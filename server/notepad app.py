from typing import TypedDict, Annotated, Optional, List, Any
from langgraph.graph import add_messages, StateGraph, END
from langchain_core.messages import HumanMessage, AIMessageChunk, ToolMessage, BaseMessage
from langchain_community.tools.tavily_search import TavilySearchResults
from dotenv import load_dotenv
from fastapi import FastAPI, Query
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from langgraph.checkpoint.memory import MemorySaver
from langchain.chat_models.base import BaseChatModel
from langchain.schema import AIMessage
from langchain_core.outputs import ChatResult, ChatGeneration
from langchain_core.callbacks import CallbackManagerForLLMRun
import aiohttp
import json
import os
from uuid import uuid4

# Load environment variables
load_dotenv()
GROK_API_KEY = os.getenv("GROK_API_KEY")
GROK_API_BASE = "https://api.x.ai/v1"

if not GROK_API_KEY:
    raise ValueError("❌ Missing GROK_API_KEY in your .env file")

# ChatGrok Implementation
class ChatGrok(BaseChatModel):
    """Custom LLM wrapper for Grok (xAI)"""
    model: str = "grok-beta"
    api_key: str = GROK_API_KEY
    api_base: str = GROK_API_BASE

    @property
    def _llm_type(self) -> str:
        """Return identifier for LLM type."""
        return "grok"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Synchronous generate - not implemented, use async"""
        raise NotImplementedError("ChatGrok only supports async. Use ainvoke()")

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Async API call to Grok"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model,
            "messages": [{"role": m.type, "content": m.content} for m in messages],
            "stream": False,
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.api_base}/chat/completions", 
                headers=headers, 
                json=payload
            ) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    raise RuntimeError(f"Grok API Error {resp.status}: {text}")
                data = await resp.json()

        content = data["choices"][0]["message"]["content"]
        message = AIMessage(content=content)
        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])

    async def ainvoke(self, messages, **kwargs):
        """Async invoke method"""
        result = await self._agenerate(messages, **kwargs)
        return result.generations[0].message

# Memory and State
memory = MemorySaver()

class State(TypedDict):
    messages: Annotated[list, add_messages]

# Tools
search_tool = TavilySearchResults(max_results=4)
tools = [search_tool]

# Initialize LLM
llm = ChatGrok(model="grok-beta")

# Graph nodes
async def model(state: State):
    result = await llm.ainvoke(state["messages"])
    return {"messages": [result]}

async def tools_router(state: State):
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and len(last_message.tool_calls) > 0:
        return "tool_node"
    else:
        return END

async def tool_node(state):
    tool_calls = getattr(state["messages"][-1], "tool_calls", [])
    tool_messages = []

    for tool_call in tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        tool_id = tool_call["id"]

        if tool_name == "tavily_search_results_json":
            search_results = await search_tool.ainvoke(tool_args)
            tool_message = ToolMessage(
                content=str(search_results),
                tool_call_id=tool_id,
                name=tool_name
            )
            tool_messages.append(tool_message)

    return {"messages": tool_messages}

# Build graph
graph_builder = StateGraph(State)
graph_builder.add_node("model", model)
graph_builder.add_node("tool_node", tool_node)
graph_builder.set_entry_point("model")
graph_builder.add_conditional_edges("model", tools_router)
graph_builder.add_edge("tool_node", "model")
graph = graph_builder.compile(checkpointer=memory)

# FastAPI app
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def serialise_ai_message_chunk(chunk):
    if isinstance(chunk, AIMessageChunk):
        return chunk.content
    raise TypeError(f"Invalid chunk type: {type(chunk).__name__}")

async def generate_chat_responses(message: str, checkpoint_id: Optional[str] = None):
    is_new_conversation = checkpoint_id is None

    if is_new_conversation:
        new_checkpoint_id = str(uuid4())
        config = {"configurable": {"thread_id": new_checkpoint_id}}
        events = graph.astream_events(
            {"messages": [HumanMessage(content=message)]},
            version="v2",
            config=config
        )
        yield f"data: {{\"type\": \"checkpoint\", \"checkpoint_id\": \"{new_checkpoint_id}\"}}\n\n"
    else:
        config = {"configurable": {"thread_id": checkpoint_id}}
        events = graph.astream_events(
            {"messages": [HumanMessage(content=message)]},
            version="v2",
            config=config
        )

    async for event in events:
        event_type = event["event"]

        if event_type == "on_chat_model_stream":
            chunk_content = serialise_ai_message_chunk(event["data"]["chunk"])
            safe_content = chunk_content.replace("'", "\\'").replace("\n", "\\n")
            yield f"data: {{\"type\": \"content\", \"content\": \"{safe_content}\"}}\n\n"

        elif event_type == "on_chat_model_end":
            tool_calls = getattr(event["data"]["output"], "tool_calls", [])
            search_calls = [c for c in tool_calls if c["name"] == "tavily_search_results_json"]
            if search_calls:
                query = search_calls[0]["args"].get("query", "")
                safe_query = query.replace('"', '\\"').replace("'", "\\'").replace("\n", "\\n")
                yield f"data: {{\"type\": \"search_start\", \"query\": \"{safe_query}\"}}\n\n"

        elif event_type == "on_tool_end" and event["name"] == "tavily_search_results_json":
            output = event["data"]["output"]
            if isinstance(output, list):
                urls = [item["url"] for item in output if isinstance(item, dict) and "url" in item]
                urls_json = json.dumps(urls)
                yield f"data: {{\"type\": \"search_results\", \"urls\": {urls_json}}}\n\n"

    yield f"data: {{\"type\": \"end\"}}\n\n"

@app.get("/chat_stream/{message}")
async def chat_stream(message: str, checkpoint_id: Optional[str] = Query(None)):
    return StreamingResponse(
        generate_chat_responses(message, checkpoint_id),
        media_type="text/event-stream"
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
