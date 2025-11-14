"""FastAPI application exposing the schedule agent endpoints."""
from __future__ import annotations

from typing import Optional

from fastapi import FastAPI, HTTPException
from langchain_core.messages import HumanMessage
from langgraph.graph import CompiledGraph
from pydantic import BaseModel

from ..config import get_settings
from ..graph.builder import build_schedule_agent_graph
from ..graph.state import AgentState
from ..models.schemas import AgentReply, ToolCall
from ..utils.guardrails import guard_and_normalize_message
from ..utils.validators import ValidationError

_GRAPH: Optional[CompiledGraph] = None


def _get_graph() -> CompiledGraph:
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_schedule_agent_graph()
    return _GRAPH

app = FastAPI(title="Schedule Agent API", version="0.1.0")


class QueryRequest(BaseModel):
    message: str
    thread_id: str | None = None


@app.get("/health")
async def health() -> dict[str, str]:
    """Return basic service health details."""

    settings = get_settings()
    return {
        "status": "ok",
        "database": "unknown",
        "llm": settings.llm_model,
    }


@app.post("/query", response_model=AgentReply)
async def query_schedule(request: QueryRequest) -> AgentReply:
    """Handle schedule queries via the LangGraph agent."""

    try:
        sanitized = guard_and_normalize_message(request.message)
    except ValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail="Failed to process query") from exc

    return await run_agent(sanitized, request.thread_id)


async def run_agent(user_query: str, thread_id: str | None) -> AgentReply:
    """Invoke the LangGraph agent to answer a query. Placeholder implementation."""

    graph = _get_graph()
    initial_state: AgentState = {
        "messages": [HumanMessage(content=user_query)],
        "user_query": user_query,
        "tool_calls": [],
        "tool_results": [],
        "retry_count": 0,
    }

    try:
        final_state = await graph.ainvoke(initial_state, config={"thread_id": thread_id or "default"})
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail="Agent execution failed") from exc

    tool_calls = [
        call if isinstance(call, ToolCall) else ToolCall.model_validate(call)
        for call in final_state.get("tool_calls", [])
    ]
    final_response = final_state.get("final_response") or "Ответ не сформирован."
    error = final_state.get("error")
    intent = final_state.get("intent")
    data = final_state.get("final_payload")
    if data is None and final_state.get("tool_results"):
        data = final_state["tool_results"][-1].get("result")

    return AgentReply(
        response=final_response,
        intent=intent,
        tool_calls=tool_calls,
        data=data,
        error=error,
    )
