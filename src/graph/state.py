"""Shared state definitions for the LangGraph schedule agent."""
from __future__ import annotations

from typing import Any, Annotated, Optional, TypedDict

from langchain_core.messages import BaseMessage, add_messages

from ..models.schemas import ExtractedEntities, QueryIntent, ToolCall


class AgentState(TypedDict, total=False):
    """State carried across LangGraph nodes."""

    messages: Annotated[list[BaseMessage], add_messages]
    user_query: str
    intent: Optional[QueryIntent]
    entities: Optional[ExtractedEntities]
    tool_calls: list[ToolCall]
    tool_results: list[dict[str, Any]]
    final_response: Optional[str]
    error: Optional[str]
    retry_count: int


AgentStateUpdate = dict[str, Any]
