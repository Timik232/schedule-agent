"""Shared state definitions for the LangGraph schedule agent."""
from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Annotated, Optional, TypedDict

from langchain_core.messages import BaseMessage

from ..models.schemas import ExtractedEntities, QueryIntent, ToolCall


def _extend_messages(current: Optional[list[BaseMessage]], new_messages: Iterable[BaseMessage] | BaseMessage | None) -> list[BaseMessage]:
    """Aggregator used by LangGraph to append messages in state updates."""

    if current is None:
        current = []

    if new_messages is None:
        return list(current)

    if isinstance(new_messages, BaseMessage):
        return [*current, new_messages]

    return [*current, *list(new_messages)]


class AgentState(TypedDict, total=False):
    """State carried across LangGraph nodes."""

    messages: Annotated[list[BaseMessage], _extend_messages]
    user_query: str
    intent: Optional[QueryIntent]
    entities: Optional[ExtractedEntities]
    tool_calls: list[ToolCall]
    tool_results: list[dict[str, Any]]
    final_response: Optional[str]
    error: Optional[str]
    retry_count: int


AgentStateUpdate = dict[str, Any]
