"""Conditional routing helpers for the LangGraph state machine."""
from __future__ import annotations

from typing import Final

from .state import AgentState

EXECUTE_TOOLS_NODE: Final[str] = "execute_tools"
FORMAT_RESPONSE_NODE: Final[str] = "format_response"
RETRY_NODE: Final[str] = "retry"


def should_execute_tools(state: AgentState) -> str:
    """Return the next node name based on whether tools should run."""

    if state.get("error"):
        return FORMAT_RESPONSE_NODE
    if state.get("tool_calls"):
        return EXECUTE_TOOLS_NODE
    return FORMAT_RESPONSE_NODE


def should_retry(state: AgentState, max_retries: int = 1) -> str:
    """Decide whether to retry the workflow after a recoverable error."""

    if state.get("error") and state.get("retry_count", 0) < max_retries:
        return RETRY_NODE
    return FORMAT_RESPONSE_NODE
