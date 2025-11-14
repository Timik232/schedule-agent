"""LangGraph builder for the schedule agent."""
from __future__ import annotations

from typing import Optional

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import CompiledGraph, END, START, StateGraph

from .edges import should_execute_tools
from .state import AgentState
from . import nodes


def build_schedule_agent_graph(checkpointer: Optional[MemorySaver] = None) -> CompiledGraph:
    """Create and compile the LangGraph for schedule queries."""

    graph = StateGraph(AgentState)

    graph.add_node("parse_query", nodes.parse_query_node)
    graph.add_node("select_tool", nodes.select_tool_node)
    graph.add_node("execute_tools", nodes.execute_tools_node)
    graph.add_node("format_response", nodes.format_response_node)

    graph.add_edge(START, "parse_query")
    graph.add_edge("parse_query", "select_tool")
    graph.add_conditional_edges("select_tool", should_execute_tools)
    graph.add_edge("execute_tools", "format_response")
    graph.add_edge("format_response", END)

    return graph.compile(checkpointer=checkpointer or MemorySaver())
