"""Prompt templates used by the LangGraph nodes."""
from __future__ import annotations

from textwrap import dedent


def build_entity_extraction_prompt(user_query: str) -> str:
    """Return prompt guiding the LLM to extract intent and entities."""

    return dedent(
        f"""
        You are a classification assistant for a university schedule agent.
        Only extract entities stated in the user's request.
        Reply using the provided JSON schema.
        Never guess or infer unstated data.

        User query: "{user_query}"
        """
    ).strip()


FORMAT_RESPONSE_SYSTEM_PROMPT = dedent(
    """
    You format deterministic schedule results.
    Use only the provided tool data. Do not invent facts.
    Keep responses concise and factual in Russian.
    """
).strip()


def build_format_response_prompt(serialized_payload: str) -> str:
    """Return the user content prompt for the formatting LLM call."""

    return dedent(
        f"""
        Convert the following structured payload into a short Russian response.
        Use bullet points only when multiple lessons are present.
        Payload:
        {serialized_payload}
        """
    ).strip()