from __future__ import annotations

import json
from datetime import date, datetime, time
from types import SimpleNamespace
from typing import Callable, cast

import pytest

from langchain_core.language_models.chat_models import BaseChatModel

from src.graph import nodes
from src.models.schemas import QueryIntent
from src.graph.state import AgentState
from src.tools import sql_templates


@pytest.mark.asyncio
async def test_get_first_class_for_group_returns_first(monkeypatch: pytest.MonkeyPatch) -> None:
    sample_rows = [
        {
            "date": date(2025, 11, 14),
            "start_time": time(9, 0),
            "end_time": time(10, 30),
            "subject": "Математический анализ",
            "room": "301",
            "teacher": "Иванов И.И.",
            "lesson_type": "Лекция",
        }
    ]

    async def fake_execute_query(sql: str, params: dict[str, object]):  # noqa: ARG001
        return sample_rows

    monkeypatch.setattr(sql_templates, "execute_query", fake_execute_query)

    result = await sql_templates.get_first_class_for_group("икмо-05-21", date(2025, 11, 14))
    assert result == sample_rows[0]


class _LLMStub:
    def __init__(self, payload: dict[str, object] | str) -> None:
        self._payload = payload

    async def ainvoke(self, *_args, **_kwargs):
        content = self._payload
        if isinstance(content, dict):
            content = json.dumps(content, ensure_ascii=False)
        return SimpleNamespace(content=content)


@pytest.mark.asyncio
async def test_parse_query_node_uses_configured_factory(monkeypatch: pytest.MonkeyPatch) -> None:
    entities_payload = {
        "intent": QueryIntent.FIRST_CLASS.value,
        "groups": ["икмо-05-21"],
        "teachers": [],
        "rooms": [],
        "date": "2025-11-14",
        "date_range": None,
        "time_range": None,
    }

    def _factory(*, streaming: bool = False):  # noqa: ARG001
        return _LLMStub(entities_payload)

    nodes.configure_llm_client_factory(cast(Callable[..., BaseChatModel], _factory))  # type: ignore[arg-type]

    try:
        state = cast(AgentState, {"user_query": "Какая первая пара у ИКМО-05-21 14 ноября?"})
        result = await nodes.parse_query_node(state)
        assert result["intent"] == QueryIntent.FIRST_CLASS
        assert result["entities"].groups == ["икмо-05-21"]
    finally:
        nodes.configure_llm_client_factory(None)
