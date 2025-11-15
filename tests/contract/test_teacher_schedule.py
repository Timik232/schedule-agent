from __future__ import annotations

import json
from datetime import date, time
from types import SimpleNamespace
from typing import Callable, cast

import pytest

from langchain_core.language_models.chat_models import BaseChatModel

from src.graph import nodes
from src.graph.state import AgentState
from src.models.schemas import ToolName
from src.tools import sql_templates


@pytest.mark.asyncio
async def test_get_teacher_schedule_orders_lessons(monkeypatch: pytest.MonkeyPatch) -> None:
    sample_rows = [
        {
            "date": date(2025, 11, 14),
            "start_time": time(9, 0),
            "end_time": time(10, 30),
            "subject": "Алгебра",
            "groups": ["икмо-05-21"],
            "room": "301",
            "lesson_type": "Лекция",
        },
        {
            "date": date(2025, 11, 14),
            "start_time": time(11, 0),
            "end_time": time(12, 30),
            "subject": "Геометрия",
            "groups": ["икмо-05-21"],
            "room": "305",
            "lesson_type": "Практика",
        },
    ]

    async def fake_execute_query(sql: str, params: dict[str, object]):  # noqa: ARG001
        return sample_rows

    monkeypatch.setattr(sql_templates, "execute_query", fake_execute_query)

    result = await sql_templates.get_teacher_schedule("иванов", date(2025, 11, 14), date(2025, 11, 14))
    assert result == sample_rows


class _LLMStub:
    def __init__(self, payload: object) -> None:
        self._payload = payload

    async def ainvoke(self, *_args, **_kwargs):  # pragma: no cover - helper stub
        content = self._payload
        if isinstance(content, dict):
            content = json.dumps(content, ensure_ascii=False)
        return SimpleNamespace(content=content)


@pytest.mark.asyncio
async def test_format_response_node_uses_factory_for_teacher_schedule(monkeypatch: pytest.MonkeyPatch) -> None:
    formatted_text = "Расписание подготовлено"

    def _factory(*, streaming: bool = False):  # noqa: ARG001
        return _LLMStub(formatted_text)

    nodes.configure_llm_client_factory(cast(Callable[..., BaseChatModel], _factory))  # type: ignore[arg-type]

    try:
        state = cast(
            AgentState,
            {
                "tool_results": [
                    {
                        "tool": ToolName.GET_TEACHER_SCHEDULE,
                        "result": {
                            "teacher": "иванов",
                            "date_start": date(2025, 11, 14),
                            "date_end": date(2025, 11, 14),
                            "lessons": [],
                        },
                    }
                ]
            },
        )
        result = await nodes.format_response_node(state)
    finally:
        nodes.configure_llm_client_factory(None)

    assert result["final_response"] == formatted_text
