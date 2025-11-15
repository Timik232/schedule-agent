from __future__ import annotations

from datetime import date, time
from typing import Generator

import pytest
from fastapi.testclient import TestClient

from src.api import server
from src.config import get_settings
from src.models.schemas import AgentReply, QueryIntent, ToolCall, ToolName


@pytest.fixture
def client(monkeypatch) -> Generator[TestClient, None, None]:
    monkeypatch.setenv("LLM_PROVIDER", "lm_studio")
    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://schedule:schedule@localhost:5432/schedule")
    monkeypatch.setenv("LOG_LEVEL", "INFO")
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("LLM_API_BASE_URL", raising=False)
    get_settings.cache_clear()
    try:
        yield TestClient(server.app)
    finally:
        get_settings.cache_clear()


def test_first_class_query_flow(monkeypatch, client: TestClient) -> None:
    expected_data = {
        "date": str(date(2025, 11, 14)),
        "start_time": str(time(9, 0)),
        "end_time": str(time(10, 30)),
        "subject": "Математический анализ",
        "teacher": "Иванов И.И.",
        "room": "301",
        "lesson_type": "Лекция",
    }

    async def fake_run_agent(user_query: str, thread_id: str | None) -> AgentReply:  # noqa: ARG001
        return AgentReply(
            response="Завтра первая пара ...",
            intent=QueryIntent.FIRST_CLASS,
            tool_calls=[ToolCall(tool=ToolName.GET_FIRST_CLASS_FOR_GROUP, parameters={})],
            data=expected_data,
            error=None,
        )

    monkeypatch.setattr(server, "run_agent", fake_run_agent)

    response = client.post("/query", json={"message": "Какая первая пара?"})
    body = response.json()

    assert response.status_code == 200
    assert body["intent"] == QueryIntent.FIRST_CLASS.value
    assert body["data"] == expected_data
    assert body["error"] is None
