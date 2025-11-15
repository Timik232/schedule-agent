from __future__ import annotations

from typing import AsyncGenerator

import pytest
from httpx import ASGITransport, AsyncClient

from src.api import server
from src.config import get_settings
from src.models.schemas import AgentReply, QueryIntent, ToolCall, ToolName


@pytest.fixture
async def client(monkeypatch: pytest.MonkeyPatch) -> AsyncGenerator[AsyncClient, None]:
    monkeypatch.setenv("LLM_PROVIDER", "lm_studio")
    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://schedule:schedule@localhost:5432/schedule")
    monkeypatch.setenv("LOG_LEVEL", "INFO")
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("LLM_API_BASE_URL", raising=False)
    get_settings.cache_clear()
    transport = ASGITransport(app=server.app)
    async with AsyncClient(transport=transport, base_url="http://test") as test_client:
        try:
            yield test_client
        finally:
            get_settings.cache_clear()


@pytest.mark.asyncio
async def test_teacher_schedule_flow_returns_ok(monkeypatch: pytest.MonkeyPatch, client: AsyncClient) -> None:
    expected_payload = {
        "teacher": "иванов",
        "date_start": "2025-11-14",
        "date_end": "2025-11-14",
        "lessons": [],
    }

    async def fake_run_agent(user_query: str, thread_id: str | None) -> AgentReply:  # noqa: ARG001
        return AgentReply(
            response="Расписание сформировано",
            intent=QueryIntent.TEACHER_SCHEDULE,
            tool_calls=[ToolCall(tool=ToolName.GET_TEACHER_SCHEDULE, parameters={})],
            data=expected_payload,
            error=None,
        )

    monkeypatch.setattr(server, "run_agent", fake_run_agent)

    response = await client.post("/query", json={"message": "Расписание Иванов с 14.11.2025 по 14.11.2025"})
    assert response.status_code == 200
    body = response.json()
    assert body["intent"] == QueryIntent.TEACHER_SCHEDULE.value
    assert body["data"] == expected_payload
    assert body["error"] is None
