from __future__ import annotations

from datetime import datetime, timedelta

from fastapi.testclient import TestClient

from src.api import server
from src.models.schemas import AgentReply, QueryIntent, ToolCall, ToolName

client = TestClient(server.app)


def test_meeting_slots_flow(monkeypatch) -> None:
    slots = [
        {
            "start": datetime(2025, 11, 14, 11, 30).isoformat(),
            "end": (datetime(2025, 11, 14, 11, 30) + timedelta(minutes=60)).isoformat(),
            "duration_minutes": 60,
            "confidence_score": 1.0,
        }
    ]

    async def fake_run_agent(user_query: str, thread_id: str | None):  # noqa: ARG001
        return AgentReply(
            response="Нашли свободные окна для встречи.",
            intent=QueryIntent.FIND_MEETING_SLOT,
            tool_calls=[ToolCall(tool=ToolName.FIND_COMMON_FREE_SLOTS, parameters={})],
            data={
                "slots": slots,
                "requested_groups": ["икмо-05-21"],
                "requested_teacher": "иванов",
            },
            error=None,
        )

    monkeypatch.setattr(server, "run_agent", fake_run_agent)

    response = client.post("/query", json={"message": "Подбери окно для встречи"})
    body = response.json()

    assert response.status_code == 200
    assert body["intent"] == QueryIntent.FIND_MEETING_SLOT.value
    assert body["data"]["slots"] == slots
    assert body["error"] is None
