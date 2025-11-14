from __future__ import annotations

from datetime import date, time

from fastapi.testclient import TestClient

from src.api import server
from src.models.schemas import AgentReply, QueryIntent, ToolCall

client = TestClient(server.app)


def test_first_class_query_flow(monkeypatch) -> None:
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
            tool_calls=[ToolCall(tool="get_first_class_for_group", parameters={})],
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
