from __future__ import annotations

from datetime import date, time

import pytest

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
