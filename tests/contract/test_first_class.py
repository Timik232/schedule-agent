from __future__ import annotations

from datetime import date, datetime, time

import pytest

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
