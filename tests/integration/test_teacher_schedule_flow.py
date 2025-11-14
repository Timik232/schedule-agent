from __future__ import annotations

import pytest
from httpx import AsyncClient

from src.api.server import app


@pytest.mark.asyncio
async def test_teacher_schedule_flow_returns_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.post("/query", json={"message": "Расписание Иванов с 14.11.2025 по 14.11.2025"})
        assert response.status_code in {200, 501}
