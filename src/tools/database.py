"""Async database utilities that enforce read-only, parameterized access."""
from __future__ import annotations

from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from ..config import get_settings
from ..utils.logging import get_logger

LOGGER = get_logger(__name__)
_ENGINE: AsyncEngine | None = None
_SESSION_FACTORY: async_sessionmaker[AsyncSession] | None = None


def _get_engine() -> AsyncEngine:
    """Return a singleton async engine backed by asyncpg."""

    global _ENGINE
    if _ENGINE is None:
        settings = get_settings()
        _ENGINE = create_async_engine(
            settings.database_url,
            echo=False,
            pool_pre_ping=True,
        )
    return _ENGINE


def _get_session_factory() -> async_sessionmaker[AsyncSession]:
    """Return a cached async session factory."""

    global _SESSION_FACTORY
    if _SESSION_FACTORY is None:
        _SESSION_FACTORY = async_sessionmaker(
            _get_engine(),
            expire_on_commit=False,
        )
    return _SESSION_FACTORY


@asynccontextmanager
async def get_session() -> AsyncIterator[AsyncSession]:
    """Yield an async session with read-only transaction semantics."""

    settings = get_settings()
    session_factory = _get_session_factory()
    async with session_factory() as session:
        async with session.begin():
            await session.execute(text("SET TRANSACTION READ ONLY"))
            await session.execute(
                text("SET LOCAL statement_timeout = :timeout_ms"),
                {"timeout_ms": settings.query_timeout_seconds * 1000},
            )
            yield session


async def execute_query(sql: str, parameters: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    """Execute a SELECT statement and return rows as dictionaries."""

    params = dict(parameters or {})
    settings = get_settings()

    async with get_session() as session:
        result = await session.execute(text(sql), params)
        rows = result.mappings().fetchall()

    row_count = len(rows)
    if row_count > settings.max_sql_rows:
        LOGGER.bind(requested=row_count, capped=settings.max_sql_rows).warning(
            "Truncated SQL result set to max rows",
        )
        rows = rows[: settings.max_sql_rows]

    return [dict(row) for row in rows]
