"""Utilities to bootstrap the PostgreSQL database from a SQLite seed file."""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Sequence

import asyncpg

from ..config import get_settings
from ..utils.logging import get_logger

LOGGER = get_logger(__name__)


@dataclass
class TableSeed:
    name: str
    select_sql: str
    insert_sql: str
    transform: Callable[[sqlite3.Row], Sequence[object]]


def _to_timestamp(value: int | None) -> datetime | None:
    if value is None:
        return None
    # Seed file stores seconds since epoch (UTC).
    return datetime.fromtimestamp(value, tz=timezone.utc).replace(tzinfo=None)


def _load_sqlite_rows(connection: sqlite3.Connection, query: str) -> list[sqlite3.Row]:
    cursor = connection.execute(query)
    return cursor.fetchall()


async def migrate_sqlite_seed_if_needed() -> None:
    """Ensure the PostgreSQL database is populated from the bundled SQLite seed."""

    settings = get_settings()
    seed_path_raw = settings.sqlite_seed_path
    if not seed_path_raw:
        LOGGER.info("No SQLite seed configured; skipping database bootstrap")
        return

    sqlite_path = Path(seed_path_raw)
    if not sqlite_path.exists():
        LOGGER.bind(path=str(sqlite_path)).warning("SQLite seed file not found; skipping bootstrap")
        return

    LOGGER.bind(sqlite_seed=str(sqlite_path)).info("Checking PostgreSQL seed status")

    try:
        sqlite_conn = sqlite3.connect(str(sqlite_path))
        sqlite_conn.row_factory = sqlite3.Row
    except Exception as exc:  # noqa: BLE001
        LOGGER.bind(error=str(exc)).error("Failed to open SQLite seed file")
        return

    try:
        await _seed_postgres(sqlite_conn, settings.postgres_dsn)
    finally:
        sqlite_conn.close()


async def _seed_postgres(sqlite_conn: sqlite3.Connection, postgres_dsn: str) -> None:
    try:
        conn = await asyncpg.connect(postgres_dsn)
    except Exception as exc:  # noqa: BLE001
        LOGGER.bind(error=str(exc)).error("Failed to connect to PostgreSQL for seeding")
        return

    try:
        lesson_table = await conn.fetchval("SELECT to_regclass('public.lesson')")
        if lesson_table:
            existing_rows = await conn.fetchval("SELECT COUNT(*) FROM lesson")
            if existing_rows and existing_rows > 0:
                LOGGER.info("PostgreSQL already contains lesson data; skipping bootstrap")
                return

        await _create_tables(conn)
        await _copy_data(conn, sqlite_conn)
        LOGGER.info("PostgreSQL seed completed successfully")
    finally:
        await conn.close()


async def _create_tables(conn: asyncpg.Connection) -> None:
    statements = [
        """
        CREATE TABLE IF NOT EXISTS academic_group (
            id INTEGER PRIMARY KEY,
            title TEXT NOT NULL,
            normalized_title TEXT NOT NULL
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS discipline (
            id INTEGER PRIMARY KEY,
            title TEXT NOT NULL
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS lesson_type (
            id INTEGER PRIMARY KEY,
            title TEXT NOT NULL
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS place (
            id INTEGER PRIMARY KEY,
            title TEXT NOT NULL,
            normalized_title TEXT NOT NULL,
            campus TEXT
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS schedule_version (
            id INTEGER PRIMARY KEY,
            start TIMESTAMP WITHOUT TIME ZONE NOT NULL,
            "end" TIMESTAMP WITHOUT TIME ZONE NOT NULL,
            period_type INTEGER NOT NULL
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS teacher (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            first_name TEXT,
            middle_name TEXT,
            last_name TEXT,
            normalized_name TEXT NOT NULL
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS lesson (
            id INTEGER PRIMARY KEY,
            start TIMESTAMP WITHOUT TIME ZONE NOT NULL,
            "end" TIMESTAMP WITHOUT TIME ZONE NOT NULL,
            lesson_number INTEGER NOT NULL,
            schedule_version_id INTEGER NOT NULL REFERENCES schedule_version(id) ON DELETE CASCADE,
            discipline_id INTEGER NOT NULL REFERENCES discipline(id) ON DELETE CASCADE,
            lesson_type_id INTEGER NOT NULL REFERENCES lesson_type(id) ON DELETE CASCADE
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS lesson_academic_group (
            lesson_id INTEGER NOT NULL REFERENCES lesson(id) ON DELETE CASCADE,
            academic_group_id INTEGER NOT NULL REFERENCES academic_group(id) ON DELETE CASCADE,
            PRIMARY KEY (lesson_id, academic_group_id)
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS lesson_place (
            lesson_id INTEGER NOT NULL REFERENCES lesson(id) ON DELETE CASCADE,
            place_id INTEGER NOT NULL REFERENCES place(id) ON DELETE CASCADE,
            PRIMARY KEY (lesson_id, place_id)
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS lesson_teacher (
            lesson_id INTEGER NOT NULL REFERENCES lesson(id) ON DELETE CASCADE,
            teacher_id INTEGER NOT NULL REFERENCES teacher(id) ON DELETE CASCADE,
            PRIMARY KEY (lesson_id, teacher_id)
        )
        """,
    ]

    for statement in statements:
        await conn.execute(statement)


async def _copy_data(conn: asyncpg.Connection, sqlite_conn: sqlite3.Connection) -> None:
    seeds: list[TableSeed] = [
        TableSeed(
            name="academic_group",
            select_sql="SELECT id, title, normalized_title FROM academic_group",
            insert_sql="""
                INSERT INTO academic_group (id, title, normalized_title)
                VALUES ($1, $2, $3)
                ON CONFLICT (id) DO NOTHING
            """,
            transform=lambda row: (row["id"], row["title"], row["normalized_title"]),
        ),
        TableSeed(
            name="discipline",
            select_sql="SELECT id, title FROM discipline",
            insert_sql="""
                INSERT INTO discipline (id, title)
                VALUES ($1, $2)
                ON CONFLICT (id) DO NOTHING
            """,
            transform=lambda row: (row["id"], row["title"]),
        ),
        TableSeed(
            name="lesson_type",
            select_sql="SELECT id, title FROM lesson_type",
            insert_sql="""
                INSERT INTO lesson_type (id, title)
                VALUES ($1, $2)
                ON CONFLICT (id) DO NOTHING
            """,
            transform=lambda row: (row["id"], row["title"]),
        ),
        TableSeed(
            name="place",
            select_sql="SELECT id, title, normalized_title, campus FROM place",
            insert_sql="""
                INSERT INTO place (id, title, normalized_title, campus)
                VALUES ($1, $2, $3, $4)
                ON CONFLICT (id) DO NOTHING
            """,
            transform=lambda row: (row["id"], row["title"], row["normalized_title"], row["campus"]),
        ),
        TableSeed(
            name="schedule_version",
            select_sql="SELECT id, start, end, period_type FROM schedule_version",
            insert_sql="""
                INSERT INTO schedule_version (id, start, "end", period_type)
                VALUES ($1, $2, $3, $4)
                ON CONFLICT (id) DO NOTHING
            """,
            transform=lambda row: (
                row["id"],
                _to_timestamp(row["start"]),
                _to_timestamp(row["end"]),
                row["period_type"],
            ),
        ),
        TableSeed(
            name="teacher",
            select_sql="SELECT id, name, first_name, middle_name, last_name, normalized_name FROM teacher",
            insert_sql="""
                INSERT INTO teacher (id, name, first_name, middle_name, last_name, normalized_name)
                VALUES ($1, $2, $3, $4, $5, $6)
                ON CONFLICT (id) DO NOTHING
            """,
            transform=lambda row: (
                row["id"],
                row["name"],
                row["first_name"],
                row["middle_name"],
                row["last_name"],
                row["normalized_name"],
            ),
        ),
        TableSeed(
            name="lesson",
            select_sql="SELECT id, start, end, lesson_number, schedule_version_id, discipline_id, lesson_type_id FROM lesson",
            insert_sql="""
                INSERT INTO lesson (id, start, "end", lesson_number, schedule_version_id, discipline_id, lesson_type_id)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                ON CONFLICT (id) DO NOTHING
            """,
            transform=lambda row: (
                row["id"],
                _to_timestamp(row["start"]),
                _to_timestamp(row["end"]),
                row["lesson_number"],
                row["schedule_version_id"],
                row["discipline_id"],
                row["lesson_type_id"],
            ),
        ),
        TableSeed(
            name="lesson_academic_group",
            select_sql="SELECT lesson_id, academic_group_id FROM lesson_academic_group",
            insert_sql="""
                INSERT INTO lesson_academic_group (lesson_id, academic_group_id)
                VALUES ($1, $2)
                ON CONFLICT (lesson_id, academic_group_id) DO NOTHING
            """,
            transform=lambda row: (row["lesson_id"], row["academic_group_id"]),
        ),
        TableSeed(
            name="lesson_place",
            select_sql="SELECT lesson_id, place_id FROM lesson_place",
            insert_sql="""
                INSERT INTO lesson_place (lesson_id, place_id)
                VALUES ($1, $2)
                ON CONFLICT (lesson_id, place_id) DO NOTHING
            """,
            transform=lambda row: (row["lesson_id"], row["place_id"]),
        ),
        TableSeed(
            name="lesson_teacher",
            select_sql="SELECT lesson_id, teacher_id FROM lesson_teacher",
            insert_sql="""
                INSERT INTO lesson_teacher (lesson_id, teacher_id)
                VALUES ($1, $2)
                ON CONFLICT (lesson_id, teacher_id) DO NOTHING
            """,
            transform=lambda row: (row["lesson_id"], row["teacher_id"]),
        ),
    ]

    seeded_tables: list[str] = []

    for seed in seeds:
        rows = _load_sqlite_rows(sqlite_conn, seed.select_sql)
        if not rows:
            continue
        payloads = [seed.transform(row) for row in rows]
        LOGGER.bind(table=seed.name, rows=len(payloads)).debug("Seeding table")
        await conn.executemany(seed.insert_sql, payloads)
        seeded_tables.append(seed.name)

    if seeded_tables:
        LOGGER.bind(tables=seeded_tables).info("Seed data copied")
