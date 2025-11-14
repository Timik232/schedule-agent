"""Parameterized SQL templates for schedule retrieval."""
from __future__ import annotations

from datetime import date, datetime
from typing import Any, Iterable

from .database import execute_query


async def get_first_class_for_group(group_norm: str, lesson_date: date) -> dict[str, Any] | None:
    """Return the first lesson for the given academic group on a specific date."""

    sql = """
        SELECT
            l.start::date AS date,
            l.start::time AS start_time,
            l."end"::time AS end_time,
            d.title AS subject,
            p.title AS room,
            t.name AS teacher,
            lt.title AS lesson_type
        FROM lesson AS l
        JOIN lesson_academic_group AS lag ON lag.lesson_id = l.id
        JOIN academic_group AS g ON g.id = lag.academic_group_id
        LEFT JOIN discipline AS d ON d.id = l.discipline_id
        LEFT JOIN lesson_type AS lt ON lt.id = l.lesson_type_id
        LEFT JOIN lesson_teacher AS ltch ON ltch.lesson_id = l.id
        LEFT JOIN teacher AS t ON t.id = ltch.teacher_id
        LEFT JOIN lesson_place AS lp ON lp.lesson_id = l.id
        LEFT JOIN place AS p ON p.id = lp.place_id
        WHERE g.normalized_title = :group_norm
          AND l.start::date = :lesson_date
        ORDER BY l.start
        LIMIT 1
    """

    rows = await execute_query(
        sql,
        {
            "group_norm": group_norm,
            "lesson_date": lesson_date,
        },
    )
    return rows[0] if rows else None


async def get_teacher_schedule(teacher_norm: str, date_start: date, date_end: date) -> list[dict[str, Any]]:
    """Return ordered lessons for a teacher between two dates."""

    sql = """
        SELECT
            l.id AS lesson_id,
            l.start::date AS date,
            l.start::time AS start_time,
            l."end"::time AS end_time,
            d.title AS subject,
            array_remove(array_agg(DISTINCT g.title), NULL) AS groups,
            array_remove(array_agg(DISTINCT p.title), NULL) AS rooms,
            lt.title AS lesson_type
        FROM lesson AS l
        JOIN lesson_teacher AS ltch ON ltch.lesson_id = l.id
        JOIN teacher AS t ON t.id = ltch.teacher_id
        LEFT JOIN lesson_academic_group AS lag ON lag.lesson_id = l.id
        LEFT JOIN academic_group AS g ON g.id = lag.academic_group_id
        LEFT JOIN lesson_place AS lp ON lp.lesson_id = l.id
        LEFT JOIN place AS p ON p.id = lp.place_id
        LEFT JOIN discipline AS d ON d.id = l.discipline_id
        LEFT JOIN lesson_type AS lt ON lt.id = l.lesson_type_id
        WHERE t.normalized_name = :teacher_norm
          AND l.start::date BETWEEN :date_start AND :date_end
        GROUP BY l.id, l.start, l."end", d.title, lt.title
        ORDER BY l.start
        LIMIT 1000
    """

    rows = await execute_query(
        sql,
        {
            "teacher_norm": teacher_norm,
            "date_start": date_start,
            "date_end": date_end,
        },
    )

    normalised_rows: list[dict[str, Any]] = []
    for row in rows:
        groups = row.get("groups") or []
        rooms = row.get("rooms") or []
        normalised_rows.append(
            {
                "lesson_id": row.get("lesson_id"),
                "date": row.get("date"),
                "start_time": row.get("start_time"),
                "end_time": row.get("end_time"),
                "subject": row.get("subject"),
                "groups": list(groups),
                "rooms": list(rooms),
                "lesson_type": row.get("lesson_type"),
            }
        )

    return normalised_rows


async def get_group_schedule(group_norm: str, date_start: date, date_end: date) -> list[dict[str, Any]]:
    """Return ordered lessons for a group between two dates."""

    sql = """
        SELECT
            l.start::date AS date,
            l.start::time AS start_time,
            l."end"::time AS end_time,
            d.title AS subject,
            array_remove(array_agg(DISTINCT t.name), NULL) AS teachers,
            array_remove(array_agg(DISTINCT p.title), NULL) AS rooms,
            lt.title AS lesson_type
        FROM lesson AS l
        JOIN lesson_academic_group AS lag ON lag.lesson_id = l.id
        JOIN academic_group AS g ON g.id = lag.academic_group_id
        LEFT JOIN discipline AS d ON d.id = l.discipline_id
        LEFT JOIN lesson_type AS lt ON lt.id = l.lesson_type_id
        LEFT JOIN lesson_teacher AS ltch ON ltch.lesson_id = l.id
        LEFT JOIN teacher AS t ON t.id = ltch.teacher_id
        LEFT JOIN lesson_place AS lp ON lp.lesson_id = l.id
        LEFT JOIN place AS p ON p.id = lp.place_id
        WHERE g.normalized_title = :group_norm
          AND l.start::date BETWEEN :date_start AND :date_end
        GROUP BY l.id, l.start, l."end", d.title, lt.title
        ORDER BY l.start
        LIMIT 1000
    """

    rows = await execute_query(
        sql,
        {
            "group_norm": group_norm,
            "date_start": date_start,
            "date_end": date_end,
        },
    )

    lessons: list[dict[str, Any]] = []
    for row in rows:
        lessons.append(
            {
                "date": row.get("date"),
                "start_time": row.get("start_time"),
                "end_time": row.get("end_time"),
                "subject": row.get("subject"),
                "teachers": list(row.get("teachers") or []),
                "rooms": list(row.get("rooms") or []),
                "lesson_type": row.get("lesson_type"),
            }
        )

    return lessons


async def get_room_schedule(room_norm: str, date_start: date, date_end: date) -> list[dict[str, Any]]:
    """Return ordered lessons for a room between two dates."""

    sql = """
        SELECT
            l.start::date AS date,
            l.start::time AS start_time,
            l."end"::time AS end_time,
            d.title AS subject,
            array_remove(array_agg(DISTINCT g.title), NULL) AS groups,
            array_remove(array_agg(DISTINCT t.name), NULL) AS teachers,
            lt.title AS lesson_type
        FROM lesson AS l
        JOIN lesson_place AS lp ON lp.lesson_id = l.id
        JOIN place AS p ON p.id = lp.place_id
        LEFT JOIN discipline AS d ON d.id = l.discipline_id
        LEFT JOIN lesson_type AS lt ON lt.id = l.lesson_type_id
        LEFT JOIN lesson_academic_group AS lag ON lag.lesson_id = l.id
        LEFT JOIN academic_group AS g ON g.id = lag.academic_group_id
        LEFT JOIN lesson_teacher AS ltch ON ltch.lesson_id = l.id
        LEFT JOIN teacher AS t ON t.id = ltch.teacher_id
        WHERE p.normalized_title = :room_norm
          AND l.start::date BETWEEN :date_start AND :date_end
        GROUP BY l.id, l.start, l."end", d.title, lt.title
        ORDER BY l.start
        LIMIT 1000
    """

    rows = await execute_query(
        sql,
        {
            "room_norm": room_norm,
            "date_start": date_start,
            "date_end": date_end,
        },
    )

    lessons: list[dict[str, Any]] = []
    for row in rows:
        lessons.append(
            {
                "date": row.get("date"),
                "start_time": row.get("start_time"),
                "end_time": row.get("end_time"),
                "subject": row.get("subject"),
                "groups": list(row.get("groups") or []),
                "teachers": list(row.get("teachers") or []),
                "lesson_type": row.get("lesson_type"),
            }
        )

    return lessons


async def get_teacher_busy_intervals(
    teacher_norm: str,
    range_start: datetime,
    range_end: datetime,
) -> list[tuple[datetime, datetime]]:
    """Return busy intervals for a teacher within the supplied datetime window."""

    sql = """
        SELECT l.start, l."end"
        FROM lesson AS l
        JOIN lesson_teacher AS lt ON lt.lesson_id = l.id
        JOIN teacher AS t ON t.id = lt.teacher_id
        WHERE t.normalized_name = :teacher_norm
          AND l.start < :range_end
          AND l."end" > :range_start
        ORDER BY l.start
    """

    rows = await execute_query(
        sql,
        {
            "teacher_norm": teacher_norm,
            "range_start": range_start,
            "range_end": range_end,
        },
    )
    return [(row["start"], row["end"]) for row in rows]


async def get_group_busy_intervals(
    group_norms: Iterable[str],
    range_start: datetime,
    range_end: datetime,
) -> dict[str, list[tuple[datetime, datetime]]]:
    """Return busy intervals per group within the supplied datetime window."""

    sql = """
        SELECT g.normalized_title AS group_norm, l.start, l."end"
        FROM lesson AS l
        JOIN lesson_academic_group AS lag ON lag.lesson_id = l.id
        JOIN academic_group AS g ON g.id = lag.academic_group_id
        WHERE g.normalized_title = ANY(:group_norms)
          AND l.start < :range_end
          AND l."end" > :range_start
        ORDER BY g.normalized_title, l.start
    """

    rows = await execute_query(
        sql,
        {
            "group_norms": list(group_norms),
            "range_start": range_start,
            "range_end": range_end,
        },
    )

    intervals: dict[str, list[tuple[datetime, datetime]]] = {}
    for row in rows:
        norm = row["group_norm"]
        intervals.setdefault(norm, []).append((row["start"], row["end"]))
    return intervals
