"""Helpers for computing meeting slots using deterministic data."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Sequence

from ..utils.logging import get_logger

from . import sql_templates
from .scheduler import BusyInterval, find_common_free_slots

LOGGER = get_logger(__name__)


async def plan_meeting_slots(
    *,
    teacher_norm: str | None,
    group_norms: Sequence[str],
    search_start: datetime,
    search_end: datetime,
    slot_duration_minutes: int = 90,
    step_minutes: int = 15,
    max_results: int = 10,
) -> dict[str, Any]:
    """Aggregate busy intervals and compute common free slots."""

    normalized_teacher = teacher_norm.strip().lower() if teacher_norm else None

    LOGGER.bind(teacher=normalized_teacher, groups=len(group_norms)).debug("Planning meeting slots")

    teacher_busy: list[BusyInterval] = []
    if normalized_teacher:
        teacher_intervals = await sql_templates.get_teacher_busy_intervals(
            teacher_norm=normalized_teacher,
            range_start=search_start,
            range_end=search_end,
        )
        teacher_busy = [BusyInterval(start=start, end=end) for start, end in teacher_intervals]

    normalized_groups = [group.strip().lower() for group in group_norms if group]
    group_busy_map = (
        await sql_templates.get_group_busy_intervals(
            normalized_groups,
            range_start=search_start,
            range_end=search_end,
        )
        if normalized_groups
        else {}
    )
    groups_busy: list[list[BusyInterval]] = []
    for group in normalized_groups:
        intervals = group_busy_map.get(group, [])
        groups_busy.append([BusyInterval(start=start, end=end) for start, end in intervals])

    slots = find_common_free_slots(
        teacher_busy=teacher_busy,
        groups_busy=groups_busy,
        search_start=search_start,
        search_end=search_end,
        slot_duration_minutes=slot_duration_minutes,
        step_minutes=step_minutes,
        max_results=max_results,
    )

    LOGGER.bind(slot_count=len(slots)).debug("Computed meeting slots")

    return {
        "slots": slots,
        "requested_groups": normalized_groups,
        "requested_teacher": normalized_teacher,
    }
