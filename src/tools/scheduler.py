"""Scheduling helpers for computing shared availability."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time, timedelta
from typing import Any, Iterable, Sequence

from ..utils.logging import get_logger

LOGGER = get_logger(__name__)


@dataclass(frozen=True)
class BusyInterval:
    """Closed-open interval representing an occupied time span."""

    start: datetime
    end: datetime

    def overlaps(self, other_start: datetime, other_end: datetime) -> bool:
        return self.start < other_end and other_start < self.end


def _normalize_window_bounds(search_start: datetime, search_end: datetime) -> tuple[datetime, datetime]:
    """Clamp the search window to working hours (08:00-20:00)."""

    if search_start.date() == search_end.date():
        day_start = datetime.combine(search_start.date(), time(8, 0))
        day_end = datetime.combine(search_start.date(), time(20, 0))
        return max(search_start, day_start), min(search_end, day_end)

    clamped_start = max(search_start, datetime.combine(search_start.date(), time(8, 0)))
    clamped_end = min(search_end, datetime.combine(search_end.date(), time(20, 0)))
    return clamped_start, clamped_end


def _generate_candidate_slots(
    window_start: datetime,
    window_end: datetime,
    slot_duration: timedelta,
    step: timedelta,
) -> list[tuple[datetime, datetime]]:
    """Generate candidate slots between two datetimes respecting working hours."""

    slots: list[tuple[datetime, datetime]] = []
    current_day = window_start.date()
    final_day = window_end.date()

    while current_day <= final_day:
        day_start = datetime.combine(current_day, time(8, 0))
        day_end = datetime.combine(current_day, time(20, 0))

        current_window_start = max(window_start, day_start)
        current_window_end = min(window_end, day_end)
        latest_start = current_window_end - slot_duration

        cursor = day_start
        while cursor <= latest_start:
            slot_end = cursor + slot_duration
            if cursor >= current_window_start and slot_end <= current_window_end:
                slots.append((cursor, slot_end))
            cursor += step

        current_day += timedelta(days=1)

    return slots


def find_common_free_slots(
    teacher_busy: Sequence[BusyInterval],
    groups_busy: Sequence[Sequence[BusyInterval]],
    search_start: datetime,
    search_end: datetime,
    slot_duration_minutes: int = 90,
    step_minutes: int = 15,
    max_results: int = 10,
) -> list[dict[str, Any]]:
    """Return shared free slots for a teacher and multiple groups."""

    if search_start >= search_end:
        LOGGER.warning("Empty search window", start=search_start, end=search_end)
        return []

    slot_duration = timedelta(minutes=slot_duration_minutes)
    step = timedelta(minutes=step_minutes)
    clamped_start, clamped_end = _normalize_window_bounds(search_start, search_end)
    if clamped_start >= clamped_end:
        LOGGER.warning("Search window outside working hours", start=search_start, end=search_end)
        return []

    candidates = _generate_candidate_slots(clamped_start, clamped_end, slot_duration, step)

    def _is_slot_free(start: datetime, end: datetime) -> bool:
        if any(interval.overlaps(start, end) for interval in teacher_busy):
            return False
        for group_intervals in groups_busy:
            if any(interval.overlaps(start, end) for interval in group_intervals):
                return False
        return True

    free_slots: list[dict[str, datetime]] = []
    for slot_start, slot_end in candidates:
        if _is_slot_free(slot_start, slot_end):
            free_slots.append(
                {
                    "start": slot_start,
                    "end": slot_end,
                    "duration_minutes": slot_duration_minutes,
                    "confidence_score": 1.0,
                }
            )
        if len(free_slots) >= max_results:
            break

    return free_slots
