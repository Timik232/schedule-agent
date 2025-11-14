from __future__ import annotations

from datetime import datetime

from src.tools.scheduler import BusyInterval, find_common_free_slots


def test_find_common_free_slots_respects_busy_intervals() -> None:
    teacher_busy = [
        BusyInterval(start=datetime(2025, 11, 14, 9, 0), end=datetime(2025, 11, 14, 10, 0)),
        BusyInterval(start=datetime(2025, 11, 14, 14, 0), end=datetime(2025, 11, 14, 15, 30)),
    ]
    groups_busy = [
        [
            BusyInterval(start=datetime(2025, 11, 14, 10, 0), end=datetime(2025, 11, 14, 11, 30)),
        ],
        [
            BusyInterval(start=datetime(2025, 11, 14, 16, 0), end=datetime(2025, 11, 14, 18, 0)),
        ],
    ]

    search_start = datetime(2025, 11, 14, 8, 0)
    search_end = datetime(2025, 11, 14, 20, 0)

    slots = find_common_free_slots(
        teacher_busy=teacher_busy,
        groups_busy=groups_busy,
        search_start=search_start,
        search_end=search_end,
        slot_duration_minutes=60,
        step_minutes=30,
        max_results=3,
    )

    expected_starts = [
        datetime(2025, 11, 14, 8, 0),
        datetime(2025, 11, 14, 11, 30),
        datetime(2025, 11, 14, 12, 0),
    ]

    assert [slot["start"] for slot in slots] == expected_starts
    assert all(slot["duration_minutes"] == 60 for slot in slots)
    assert all(slot["confidence_score"] == 1.0 for slot in slots)

    for slot in slots:
        for interval in teacher_busy:
            assert not interval.overlaps(slot["start"], slot["end"])
        for group_intervals in groups_busy:
            for interval in group_intervals:
                assert not interval.overlaps(slot["start"], slot["end"])
