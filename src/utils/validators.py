from __future__ import annotations

import re
from datetime import time
from typing import Iterable

MAX_USER_MESSAGE_LENGTH = 500
FORBIDDEN_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\b(drop|delete|update|insert|alter)\b", re.IGNORECASE),
    re.compile(r"--"),
    re.compile(r";"),
    re.compile(r"system:\s*", re.IGNORECASE),
    re.compile(r"ignore\s+all\s+previous", re.IGNORECASE),
)

WORKING_DAY_START = time(8, 0)
WORKING_DAY_END = time(20, 0)


class ValidationError(ValueError):
    """Raised when user input violates guardrails."""


def validate_user_input(message: str) -> None:
    """Validate user-supplied text for length and malicious patterns."""

    if not message:
        raise ValidationError("Message must not be empty.")

    if len(message) > MAX_USER_MESSAGE_LENGTH:
        raise ValidationError("Message is too long.")

    for pattern in FORBIDDEN_PATTERNS:
        if pattern.search(message):
            raise ValidationError("Message contains restricted content.")


def ensure_whitelisted_identifier(value: str, allowed: Iterable[str]) -> str:
    """Return the identifier if it matches one of the allowed options."""

    normalized = value.strip().lower()
    if normalized not in {item.strip().lower() for item in allowed}:
        raise ValidationError("Identifier is not allowed.")
    return normalized


def normalize_time_range(
    start: time | None,
    end: time | None,
    *,
    default_start: time = WORKING_DAY_START,
    default_end: time = WORKING_DAY_END,
) -> tuple[time, time]:
    """Validate and normalize a time window within working hours."""

    start_time = start or default_start
    end_time = end or default_end

    if start_time < WORKING_DAY_START or start_time > WORKING_DAY_END:
        raise ValidationError("Start time is outside working hours.")

    if end_time < WORKING_DAY_START or end_time > WORKING_DAY_END:
        raise ValidationError("End time is outside working hours.")

    return start_time, end_time
