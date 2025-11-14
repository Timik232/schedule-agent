from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

from .logging import get_logger
from .validators import ValidationError, validate_user_input

LOGGER = get_logger(__name__)


def apply_input_guardrails(message: str) -> None:
    """Validate user input and log violations."""

    try:
        validate_user_input(message)
    except ValidationError as exc:
        LOGGER.bind(length=len(message)).warning("Rejected user input", error=str(exc))
        raise


def guard_and_normalize_message(message: str) -> str:
    """Return sanitized user input after enforcing guardrails."""

    normalized = message.strip()
    apply_input_guardrails(normalized)
    return normalized


def redact_sensitive_values(payload: dict[str, Any], sensitive_keys: Sequence[str]) -> dict[str, Any]:
    """Return a copy with sensitive fields replaced by placeholders."""

    redacted: dict[str, Any] = {}
    for key, value in payload.items():
        if key in sensitive_keys and isinstance(value, str):
            redacted[key] = "***"
            continue
        redacted[key] = value
    return redacted
