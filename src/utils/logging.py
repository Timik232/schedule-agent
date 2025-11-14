"""Utilities for configuring structured application logging."""
from __future__ import annotations

import logging
from typing import Any, Mapping

_DEFAULT_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


def configure_logging(level: str = "INFO") -> None:
    """Configure root logging with a consistent structured format."""

    logging.basicConfig(level=level.upper(), format=_DEFAULT_FORMAT)


class StructuredLoggerAdapter(logging.LoggerAdapter):
    """Logger adapter that appends context key-value pairs to each message."""

    def process(self, msg: str, kwargs: Mapping[str, Any]):  # type: ignore[override]
        context_bits = [f"{key}={value!r}" for key, value in self.extra.items()]
        if context_bits:
            msg = f"{msg} [{' '.join(context_bits)}]"
        return msg, kwargs

    def bind(self, **extra: Any) -> "StructuredLoggerAdapter":
        """Return a new adapter with additional contextual fields."""

        merged = {**self.extra, **extra}
        return StructuredLoggerAdapter(self.logger, merged)


def get_logger(name: str, **context: Any) -> StructuredLoggerAdapter:
    """Return a logger adapter enriched with optional structured context."""

    logger = logging.getLogger(name)
    return StructuredLoggerAdapter(logger, context)
