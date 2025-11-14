"""LLM-powered fallback for text-to-SQL with strict validation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from ..utils.logging import get_logger

LOGGER = get_logger(__name__)


class GeneratedSQL(BaseModel):
    """Schema describing an LLM-generated SQL statement."""

    statement: str = Field(..., description="Parameterized SQL query limited to SELECT")
    parameters: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def validate_statement(cls, sql: str) -> str:
        candidate = sql.strip().lower()
        if not candidate.startswith("select"):
            raise ValueError("Only SELECT statements are permitted")
        if ";" in candidate:
            raise ValueError("Semicolons are not allowed in generated SQL")
        return sql

    def model_post_init(self, __context: Any) -> None:  # type: ignore[override]
        self.statement = self.validate_statement(self.statement)


@dataclass
class TextToSQLFallback:
    """Orchestrates invocation of an LLM to create SQL safely."""

    llm: Any

    async def generate_and_execute(self, user_query: str) -> list[dict[str, Any]]:
        """Generate SQL from natural language and execute it.

        This is a scaffold implementation; concrete execution will be added during
        user story tasks where fallback coverage is required.
        """

        LOGGER.warning("Text-to-SQL fallback invoked without concrete implementation", query=user_query)
        raise NotImplementedError("text-to-SQL fallback is not yet implemented")


__all__ = ["GeneratedSQL", "TextToSQLFallback"]
