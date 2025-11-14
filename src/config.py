from functools import lru_cache
from typing import Any

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration loaded from environment variables."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    openai_api_key: str = Field(..., alias="OPENAI_API_KEY")
    llm_model: str = Field("gpt-4o-mini", alias="LLM_MODEL")
    llm_temperature: float = Field(0.0, alias="LLM_TEMPERATURE", ge=0.0, le=1.0)

    database_url: str = Field(..., alias="DATABASE_URL")
    max_sql_rows: int = Field(1000, alias="MAX_SQL_ROWS", gt=0, le=1000)
    query_timeout_seconds: int = Field(30, alias="QUERY_TIMEOUT_SECONDS", ge=1)

    log_level: str = Field("INFO", alias="LOG_LEVEL")

    request_timeout_seconds: int = Field(15, alias="REQUEST_TIMEOUT_SECONDS", ge=1)

    @field_validator("log_level", mode="before")
    @classmethod
    def _normalize_log_level(cls, value: str) -> str:
        level = value.upper().strip()
        valid_levels = {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"}
        if level not in valid_levels:
            raise ValueError(f"Unsupported log level '{value}'. Choose from {sorted(valid_levels)}")
        return level

    @field_validator("database_url")
    @classmethod
    def _validate_database_url(cls, value: str) -> str:
        if "asyncpg" not in value:
            raise ValueError("DATABASE_URL must use the asyncpg driver (postgresql+asyncpg://)")
        return value

    @property
    def llm_kwargs(self) -> dict[str, Any]:
        """Return kwargs for initializing the default ChatOpenAI client."""
        return {
            "model": self.llm_model,
            "temperature": self.llm_temperature,
            "api_key": self.openai_api_key,
            "timeout": self.request_timeout_seconds,
        }

    @property
    def db_execution_kwargs(self) -> dict[str, Any]:
        """Parameters shared across database query helpers."""

        return {
            "timeout": self.query_timeout_seconds,
            "max_rows": self.max_sql_rows,
        }


@lru_cache(1)
def get_settings() -> Settings:
    """Return cached settings instance to avoid repeated env parsing."""
    return Settings()
