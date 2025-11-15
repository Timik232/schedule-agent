from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from typing import Any

from pydantic import AliasChoices, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMProvider(str, Enum):
    COPILOT = "copilot"
    LM_STUDIO = "lm_studio"


@dataclass(frozen=True)
class LLMClientConfig:
    provider: LLMProvider
    api_base_url: str
    api_key: str | None
    model: str
    temperature: float
    timeout: int


class Settings(BaseSettings):
    """Runtime configuration loaded from environment variables."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    llm_provider: LLMProvider = Field(default=LLMProvider.COPILOT, alias="LLM_PROVIDER")
    llm_api_base_url: str | None = Field(default=None, validation_alias=AliasChoices("LLM_API_BASE_URL"))
    llm_api_key: str | None = Field(default=None, validation_alias=AliasChoices("LLM_API_KEY", "OPENAI_API_KEY"))
    llm_model: str = Field("gpt-4o-mini-copilot", alias="LLM_MODEL")
    llm_temperature: float = Field(0.0, alias="LLM_TEMPERATURE", ge=0.0, le=1.0)

    sqlite_seed_path: str | None = Field(default=None, alias="SQLITE_SEED_PATH")

    database_url: str = Field(..., alias="DATABASE_URL")
    max_sql_rows: int = Field(1000, alias="MAX_SQL_ROWS", gt=0, le=1000)
    query_timeout_seconds: int = Field(30, alias="QUERY_TIMEOUT_SECONDS", ge=1)

    log_level: str = Field("INFO", alias="LOG_LEVEL")

    request_timeout_seconds: int = Field(15, alias="REQUEST_TIMEOUT_SECONDS", ge=1)

    @model_validator(mode="after")
    def _apply_llm_defaults(self) -> "Settings":
        if self.llm_provider == LLMProvider.COPILOT:
            if not self.llm_api_key:
                raise ValueError("LLM_API_KEY is required when LLM_PROVIDER=copilot")
            if not self.llm_api_base_url:
                self.llm_api_base_url = "https://api.githubcopilot.com"
        elif self.llm_provider == LLMProvider.LM_STUDIO:
            if not self.llm_api_base_url:
                self.llm_api_base_url = "http://localhost:1234/v1"
        else:  # pragma: no cover - defensive branch
            raise ValueError(f"Unsupported LLM provider '{self.llm_provider}'")

        return self

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
    def llm_client_config(self) -> LLMClientConfig:
        """Return provider-specific configuration for the LLM factory."""

        if self.llm_api_base_url is None:  # pragma: no cover - should be set by validator
            raise ValueError("LLM_API_BASE_URL must be configured")

        return LLMClientConfig(
            provider=self.llm_provider,
            api_base_url=self.llm_api_base_url,
            api_key=self.llm_api_key,
            model=self.llm_model,
            temperature=self.llm_temperature,
            timeout=self.request_timeout_seconds,
        )

    @property
    def db_execution_kwargs(self) -> dict[str, Any]:
        """Parameters shared across database query helpers."""

        return {
            "timeout": self.query_timeout_seconds,
            "max_rows": self.max_sql_rows,
        }

    @property
    def postgres_dsn(self) -> str:
        """Return DSN suitable for asyncpg connections."""

        if self.database_url.startswith("postgresql+asyncpg://"):
            return "postgresql://" + self.database_url.split("://", 1)[1]
        return self.database_url


@lru_cache(1)
def get_settings() -> Settings:
    """Return cached settings instance to avoid repeated env parsing."""
    return Settings()  # type: ignore[call-arg]
