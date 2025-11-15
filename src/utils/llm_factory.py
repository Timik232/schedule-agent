"""Factory helpers for constructing provider-specific chat models."""
from __future__ import annotations

from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI

from ..config import LLMClientConfig, LLMProvider, get_settings


class LLMProviderConfigurationError(RuntimeError):
    """Raised when an unsupported or incomplete LLM provider configuration is detected."""


def _build_common_kwargs(config: LLMClientConfig, *, streaming: bool, model: str | None) -> dict[str, Any]:
    return {
        "model": model or config.model,
        "temperature": config.temperature,
        "timeout": config.timeout,
        "streaming": streaming,
        "max_retries": 2,
    }


def _build_copilot_client(config: LLMClientConfig, *, streaming: bool, model: str | None) -> BaseChatModel:
    if not config.api_key:
        raise LLMProviderConfigurationError("LLM_API_KEY is required when using the Copilot provider")

    kwargs = _build_common_kwargs(config, streaming=streaming, model=model)
    kwargs.update(
        {
            "api_key": config.api_key,
            "base_url": config.api_base_url,
            "default_headers": {
                "GitHub-Assistant-Id": "schedule-agent",
            },
        }
    )
    return ChatOpenAI(**kwargs)


def _build_lm_studio_client(config: LLMClientConfig, *, streaming: bool, model: str | None) -> BaseChatModel:
    api_key = config.api_key or "lm-studio"
    kwargs = _build_common_kwargs(config, streaming=streaming, model=model)
    kwargs.update(
        {
            "api_key": api_key,
            "base_url": config.api_base_url,
        }
    )
    return ChatOpenAI(**kwargs)


def get_chat_model(*, streaming: bool = False, model: str | None = None) -> BaseChatModel:
    """Return a chat model implementation for the configured provider."""

    settings = get_settings()
    config = settings.llm_client_config

    if config.provider == LLMProvider.COPILOT:
        return _build_copilot_client(config, streaming=streaming, model=model)
    if config.provider == LLMProvider.LM_STUDIO:
        return _build_lm_studio_client(config, streaming=streaming, model=model)

    raise LLMProviderConfigurationError(f"Unsupported LLM provider: {config.provider}")


__all__ = ["get_chat_model", "LLMProviderConfigurationError"]
