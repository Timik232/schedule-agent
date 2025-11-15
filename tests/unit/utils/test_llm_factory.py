from __future__ import annotations

from typing import Generator

import pytest

from src.config import get_settings
from src.utils.llm_factory import get_chat_model


class _StubChatModel:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    async def ainvoke(self, *_args, **_kwargs):  # pragma: no cover - helper stub
        raise NotImplementedError


@pytest.fixture(autouse=True)
def _reset_settings_cache() -> Generator[None, None, None]:
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def _set_base_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://tester:pass@localhost:5432/db")
    monkeypatch.setenv("LOG_LEVEL", "INFO")
    monkeypatch.setenv("REQUEST_TIMEOUT_SECONDS", "20")


def test_get_chat_model_for_copilot_uses_default_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_base_environment(monkeypatch)
    monkeypatch.setenv("LLM_PROVIDER", "copilot")
    monkeypatch.setenv("LLM_API_KEY", "test-token")
    monkeypatch.delenv("LLM_API_BASE_URL", raising=False)
    monkeypatch.setenv("LLM_MODEL", "gpt-copilot-test")

    captured: dict[str, object] = {}

    def _capture_kwargs(**kwargs: object) -> _StubChatModel:
        captured.update(kwargs)
        return _StubChatModel(**kwargs)

    monkeypatch.setenv("LLM_TEMPERATURE", "0.1")
    monkeypatch.setattr("src.utils.llm_factory.ChatOpenAI", _capture_kwargs)

    get_chat_model()

    assert captured["base_url"] == "https://api.githubcopilot.com"
    assert captured["api_key"] == "test-token"
    assert captured["model"] == "gpt-copilot-test"
    assert captured["default_headers"] == {"GitHub-Assistant-Id": "schedule-agent"}
    assert captured["timeout"] == 20
    assert captured["temperature"] == 0.1


def test_get_chat_model_for_lm_studio_supplies_fallback_key(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_base_environment(monkeypatch)
    monkeypatch.setenv("LLM_PROVIDER", "lm_studio")
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("LLM_API_BASE_URL", raising=False)
    monkeypatch.setenv("LLM_MODEL", "local-model")

    captured: dict[str, object] = {}

    def _capture_kwargs(**kwargs: object) -> _StubChatModel:
        captured.update(kwargs)
        return _StubChatModel(**kwargs)

    monkeypatch.setattr("src.utils.llm_factory.ChatOpenAI", _capture_kwargs)

    get_chat_model()

    assert captured["base_url"] == "http://localhost:1234/v1"
    assert captured["api_key"] == "lm-studio"
    assert captured["model"] == "local-model"
    assert "default_headers" not in captured


def test_copilot_provider_without_api_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_base_environment(monkeypatch)
    monkeypatch.setenv("LLM_PROVIDER", "copilot")
    monkeypatch.delenv("LLM_API_KEY", raising=False)

    with pytest.raises(ValueError):
        get_chat_model()