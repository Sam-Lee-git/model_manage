"""LLM client factory — creates the right provider from settings or explicit args."""

from __future__ import annotations

import os
from enum import Enum
from typing import Optional

from model_manager.agent.base import LLMClient
from model_manager.core.exceptions import APIKeyMissingError


class LLMProvider(str, Enum):
    CLAUDE   = "claude"
    OPENAI   = "openai"
    GEMINI   = "gemini"
    QWEN     = "qwen"
    MINIMAX  = "minimax"
    DEEPSEEK = "deepseek"


# Standard environment variable name per provider
PROVIDER_ENV_VARS: dict[LLMProvider, str] = {
    LLMProvider.CLAUDE:   "ANTHROPIC_API_KEY",
    LLMProvider.OPENAI:   "OPENAI_API_KEY",
    LLMProvider.GEMINI:   "GEMINI_API_KEY",
    LLMProvider.QWEN:     "QWEN_API_KEY",
    LLMProvider.MINIMAX:  "MINIMAX_API_KEY",
    LLMProvider.DEEPSEEK: "DEEPSEEK_API_KEY",
}

# Priority order when multiple keys are present (higher index = lower priority)
PROVIDER_PRIORITY: list[LLMProvider] = [
    LLMProvider.CLAUDE,
    LLMProvider.DEEPSEEK,
    LLMProvider.QWEN,
    LLMProvider.OPENAI,
    LLMProvider.GEMINI,
    LLMProvider.MINIMAX,
]

# Default model IDs per provider
DEFAULT_MODELS: dict[LLMProvider, str] = {
    LLMProvider.CLAUDE:   "claude-sonnet-4-6",
    LLMProvider.OPENAI:   "gpt-4o",
    LLMProvider.GEMINI:   "gemini-2.0-flash",
    LLMProvider.QWEN:     "qwen-max",
    LLMProvider.MINIMAX:  "abab6.5s-chat",
    LLMProvider.DEEPSEEK: "deepseek-chat",
}


def detect_available_providers() -> list[tuple[LLMProvider, str]]:
    """
    Scan system environment variables and return all providers whose API key is set.
    Returns a list of (provider, api_key) sorted by PROVIDER_PRIORITY.
    """
    found: list[tuple[LLMProvider, str]] = []
    for provider in PROVIDER_PRIORITY:
        env_var = PROVIDER_ENV_VARS[provider]
        key = os.environ.get(env_var, "").strip()
        if key:
            found.append((provider, key))
    return found


def auto_select_provider() -> tuple[LLMProvider, str]:
    """
    Pick the highest-priority provider whose API key exists in env vars.
    Raises APIKeyMissingError if none are found.
    """
    available = detect_available_providers()
    if not available:
        env_var_list = ", ".join(PROVIDER_ENV_VARS.values())
        raise APIKeyMissingError(
            "No LLM provider API key found in environment variables.\n"
            f"Please set one of: {env_var_list}"
        )
    return available[0]


def create_client(
    provider: LLMProvider | str,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
) -> LLMClient:
    """
    Instantiate the appropriate LLM client.

    API key resolution order:
      1. Explicit `api_key` argument
      2. System environment variable for the provider
      3. Settings object (pydantic-settings, reads .env)
    """
    provider = LLMProvider(provider) if isinstance(provider, str) else provider
    settings = _get_settings()

    # ── Claude ────────────────────────────────────────────────────────────────
    if provider == LLMProvider.CLAUDE:
        key = (api_key
               or os.environ.get(PROVIDER_ENV_VARS[provider], "").strip()
               or settings.anthropic_api_key
               or "")
        if not key:
            raise APIKeyMissingError(
                f"ANTHROPIC_API_KEY not found. Set it in your system environment variables."
            )
        from model_manager.agent.providers.anthropic_provider import AnthropicProvider
        return AnthropicProvider(
            api_key=key,
            model=model or settings.claude_model or DEFAULT_MODELS[provider],
        )

    # ── OpenAI-compatible providers ───────────────────────────────────────────
    from model_manager.agent.providers.openai_compat import (
        OpenAICompatProvider, PROVIDER_BASE_URLS, PROVIDER_DEFAULT_MODELS,
    )

    provider_name = provider.value
    env_var       = PROVIDER_ENV_VARS[provider]
    settings_attr = f"{provider_name}_api_key"

    key = (api_key
           or os.environ.get(env_var, "").strip()
           or getattr(settings, settings_attr, "") or "")

    if not key:
        raise APIKeyMissingError(
            f"{env_var} not found. Set it in your system environment variables."
        )

    resolved_url = (base_url
                    or getattr(settings, f"{provider_name}_base_url", None)
                    or PROVIDER_BASE_URLS[provider_name])
    resolved_model = (model
                      or getattr(settings, f"{provider_name}_model", None)
                      or PROVIDER_DEFAULT_MODELS.get(provider_name, ""))

    return OpenAICompatProvider(
        api_key=key,
        model=resolved_model,
        base_url=resolved_url,
        provider_name=provider_name,
    )


def create_client_from_settings() -> LLMClient:
    """
    Create a client based on settings.

    If llm_provider is "auto" (the default), scan system env vars and pick
    the highest-priority available provider automatically.
    """
    settings = _get_settings()
    provider_setting = (settings.llm_provider or "auto").strip().lower()

    if provider_setting == "auto":
        provider, key = auto_select_provider()
        return create_client(provider=provider, api_key=key)

    return create_client(provider=provider_setting)


def _get_settings():
    from model_manager.config.settings import get_settings
    return get_settings()
