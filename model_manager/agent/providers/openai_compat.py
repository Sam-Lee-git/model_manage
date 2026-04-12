"""
OpenAI-compatible provider.

Covers: OpenAI, Qwen (DashScope), DeepSeek, MiniMax, Gemini (via their OpenAI-compat endpoint).
All of these accept the OpenAI messages format where the system prompt is the first message
with role="system".
"""

from __future__ import annotations

from typing import AsyncIterator

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from model_manager.agent.base import LLMClient


# ── Default endpoints per provider ────────────────────────────────────────────
PROVIDER_BASE_URLS: dict[str, str] = {
    "openai":   "https://api.openai.com/v1",
    "qwen":     "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "deepseek": "https://api.deepseek.com/v1",
    "minimax":  "https://api.minimax.chat/v1",
    "gemini":   "https://generativelanguage.googleapis.com/v1beta/openai/",
}

PROVIDER_DEFAULT_MODELS: dict[str, str] = {
    "openai":   "gpt-4o",
    "qwen":     "qwen-max",
    "deepseek": "deepseek-chat",
    "minimax":  "abab6.5s-chat",
    "gemini":   "gemini-2.0-flash",
}


class OpenAICompatProvider(LLMClient):
    """
    Wraps `openai.AsyncOpenAI` with a configurable base_url.
    The system prompt is prepended to messages as role="system".
    """

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str,
        provider_name: str = "openai",
    ) -> None:
        try:
            from openai import AsyncOpenAI, RateLimitError, APIStatusError
        except ImportError:
            raise ImportError("openai package not installed. Run: pip install openai")

        self._client        = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self._model         = model
        self._provider_name = provider_name
        self._RateLimitError  = RateLimitError
        self._APIStatusError  = APIStatusError

    def _build_messages(self, messages: list[dict], system: str) -> list[dict]:
        """Prepend system as first message in the OpenAI format."""
        return [{"role": "system", "content": system}] + messages

    @retry(
        wait=wait_exponential(multiplier=1, min=2, max=60),
        stop=stop_after_attempt(5),
        reraise=True,
    )
    async def chat(
        self,
        messages: list[dict],
        system: str,
        max_tokens: int = 4096,
    ) -> str:
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=self._build_messages(messages, system),
            max_tokens=max_tokens,
            stream=False,
        )
        return response.choices[0].message.content or ""

    async def stream(
        self,
        messages: list[dict],
        system: str,
        max_tokens: int = 2048,
    ) -> AsyncIterator[str]:
        # Works with both openai 1.x and 2.x:
        # create() with stream=True returns an AsyncStream (async iterable).
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=self._build_messages(messages, system),
            max_tokens=max_tokens,
            stream=True,
        )
        async for chunk in response:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta and delta.content:
                yield delta.content
