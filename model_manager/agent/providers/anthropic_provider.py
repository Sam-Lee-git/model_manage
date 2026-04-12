"""Anthropic Claude provider.

Uses the `anthropic` SDK when available; falls back to direct httpx calls otherwise.
"""

from __future__ import annotations

from typing import AsyncIterator, Optional

from tenacity import retry, stop_after_attempt, wait_exponential

from model_manager.agent.base import LLMClient

_ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
_ANTHROPIC_VERSION = "2023-06-01"


class AnthropicProvider(LLMClient):
    def __init__(self, api_key: str, model: str = "claude-sonnet-4-6") -> None:
        self._api_key = api_key
        self._model   = model
        self._use_sdk = _sdk_available()

    @retry(wait=wait_exponential(min=2, max=60), stop=stop_after_attempt(5), reraise=True)
    async def chat(self, messages: list[dict], system: str, max_tokens: int = 4096) -> str:
        if self._use_sdk:
            return await self._chat_sdk(messages, system, max_tokens)
        return await self._chat_httpx(messages, system, max_tokens)

    async def stream(
        self, messages: list[dict], system: str, max_tokens: int = 2048
    ) -> AsyncIterator[str]:
        if self._use_sdk:
            async for chunk in self._stream_sdk(messages, system, max_tokens):
                yield chunk
        else:
            async for chunk in self._stream_httpx(messages, system, max_tokens):
                yield chunk

    # ── SDK path ──────────────────────────────────────────────────────────────

    async def _chat_sdk(self, messages, system, max_tokens) -> str:
        import anthropic
        client = anthropic.AsyncAnthropic(api_key=self._api_key)
        response = await client.messages.create(
            model=self._model, max_tokens=max_tokens, system=system, messages=messages
        )
        return "\n".join(b.text for b in response.content if hasattr(b, "text"))

    async def _stream_sdk(self, messages, system, max_tokens) -> AsyncIterator[str]:
        import anthropic
        client = anthropic.AsyncAnthropic(api_key=self._api_key)
        async with client.messages.stream(
            model=self._model, max_tokens=max_tokens, system=system, messages=messages
        ) as s:
            async for chunk in s.text_stream:
                yield chunk

    # ── httpx fallback path ───────────────────────────────────────────────────

    def _headers(self) -> dict:
        return {
            "x-api-key": self._api_key,
            "anthropic-version": _ANTHROPIC_VERSION,
            "content-type": "application/json",
        }

    def _payload(self, messages, system, max_tokens, stream=False) -> dict:
        return {
            "model":      self._model,
            "max_tokens": max_tokens,
            "system":     system,
            "messages":   messages,
            "stream":     stream,
        }

    async def _chat_httpx(self, messages, system, max_tokens) -> str:
        import httpx
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                _ANTHROPIC_API_URL,
                headers=self._headers(),
                json=self._payload(messages, system, max_tokens, stream=False),
            )
        resp.raise_for_status()
        data = resp.json()
        return "\n".join(
            b["text"] for b in data.get("content", []) if b.get("type") == "text"
        )

    async def _stream_httpx(self, messages, system, max_tokens) -> AsyncIterator[str]:
        import httpx, json
        async with httpx.AsyncClient(timeout=120) as client:
            async with client.stream(
                "POST",
                _ANTHROPIC_API_URL,
                headers=self._headers(),
                json=self._payload(messages, system, max_tokens, stream=True),
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    raw = line[5:].strip()
                    if raw in ("", "[DONE]"):
                        continue
                    try:
                        event = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    if event.get("type") == "content_block_delta":
                        delta = event.get("delta", {})
                        if delta.get("type") == "text_delta":
                            yield delta.get("text", "")


def _sdk_available() -> bool:
    try:
        import anthropic  # noqa: F401
        return True
    except ImportError:
        return False
