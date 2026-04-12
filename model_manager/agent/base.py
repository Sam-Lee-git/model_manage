"""Abstract LLM client interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import AsyncIterator, Optional


def load_prompt(name: str) -> str:
    path = Path(__file__).parent / "prompts" / name
    return path.read_text(encoding="utf-8")


class LLMClient(ABC):
    """
    Uniform interface over multiple LLM providers.

    All implementations accept `messages` in the OpenAI-style format
    (user/assistant turns only, no system message in the list) plus a
    separate `system` string that each provider handles natively.
    """

    @abstractmethod
    async def chat(
        self,
        messages: list[dict],
        system: str,
        max_tokens: int = 4096,
    ) -> str:
        """Blocking (non-streaming) chat. Returns full response text."""
        ...

    @abstractmethod
    async def stream(
        self,
        messages: list[dict],
        system: str,
        max_tokens: int = 2048,
    ) -> AsyncIterator[str]:
        """Streaming chat. Yields text chunks."""
        ...

    def get_error_diagnosis_system(self) -> str:
        return load_prompt("system_error_diagnosis.txt")

    def get_recommendation_system(self) -> str:
        return load_prompt("system_recommendation.txt")
