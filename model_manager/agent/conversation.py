"""Stateful conversation manager — provider-agnostic."""

from __future__ import annotations

from typing import Optional

from model_manager.agent.base import LLMClient
from model_manager.core.events import ChatResponseChunkEvent, bus


MAX_HISTORY_TURNS = 20


class ConversationManager:
    def __init__(self, client: Optional[LLMClient] = None) -> None:
        if client is None:
            from model_manager.agent.factory import create_client_from_settings
            client = create_client_from_settings()
        self._client  = client
        self._history: list[dict] = []

    def add_user_message(self, text: str) -> None:
        self._history.append({"role": "user", "content": text})
        self._trim()

    def add_assistant_message(self, text: str) -> None:
        self._history.append({"role": "assistant", "content": text})

    def _trim(self) -> None:
        if len(self._history) > MAX_HISTORY_TURNS * 2:
            self._history = self._history[-(MAX_HISTORY_TURNS * 2):]

    async def stream_response(
        self,
        user_message: str,
        system: Optional[str] = None,
    ) -> str:
        self.add_user_message(user_message)
        system_prompt = system or self._client.get_recommendation_system()

        full_response = ""
        async for chunk in self._client.stream(
            messages=self._history,
            system=system_prompt,
        ):
            full_response += chunk
            await bus.emit(ChatResponseChunkEvent(chunk=chunk))

        await bus.emit(ChatResponseChunkEvent(chunk="", is_final=True))
        self.add_assistant_message(full_response)
        return full_response

    def clear(self) -> None:
        self._history.clear()
