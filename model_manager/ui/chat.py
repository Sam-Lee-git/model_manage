"""Async input widget — uses prompt_toolkit if available, falls back to asyncio stdin."""

from __future__ import annotations

import asyncio
import sys
from typing import Callable, Coroutine

try:
    from prompt_toolkit import PromptSession
    from prompt_toolkit.formatted_text import HTML
    from prompt_toolkit.history import InMemoryHistory
    from prompt_toolkit.patch_stdout import patch_stdout
    _HAS_PROMPT_TOOLKIT = True
except ImportError:
    _HAS_PROMPT_TOOLKIT = False


SLASH_COMMANDS = {
    "/help":     "Show this help message",
    "/status":   "Show current installation status",
    "/sessions": "List past installation sessions",
    "/cancel":   "Cancel current operation",
    "/exit":     "Exit model manager",
}

MessageHandler = Callable[[str], Coroutine]


class ChatInput:
    def __init__(self) -> None:
        if _HAS_PROMPT_TOOLKIT:
            self._session = PromptSession(history=InMemoryHistory())
        else:
            self._session = None

    async def get_input(self, prompt: str = "You: ") -> str:
        if _HAS_PROMPT_TOOLKIT:
            with patch_stdout():
                text = await self._session.prompt_async(
                    HTML(f"<ansiblue><b>{prompt}</b></ansiblue>"),
                )
            return text.strip()
        # Fallback: blocking readline wrapped in executor
        loop = asyncio.get_event_loop()
        try:
            sys.stdout.write(prompt)
            sys.stdout.flush()
            text = await loop.run_in_executor(None, sys.stdin.readline)
            return text.rstrip("\n").strip()
        except (EOFError, KeyboardInterrupt):
            return "/exit"

    async def run_loop(self, handler: MessageHandler) -> None:
        while True:
            try:
                text = await self.get_input()
            except (EOFError, KeyboardInterrupt):
                break

            if not text:
                continue
            if text.lower() in ("/exit", "/quit", "exit", "quit"):
                break
            await handler(text)
