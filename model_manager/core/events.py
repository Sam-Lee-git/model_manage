"""Lightweight async event bus for decoupling subsystems from the UI."""

from __future__ import annotations

import asyncio
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine


# ── Event base ────────────────────────────────────────────────────────────────

@dataclass
class Event:
    """Base class for all internal events."""
    pass


# ── Concrete events ───────────────────────────────────────────────────────────

@dataclass
class StateChangedEvent(Event):
    previous: str
    current: str

@dataclass
class StepStartedEvent(Event):
    step_name: str
    step_index: int
    total_steps: int

@dataclass
class StepCompletedEvent(Event):
    step_name: str
    step_index: int

@dataclass
class StepFailedEvent(Event):
    step_name: str
    step_index: int
    error: str

@dataclass
class ErrorCapturedEvent(Event):
    step_name: str
    error_summary: str

@dataclass
class DiagnosisStartedEvent(Event):
    step_name: str

@dataclass
class DiagnosisReadyEvent(Event):
    root_cause: str
    confidence: float
    requires_user_decision: bool
    decision_options: list[str] = field(default_factory=list)

@dataclass
class BranchStartedEvent(Event):
    depth: int
    fix_steps: int

@dataclass
class BranchStepEvent(Event):
    step_description: str
    step_index: int

@dataclass
class BranchSucceededEvent(Event):
    depth: int

@dataclass
class BranchFailedEvent(Event):
    depth: int
    reason: str

@dataclass
class ResumeEvent(Event):
    step_index: int

@dataclass
class DownloadProgressEvent(Event):
    filename: str
    downloaded_bytes: int
    total_bytes: int
    speed_mbps: float

@dataclass
class LogLineEvent(Event):
    level: str   # "info" | "warning" | "error"
    message: str

@dataclass
class ChatResponseChunkEvent(Event):
    chunk: str
    is_final: bool = False

@dataclass
class UserDecisionRequiredEvent(Event):
    question: str
    options: list[str]


# ── Bus ───────────────────────────────────────────────────────────────────────

Handler = Callable[[Event], Coroutine[Any, Any, None]]


class EventBus:
    """Simple async pub/sub event bus. Handlers are called in registration order."""

    def __init__(self) -> None:
        self._handlers: dict[type[Event], list[Handler]] = defaultdict(list)

    def subscribe(self, event_type: type[Event], handler: Handler) -> None:
        self._handlers[event_type].append(handler)

    def unsubscribe(self, event_type: type[Event], handler: Handler) -> None:
        self._handlers[event_type].remove(handler)

    async def emit(self, event: Event) -> None:
        for handler in self._handlers[type(event)]:
            await handler(event)

    async def emit_nowait(self, event: Event) -> None:
        """Fire-and-forget: schedule emission without awaiting handlers."""
        asyncio.get_event_loop().create_task(self.emit(event))


# Singleton instance shared across all subsystems
bus = EventBus()
