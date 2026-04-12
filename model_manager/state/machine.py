"""Installation state machine."""

from __future__ import annotations

import asyncio
from typing import Optional

from model_manager.core.constants import SessionState
from model_manager.core.events import StateChangedEvent, bus
from model_manager.state.models import InstallationState
from model_manager.state.store import StateStore


TRANSITIONS = [
    # Normal flow
    {"trigger": "start",               "source": SessionState.IDLE,                    "dest": SessionState.DETECTING_HARDWARE},
    {"trigger": "hardware_detected",   "source": SessionState.DETECTING_HARDWARE,      "dest": SessionState.BROWSING_CATALOG},
    {"trigger": "model_selected",      "source": SessionState.BROWSING_CATALOG,        "dest": SessionState.MODEL_SELECTED},
    {"trigger": "user_exit",           "source": SessionState.BROWSING_CATALOG,        "dest": SessionState.ABORTED},
    {"trigger": "storage_analyzed",    "source": SessionState.MODEL_SELECTED,          "dest": SessionState.ANALYZING_STORAGE},
    {"trigger": "plan_ready",          "source": SessionState.ANALYZING_STORAGE,       "dest": SessionState.CONFIRMING_PLAN},
    {"trigger": "plan_confirmed",      "source": SessionState.CONFIRMING_PLAN,         "dest": SessionState.SELECTING_BACKEND},
    {"trigger": "plan_rejected",       "source": SessionState.CONFIRMING_PLAN,         "dest": SessionState.BROWSING_CATALOG},
    {"trigger": "backend_selected",    "source": SessionState.SELECTING_BACKEND,       "dest": SessionState.RESOLVING_REPOS},
    {"trigger": "repos_resolved",      "source": SessionState.RESOLVING_REPOS,         "dest": SessionState.INSTALLING_DEPENDENCIES},
    {"trigger": "deps_installed",      "source": SessionState.INSTALLING_DEPENDENCIES, "dest": SessionState.DOWNLOADING_MODEL},
    {"trigger": "download_complete",   "source": SessionState.DOWNLOADING_MODEL,       "dest": SessionState.VERIFYING_INSTALL},
    {"trigger": "verified",            "source": SessionState.VERIFYING_INSTALL,       "dest": SessionState.COMPLETED},
    # Error recovery (from any active state)
    {
        "trigger": "error_captured",
        "source": [
            SessionState.DETECTING_HARDWARE,
            SessionState.BROWSING_CATALOG,
            SessionState.MODEL_SELECTED,
            SessionState.ANALYZING_STORAGE,
            SessionState.CONFIRMING_PLAN,
            SessionState.SELECTING_BACKEND,
            SessionState.RESOLVING_REPOS,
            SessionState.INSTALLING_DEPENDENCIES,
            SessionState.DOWNLOADING_MODEL,
            SessionState.VERIFYING_INSTALL,
        ],
        "dest": SessionState.ERROR_RECOVERY,
    },
    {"trigger": "branch_verified",     "source": SessionState.ERROR_RECOVERY, "dest": None},   # dynamic
    {"trigger": "branch_needs_user",   "source": SessionState.ERROR_RECOVERY, "dest": SessionState.PAUSED},
    {"trigger": "branch_fatal",        "source": SessionState.ERROR_RECOVERY, "dest": SessionState.FAILED},
    # Resume / pause
    {"trigger": "user_resumes",        "source": SessionState.PAUSED,         "dest": None},    # dynamic
    {"trigger": "user_cancels",        "source": SessionState.PAUSED,         "dest": SessionState.ABORTED},
]


class StateMachine:
    """
    Thin state machine that wraps InstallationState.
    Uses plain dicts (not `transitions` library) to stay dependency-light.
    """

    def __init__(self, state: InstallationState, store: StateStore) -> None:
        self._state   = state
        self._store   = store
        self._resume_target: Optional[str] = None  # set before branch_verified trigger

    @property
    def current(self) -> str:
        return self._state.current_state

    def set_resume_target(self, target: SessionState) -> None:
        self._resume_target = target.value

    async def trigger(self, trigger_name: str) -> None:
        current = self._state.current_state
        dest    = self._resolve_dest(trigger_name, current)
        if dest is None:
            raise ValueError(f"No transition '{trigger_name}' from state '{current}'")

        prev = current
        self._state.current_state = dest
        self._store.save(self._state)
        await bus.emit(StateChangedEvent(previous=prev, current=dest))

    def _resolve_dest(self, trigger_name: str, current: str) -> Optional[str]:
        for t in TRANSITIONS:
            sources = t["source"]
            if isinstance(sources, list):
                sources = [s.value if isinstance(s, SessionState) else s for s in sources]
            else:
                sources = [sources.value if isinstance(sources, SessionState) else sources]

            if trigger_name == t["trigger"] and current in sources:
                dest = t["dest"]
                if dest is None:
                    # Dynamic destination
                    return self._resume_target
                return dest.value if isinstance(dest, SessionState) else dest
        return None
