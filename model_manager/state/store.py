"""Atomic JSON state persistence."""

from __future__ import annotations

import dataclasses
import json
import os
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from model_manager.config.paths import get_sessions_dir, ensure_dirs
from model_manager.core.exceptions import SessionNotFoundError
from model_manager.state.models import BranchState, BranchStep, InstallStep, InstallationState


class _Encoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        if dataclasses.is_dataclass(obj):
            return dataclasses.asdict(obj)
        if isinstance(obj, Path):
            return str(obj)
        return super().default(obj)


def _decode(data: dict) -> InstallationState:
    """Best-effort decode — tolerates missing fields for forward/backward compat."""
    steps = [InstallStep(**s) for s in data.pop("steps", [])]
    branches_raw = data.pop("branch_history", [])
    branches = []
    for b in branches_raw:
        fix_steps = [BranchStep(**s) for s in b.pop("fix_steps", [])]
        completed  = [BranchStep(**s) for s in b.pop("completed_steps", [])]
        branches.append(BranchState(fix_steps=fix_steps, completed_steps=completed, **b))

    for dt_field in ("created_at", "updated_at"):
        if isinstance(data.get(dt_field), str):
            data[dt_field] = datetime.fromisoformat(data[dt_field])

    return InstallationState(steps=steps, branch_history=branches, **data)


class StateStore:
    def __init__(self, sessions_dir: Optional[Path] = None) -> None:
        self._dir = sessions_dir or get_sessions_dir()

    def _path(self, session_id: str) -> Path:
        return self._dir / f"{session_id}.json"

    def save(self, state: InstallationState) -> None:
        ensure_dirs()
        state.updated_at = datetime.utcnow()
        target = self._path(state.session_id)
        # Atomic write: write to temp file then rename
        fd, tmp = tempfile.mkstemp(dir=self._dir, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(dataclasses.asdict(state), f, cls=_Encoder, indent=2)
            os.replace(tmp, target)
        except Exception:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    def load(self, session_id: str) -> InstallationState:
        path = self._path(session_id)
        if not path.exists():
            raise SessionNotFoundError(f"Session {session_id} not found at {path}")
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return _decode(data)

    def list_sessions(self) -> list[dict]:
        ensure_dirs()
        sessions = []
        for p in sorted(self._dir.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
            try:
                with open(p, encoding="utf-8") as f:
                    d = json.load(f)
                sessions.append({
                    "session_id": d.get("session_id", p.stem),
                    "current_state": d.get("current_state", "?"),
                    "model": d.get("selected_model_id", "—"),
                    "updated_at": d.get("updated_at", ""),
                })
            except Exception:
                continue
        return sessions

    def delete(self, session_id: str) -> None:
        p = self._path(session_id)
        if p.exists():
            p.unlink()
