"""State machine dataclasses."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from model_manager.core.constants import SessionState, StepStatus


@dataclass
class InstallStep:
    step_id: str
    step_type: str
    description: str
    status: StepStatus = StepStatus.PENDING
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None
    artifacts: dict[str, Any] = field(default_factory=dict)
    retry_count: int = 0


@dataclass
class BranchStep:
    description: str
    action_type: str
    action_params: dict[str, Any]
    requires_elevation: bool = False
    timeout_seconds: int = 120
    verify_command: Optional[str] = None
    status: StepStatus = StepStatus.PENDING
    output: str = ""
    error: Optional[str] = None


@dataclass
class BranchState:
    branch_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    parent_session_id: str = ""
    depth: int = 0
    fix_steps: list[BranchStep] = field(default_factory=list)
    completed_steps: list[BranchStep] = field(default_factory=list)
    succeeded: bool = False
    environmental_changes: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class InstallationState:
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    # Selections
    hardware_profile: Optional[dict] = None     # serialised HardwareProfile
    selected_model_id: Optional[str] = None
    selected_quant: Optional[str] = None
    install_path: Optional[str] = None
    backend: Optional[str] = None

    # Progress
    current_state: str = SessionState.IDLE.value
    steps: list[InstallStep] = field(default_factory=list)
    current_step_index: int = 0

    # Recovery
    branch_history: list[BranchState] = field(default_factory=list)
    error_count: int = 0

    # Conversation
    conversation_session_id: Optional[str] = None

    def apply_branch_outcomes(self, branch: BranchState) -> None:
        for change in branch.environmental_changes:
            kind = change.get("kind")
            if kind == "env_var":
                import os
                os.environ[change["key"]] = change["value"]
        self.branch_history.append(branch)
        self.updated_at = datetime.utcnow()
