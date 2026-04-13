"""BranchExecutor — runs fix steps in an isolated environment."""

from __future__ import annotations

import asyncio
import os
import subprocess
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from model_manager.core.constants import MAX_BRANCH_DEPTH, StepStatus
from model_manager.core.events import (
    BranchStartedEvent, BranchStepEvent, BranchSucceededEvent,
    BranchFailedEvent, bus,
)
from model_manager.core.exceptions import BranchDepthExceededError, BranchFailedError
from model_manager.recovery.context import ErrorContext
from model_manager.state.models import BranchState, BranchStep
from model_manager.state.store import StateStore


@dataclass
class DiagnosisResult:
    error_category: str
    root_cause: str
    confidence: float
    fix_plan: list[dict[str, Any]]          # raw dicts from Claude
    alternative_plans: list[list[dict[str, Any]]]
    user_explanation: str
    requires_user_decision: bool = False
    decision_options: list[str] = None

    def __post_init__(self):
        if self.decision_options is None:
            self.decision_options = []


@dataclass
class BranchResult:
    success: bool
    branch_state: Optional[BranchState] = None
    error: Optional[str] = None


def _parse_fix_steps(raw: list[dict]) -> list[BranchStep]:
    steps = []
    for item in raw:
        steps.append(BranchStep(
            description=item.get("description", ""),
            action_type=item.get("action_type", "run_command"),
            action_params=item.get("action_params", {}),
            requires_elevation=item.get("requires_elevation", False),
            timeout_seconds=item.get("timeout_seconds", 120),
            verify_command=item.get("verify_command"),
        ))
    return steps


class BranchExecutor:
    def __init__(self, store: StateStore) -> None:
        self._store = store

    async def execute(
        self,
        diagnosis: DiagnosisResult,
        context: ErrorContext,
        depth: int = 0,
    ) -> BranchResult:
        if depth >= MAX_BRANCH_DEPTH:
            raise BranchDepthExceededError(
                f"Branch depth {depth} exceeds maximum {MAX_BRANCH_DEPTH}"
            )

        fix_steps = _parse_fix_steps(diagnosis.fix_plan)
        branch = BranchState(
            parent_session_id=context.session_id,
            depth=depth,
            fix_steps=fix_steps,
        )

        await bus.emit(BranchStartedEvent(depth=depth, fix_steps=len(fix_steps)))

        for i, step in enumerate(fix_steps):
            await bus.emit(BranchStepEvent(step_description=step.description, step_index=i))
            try:
                output, env_changes = await self._execute_step(step)
                step.status = StepStatus.COMPLETED
                step.output = output
                branch.completed_steps.append(step)
                branch.environmental_changes.extend(env_changes)
                self._save_branch(context.session_id, branch)
            except Exception as e:
                step.status = StepStatus.FAILED
                step.error  = str(e)

                # Try alternative plans (recursive branch, depth+1)
                if diagnosis.alternative_plans:
                    alt_diagnosis = DiagnosisResult(
                        error_category=diagnosis.error_category,
                        root_cause=diagnosis.root_cause,
                        confidence=diagnosis.confidence,
                        fix_plan=diagnosis.alternative_plans[0],
                        alternative_plans=diagnosis.alternative_plans[1:],
                        user_explanation=diagnosis.user_explanation,
                    )
                    try:
                        return await self.execute(alt_diagnosis, context, depth + 1)
                    except BranchDepthExceededError:
                        pass

                await bus.emit(BranchFailedEvent(depth=depth, reason=str(e)))
                return BranchResult(success=False, branch_state=branch, error=str(e))

        branch.succeeded = True
        await bus.emit(BranchSucceededEvent(depth=depth))
        return BranchResult(success=True, branch_state=branch)

    async def _execute_step(
        self, step: BranchStep
    ) -> tuple[str, list[dict]]:
        """Execute one fix step. Returns (output, env_changes)."""
        env_changes: list[dict] = []
        atype = step.action_type

        if atype == "run_command":
            cmd = step.action_params.get("command", "")
            output = await self._run_command(cmd, step.timeout_seconds)

        elif atype == "install_package":
            import sys
            pkgs = step.action_params.get("packages", [])
            index_url = step.action_params.get("index_url", "")
            cmd = [sys.executable, "-m", "pip", "install", "--quiet"] + pkgs
            if index_url:
                cmd += ["--extra-index-url", index_url]
            output = await self._run_command(cmd, step.timeout_seconds)

        elif atype == "modify_env_var":
            key = step.action_params.get("key", "")
            value = step.action_params.get("value", "")
            os.environ[key] = value
            env_changes.append({"kind": "env_var", "key": key, "value": value})
            output = f"Set {key}={value}"

        elif atype == "write_config":
            path_str = step.action_params.get("path", "")
            content  = step.action_params.get("content", "")
            path = Path(path_str).expanduser()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
            output = f"Wrote {path}"

        else:
            output = f"Unknown action_type '{atype}' — skipped"

        # Run verify command if specified
        if step.verify_command:
            verify_out = await self._run_command(step.verify_command, 30)
            output += f"\nVerify: {verify_out}"

        return output, env_changes

    async def _run_command(
        self,
        cmd: str | list[str],
        timeout: int = 120,
    ) -> str:
        # Explicitly pass current os.environ so runtime changes (e.g. HF_TOKEN set
        # by the user mid-session) are visible to the subprocess.
        current_env = os.environ.copy()
        if isinstance(cmd, str):
            proc = await asyncio.create_subprocess_shell(
                cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                env=current_env,
            )
        else:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                env=current_env,
            )
        try:
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            proc.kill()
            raise RuntimeError(f"Command timed out after {timeout}s: {cmd}")

        output = stdout.decode(errors="replace")
        if proc.returncode != 0:
            raise RuntimeError(
                f"Command failed (exit {proc.returncode}): {cmd}\n{output[-2000:]}"
            )
        return output

    def _save_branch(self, session_id: str, branch: BranchState) -> None:
        try:
            state = self._store.load(session_id)
            # Replace or append
            for i, b in enumerate(state.branch_history):
                if b.branch_id == branch.branch_id:
                    state.branch_history[i] = branch
                    self._store.save(state)
                    return
            state.branch_history.append(branch)
            self._store.save(state)
        except Exception:
            pass
