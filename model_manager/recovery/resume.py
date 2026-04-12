"""ResumeCoordinator — merges branch outcomes and resumes main flow."""

from __future__ import annotations

from model_manager.core.constants import StepStatus
from model_manager.core.events import ResumeEvent, bus
from model_manager.recovery.branch import BranchResult
from model_manager.state.models import InstallationState
from model_manager.state.store import StateStore


class ResumeCoordinator:
    def __init__(self, store: StateStore) -> None:
        self._store = store

    async def resume(
        self,
        session: InstallationState,
        branch_result: BranchResult,
        failed_step_index: int,
    ) -> None:
        # 1. Merge environmental changes
        if branch_result.branch_state:
            session.apply_branch_outcomes(branch_result.branch_state)

        # 2. Mark failed step for retry
        if 0 <= failed_step_index < len(session.steps):
            session.steps[failed_step_index].status = StepStatus.PENDING_RETRY
            session.steps[failed_step_index].retry_count += 1

        # 3. Persist
        self._store.save(session)

        # 4. Signal main loop
        await bus.emit(ResumeEvent(step_index=failed_step_index))
