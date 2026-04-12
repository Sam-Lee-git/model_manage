"""Abstract PermissionManager."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from model_manager.core.constants import PrivilegeLevel


class PermissionManager(ABC):

    @abstractmethod
    def get_privilege_level(self) -> PrivilegeLevel: ...

    @abstractmethod
    def check_write_access(self, path: Path) -> bool: ...

    @abstractmethod
    async def ensure_write_access(self, path: Path, reason: str) -> bool:
        """Returns True if access was granted (possibly after elevation)."""
        ...

    @abstractmethod
    async def run_elevated(self, command: list[str], reason: str) -> tuple[int, str]:
        """Run a command with elevated privileges. Returns (returncode, output)."""
        ...

    def can_write_system_paths(self) -> bool:
        return self.get_privilege_level() in (PrivilegeLevel.ELEVATED, PrivilegeLevel.ROOT)

    async def request_elevation(self, reason: str) -> bool:
        """
        Ask user to confirm elevation, then attempt it.
        Returns True if elevation succeeded or is already elevated.
        """
        level = self.get_privilege_level()
        if level in (PrivilegeLevel.ELEVATED, PrivilegeLevel.ROOT):
            return True
        return await self._do_elevate(reason)

    @abstractmethod
    async def _do_elevate(self, reason: str) -> bool: ...
