"""Linux permission manager (sudo / pkexec)."""

from __future__ import annotations

import asyncio
import os
import shutil
from pathlib import Path

from model_manager.core.constants import PrivilegeLevel
from model_manager.permissions.base import PermissionManager


class LinuxPermissionManager(PermissionManager):

    def get_privilege_level(self) -> PrivilegeLevel:
        return PrivilegeLevel.ROOT if os.getuid() == 0 else PrivilegeLevel.STANDARD

    def check_write_access(self, path: Path) -> bool:
        target = path if path.exists() else path.parent
        return os.access(target, os.W_OK)

    async def ensure_write_access(self, path: Path, reason: str) -> bool:
        if self.check_write_access(path):
            return True
        return await self.request_elevation(reason)

    async def run_elevated(self, command: list[str], reason: str) -> tuple[int, str]:
        if self.get_privilege_level() == PrivilegeLevel.ROOT:
            proc = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
            stdout, _ = await proc.communicate()
            return proc.returncode or 0, stdout.decode(errors="replace")

        sudo_cmd = ["sudo", "--"] + command
        proc = await asyncio.create_subprocess_exec(
            *sudo_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        stdout, _ = await proc.communicate()
        return proc.returncode or 0, stdout.decode(errors="replace")

    async def _do_elevate(self, reason: str) -> bool:
        """Test if passwordless sudo is available."""
        if not shutil.which("sudo"):
            return False
        proc = await asyncio.create_subprocess_exec(
            "sudo", "-n", "true",
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await proc.communicate()
        return proc.returncode == 0
