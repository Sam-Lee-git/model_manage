"""macOS permission manager (sudo / SIP / Gatekeeper)."""

from __future__ import annotations

import asyncio
import os
import subprocess
from pathlib import Path

from model_manager.core.constants import PrivilegeLevel
from model_manager.permissions.base import PermissionManager


class MacOSPermissionManager(PermissionManager):

    def get_privilege_level(self) -> PrivilegeLevel:
        return PrivilegeLevel.ROOT if os.getuid() == 0 else PrivilegeLevel.STANDARD

    def check_write_access(self, path: Path) -> bool:
        target = path if path.exists() else path.parent
        return os.access(target, os.W_OK)

    def is_sip_enabled(self) -> bool:
        try:
            result = subprocess.run(
                ["csrutil", "status"], capture_output=True, text=True, timeout=5
            )
            return "enabled" in result.stdout.lower()
        except Exception:
            return True   # assume SIP on if we can't check

    def remove_quarantine(self, path: Path) -> None:
        """Remove Gatekeeper quarantine xattr from a downloaded file."""
        try:
            subprocess.run(
                ["xattr", "-d", "com.apple.quarantine", str(path)],
                capture_output=True, timeout=5,
            )
        except Exception:
            pass

    def homebrew_prefix(self) -> Path:
        import platform
        if platform.machine() == "arm64":
            return Path("/opt/homebrew")
        return Path("/usr/local")

    async def ensure_write_access(self, path: Path, reason: str) -> bool:
        if self.check_write_access(path):
            return True
        # Check SIP — if path is under /System, we cannot gain access
        if self.is_sip_enabled() and str(path).startswith("/System"):
            return False
        return await self.request_elevation(reason)

    async def run_elevated(self, command: list[str], reason: str) -> tuple[int, str]:
        prefix = [] if self.get_privilege_level() == PrivilegeLevel.ROOT else ["sudo", "--"]
        proc = await asyncio.create_subprocess_exec(
            *(prefix + command),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        stdout, _ = await proc.communicate()
        return proc.returncode or 0, stdout.decode(errors="replace")

    async def _do_elevate(self, reason: str) -> bool:
        import shutil
        if not shutil.which("sudo"):
            return False
        proc = await asyncio.create_subprocess_exec(
            "sudo", "-n", "true",
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await proc.communicate()
        return proc.returncode == 0
