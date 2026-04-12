"""Windows permission manager (UAC / PowerShell)."""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from pathlib import Path

from model_manager.core.constants import PrivilegeLevel
from model_manager.permissions.base import PermissionManager


class WindowsPermissionManager(PermissionManager):

    def get_privilege_level(self) -> PrivilegeLevel:
        try:
            import ctypes
            if ctypes.windll.shell32.IsUserAnAdmin():
                return PrivilegeLevel.ELEVATED
        except Exception:
            pass
        return PrivilegeLevel.STANDARD

    def check_write_access(self, path: Path) -> bool:
        test = path if path.exists() else path.parent
        try:
            import tempfile
            with tempfile.NamedTemporaryFile(dir=test, delete=True):
                pass
            return True
        except (PermissionError, OSError):
            return False

    async def ensure_write_access(self, path: Path, reason: str) -> bool:
        if self.check_write_access(path):
            return True
        return await self.request_elevation(reason)

    async def run_elevated(self, command: list[str], reason: str) -> tuple[int, str]:
        if self.get_privilege_level() == PrivilegeLevel.ELEVATED:
            proc = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
            stdout, _ = await proc.communicate()
            return proc.returncode, stdout.decode(errors="replace")

        # Re-launch as admin — cannot capture output easily via ShellExecute,
        # so we write a temp script and run it
        import tempfile
        cmd_str = subprocess.list2cmdline(command)
        with tempfile.NamedTemporaryFile("w", suffix=".bat", delete=False) as f:
            f.write(f"@echo off\n{cmd_str}\n")
            bat = f.name

        try:
            import ctypes
            rc = ctypes.windll.shell32.ShellExecuteW(
                None, "runas", "cmd.exe", f"/c {bat}", None, 1
            )
            # ShellExecuteW returns > 32 on success
            return (0 if rc > 32 else 1), ""
        finally:
            try:
                os.unlink(bat)
            except OSError:
                pass

    async def _do_elevate(self, reason: str) -> bool:
        # On Windows, elevation requires re-launching the process.
        # We signal the caller that elevation is needed via a raised exception
        # rather than trying to elevate the current process in-place.
        from model_manager.core.exceptions import ElevationFailedError
        raise ElevationFailedError(
            "Cannot elevate the current process on Windows. "
            "Please re-run the application as Administrator."
        )

    def fix_powershell_policy(self) -> None:
        """Set PowerShell execution policy to RemoteSigned for current user."""
        subprocess.run(
            ["powershell", "-Command",
             "Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned -Force"],
            capture_output=True, timeout=15,
        )
