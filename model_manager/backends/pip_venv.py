"""pip + venv backend."""

from __future__ import annotations

import asyncio
import sys
import venv
from pathlib import Path
from typing import Optional

from model_manager.backends.base import (
    EnvInfo, InstallBackend, InstallResult, PackageSpec,
)


class PipVenvBackend(InstallBackend):
    name = "pip_venv"

    async def is_available(self) -> bool:
        return True   # sys.executable is always available

    async def create_environment(
        self, env_name: str, python_version: str = "3.11", base_path: Optional[Path] = None
    ) -> EnvInfo:
        root = base_path or Path.home() / ".model_manager" / "envs"
        env_path = root / env_name
        await asyncio.to_thread(_create_venv, env_path)

        if sys.platform == "win32":
            python_exe = env_path / "Scripts" / "python.exe"
        else:
            python_exe = env_path / "bin" / "python"

        return EnvInfo(
            name=env_name,
            path=env_path,
            python_executable=python_exe,
            backend_type=self.name,
        )

    async def install_packages(
        self,
        packages: list[PackageSpec],
        env: EnvInfo,
        extra_index_urls: list[str] | None = None,
    ) -> InstallResult:
        pip_specs = [p.to_pip_string() for p in packages]
        cmd = [str(env.python_executable), "-m", "pip", "install", "--quiet"] + pip_specs

        for url in (extra_index_urls or []):
            cmd += ["--extra-index-url", url]

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        stdout, _ = await proc.communicate()
        output = stdout.decode(errors="replace")

        if proc.returncode != 0:
            from model_manager.core.exceptions import InstallError
            raise InstallError(
                f"pip install failed (exit {proc.returncode})",
                stderr=output,
            )

        return InstallResult(success=True, installed=pip_specs, output=output)

    async def run_in_env(self, command: list[str], env: EnvInfo) -> tuple[int, str]:
        # Replace 'python' with env python
        if command and command[0] in ("python", "python3"):
            command = [str(env.python_executable)] + command[1:]

        proc = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        stdout, _ = await proc.communicate()
        return proc.returncode or 0, stdout.decode(errors="replace")

    def get_env_activation_command(self, env: EnvInfo) -> str:
        if sys.platform == "win32":
            return str(env.path / "Scripts" / "activate.bat")
        return f"source {env.path / 'bin' / 'activate'}"


def _create_venv(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    venv.create(str(path), with_pip=True, clear=False)
