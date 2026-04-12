"""Conda / mamba backend."""

from __future__ import annotations

import asyncio
import json
import shutil
import sys
from pathlib import Path
from typing import Optional

from model_manager.backends.base import (
    EnvInfo, InstallBackend, InstallResult, PackageSpec,
)


class CondaBackend(InstallBackend):
    name = "conda"

    def __init__(self) -> None:
        self._exe = shutil.which("mamba") or shutil.which("conda") or "conda"

    async def is_available(self) -> bool:
        return shutil.which("conda") is not None or shutil.which("mamba") is not None

    async def create_environment(
        self, env_name: str, python_version: str = "3.11", base_path: Optional[Path] = None
    ) -> EnvInfo:
        cmd = [self._exe, "create", "-n", env_name, f"python={python_version}", "-y", "--quiet"]
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        stdout, _ = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(
                f"conda create failed:\n{stdout.decode(errors='replace')}"
            )

        env_path = await self._get_env_path(env_name)
        python_exe = env_path / ("python.exe" if sys.platform == "win32" else "bin/python")

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
        conda_pkgs = [p for p in packages if p.install_via == "conda"]
        pip_pkgs   = [p for p in packages if p.install_via != "conda"]

        output = ""
        installed: list[str] = []

        if conda_pkgs:
            channels = set()
            specs = []
            for p in conda_pkgs:
                specs.append(p.to_conda_string())
                if p.conda_channel:
                    channels.add(p.conda_channel)
            channels.add("conda-forge")

            cmd = [self._exe, "install", "-n", env.name, "-y", "--quiet"]
            for ch in channels:
                cmd += ["-c", ch]
            cmd += specs

            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
            out, _ = await proc.communicate()
            output += out.decode(errors="replace")
            if proc.returncode != 0:
                from model_manager.core.exceptions import InstallError
                raise InstallError("conda install failed", stderr=output)
            installed += specs

        if pip_pkgs:
            pip_specs = [p.to_pip_string() for p in pip_pkgs]
            cmd = [str(env.python_executable), "-m", "pip", "install", "--quiet"] + pip_specs
            for url in (extra_index_urls or []):
                cmd += ["--extra-index-url", url]

            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
            out, _ = await proc.communicate()
            output += out.decode(errors="replace")
            if proc.returncode != 0:
                from model_manager.core.exceptions import InstallError
                raise InstallError("pip install (in conda env) failed", stderr=output)
            installed += pip_specs

        return InstallResult(success=True, installed=installed, output=output)

    async def run_in_env(self, command: list[str], env: EnvInfo) -> tuple[int, str]:
        cmd = [self._exe, "run", "-n", env.name, "--no-capture-output"] + command
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        stdout, _ = await proc.communicate()
        return proc.returncode or 0, stdout.decode(errors="replace")

    def get_env_activation_command(self, env: EnvInfo) -> str:
        return f"conda activate {env.name}"

    async def _get_env_path(self, env_name: str) -> Path:
        proc = await asyncio.create_subprocess_exec(
            self._exe, "env", "list", "--json",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        stdout, _ = await proc.communicate()
        data = json.loads(stdout.decode(errors="replace"))
        for p in data.get("envs", []):
            if Path(p).name == env_name:
                return Path(p)
        raise RuntimeError(f"Conda env '{env_name}' not found after creation")
