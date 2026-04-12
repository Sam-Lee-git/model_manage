"""Abstract InstallBackend."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import AsyncIterator, Optional


@dataclass
class EnvInfo:
    name: str
    path: Path
    python_executable: Path
    backend_type: str


@dataclass
class PackageSpec:
    name: str
    version_constraint: Optional[str] = None    # ">=2.1.0,<3.0"
    extras: list[str] = field(default_factory=list)
    install_via: str = "pip"                    # "conda" | "pip" | "system" | "git"
    index_url: Optional[str] = None
    conda_channel: Optional[str] = None
    pre_release: bool = False

    def to_pip_string(self) -> str:
        base = self.name
        if self.extras:
            base += f"[{','.join(self.extras)}]"
        if self.version_constraint:
            base += self.version_constraint
        return base

    def to_conda_string(self) -> str:
        base = self.name
        if self.version_constraint:
            # conda uses spaces: "torch>=2.0"
            base += self.version_constraint
        return base


@dataclass
class InstallResult:
    success: bool
    installed: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)
    output: str = ""


class InstallBackend(ABC):
    name: str = "base"

    @abstractmethod
    async def is_available(self) -> bool: ...

    @abstractmethod
    async def create_environment(
        self, env_name: str, python_version: str = "3.11", base_path: Optional[Path] = None
    ) -> EnvInfo: ...

    @abstractmethod
    async def install_packages(
        self,
        packages: list[PackageSpec],
        env: EnvInfo,
        extra_index_urls: list[str] | None = None,
    ) -> InstallResult: ...

    @abstractmethod
    async def run_in_env(
        self, command: list[str], env: EnvInfo
    ) -> tuple[int, str]: ...

    @abstractmethod
    def get_env_activation_command(self, env: EnvInfo) -> str: ...

    async def stream_output(
        self, command: list[str], env: Optional[EnvInfo] = None
    ) -> AsyncIterator[str]:
        """Run a command and yield stdout lines in real time."""
        import asyncio
        env_exe = str(env.python_executable) if env else None
        cmd = command

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        assert proc.stdout is not None
        async for line in proc.stdout:
            yield line.decode(errors="replace").rstrip()
        await proc.wait()
