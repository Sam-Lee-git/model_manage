"""ErrorContext — snapshot captured at the moment of failure."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from model_manager.hardware.profile import HardwareProfile


@dataclass
class EnvironmentSnapshot:
    python_version: str
    pip_packages: dict[str, str]   # name -> version
    conda_envs: Optional[list[str]]
    path_env: str
    cuda_home: Optional[str]
    conda_prefix: Optional[str]
    platform: str
    available_disk_gb: float
    available_ram_gb: float
    hf_token_present: bool = False  # True if HF_TOKEN / HUGGING_FACE_HUB_TOKEN is set


@dataclass
class ErrorContext:
    session_id: str
    timestamp: datetime
    step_name: str
    step_index: int
    exception_type: str
    exception_message: str
    traceback_str: str
    stdout_tail: str
    stderr_tail: str
    environment: EnvironmentSnapshot
    hardware: Optional[dict]          # serialised HardwareProfile
    attempted_command: Optional[str]
    working_directory: str
    elapsed_seconds: float


def _tail(text: str, lines: int = 200) -> str:
    return "\n".join(text.splitlines()[-lines:])


def _get_pip_packages() -> dict[str, str]:
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "list", "--format=json"],
            capture_output=True, text=True, timeout=30,
        )
        import json
        pkgs = json.loads(result.stdout)
        return {p["name"]: p["version"] for p in pkgs}
    except Exception:
        return {}


def _get_conda_envs() -> Optional[list[str]]:
    if not shutil.which("conda"):
        return None
    try:
        result = subprocess.run(
            ["conda", "env", "list", "--json"],
            capture_output=True, text=True, timeout=15,
        )
        import json
        data = json.loads(result.stdout)
        return data.get("envs", [])
    except Exception:
        return None


def build_error_context(
    *,
    session_id: str,
    step_name: str,
    step_index: int,
    exc: BaseException,
    stdout_tail: str = "",
    stderr_tail: str = "",
    hardware: Optional[HardwareProfile] = None,
    attempted_command: Optional[str] = None,
    elapsed_seconds: float = 0.0,
) -> ErrorContext:
    import psutil

    mem = psutil.virtual_memory()
    disk_free = 0.0
    try:
        disk_free = psutil.disk_usage(os.getcwd()).free / 1e9
    except Exception:
        pass

    hf_token_present = bool(
        os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    )
    env = EnvironmentSnapshot(
        python_version=sys.version,
        pip_packages=_get_pip_packages(),
        conda_envs=_get_conda_envs(),
        path_env=os.environ.get("PATH", ""),
        cuda_home=os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH"),
        conda_prefix=os.environ.get("CONDA_PREFIX"),
        platform=sys.platform,
        available_disk_gb=disk_free,
        available_ram_gb=mem.available / 1e9,
        hf_token_present=hf_token_present,
    )

    return ErrorContext(
        session_id=session_id,
        timestamp=datetime.utcnow(),
        step_name=step_name,
        step_index=step_index,
        exception_type=type(exc).__name__,
        exception_message=str(exc),
        traceback_str=traceback.format_exc(),
        stdout_tail=_tail(stdout_tail),
        stderr_tail=_tail(stderr_tail),
        environment=env,
        hardware=hardware.to_dict() if hardware else None,
        attempted_command=attempted_command,
        working_directory=os.getcwd(),
        elapsed_seconds=elapsed_seconds,
    )
