"""BackendSelector — auto-detect best installation backend based on hardware and environment."""

from __future__ import annotations

import shutil
import sys
from typing import Optional

from model_manager.backends.base import InstallBackend
from model_manager.backends.conda import CondaBackend
from model_manager.backends.pip_venv import PipVenvBackend
from model_manager.core.constants import InstallBackendType
from model_manager.hardware.profile import HardwareProfile


# Recommendation reasons surfaced to the user in logs
_REASONS: dict[InstallBackendType, str] = {
    InstallBackendType.CONDA:    "NVIDIA GPU detected — conda manages CUDA toolkit versions cleanly",
    InstallBackendType.PIP_VENV: "pip + venv — lightweight, no extra tooling required",
    InstallBackendType.DOCKER:   "Linux server / production deployment — Docker provides full isolation",
}


class BackendSelector:

    async def select(
        self,
        hardware: Optional[HardwareProfile],
        requires_system_libs: bool = False,
        user_preference: Optional[InstallBackendType] = None,
    ) -> tuple[InstallBackend, str]:
        """
        Return (backend, reason_string).

        Selection priority:
          1. User explicit preference (--backend flag)
          2. NVIDIA GPU + conda available  → conda
             (conda handles cudatoolkit / cudnn automatically)
          3. Linux + Docker available + no conda → Docker
             (good for server / headless deployments)
          4. Fallback: pip + venv (always available)

        Docker is NOT used as an install backend here (packages are still installed
        via pip inside the container); the Docker option is surfaced in launch
        instructions after installation completes.
        """
        if user_preference:
            backend = await self._get_backend(user_preference)
            reason  = _REASONS.get(user_preference, "user-specified")
            return backend, reason

        conda = CondaBackend()
        pip   = PipVenvBackend()

        # Rule 1: NVIDIA GPU → prefer conda (CUDA toolkit management)
        if hardware and hardware.has_nvidia_gpu and await conda.is_available():
            return conda, _REASONS[InstallBackendType.CONDA]

        # Rule 2: AMD GPU (ROCm) → conda for ROCm package availability
        if hardware and hardware.has_amd_gpu and await conda.is_available():
            return conda, "AMD/ROCm GPU detected — conda provides best ROCm compatibility"

        # Rule 3: Linux + no GPU + Docker available → note it, but still use pip
        # (Docker as full install backend is not yet implemented; it is surfaced
        #  in launch instructions so server users can choose it post-install)
        if sys.platform == "linux" and _docker_available():
            return pip, (
                "pip + venv (Docker also available — see launch instructions "
                "for a Docker-based deployment option)"
            )

        # Rule 4: macOS with Apple Silicon → pip is fine (Metal via llama-cpp-python)
        if sys.platform == "darwin":
            return pip, "pip + venv — llama-cpp-python supports Apple Metal automatically"

        # Default
        return pip, _REASONS[InstallBackendType.PIP_VENV]

    async def _get_backend(self, pref: InstallBackendType) -> InstallBackend:
        if pref == InstallBackendType.CONDA:
            b = CondaBackend()
            if not await b.is_available():
                raise RuntimeError("conda/mamba not found in PATH")
            return b
        if pref == InstallBackendType.DOCKER:
            raise NotImplementedError(
                "Docker install backend is not yet implemented. "
                "Use pip_venv or conda, then follow the Docker launch instructions."
            )
        return PipVenvBackend()


def _docker_available() -> bool:
    return shutil.which("docker") is not None
