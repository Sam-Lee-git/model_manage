"""BackendSelector — auto-detect best installation backend."""

from __future__ import annotations

from typing import Optional

from model_manager.backends.base import InstallBackend
from model_manager.backends.conda import CondaBackend
from model_manager.backends.pip_venv import PipVenvBackend
from model_manager.core.constants import InstallBackendType
from model_manager.hardware.profile import HardwareProfile


class BackendSelector:

    async def select(
        self,
        hardware: HardwareProfile,
        requires_system_libs: bool = False,
        user_preference: Optional[InstallBackendType] = None,
    ) -> InstallBackend:
        if user_preference:
            return await self._get_backend(user_preference)

        conda = CondaBackend()
        pip   = PipVenvBackend()

        # Conda preferred when NVIDIA GPU present (best CUDA toolkit management)
        if hardware.has_nvidia_gpu and await conda.is_available():
            return conda

        # pip/venv as universal fallback
        return pip

    async def _get_backend(self, pref: InstallBackendType) -> InstallBackend:
        if pref == InstallBackendType.CONDA:
            b = CondaBackend()
            if not await b.is_available():
                raise RuntimeError("conda/mamba not found in PATH")
            return b
        if pref == InstallBackendType.DOCKER:
            # Docker backend placeholder
            raise NotImplementedError("Docker backend not yet implemented")
        return PipVenvBackend()
