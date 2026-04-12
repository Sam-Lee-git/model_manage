"""HardwareDetector orchestrator."""

from __future__ import annotations

import asyncio
import sys

from model_manager.hardware.cpu    import detect_cpu
from model_manager.hardware.memory import detect_memory
from model_manager.hardware.gpu    import detect_gpus
from model_manager.hardware.disk   import detect_drives
from model_manager.hardware.profile import HardwareProfile


class HardwareDetector:
    async def detect(self) -> HardwareProfile:
        cpu, (ram_total, ram_avail), gpus, drives = await asyncio.gather(
            detect_cpu(),
            detect_memory(),
            detect_gpus(),
            detect_drives(),
        )
        return HardwareProfile(
            cpu=cpu,
            ram_total_gb=ram_total,
            ram_available_gb=ram_avail,
            gpus=gpus,
            drives=drives,
            os_platform=sys.platform,
            os_version=_os_version(),
        )


def _os_version() -> str:
    import platform
    if sys.platform == "win32":
        return platform.version()
    if sys.platform == "darwin":
        return platform.mac_ver()[0]
    return platform.release()
