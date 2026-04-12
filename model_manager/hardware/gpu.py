"""GPU detection across CUDA, ROCm, Metal, and CPU-only environments."""

from __future__ import annotations

import asyncio
import json
import shutil
import subprocess
import sys
from typing import Optional

from model_manager.core.constants import ComputeBackend
from model_manager.hardware.profile import GPUInfo


async def detect_gpus() -> list[GPUInfo]:
    return await asyncio.to_thread(_detect_gpus_sync)


def _detect_gpus_sync() -> list[GPUInfo]:
    gpus: list[GPUInfo] = []

    # 1. NVIDIA via nvidia-smi
    nvidia = _try_nvidia()
    if nvidia:
        return nvidia

    # 2. AMD via rocm-smi
    amd = _try_rocm()
    if amd:
        return amd

    # 3. Apple Metal (macOS)
    if sys.platform == "darwin":
        metal = _try_metal()
        if metal:
            return metal

    # 4. CPU-only sentinel
    return [GPUInfo(name="CPU only", vram_gb=0.0, compute_backend=ComputeBackend.CPU)]


def _try_nvidia() -> list[GPUInfo]:
    if not shutil.which("nvidia-smi"):
        return []
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            return []

        # Detect CUDA version from nvidia-smi header
        cuda_ver = _get_cuda_version()

        gpus = []
        for i, line in enumerate(result.stdout.strip().splitlines()):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 3:
                continue
            name, vram_mib, driver = parts[0], parts[1], parts[2]
            gpus.append(GPUInfo(
                name=name,
                vram_gb=float(vram_mib) / 1024,
                compute_backend=ComputeBackend.CUDA,
                cuda_version=cuda_ver,
                driver_version=driver,
                device_index=i,
            ))
        return gpus
    except Exception:
        return []


def _get_cuda_version() -> Optional[str]:
    try:
        result = subprocess.run(
            ["nvidia-smi"], capture_output=True, text=True, timeout=5
        )
        for line in result.stdout.splitlines():
            if "CUDA Version:" in line:
                return line.split("CUDA Version:")[1].strip().split()[0]
    except Exception:
        pass
    return None


def _try_rocm() -> list[GPUInfo]:
    if not shutil.which("rocm-smi"):
        return []
    try:
        result = subprocess.run(
            ["rocm-smi", "--showproductname", "--showmeminfo", "vram", "--json"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            return []
        data = json.loads(result.stdout)
        gpus = []
        for i, (card, info) in enumerate(data.items()):
            name = info.get("Card series", info.get("Card SKU", f"AMD GPU {i}"))
            vram_bytes = int(info.get("VRAM Total Memory (B)", 0))
            gpus.append(GPUInfo(
                name=name,
                vram_gb=vram_bytes / 1e9,
                compute_backend=ComputeBackend.ROCM,
                device_index=i,
            ))
        return gpus
    except Exception:
        return []


def _try_metal() -> list[GPUInfo]:
    try:
        result = subprocess.run(
            ["system_profiler", "SPDisplaysDataType", "-json"],
            capture_output=True, text=True, timeout=10,
        )
        data = json.loads(result.stdout)
        displays = data.get("SPDisplaysDataType", [])
        gpus = []
        for i, d in enumerate(displays):
            name = d.get("sppci_model", "Apple GPU")
            # VRAM for discrete; integrated shares system RAM
            vram_str = d.get("spdisplays_vram", "0 MB")
            vram_gb = _parse_vram(vram_str)
            gpus.append(GPUInfo(
                name=name,
                vram_gb=vram_gb,
                compute_backend=ComputeBackend.METAL,
                device_index=i,
            ))
        return gpus
    except Exception:
        return []


def _parse_vram(s: str) -> float:
    """Parse '8 GB' or '512 MB' -> float GB."""
    parts = s.split()
    if not parts:
        return 0.0
    try:
        val = float(parts[0])
        unit = parts[1].upper() if len(parts) > 1 else "MB"
        return val if "GB" in unit else val / 1024
    except (ValueError, IndexError):
        return 0.0
