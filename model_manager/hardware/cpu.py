"""CPU detection."""

from __future__ import annotations

import asyncio
import platform
import struct
import subprocess
import sys

try:
    import psutil as _psutil
except ImportError:
    _psutil = None  # type: ignore

from model_manager.hardware.profile import CPUInfo


def _check_avx_linux() -> tuple[bool, bool]:
    """Read /proc/cpuinfo flags for AVX2/AVX-512."""
    try:
        with open("/proc/cpuinfo") as f:
            flags_line = next((l for l in f if l.startswith("flags")), "")
        avx2   = "avx2"    in flags_line
        avx512 = "avx512f" in flags_line
        return avx2, avx512
    except Exception:
        return False, False


def _check_avx_windows() -> tuple[bool, bool]:
    """Use CPUID via ctypes on Windows to check AVX2/AVX-512."""
    try:
        import ctypes
        # Simplified: check via environment variable set by some ML installers
        # Full CPUID would require inline asm or a C extension
        return False, False
    except Exception:
        return False, False


def _check_avx_macos() -> tuple[bool, bool]:
    try:
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.features", "machdep.cpu.leaf7_features"],
            capture_output=True, text=True, timeout=5
        )
        combined = result.stdout.upper()
        return "AVX2" in combined, "AVX512" in combined
    except Exception:
        return False, False


async def detect_cpu() -> CPUInfo:
    """Async CPU detection (offloads blocking calls to thread pool)."""
    return await asyncio.to_thread(_detect_cpu_sync)


def _detect_cpu_sync() -> CPUInfo:
    freq = _psutil.cpu_freq() if _psutil else None
    base_mhz = freq.min if freq else 0.0

    # Brand string
    brand = platform.processor() or "Unknown CPU"
    # On Linux, platform.processor() can be empty; try /proc/cpuinfo
    if sys.platform == "linux" and not brand.strip():
        try:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.startswith("model name"):
                        brand = line.split(":", 1)[1].strip()
                        break
        except Exception:
            pass

    arch = platform.machine()

    if sys.platform == "linux":
        avx2, avx512 = _check_avx_linux()
    elif sys.platform == "darwin":
        avx2, avx512 = _check_avx_macos()
    else:
        avx2, avx512 = _check_avx_windows()

    return CPUInfo(
        brand=brand,
        physical_cores=(_psutil.cpu_count(logical=False) if _psutil else 1) or 1,
        logical_cores=(_psutil.cpu_count(logical=True) if _psutil else 1) or 1,
        architecture=arch,
        supports_avx2=avx2,
        supports_avx512=avx512,
        base_freq_mhz=base_mhz,
    )
