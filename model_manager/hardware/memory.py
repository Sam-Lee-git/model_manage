"""RAM detection."""

from __future__ import annotations

import asyncio

try:
    import psutil as _psutil
except ImportError:
    _psutil = None  # type: ignore


async def detect_memory() -> tuple[float, float]:
    """Return (total_gb, available_gb)."""
    return await asyncio.to_thread(_detect_memory_sync)


def _detect_memory_sync() -> tuple[float, float]:
    if _psutil:
        mem = _psutil.virtual_memory()
        return mem.total / 1e9, mem.available / 1e9
    # Fallback: read /proc/meminfo on Linux, or return unknowns
    try:
        with open("/proc/meminfo") as f:
            data = {k: int(v.split()[0]) for k, _, *v in
                    (line.partition(":") for line in f) if v}
        total = data.get("MemTotal", 0) / 1e6
        avail = data.get("MemAvailable", data.get("MemFree", 0)) / 1e6
        return total, avail
    except Exception:
        return 0.0, 0.0
