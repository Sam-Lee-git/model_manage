"""Disk/drive detection."""

from __future__ import annotations

import asyncio
import sys

try:
    import psutil as _psutil
except ImportError:
    _psutil = None  # type: ignore

from model_manager.hardware.profile import DriveInfo


async def detect_drives() -> list[DriveInfo]:
    return await asyncio.to_thread(_detect_drives_sync)


def _detect_drives_sync() -> list[DriveInfo]:
    if not _psutil:
        return _detect_drives_fallback()
    drives: list[DriveInfo] = []
    for part in _psutil.disk_partitions(all=False):
        try:
            usage = _psutil.disk_usage(part.mountpoint)
        except (PermissionError, OSError):
            continue

        opts = part.opts.lower()
        is_removable = "removable" in opts or _is_removable(part.device)
        is_network = any(t in part.fstype.lower() for t in ("nfs", "cifs", "smb", "smbfs", "davfs"))

        drives.append(DriveInfo(
            path=part.mountpoint,
            total_gb=usage.total / 1e9,
            free_gb=usage.free / 1e9,
            filesystem=part.fstype,
            is_removable=is_removable,
            is_network=is_network,
        ))
    return drives


def _detect_drives_fallback() -> list[DriveInfo]:
    """Best-effort drive detection without psutil."""
    import shutil, sys
    drives = []
    if sys.platform == "win32":
        import string
        for letter in string.ascii_uppercase:
            path = f"{letter}:\\"
            try:
                total, used, free = shutil.disk_usage(path)
                drives.append(DriveInfo(
                    path=path, total_gb=total/1e9, free_gb=free/1e9, filesystem="ntfs"
                ))
            except OSError:
                continue
    else:
        try:
            total, used, free = shutil.disk_usage("/")
            drives.append(DriveInfo(
                path="/", total_gb=total/1e9, free_gb=free/1e9, filesystem="unknown"
            ))
        except OSError:
            pass
    return drives


def _is_removable(device: str) -> bool:
    """Best-effort check whether a device is removable."""
    if sys.platform == "linux":
        try:
            dev_name = device.split("/")[-1].rstrip("0123456789")
            with open(f"/sys/block/{dev_name}/removable") as f:
                return f.read().strip() == "1"
        except Exception:
            pass
    return False
