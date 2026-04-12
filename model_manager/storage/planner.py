"""Storage path planner — suggests optimal install paths."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from model_manager.core.exceptions import InsufficientDiskSpaceError
from model_manager.hardware.profile import DriveInfo, HardwareProfile


@dataclass
class SuggestedPath:
    path: Path
    drive: DriveInfo
    available_gb: float
    score: float
    reason: str
    requires_elevation: bool = False


class StoragePlanner:

    def suggest(
        self,
        profile: HardwareProfile,
        required_gb: float,
        model_name: str,
    ) -> list[SuggestedPath]:
        candidates: list[SuggestedPath] = []

        for drive in profile.drives:
            if drive.is_network or drive.is_removable:
                continue
            if drive.free_gb < required_gb * 1.1:   # 10% buffer
                continue

            # Build a sensible default path on this drive
            path = self._default_path(drive, model_name)
            score = self._score(drive, required_gb)
            reason = self._reason(drive, required_gb)

            candidates.append(SuggestedPath(
                path=path,
                drive=drive,
                available_gb=drive.free_gb,
                score=score,
                reason=reason,
            ))

        candidates.sort(key=lambda c: c.score, reverse=True)

        if not candidates:
            # Find the drive with most free space for a better error message
            best = max(profile.drives, key=lambda d: d.free_gb, default=None)
            avail = best.free_gb if best else 0.0
            raise InsufficientDiskSpaceError(
                required_gb=required_gb,
                available_gb=avail,
                path=best.path if best else "?",
            )

        return candidates

    def _default_path(self, drive: DriveInfo, model_name: str) -> Path:
        import sys, os
        safe_name = model_name.replace("/", "--").replace(" ", "_")
        if sys.platform == "win32":
            base = Path(os.environ.get("LOCALAPPDATA", drive.path)) / "model_manager" / "models"
        else:
            base = Path(drive.path) / "models"
        return base / safe_name

    def _score(self, drive: DriveInfo, required_gb: float) -> float:
        headroom = (drive.free_gb - required_gb) / max(drive.free_gb, 1)
        return min(headroom, 1.0) * 100

    def _reason(self, drive: DriveInfo, required_gb: float) -> str:
        headroom = drive.free_gb - required_gb
        return (
            f"{drive.free_gb:.1f} GB free on {drive.path} "
            f"({headroom:.1f} GB remaining after install)"
        )
