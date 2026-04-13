"""Hardware profile dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from model_manager.core.constants import ComputeBackend


@dataclass
class CPUInfo:
    brand: str
    physical_cores: int
    logical_cores: int
    architecture: str          # "x86_64" | "arm64"
    supports_avx2: bool = False
    supports_avx512: bool = False
    base_freq_mhz: float = 0.0


@dataclass
class GPUInfo:
    name: str
    vram_gb: float
    compute_backend: ComputeBackend
    cuda_version: Optional[str] = None
    driver_version: Optional[str] = None
    device_index: int = 0


@dataclass
class DriveInfo:
    path: str                  # mount point or drive letter root
    total_gb: float
    free_gb: float
    filesystem: str            # "ntfs" | "ext4" | "apfs" | ...
    is_removable: bool = False
    is_network: bool = False


@dataclass
class HardwareProfile:
    cpu: CPUInfo
    ram_total_gb: float
    ram_available_gb: float
    gpus: list[GPUInfo]
    drives: list[DriveInfo]
    os_platform: str           # "windows" | "linux" | "darwin"
    os_version: str
    detected_at: datetime = field(default_factory=datetime.utcnow)

    # ── Convenience properties ────────────────────────────────────────────────

    @property
    def primary_compute_backend(self) -> ComputeBackend:
        for backend in (ComputeBackend.CUDA, ComputeBackend.ROCM, ComputeBackend.METAL):
            if any(g.compute_backend == backend for g in self.gpus):
                return backend
        return ComputeBackend.CPU

    @property
    def total_vram_gb(self) -> float:
        return sum(g.vram_gb for g in self.gpus)

    @property
    def has_nvidia_gpu(self) -> bool:
        return any(g.compute_backend == ComputeBackend.CUDA for g in self.gpus)

    @property
    def has_amd_gpu(self) -> bool:
        return any(g.compute_backend == ComputeBackend.ROCM for g in self.gpus)

    @property
    def best_drive(self) -> Optional[DriveInfo]:
        """Drive with most free space, excluding network/removable."""
        candidates = [d for d in self.drives if not d.is_network and not d.is_removable]
        return max(candidates, key=lambda d: d.free_gb) if candidates else None

    def to_dict(self) -> dict:
        import dataclasses
        return dataclasses.asdict(self)
