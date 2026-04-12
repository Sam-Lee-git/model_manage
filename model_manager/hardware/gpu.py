"""GPU detection across CUDA, ROCm, Metal, and CPU-only environments.

Detection order:
  1. nvidia-smi (NVIDIA/CUDA)
     - If missing but NVIDIA hardware found → attempt driver install → retry
     - If runs but errors → diagnose and surface actionable message
  2. rocm-smi (AMD/ROCm)
  3. system_profiler (Apple Metal, macOS only)
  4. CPU-only sentinel fallback
"""

from __future__ import annotations

import asyncio
import json
import shutil
import subprocess
import sys
from typing import Optional

from model_manager.core.constants import ComputeBackend
from model_manager.hardware.profile import GPUInfo


# ── Public API ────────────────────────────────────────────────────────────────

async def detect_gpus() -> list[GPUInfo]:
    return await asyncio.to_thread(_detect_gpus_sync)


# ── Top-level sync dispatcher ─────────────────────────────────────────────────

def _detect_gpus_sync() -> list[GPUInfo]:
    nvidia = _try_nvidia_with_recovery()
    if nvidia is not None:
        return nvidia

    amd = _try_rocm()
    if amd:
        return amd

    if sys.platform == "darwin":
        metal = _try_metal()
        if metal:
            return metal

    return [GPUInfo(name="CPU only", vram_gb=0.0, compute_backend=ComputeBackend.CPU)]


# ── NVIDIA detection + recovery ───────────────────────────────────────────────

def _try_nvidia_with_recovery() -> Optional[list[GPUInfo]]:
    """
    Returns:
      list[GPUInfo]  — NVIDIA GPUs found and working
      []             — NVIDIA hardware present but driver broken (already warned user)
      None           — No NVIDIA hardware at all; caller should try next backend
    """
    smi = shutil.which("nvidia-smi")

    if smi:
        gpus, error = _run_nvidia_smi_query(smi)
        if gpus:
            return gpus
        # nvidia-smi exists but returned an error
        _warn_driver_broken(error)
        return []

    # nvidia-smi not found — check whether NVIDIA hardware is actually present
    if not _nvidia_hardware_present():
        return None   # genuinely no NVIDIA GPU

    # Hardware found but driver / nvidia-smi missing
    _print(
        "[warning]NVIDIA GPU detected, but nvidia-smi not found "
        "(driver not installed or not in PATH).[/warning]"
    )

    if sys.platform == "linux":
        if _try_install_driver_linux():
            smi = shutil.which("nvidia-smi")
            if smi:
                gpus, _ = _run_nvidia_smi_query(smi)
                if gpus:
                    _print("[success]Driver installed successfully — GPU is ready.[/success]")
                    return gpus
        _print_linux_manual_instructions()
    elif sys.platform == "win32":
        _print_windows_manual_instructions()
    elif sys.platform == "darwin":
        _print(
            "[muted]NVIDIA GPUs are not supported on macOS. "
            "Metal (Apple GPU) will be used instead.[/muted]"
        )

    return []   # hardware present but unusable for now


def _run_nvidia_smi_query(smi_path: str) -> tuple[list[GPUInfo], str]:
    """Run nvidia-smi and parse GPU list. Returns (gpus, error_message)."""
    try:
        result = subprocess.run(
            [
                smi_path,
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True, text=True, timeout=15,
        )
    except FileNotFoundError:
        return [], "nvidia-smi not found"
    except subprocess.TimeoutExpired:
        return [], "nvidia-smi timed out (driver may be hung)"

    if result.returncode != 0:
        stderr = (result.stderr or result.stdout or "").strip()
        return [], _diagnose_nvidia_smi_error(result.returncode, stderr)

    cuda_ver = _get_cuda_version(smi_path)
    gpus: list[GPUInfo] = []
    for i, line in enumerate(result.stdout.strip().splitlines()):
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        name, vram_mib, driver = parts[0], parts[1], parts[2]
        try:
            vram_gb = float(vram_mib) / 1024
        except ValueError:
            vram_gb = 0.0
        gpus.append(GPUInfo(
            name=name,
            vram_gb=vram_gb,
            compute_backend=ComputeBackend.CUDA,
            cuda_version=cuda_ver,
            driver_version=driver,
            device_index=i,
        ))
    return gpus, ""


def _diagnose_nvidia_smi_error(returncode: int, stderr: str) -> str:
    """Map known nvidia-smi error patterns to human-readable messages."""
    low = stderr.lower()
    if "driver/library version mismatch" in low or "version mismatch" in low:
        return (
            "Driver/library version mismatch — the NVIDIA kernel driver and "
            "userspace library are out of sync. Try rebooting, or reinstall the driver."
        )
    if "no devices were found" in low or "no nvidia" in low:
        return "nvidia-smi found but reports no NVIDIA devices (GPU may be disabled in BIOS)."
    if "couldn't open" in low or "unable to determine" in low:
        return (
            "nvidia-smi cannot access the GPU. "
            "Try: sudo rmmod nouveau && sudo modprobe nvidia"
        )
    if "permission" in low:
        return (
            "Permission denied when accessing GPU device. "
            "Add your user to the 'video' group: sudo usermod -aG video $USER"
        )
    return f"nvidia-smi exited with code {returncode}: {stderr[:200]}"


def _warn_driver_broken(error: str) -> None:
    _print(f"[warning]NVIDIA GPU driver issue:[/warning] {error}")
    _print("[muted]Falling back to CPU mode. Fix the driver and re-run mm.[/muted]")


# ── Check NVIDIA hardware without nvidia-smi ──────────────────────────────────

def _nvidia_hardware_present() -> bool:
    """Return True if an NVIDIA PCI device is found, even without a driver."""
    if sys.platform == "linux":
        return _nvidia_present_linux()
    if sys.platform == "win32":
        return _nvidia_present_windows()
    return False


def _nvidia_present_linux() -> bool:
    try:
        result = subprocess.run(
            ["lspci"], capture_output=True, text=True, timeout=5
        )
        return "nvidia" in result.stdout.lower() or "geforce" in result.stdout.lower()
    except Exception:
        pass
    # Fallback: scan /sys PCI class 0x0300 (VGA) / 0x0302 (3D)
    try:
        import os, glob
        for vendor_path in glob.glob("/sys/bus/pci/devices/*/vendor"):
            with open(vendor_path) as f:
                if f.read().strip().lower() == "0x10de":  # NVIDIA vendor ID
                    return True
    except Exception:
        pass
    return False


def _nvidia_present_windows() -> bool:
    try:
        result = subprocess.run(
            ["wmic", "path", "win32_VideoController", "get", "name"],
            capture_output=True, text=True, timeout=10,
        )
        output = result.stdout.lower()
        return "nvidia" in output or "geforce" in output or "quadro" in output or "tesla" in output
    except Exception:
        pass
    return False


# ── Linux driver auto-install ─────────────────────────────────────────────────

def _try_install_driver_linux() -> bool:
    """Attempt automatic NVIDIA driver installation on Ubuntu/Debian Linux.
    Returns True if installation succeeded."""
    _print("[info]Attempting automatic NVIDIA driver installation...[/info]")

    # Try ubuntu-drivers first (Ubuntu / Linux Mint)
    if shutil.which("ubuntu-drivers"):
        _print("  Running: ubuntu-drivers autoinstall")
        try:
            result = subprocess.run(
                ["sudo", "-n", "ubuntu-drivers", "autoinstall"],
                capture_output=True, text=True, timeout=300,
            )
            if result.returncode == 0:
                _print("[success]  ubuntu-drivers autoinstall succeeded.[/success]")
                _print("[muted]  A reboot may be required for the driver to activate.[/muted]")
                return True
            _print(f"[muted]  ubuntu-drivers failed: {result.stderr.strip()[:120]}[/muted]")
        except subprocess.TimeoutExpired:
            _print("[warning]  ubuntu-drivers timed out.[/warning]")

    # Try apt-get with the recommended driver (535 is current LTS as of 2025)
    for driver_pkg in ("nvidia-driver-560", "nvidia-driver-535", "nvidia-driver-525"):
        _print(f"  Trying: apt-get install -y {driver_pkg}")
        try:
            result = subprocess.run(
                ["sudo", "-n", "apt-get", "install", "-y", driver_pkg],
                capture_output=True, text=True, timeout=300,
            )
            if result.returncode == 0:
                _print(f"[success]  {driver_pkg} installed.[/success]")
                _print("[muted]  Reboot required: sudo reboot[/muted]")
                return True
            _print(f"[muted]  {driver_pkg} not available, trying next...[/muted]")
        except subprocess.TimeoutExpired:
            break

    return False


# ── Manual instruction helpers ────────────────────────────────────────────────

def _print_linux_manual_instructions() -> None:
    _print("\n[warning]Could not auto-install NVIDIA driver. Please install manually:[/warning]")
    _print("  [info]Ubuntu/Debian:[/info]")
    _print("    sudo apt update")
    _print("    sudo ubuntu-drivers autoinstall   # recommended")
    _print("    # or: sudo apt install nvidia-driver-535")
    _print("    sudo reboot")
    _print("  [info]RHEL/Fedora:[/info]")
    _print("    sudo dnf install akmod-nvidia")
    _print("    sudo reboot")
    _print("  [info]Arch:[/info]")
    _print("    sudo pacman -S nvidia nvidia-utils")
    _print("    sudo reboot")
    _print("  After rebooting, re-run: mm\n")


def _print_windows_manual_instructions() -> None:
    _print("\n[warning]NVIDIA GPU detected but driver not installed.[/warning]")
    _print("  Download and install the driver from:")
    _print("    https://www.nvidia.com/Download/index.aspx")
    _print("  Or use GeForce Experience for automatic driver updates.")
    _print("  After installing, re-run: mm\n")


# ── CUDA version helper ───────────────────────────────────────────────────────

def _get_cuda_version(smi_path: str) -> Optional[str]:
    try:
        result = subprocess.run(
            [smi_path], capture_output=True, text=True, timeout=5
        )
        for line in result.stdout.splitlines():
            if "CUDA Version:" in line:
                return line.split("CUDA Version:")[1].strip().split()[0]
    except Exception:
        pass
    return None


# ── AMD ROCm detection ────────────────────────────────────────────────────────

def _try_rocm() -> list[GPUInfo]:
    if not shutil.which("rocm-smi"):
        # Check if AMD GPU hardware is present but rocm not installed
        if _amd_hardware_present():
            _print("[warning]AMD GPU detected but rocm-smi not found.[/warning]")
            _print("  Install ROCm: https://rocm.docs.amd.com/en/latest/deploy/linux/index.html")
        return []
    try:
        result = subprocess.run(
            ["rocm-smi", "--showproductname", "--showmeminfo", "vram", "--json"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            _print(f"[warning]rocm-smi error: {result.stderr.strip()[:120]}[/warning]")
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


def _amd_hardware_present() -> bool:
    if sys.platform == "linux":
        try:
            result = subprocess.run(["lspci"], capture_output=True, text=True, timeout=5)
            out = result.stdout.lower()
            return "amd" in out or "radeon" in out or "advanced micro devices" in out
        except Exception:
            pass
    if sys.platform == "win32":
        try:
            result = subprocess.run(
                ["wmic", "path", "win32_VideoController", "get", "name"],
                capture_output=True, text=True, timeout=10,
            )
            out = result.stdout.lower()
            return "amd" in out or "radeon" in out
        except Exception:
            pass
    return False


# ── Apple Metal detection ─────────────────────────────────────────────────────

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
    parts = s.split()
    if not parts:
        return 0.0
    try:
        val = float(parts[0])
        unit = parts[1].upper() if len(parts) > 1 else "MB"
        return val if "GB" in unit else val / 1024
    except (ValueError, IndexError):
        return 0.0


# ── Internal print helper (avoids circular import with event bus) ─────────────

def _print(msg: str) -> None:
    try:
        from model_manager.ui.console import console
        console.print(msg)
    except Exception:
        import re
        print(re.sub(r'\[/?[^\]]+\]', '', msg))
