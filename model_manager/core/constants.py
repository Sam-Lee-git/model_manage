"""Platform constants and enums."""

import platform
import sys
from enum import Enum, auto


# ── Platform ──────────────────────────────────────────────────────────────────

class Platform(str, Enum):
    WINDOWS = "windows"
    LINUX   = "linux"
    MACOS   = "darwin"

    @classmethod
    def current(cls) -> "Platform":
        s = sys.platform
        if s == "win32":
            return cls.WINDOWS
        if s == "darwin":
            return cls.MACOS
        return cls.LINUX


CURRENT_PLATFORM = Platform.current()


# ── Compute backends ──────────────────────────────────────────────────────────

class ComputeBackend(str, Enum):
    CUDA  = "cuda"
    ROCM  = "rocm"
    METAL = "metal"
    CPU   = "cpu"


# ── Installation backends ─────────────────────────────────────────────────────

class InstallBackendType(str, Enum):
    PIP_VENV = "pip_venv"
    CONDA    = "conda"
    DOCKER   = "docker"


# ── Installation step status ──────────────────────────────────────────────────

class StepStatus(str, Enum):
    PENDING       = "pending"
    RUNNING       = "running"
    COMPLETED     = "completed"
    FAILED        = "failed"
    SKIPPED       = "skipped"
    PENDING_RETRY = "pending_retry"


# ── Session / state machine states ───────────────────────────────────────────

class SessionState(str, Enum):
    IDLE                    = "IDLE"
    DETECTING_HARDWARE      = "DETECTING_HARDWARE"
    BROWSING_CATALOG        = "BROWSING_CATALOG"
    MODEL_SELECTED          = "MODEL_SELECTED"
    ANALYZING_STORAGE       = "ANALYZING_STORAGE"
    CONFIRMING_PLAN         = "CONFIRMING_PLAN"
    SELECTING_BACKEND       = "SELECTING_BACKEND"
    RESOLVING_REPOS         = "RESOLVING_REPOS"
    INSTALLING_DEPENDENCIES = "INSTALLING_DEPENDENCIES"
    DOWNLOADING_MODEL       = "DOWNLOADING_MODEL"
    VERIFYING_INSTALL       = "VERIFYING_INSTALL"
    COMPLETED               = "COMPLETED"
    PAUSED                  = "PAUSED"
    ERROR_RECOVERY          = "ERROR_RECOVERY"
    FAILED                  = "FAILED"
    ABORTED                 = "ABORTED"


# ── Error categories ──────────────────────────────────────────────────────────

class ErrorCategory(str, Enum):
    NETWORK             = "network"
    DEPENDENCY_CONFLICT = "dependency_conflict"
    CUDA_MISMATCH       = "cuda_mismatch"
    PERMISSION          = "permission"
    DISK_SPACE          = "disk_space"
    UNKNOWN             = "unknown"


# ── Privilege levels ──────────────────────────────────────────────────────────

class PrivilegeLevel(str, Enum):
    STANDARD = "standard"
    ELEVATED = "elevated"
    ROOT     = "root"


# ── Misc ──────────────────────────────────────────────────────────────────────

MAX_BRANCH_DEPTH = 2
STATE_DIR_NAME   = ".model_manager"
SESSIONS_DIR     = "sessions"
CONFIG_FILE      = "config.toml"
CATALOG_FILE     = "catalog.json"
APP_NAME         = "mm"
