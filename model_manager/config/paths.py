"""Platform-aware path resolution for config and data directories."""

from __future__ import annotations

import os
from pathlib import Path

from model_manager.core.constants import CURRENT_PLATFORM, Platform, STATE_DIR_NAME


def get_data_dir() -> Path:
    """Return the root data directory (~/.model_manager or platform equivalent)."""
    env_override = os.environ.get("MM_DATA_DIR")
    if env_override:
        return Path(env_override).expanduser().resolve()

    if CURRENT_PLATFORM == Platform.WINDOWS:
        base = Path(os.environ.get("APPDATA", Path.home()))
    elif CURRENT_PLATFORM == Platform.MACOS:
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))

    return base / STATE_DIR_NAME


def get_sessions_dir() -> Path:
    return get_data_dir() / "sessions"


def get_config_file() -> Path:
    return get_data_dir() / "config.toml"


def get_logs_dir() -> Path:
    return get_data_dir() / "logs"


def get_catalog_cache() -> Path:
    return get_data_dir() / "catalog.json"


def ensure_dirs() -> None:
    """Create all required directories if they don't exist."""
    for d in [get_data_dir(), get_sessions_dir(), get_logs_dir()]:
        d.mkdir(parents=True, exist_ok=True)
