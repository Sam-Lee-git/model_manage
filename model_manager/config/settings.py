"""Application settings — works with or without pydantic-settings."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from model_manager.config.paths import get_data_dir


class Settings:
    """
    Reads configuration from environment variables and .env file.
    Falls back to sensible defaults when pydantic-settings is unavailable.
    """

    def __init__(self) -> None:
        self._load_dotenv()

    def _load_dotenv(self) -> None:
        """Load .env file if python-dotenv is available."""
        try:
            from dotenv import load_dotenv
            load_dotenv(override=False)   # don't overwrite already-set env vars
        except ImportError:
            pass

    def _get(self, key: str, default: str = "") -> str:
        return os.environ.get(key, default).strip()

    def _get_optional(self, key: str) -> Optional[str]:
        v = os.environ.get(key, "").strip()
        return v if v else None

    # ── Active LLM provider ───────────────────────────────────────────────────
    @property
    def llm_provider(self) -> str:
        return self._get("MM_LLM_PROVIDER", "auto")

    # ── API keys ──────────────────────────────────────────────────────────────
    @property
    def anthropic_api_key(self) -> Optional[str]:
        return self._get_optional("ANTHROPIC_API_KEY")

    @property
    def openai_api_key(self) -> Optional[str]:
        return self._get_optional("OPENAI_API_KEY")

    @property
    def gemini_api_key(self) -> Optional[str]:
        return self._get_optional("GEMINI_API_KEY")

    @property
    def qwen_api_key(self) -> Optional[str]:
        return self._get_optional("QWEN_API_KEY")

    @property
    def minimax_api_key(self) -> Optional[str]:
        return self._get_optional("MINIMAX_API_KEY")

    @property
    def deepseek_api_key(self) -> Optional[str]:
        return self._get_optional("DEEPSEEK_API_KEY")

    # ── Model overrides ───────────────────────────────────────────────────────
    @property
    def claude_model(self) -> str:
        return self._get("MM_CLAUDE_MODEL", "claude-sonnet-4-6")

    @property
    def openai_model(self) -> str:
        return self._get("MM_OPENAI_MODEL", "gpt-4o")

    @property
    def gemini_model(self) -> str:
        return self._get("MM_GEMINI_MODEL", "gemini-2.0-flash")

    @property
    def qwen_model(self) -> str:
        return self._get("MM_QWEN_MODEL", "qwen-max")

    @property
    def minimax_model(self) -> str:
        return self._get("MM_MINIMAX_MODEL", "abab6.5s-chat")

    @property
    def deepseek_model(self) -> str:
        return self._get("MM_DEEPSEEK_MODEL", "deepseek-chat")

    # ── Base URL overrides ────────────────────────────────────────────────────
    @property
    def openai_base_url(self) -> Optional[str]:
        return self._get_optional("MM_OPENAI_BASE_URL")

    @property
    def gemini_base_url(self) -> Optional[str]:
        return self._get_optional("MM_GEMINI_BASE_URL")

    @property
    def qwen_base_url(self) -> Optional[str]:
        return self._get_optional("MM_QWEN_BASE_URL")

    @property
    def minimax_base_url(self) -> Optional[str]:
        return self._get_optional("MM_MINIMAX_BASE_URL")

    @property
    def deepseek_base_url(self) -> Optional[str]:
        return self._get_optional("MM_DEEPSEEK_BASE_URL")

    # ── Paths ─────────────────────────────────────────────────────────────────
    @property
    def data_dir(self) -> Path:
        v = self._get_optional("MM_DATA_DIR")
        return Path(v).expanduser() if v else get_data_dir()

    # ── Other ─────────────────────────────────────────────────────────────────
    @property
    def log_level(self) -> str:
        return self._get("MM_LOG_LEVEL", "INFO")

    @property
    def catalog_remote_url(self) -> str:
        return self._get(
            "MM_CATALOG_REMOTE_URL",
            "https://raw.githubusercontent.com/your-org/model-catalog/main/catalog.json",
        )

    @property
    def catalog_update_on_start(self) -> bool:
        return self._get("MM_CATALOG_UPDATE_ON_START", "true").lower() == "true"

    @property
    def download_chunk_size(self) -> int:
        return int(self._get("MM_DOWNLOAD_CHUNK_SIZE", str(1024 * 1024)))

    @property
    def download_max_retries(self) -> int:
        return int(self._get("MM_DOWNLOAD_MAX_RETRIES", "5"))

    @property
    def max_concurrent_downloads(self) -> int:
        return int(self._get("MM_MAX_CONCURRENT_DOWNLOADS", "3"))

    @property
    def max_branch_depth(self) -> int:
        return int(self._get("MM_MAX_BRANCH_DEPTH", "2"))

    @property
    def max_recovery_attempts(self) -> int:
        return int(self._get("MM_MAX_RECOVERY_ATTEMPTS", "3"))


_settings: Optional[Settings] = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reset_settings() -> None:
    global _settings
    _settings = None
