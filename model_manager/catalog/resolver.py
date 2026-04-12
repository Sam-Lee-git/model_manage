"""Repo URL resolver with mirror support."""

from __future__ import annotations

import os

from model_manager.catalog.models import ModelEntry, QuantizationOption


# Mirror substitution map: original_host -> mirror_host
MIRRORS = {
    "huggingface.co": [
        "hf-mirror.com",        # CN mainland mirror
    ],
}


def _prefer_mirror() -> bool:
    """Heuristic: use mirror if MODELSCOPE_CACHE or HF_ENDPOINT env var set, or if in CN."""
    if os.environ.get("HF_ENDPOINT") or os.environ.get("HF_MIRROR"):
        return True
    return False


def resolve_repo_url(model: ModelEntry, quant: QuantizationOption) -> str:
    """Return the best download URL for the given model + quantization."""
    base_url = quant.repo_url

    if _prefer_mirror():
        for original, mirrors in MIRRORS.items():
            if original in base_url:
                base_url = base_url.replace(original, mirrors[0], 1)
                break

    return base_url


def resolve_hf_url(repo_id: str, filename: str, use_mirror: bool = False) -> str:
    host = "hf-mirror.com" if (use_mirror or _prefer_mirror()) else "huggingface.co"
    return f"https://{host}/{repo_id}/resolve/main/{filename}"
