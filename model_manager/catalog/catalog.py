"""ModelCatalog — load, cache, and query the model catalog."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from model_manager.catalog.models import ModelEntry
from model_manager.config.paths import get_catalog_cache
from model_manager.core.exceptions import CatalogLoadError, ModelNotFoundError

# Bundled minimal catalog (shipped with the package)
_BUNDLED_CATALOG = Path(__file__).parent / "data" / "catalog.json"


class ModelCatalog:
    def __init__(self) -> None:
        self._models: dict[str, ModelEntry] = {}

    # ── Loading ───────────────────────────────────────────────────────────────

    def load(self) -> None:
        """Load catalog from cache or bundled fallback."""
        cache = get_catalog_cache()
        path  = cache if cache.exists() else _BUNDLED_CATALOG
        if not path.exists():
            raise CatalogLoadError(f"No catalog found at {path}")
        self._load_file(path)

    def _load_file(self, path: Path) -> None:
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            raise CatalogLoadError(f"Failed to load catalog: {e}") from e

        self._models = {}
        for item in data.get("models", []):
            try:
                entry = ModelEntry.from_dict(item.copy())
                self._models[entry.model_id] = entry
            except Exception:
                continue   # skip malformed entries

    async def validate_repos(
        self,
        token: Optional[str] = None,
        timeout: float = 5.0,
    ) -> dict[str, bool]:
        """
        Concurrently HEAD-check each model's primary GGUF repo_url.
        Returns {model_id: True/False}.
        Removes models with unreachable repos (404) from self._models in-place.
        Network errors (timeout/no connectivity) keep the model to avoid false removals.
        """
        import asyncio
        import httpx

        results: dict[str, bool] = {}

        async def _check_one(client: httpx.AsyncClient, model: ModelEntry) -> tuple[str, bool]:
            repo_url = next(
                (q.repo_url for q in model.quantizations if q.repo_url),
                None,
            )
            if not repo_url:
                return model.model_id, True  # no URL to check — keep

            repo_id = repo_url.split("huggingface.co/")[-1].strip("/")
            api_url = f"https://huggingface.co/api/models/{repo_id}"
            headers = {"Authorization": f"Bearer {token}"} if token else {}

            try:
                r = await client.head(api_url, headers=headers)
                return model.model_id, r.status_code < 400
            except Exception:
                return model.model_id, True  # network error — keep model

        async with httpx.AsyncClient(timeout=timeout) as client:
            tasks = [_check_one(client, m) for m in list(self._models.values())]
            pairs = await asyncio.gather(*tasks)

        to_remove: list[str] = []
        for model_id, ok in pairs:
            results[model_id] = ok
            if not ok:
                to_remove.append(model_id)

        for mid in to_remove:
            self._models.pop(mid, None)

        return results

    async def update_from_remote(self, url: str) -> bool:
        """Download updated catalog and save to cache. Returns True on success."""
        import httpx
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.get(url)
                resp.raise_for_status()
                data = resp.json()
            cache = get_catalog_cache()
            cache.parent.mkdir(parents=True, exist_ok=True)
            with open(cache, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            self._load_file(cache)
            return True
        except Exception:
            return False

    def add_entry(self, entry: "ModelEntry") -> None:
        """Add or replace a model entry (used for dynamically validated LLM recommendations)."""
        self._models[entry.model_id] = entry

    # ── Query ─────────────────────────────────────────────────────────────────

    def get_by_id(self, model_id: str) -> ModelEntry:
        entry = self._models.get(model_id)
        if entry is None:
            raise ModelNotFoundError(f"Model '{model_id}' not found in catalog")
        return entry

    def all(self) -> list[ModelEntry]:
        return list(self._models.values())

    def filter(
        self,
        max_vram_gb: Optional[float] = None,
        min_ram_gb: Optional[float] = None,
        backends: Optional[list[str]] = None,
        capabilities: Optional[list[str]] = None,
        search: Optional[str] = None,
    ) -> list[ModelEntry]:
        results = list(self._models.values())

        if max_vram_gb is not None:
            results = [m for m in results if m.min_vram_gb <= max_vram_gb]
        if min_ram_gb is not None:
            results = [m for m in results if m.min_ram_gb <= min_ram_gb]
        if backends:
            results = [
                m for m in results
                if any(b in m.supported_backends for b in backends)
            ]
        if capabilities:
            results = [
                m for m in results
                if all(c in m.capabilities for c in capabilities)
            ]
        if search:
            q = search.lower()
            results = [
                m for m in results
                if q in m.display_name.lower()
                or q in m.description.lower()
                or q in m.family.lower()
                or any(q in tag for tag in m.tags)
            ]
        return results
