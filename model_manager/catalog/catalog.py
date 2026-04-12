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
