"""Hardware-aware model recommender."""

from __future__ import annotations

from dataclasses import dataclass

from model_manager.catalog.catalog import ModelCatalog
from model_manager.catalog.models import ModelEntry, QuantizationOption
from model_manager.hardware.profile import HardwareProfile


@dataclass
class RankedModel:
    model: ModelEntry
    best_quant: QuantizationOption | None
    score: float
    reason: str


class ModelRecommender:
    def __init__(self, catalog: ModelCatalog) -> None:
        self._catalog = catalog

    def recommend(self, profile: HardwareProfile, top_n: int = 5) -> list[RankedModel]:
        backend = profile.primary_compute_backend.value
        candidates = self._catalog.filter(
            max_vram_gb=profile.total_vram_gb,
            backends=[backend, "cpu"],
        )

        ranked: list[RankedModel] = []
        for model in candidates:
            quant = model.best_quant_for(profile.total_vram_gb)
            if quant is None and profile.total_vram_gb > 0:
                continue  # no fitting quantization

            score = self._score(model, quant, profile)
            reason = self._reason(model, quant, profile)
            ranked.append(RankedModel(model=model, best_quant=quant, score=score, reason=reason))

        ranked.sort(key=lambda r: r.score, reverse=True)
        return ranked[:top_n]

    def _score(
        self,
        model: ModelEntry,
        quant: QuantizationOption | None,
        profile: HardwareProfile,
    ) -> float:
        score = 0.0
        # Larger models score higher (capability proxy)
        score += min(model.parameter_count_b / 70.0, 1.0) * 40

        # Better quantization quality scores higher
        if quant:
            score += quant.quality_score * 30

        # VRAM headroom bonus (not squeezing too tight)
        if quant and profile.total_vram_gb > 0:
            headroom = (profile.total_vram_gb - quant.min_vram_gb) / profile.total_vram_gb
            score += min(headroom, 0.5) * 20

        # RAM adequacy
        if profile.ram_total_gb >= model.min_ram_gb * 1.5:
            score += 10

        return score

    def _reason(
        self,
        model: ModelEntry,
        quant: QuantizationOption | None,
        profile: HardwareProfile,
    ) -> str:
        parts = [f"{model.parameter_count_b:.0f}B parameter model"]
        if quant:
            parts.append(f"{quant.quant_type} quantization ({quant.file_size_gb:.1f} GB)")
            vram_used = f"{quant.min_vram_gb:.1f}/{profile.total_vram_gb:.1f} GB VRAM"
            parts.append(vram_used)
        else:
            parts.append("CPU inference")
        return ", ".join(parts)
