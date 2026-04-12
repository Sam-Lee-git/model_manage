"""Model catalog dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class QuantizationOption:
    quant_type: str           # "Q4_K_M", "Q8_0", "fp16", "int4", "awq"
    file_size_gb: float
    min_vram_gb: float        # 0.0 for CPU quants
    quality_score: float      # 0-1 relative to fp16
    repo_url: str
    filename_pattern: str     # glob, e.g. "*.Q4_K_M.gguf"


@dataclass
class ModelEntry:
    model_id: str
    display_name: str
    family: str               # "llama", "mistral", "qwen", "phi", ...
    parameter_count_b: float
    modality: list[str]       # ["text"] | ["text", "vision"]
    capabilities: list[str]   # ["chat", "code", "reasoning"]
    license: str

    # Requirements
    min_ram_gb: float
    min_vram_gb: float
    min_disk_gb: float
    supported_backends: list[str]  # ["cuda", "rocm", "metal", "cpu"]

    # Quantization options
    quantizations: list[QuantizationOption] = field(default_factory=list)

    # Repository sources
    hf_repo_id: Optional[str]       = None
    modelscope_repo_id: Optional[str] = None
    github_url: Optional[str]       = None

    # Metadata
    context_length: int            = 4096
    languages: list[str]           = field(default_factory=lambda: ["en"])
    tags: list[str]                = field(default_factory=list)
    description: str               = ""

    def best_quant_for(self, available_vram_gb: float) -> Optional[QuantizationOption]:
        """Highest quality quantization that fits in available VRAM."""
        fitting = [q for q in self.quantizations if q.min_vram_gb <= available_vram_gb]
        return max(fitting, key=lambda q: q.quality_score) if fitting else None

    @classmethod
    def from_dict(cls, d: dict) -> "ModelEntry":
        quants = [QuantizationOption(**q) for q in d.pop("quantizations", [])]
        return cls(quantizations=quants, **d)
