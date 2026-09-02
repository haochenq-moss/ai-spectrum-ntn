"""Offline diagnosis of network state severity and allocation strategy."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import numpy as np

from .memory_rag import MemoryRAG


@dataclass(frozen=True)
class Diagnosis:
    network_condition: str
    severity: str
    recommended_strategy: str
    confidence: float
    reasoning: str

    def to_dict(self) -> dict:
        return asdict(self)


class ReasoningEngine:
    """Choose a reproducible optimization strategy from a 256-dimensional state."""

    def __init__(self, memory: MemoryRAG) -> None:
        self.memory = memory

    def reason(self, state_vector: np.ndarray) -> Diagnosis:
        if np.asarray(state_vector).shape != (256,):
            raise ValueError("Agent reasoning requires a 256-dimensional state vector")
        interference_level = float(np.mean(np.abs(state_vector[224:232])))
        jammer_level = float(np.mean(np.abs(state_vector[232:248])))
        if jammer_level > 0.25:
            condition, severity, strategy = "adversarial_jamming_detected", "high", "adversarial_wf"
        elif interference_level > 0.75:
            condition, severity, strategy = "high_interference", "medium", "diffract"
        else:
            condition, severity, strategy = "nominal", "low", "classical_wf"
        context = self.memory.retrieve(f"{condition} {strategy}", top_k=1)
        explanation = context[0].text if context else "Use the selected allocation strategy."
        confidence = float(min(0.95, 0.55 + max(interference_level, jammer_level) * 0.2))
        return Diagnosis(condition, severity, strategy, confidence, explanation)