"""Translate allocation strategy into concrete spectrum-management commands."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import numpy as np

from src.optimization.adversarial_water_filling import adversarial_water_filling
from src.optimization.classical_water_filling import water_filling
from src.optimization.diffract_optimizer import diffract_optimize


@dataclass(frozen=True)
class NetworkActions:
    strategy: str
    power_allocation: np.ndarray
    spectrum_assignment: np.ndarray
    beam_mode: str

    def summary(self) -> dict:
        return {**asdict(self), "power_allocation": self.power_allocation.tolist(), "spectrum_assignment": self.spectrum_assignment.tolist()}


def _gains_from_state(state_vector: np.ndarray) -> np.ndarray:
    values = np.log1p(np.clip(np.abs(state_vector[64:192]), 0.0, 1e6))
    values = (values - values.min()) / max(float(np.ptp(values)), 1e-12)
    return (0.05 + 0.95 * np.resize(values, 512)).reshape(16, 32)


class ActionGenerator:
    """Generate safe, normalized resource commands from a chosen strategy."""

    def generate(self, state_vector: np.ndarray, strategy: str) -> NetworkActions:
        gains = _gains_from_state(state_vector)
        if strategy == "adversarial_wf":
            allocation, _ = adversarial_water_filling(gains, 1e-3, 1.0, 1.0, max_iterations=20)
            beam_mode = "adaptive_robust"
        elif strategy == "diffract":
            allocation, beam_mode = diffract_optimize(gains, 1e-3, 1.0, iterations=50), "adaptive_fast"
        else:
            allocation, beam_mode = water_filling(gains, 1e-3, 1.0), "conservative"
        assignment = allocation.argmax(axis=0).astype(np.int32)
        return NetworkActions(strategy, allocation, assignment, beam_mode)