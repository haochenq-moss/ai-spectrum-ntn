"""Alternating minimax water-filling against a power-bounded jammer."""

from __future__ import annotations

from typing import Tuple

import numpy as np

from .classical_water_filling import water_filling


def adversarial_water_filling(channel_gains: np.ndarray, noise_power: float,
                              power_budget: float, jammer_budget: float,
                              max_iterations: int = 100,
                              tolerance: float = 1e-5) -> Tuple[np.ndarray, np.ndarray]:
    """Return allocator and jammer strategies for an alternating minimax game."""
    gains = np.maximum(np.asarray(channel_gains, dtype=np.float64), 1e-12)
    jammer = np.full(gains.shape[1], jammer_budget / gains.shape[1], dtype=np.float64)
    allocation = water_filling(gains, noise_power, power_budget, jammer)
    for _ in range(max_iterations):
        previous_jammer = jammer.copy()
        allocation = water_filling(gains, noise_power, power_budget, jammer)
        received_power = (allocation * gains).sum(axis=0)
        jammer = jammer_budget * received_power / max(float(received_power.sum()), 1e-12)
        jammer = 0.5 * jammer + 0.5 * previous_jammer
        if np.max(np.abs(jammer - previous_jammer)) < tolerance:
            break
    return allocation.astype(np.float32), jammer.astype(np.float32)