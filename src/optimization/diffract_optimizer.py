"""Projected-gradient differentiable resource allocation baseline."""

from __future__ import annotations

import numpy as np


def diffract_optimize(channel_gains: np.ndarray, noise_power: float, power_budget: float,
                      iterations: int = 150, learning_rate: float = 0.05) -> np.ndarray:
    """Optimize aggregate capacity using projected gradient ascent on power."""
    gains = np.maximum(np.asarray(channel_gains, dtype=np.float64), 1e-12)
    allocation = np.full(gains.shape, power_budget / gains.size, dtype=np.float64)
    for _ in range(iterations):
        gradient = gains / (max(noise_power, 1e-12) + allocation * gains)
        allocation = np.maximum(allocation + learning_rate * gradient, 0.0)
        allocation *= power_budget / max(float(allocation.sum()), 1e-12)
    return allocation.astype(np.float32)