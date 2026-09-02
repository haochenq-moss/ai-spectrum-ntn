"""Classical water-filling for non-negative spectrum allocation."""

from __future__ import annotations

import numpy as np


def water_filling(channel_gains: np.ndarray, noise_power: float, power_budget: float,
                  interference: np.ndarray | None = None) -> np.ndarray:
    """Maximize aggregate Shannon capacity with a total power constraint."""
    gains = np.maximum(np.asarray(channel_gains, dtype=np.float64), 1e-12)
    effective_noise = np.full_like(gains, max(noise_power, 1e-12))
    if interference is not None:
        effective_noise += np.broadcast_to(np.asarray(interference, dtype=np.float64), gains.shape)
    inverse_sinr = effective_noise / gains
    sorted_inverse = np.sort(inverse_sinr.ravel())
    cumulative = np.cumsum(sorted_inverse)
    candidate_levels = (power_budget + cumulative) / np.arange(1, sorted_inverse.size + 1)
    active = np.nonzero(candidate_levels > sorted_inverse)[0]
    water_level = candidate_levels[active[-1]] if active.size else power_budget + sorted_inverse[0]
    allocation = np.maximum(water_level - inverse_sinr, 0.0)
    allocation *= power_budget / max(float(allocation.sum()), 1e-12)
    return allocation.astype(np.float32)


def spectral_efficiency(channel_gains: np.ndarray, allocation: np.ndarray,
                        noise_power: float, interference: np.ndarray | None = None) -> float:
    """Return total Shannon spectral efficiency in bits/s/Hz."""
    denominator = max(noise_power, 1e-12)
    if interference is not None:
        denominator = denominator + np.broadcast_to(np.asarray(interference), allocation.shape)
    sinr = np.asarray(allocation) * np.asarray(channel_gains) / denominator
    return float(np.log2(1.0 + np.maximum(sinr, 0.0)).sum())