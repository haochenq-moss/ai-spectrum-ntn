"""Validated CPU implementations of the Step 6 wireless tool contract."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from src.agent.action_generator import _gains_from_state
from src.optimization.adversarial_water_filling import adversarial_water_filling
from src.optimization.classical_water_filling import spectral_efficiency, water_filling
from src.optimization.diffract_optimizer import diffract_optimize


class ToolRegistry:
    """Registry that validates tool names and dispatches deterministic calls."""

    def __init__(self) -> None:
        self._tools: dict[str, Callable[..., Any]] = {}

    def register(self, name: str, tool: Callable[..., Any]) -> None:
        self._tools[name] = tool

    def execute(self, name: str, state_vector: np.ndarray, **arguments: Any) -> Any:
        if name not in self._tools:
            raise KeyError(f"Unknown agent tool: {name}")
        return self._tools[name](state_vector, **arguments)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._tools))


def estimate_channel(state_vector: np.ndarray, method: str = "pilot_based") -> dict[str, Any]:
    gains = _gains_from_state(state_vector)
    return {"method": method, "channel_gains": gains, "mean_gain": float(gains.mean())}


def detect_jamming(state_vector: np.ndarray, sensitivity: str = "high") -> dict[str, Any]:
    profile = np.abs(np.asarray(state_vector[232:248], dtype=np.float32))
    power = float(profile.mean())
    threshold = 0.15 if sensitivity == "high" else 0.30
    return {"detected": power > threshold, "power": power, "type": "broadband" if power > threshold else "none"}


def predict_traffic(state_vector: np.ndarray, horizon: int = 1) -> dict[str, Any]:
    features = np.asarray(state_vector[192:224], dtype=np.float32)
    return {"horizon": horizon, "predicted_demand": float(np.maximum(features, 0).mean()), "urgency": float(np.abs(features).std())}


def allocate_power(state_vector: np.ndarray, method: str = "classical_wf") -> dict[str, Any]:
    gains = _gains_from_state(state_vector)
    if method == "adversarial_wf":
        allocation, jammer = adversarial_water_filling(gains, 1e-3, 1.0, 1.0, max_iterations=20)
    elif method == "diffract":
        allocation, jammer = diffract_optimize(gains, 1e-3, 1.0, iterations=50), np.zeros(32, dtype=np.float32)
    elif method == "classical_wf":
        allocation, jammer = water_filling(gains, 1e-3, 1.0), np.zeros(32, dtype=np.float32)
    else:
        raise ValueError(f"Unsupported allocation method: {method}")
    return {"method": method, "allocation": allocation, "jammer_allocation": jammer}


def allocate_spectrum(state_vector: np.ndarray, fairness: str = "proportional",
                      allocation: np.ndarray | None = None) -> dict[str, Any]:
    allocation = allocation if allocation is not None else allocate_power(state_vector)["allocation"]
    return {"fairness": fairness, "user_per_subcarrier": allocation.argmax(axis=0).astype(np.int32)}


def reconfigure_beam(state_vector: np.ndarray, mode: str = "adaptive") -> dict[str, Any]:
    phases = np.angle(np.fft.fft(np.resize(state_vector[64:192], 64))).astype(np.float32)
    return {"mode": mode, "weights": np.exp(1j * phases).astype(np.complex64)}


def handover_user(state_vector: np.ndarray, user: int = 0) -> dict[str, Any]:
    satellite_count = max(1, int(round(abs(float(state_vector[250])))))
    return {"user": user, "satellite_index": user % satellite_count}


def evaluate_policy(state_vector: np.ndarray, action: np.ndarray) -> dict[str, float]:
    gains = _gains_from_state(state_vector)
    reward = spectral_efficiency(gains, action, 1e-3)
    user_rates = np.log2(1 + (action * gains / 1e-3)).sum(axis=1)
    fairness = float(user_rates.sum() ** 2 / max(len(user_rates) * np.square(user_rates).sum(), 1e-12))
    return {"spectral_efficiency": reward, "fairness_jain": fairness}


def default_tool_registry() -> ToolRegistry:
    """Create a registry containing all documented Step 6 tools."""
    registry = ToolRegistry()
    for tool in [estimate_channel, detect_jamming, predict_traffic, allocate_power,
                 allocate_spectrum, reconfigure_beam, handover_user, evaluate_policy]:
        registry.register(tool.__name__, tool)
    return registry