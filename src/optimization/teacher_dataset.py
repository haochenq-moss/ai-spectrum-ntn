"""Generate HDF5 teacher labels from processed network states."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np

from .adversarial_water_filling import adversarial_water_filling
from .classical_water_filling import spectral_efficiency, water_filling
from .diffract_optimizer import diffract_optimize


def _state_to_channel_gains(state: np.ndarray, num_users: int, num_subcarriers: int) -> np.ndarray:
    """Convert the Step 2 channel encoding into normalized solver gain values."""
    channel_encoding = np.abs(np.asarray(state[64:192], dtype=np.float64))
    compressed = np.log1p(np.clip(channel_encoding, 0.0, 1e6))
    relative_gains = compressed - compressed.min()
    relative_gains /= max(float(relative_gains.max()), 1e-12)
    samples = np.interp(
        np.linspace(0, relative_gains.size - 1, num_users * num_subcarriers),
        np.arange(relative_gains.size),
        relative_gains,
    )
    return (0.05 + 0.95 * samples).reshape(num_users, num_subcarriers)


def generate_teacher_dataset(method: str = "adversarial_wf", num_scenarios: int | None = None,
                             output_dir: str = "data/ground_truth/",
                             processed_dir: str = "data/processed/",
                             optimization_config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Create teacher states, power actions, jammer actions, and rewards in HDF5."""
    config = optimization_config or {}
    method_aliases = {"classical_wf": "classical_wf", "adversarial_wf": "adversarial_wf", "diffract": "diffract"}
    if method not in method_aliases:
        raise ValueError(f"Unsupported optimization method: {method}")
    payload = np.load(Path(processed_dir) / "learning_states.npz")
    states, scenario_ids = payload["states"], payload["scenario_ids"]
    if num_scenarios is not None:
        states, scenario_ids = states[:num_scenarios], scenario_ids[:num_scenarios]
    if not len(states):
        raise ValueError("No processed states available for teacher-label generation")
    num_users, num_subcarriers = 16, 32
    power_budget = 10 ** (float(config.get("power_budget_dbm", 30.0)) / 10.0) / 1000.0
    jammer_budget = 10 ** (float(config.get("jammer_budget_dbm", 30.0)) / 10.0) / 1000.0
    noise_power = 1e-3
    actions, jammers, rewards = [], [], []
    for state in states:
        gains = _state_to_channel_gains(state, num_users, num_subcarriers)
        if method == "adversarial_wf":
            action, jammer = adversarial_water_filling(gains, noise_power, power_budget, jammer_budget)
        elif method == "diffract":
            action, jammer = diffract_optimize(gains, noise_power, power_budget), np.zeros(num_subcarriers, dtype=np.float32)
        else:
            action, jammer = water_filling(gains, noise_power, power_budget), np.zeros(num_subcarriers, dtype=np.float32)
        actions.append(action)
        jammers.append(jammer)
        rewards.append(spectral_efficiency(gains, action, noise_power, jammer))
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    with h5py.File(destination / "teacher_dataset.h5", "w") as output:
        output.create_dataset("states", data=states, compression="gzip")
        output.create_dataset("actions", data=np.asarray(actions), compression="gzip")
        output.create_dataset("jammer_actions", data=np.asarray(jammers), compression="gzip")
        output.create_dataset("rewards", data=np.asarray(rewards, dtype=np.float32))
        output.create_dataset("scenario_ids", data=scenario_ids)
        output.attrs["method"] = method
        output.attrs["power_budget_w"] = power_budget
        output.attrs["num_users"] = num_users
        output.attrs["num_subcarriers"] = num_subcarriers
    return {"num_scenarios": len(states), "method": method, "output_file": str(destination / "teacher_dataset.h5"), "mean_reward": float(np.mean(rewards))}