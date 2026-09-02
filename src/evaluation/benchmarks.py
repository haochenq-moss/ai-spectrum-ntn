"""Reproducible CPU benchmarks for spectrum-allocation policies."""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any, Callable

import numpy as np
import pandas as pd

from src.agent.action_generator import ActionGenerator, _gains_from_state
from src.agent.ai_agent import AIAgent
from src.optimization.adversarial_water_filling import adversarial_water_filling
from src.optimization.classical_water_filling import spectral_efficiency, water_filling
from src.optimization.diffract_optimizer import diffract_optimize


def _jains_index(rates: np.ndarray) -> float:
    return float(rates.sum() ** 2 / max(len(rates) * np.square(rates).sum(), 1e-12))


def _metrics(state: np.ndarray, allocation: np.ndarray, latency_ms: float) -> dict[str, float]:
    gains = _gains_from_state(state)
    rates = np.log2(1.0 + allocation * gains / 1e-3).sum(axis=1)
    efficiency = spectral_efficiency(gains, allocation, 1e-3)
    sinr = allocation * gains / 1e-3
    return {
        "spectral_efficiency": efficiency,
        "latency_ms": latency_ms,
        "fairness_jain": _jains_index(rates),
        "energy_efficiency": efficiency / max(float(allocation.sum()), 1e-12),
        "outage_probability": float(np.mean(sinr < 1.0)),
    }


def _allocation_functions() -> dict[str, Callable[[np.ndarray], np.ndarray]]:
    agent = AIAgent("hybrid")
    actions = ActionGenerator()
    return {
        "classical_wf": lambda state: water_filling(_gains_from_state(state), 1e-3, 1.0),
        "adversarial_wf": lambda state: adversarial_water_filling(_gains_from_state(state), 1e-3, 1.0, 1.0, max_iterations=20)[0],
        "diffract": lambda state: diffract_optimize(_gains_from_state(state), 1e-3, 1.0, iterations=50),
        "llm": lambda state: np.asarray(agent.action_generator.generate(state, "classical_wf").power_allocation),
        "proposed": lambda state: np.asarray(agent.action_generator.generate(state, agent.reasoning.reason(state).recommended_strategy).power_allocation),
    }


def _rl_allocation(state: np.ndarray) -> np.ndarray:
    """Load the trained PPO policy when present, otherwise use equal power."""
    artifact = Path("data/models/rl_policy.zip")
    if not artifact.is_file():
        return np.full((16, 32), 1 / 512, dtype=np.float32)
    from stable_baselines3 import PPO
    action, _ = PPO.load(str(artifact)).predict(state, deterministic=True)
    allocation = np.maximum(np.asarray(action, dtype=np.float32), 0.0).reshape(16, 32)
    return allocation / max(float(allocation.sum()), 1e-12)


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    frame = pd.DataFrame(rows)
    metric_names = ["spectral_efficiency", "latency_ms", "fairness_jain", "energy_efficiency", "outage_probability"]
    return {
        metric: {key: float(getattr(frame[metric], key)()) for key in ["mean", "std", "min", "max"]}
        for metric in metric_names
    }


def run_benchmark(baselines: str | list[str] | None = None, num_scenarios: int = 100,
                  output_dir: str = "experiments/results/",
                  processed_path: str = "data/processed/learning_states.npz",
                  agent: Any | None = None, compiled_workflow: str | None = None) -> dict[str, dict[str, dict[str, float]]]:
    """Evaluate baseline allocations on processed states and persist metrics."""
    del agent, compiled_workflow
    available = _allocation_functions()
    available["rl"] = _rl_allocation
    selected = list(available) if baselines is None or baselines == "all" else ([baselines] if isinstance(baselines, str) else baselines)
    unknown = set(selected) - set(available)
    if unknown:
        raise ValueError(f"Unknown baselines: {sorted(unknown)}")
    states = np.load(processed_path)["states"][:num_scenarios]
    if not len(states):
        raise ValueError("No processed states available for benchmarking")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    summary: dict[str, dict[str, dict[str, float]]] = {}
    all_rows: list[dict[str, Any]] = []
    for name in selected:
        rows = []
        for scenario_index, state in enumerate(states):
            started = time.perf_counter()
            allocation = available[name](state)
            latency_ms = (time.perf_counter() - started) * 1000
            row = {"baseline": name, "scenario_index": scenario_index, **_metrics(state, allocation, latency_ms)}
            rows.append(row)
            all_rows.append(row)
        summary[name] = _aggregate(rows)
    pd.DataFrame(all_rows).to_csv(output_path / "benchmark_per_scenario.csv", index=False)
    (output_path / "benchmark_summary.json").write_text(json.dumps(summary, indent=2))
    return summary