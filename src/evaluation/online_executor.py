"""Bounded real-time spectrum-management execution with safe fallback actions."""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any

import numpy as np

from src.agent.ai_agent import AIAgent, build_agent


class OnlineExecutor:
    """Execute spectrum decisions periodically against processed scenario observations."""

    def __init__(self, agent_type: str = "compiled", agent: AIAgent | None = None,
                 compiled_workflow: str | None = None, log_file: str = "logs/online_execution.log",
                 control_interval_ms: int = 100, confidence_threshold: float = 0.8) -> None:
        self.agent_type = agent_type
        self.agent = agent or build_agent("hybrid")
        self.compiled_workflow = compiled_workflow
        self.log_file = Path(log_file)
        self.control_interval_seconds = control_interval_ms / 1000
        self.confidence_threshold = confidence_threshold

    @staticmethod
    def _fallback_decision(scenario_id: int, reason: str) -> dict[str, Any]:
        allocation = np.full((16, 32), 1 / 512, dtype=np.float32)
        return {
            "scenario_id": scenario_id,
            "fallback": True,
            "reason": reason,
            "actions": {"strategy": "equal_power_fallback", "power_allocation": allocation.tolist()},
            "policy_evaluation": {"spectral_efficiency": 0.0, "fairness_jain": 1.0},
        }

    @staticmethod
    def _is_safe(decision: dict[str, Any]) -> bool:
        allocation = np.asarray(decision["actions"]["power_allocation"], dtype=np.float32)
        return allocation.shape == (16, 32) and np.isfinite(allocation).all() and np.all(allocation >= 0) and abs(float(allocation.sum()) - 1.0) < 1e-5

    def run(self, duration_seconds: float = 3600, processed_path: str = "data/processed/learning_states.npz") -> dict[str, Any]:
        """Execute decisions until the duration expires and return aggregate run metrics."""
        payload = np.load(processed_path)
        scenario_ids = payload["scenario_ids"]
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        started, iterations, fallbacks, rewards = time.monotonic(), 0, 0, []
        with self.log_file.open("w") as log:
            while time.monotonic() - started < duration_seconds:
                iteration_started = time.monotonic()
                scenario_index = iterations % len(scenario_ids)
                try:
                    decision = self.agent.run(scenario_index, processed_path)
                    if decision["diagnosis"]["confidence"] < self.confidence_threshold or not self._is_safe(decision):
                        raise ValueError("low confidence or unsafe allocation")
                except Exception as error:
                    fallbacks += 1
                    decision = self._fallback_decision(int(scenario_ids[scenario_index]), str(error))
                latency_ms = (time.monotonic() - iteration_started) * 1000
                decision["latency_ms"] = latency_ms
                decision["timestamp"] = time.time()
                rewards.append(float(decision["policy_evaluation"]["spectral_efficiency"]))
                log.write(json.dumps(decision) + "\n")
                iterations += 1
                remaining = self.control_interval_seconds - (time.monotonic() - iteration_started)
                if remaining > 0:
                    time.sleep(remaining)
        return {"iterations": iterations, "fallbacks": fallbacks, "mean_spectral_efficiency": float(np.mean(rewards)), "log_file": str(self.log_file)}