"""Observation adapters for processed NTN network states."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time

import numpy as np


@dataclass(frozen=True)
class EnvironmentSnapshot:
    """A timestamped network state observation."""
    state_vector: np.ndarray
    scenario_id: int
    timestamp: float


class EnvironmentPerception:
    """Load a validated state observation from Step 2 output."""

    def observe(self, scenario_index: int = 0,
                processed_path: str = "data/processed/learning_states.npz") -> EnvironmentSnapshot:
        payload = np.load(Path(processed_path))
        states, scenario_ids = payload["states"], payload["scenario_ids"]
        if not 0 <= scenario_index < len(states):
            raise IndexError(f"Scenario index {scenario_index} is outside 0..{len(states) - 1}")
        return EnvironmentSnapshot(states[scenario_index], int(scenario_ids[scenario_index]), time.time())