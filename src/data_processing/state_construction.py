"""Fixed-width network state representation used by optimization and learning."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class NetworkState:
    topology_embedding: np.ndarray
    channel_state: np.ndarray
    traffic_features: np.ndarray
    interference_features: np.ndarray
    jammer_profile: np.ndarray
    context_features: np.ndarray

    def to_vector(self) -> np.ndarray:
        """Return the normalized fixed-width 256-dimensional state vector."""
        vector = np.concatenate([
            self.topology_embedding, self.channel_state, self.traffic_features,
            self.interference_features, self.jammer_profile, self.context_features,
        ]).astype(np.float32)
        if vector.shape != (256,):
            raise ValueError(f"Expected a 256-dimensional state, received {vector.shape}")
        return vector


def build_context_features(scenario: dict, scenario_id: int, link_count: int) -> np.ndarray:
    """Build the eight scenario-level features not represented by the encoders."""
    atmosphere = scenario.get("atmospheric_conditions", {})
    return np.asarray([
        scenario_id,
        link_count,
        len(scenario.get("satellites", [])),
        len(scenario.get("ground_stations", [])),
        len(scenario.get("user_equipment", [])),
        atmosphere.get("rain_rate_mm_h", 0.0),
        atmosphere.get("temperature_c", 0.0),
        atmosphere.get("humidity_percent", 0.0),
    ], dtype=np.float32)