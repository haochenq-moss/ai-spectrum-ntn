"""PPO training environment for normalized spectrum-allocation actions."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np
from gymnasium import Env, spaces
from stable_baselines3 import PPO

from .training_utils import default_model_path, resolve_device


class SpectrumAllocationEnv(Env):
    """Sequential environment whose reward is capacity under synthetic channel gains."""

    metadata = {"render_modes": []}

    def __init__(self, teacher_path: str = "data/ground_truth/teacher_dataset.h5") -> None:
        with h5py.File(teacher_path, "r") as source:
            self.states = source["states"][()].astype(np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(self.states.shape[1],), dtype=np.float32)
        self.action_space = spaces.Box(0.0, 1.0, shape=(512,), dtype=np.float32)
        self.index = 0

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.index = 0 if options is None else int(options.get("start_index", 0)) % len(self.states)
        return self.states[self.index], {}

    def step(self, action: np.ndarray):
        allocation = np.clip(np.asarray(action, dtype=np.float32), 0.0, 1.0)
        allocation /= max(float(allocation.sum()), 1e-8)
        gains = np.exp(np.clip(self.states[self.index, 64:192], -6.0, 6.0))
        gains = np.resize(gains, allocation.size)
        reward = float(np.log2(1.0 + allocation * gains / 1e-3).sum())
        self.index += 1
        terminated = self.index >= len(self.states)
        observation = self.states[min(self.index, len(self.states) - 1)]
        return observation, reward, terminated, False, {"spectral_efficiency": reward}


def train_rl(num_episodes: int = 100, device: str = "cpu",
             training_config: dict[str, Any] | None = None,
             teacher_path: str = "data/ground_truth/teacher_dataset.h5") -> PPO:
    """Train a PPO allocation policy and persist the Stable-Baselines model."""
    del training_config
    environment = SpectrumAllocationEnv(teacher_path)
    model = PPO("MlpPolicy", environment, device=str(resolve_device(device)), n_steps=8, batch_size=8, verbose=0)
    model.learn(total_timesteps=max(8, num_episodes * len(environment.states)))
    model.save(str(default_model_path("rl_policy").with_suffix("")))
    return model