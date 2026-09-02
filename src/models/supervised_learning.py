"""Imitation-learning policy trained on Step 3 teacher allocations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .training_utils import default_model_path, resolve_device
from src.agent.action_generator import _gains_from_state
from src.optimization.classical_water_filling import spectral_efficiency


class SupervisedPolicyNetwork(nn.Module):
    """Map a 256-dimensional network state to a normalized power allocation."""

    def __init__(self, state_dim: int = 256, action_dim: int = 512) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, action_dim),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.network(state), dim=-1)


def train_supervised(epochs: int = 100, device: str = "cpu",
                     training_config: dict[str, Any] | None = None,
                     teacher_path: str = "data/ground_truth/teacher_dataset.h5",
                     split_path: str = "data/processed/splits.json",
                     model_path: str | None = None,
                     max_train_samples: int | None = None,
                     random_seed: int | None = None) -> SupervisedPolicyNetwork:
    """Train and persist the spectrum-allocation imitation policy."""
    config = training_config or {}
    if random_seed is not None:
        torch.manual_seed(random_seed)
    with h5py.File(teacher_path, "r") as source:
        states = torch.tensor(source["states"][()], dtype=torch.float32)
        actions = torch.tensor(source["actions"][()].reshape(len(states), -1), dtype=torch.float32)
    split_indices = json.loads(Path(split_path).read_text())
    train_indices = torch.tensor(split_indices["train"], dtype=torch.long)
    if max_train_samples is not None:
        train_indices = train_indices[:max_train_samples]
    if not len(train_indices):
        raise ValueError("At least one training state is required")
    train_states, train_actions = states[train_indices], actions[train_indices]
    target_device = resolve_device(device)
    model = SupervisedPolicyNetwork(states.shape[1], actions.shape[1]).to(target_device)
    generator = torch.Generator().manual_seed(random_seed) if random_seed is not None else None
    loader = DataLoader(TensorDataset(train_states, train_actions), batch_size=min(int(config.get("batch_size", 32)), len(train_states)), shuffle=True, generator=generator)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config.get("learning_rate", 1e-3)))
    for _ in range(epochs):
        for batch_states, batch_actions in loader:
            predictions = model(batch_states.to(target_device))
            loss = nn.functional.mse_loss(predictions, batch_actions.to(target_device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    model.eval()
    destination = Path(model_path) if model_path else default_model_path("supervised_policy")
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "state_dim": states.shape[1], "action_dim": actions.shape[1]}, destination)
    return model


def evaluate_supervised(model: SupervisedPolicyNetwork,
                        teacher_path: str = "data/ground_truth/teacher_dataset.h5",
                        split_path: str = "data/processed/splits.json") -> dict[str, float]:
    """Measure held-out imitation MSE, capacity, and inference latency."""
    with h5py.File(teacher_path, "r") as source:
        states = source["states"][()]
        targets = source["actions"][()].reshape(len(states), -1)
    test_indices = np.asarray(json.loads(Path(split_path).read_text())["test"], dtype=int)
    device = next(model.parameters()).device
    test_states = torch.tensor(states[test_indices], dtype=torch.float32, device=device)
    started = __import__("time").perf_counter()
    with torch.no_grad():
        predictions = model(test_states).cpu().numpy()
    latency_ms = (__import__("time").perf_counter() - started) * 1000 / len(test_indices)
    mse = float(np.mean((predictions - targets[test_indices]) ** 2))
    capacities = [
        spectral_efficiency(_gains_from_state(state), allocation.reshape(16, 32), 1e-3)
        for state, allocation in zip(states[test_indices], predictions)
    ]
    return {"test_mse": mse, "test_spectral_efficiency": float(np.mean(capacities)), "inference_latency_ms": latency_ms}