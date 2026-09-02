"""LSTM predictor for the traffic-feature segment of network states."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn

from .training_utils import default_model_path, resolve_device


class TrafficPredictor(nn.Module):
    """Predict the next 32-dimensional traffic feature vector."""

    def __init__(self, input_dim: int = 32, hidden_dim: int = 64) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.output = nn.Linear(hidden_dim, input_dim)

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        sequence, _ = self.lstm(history)
        return self.output(sequence[:, -1])


def train_traffic_predictor(epochs: int = 100, device: str = "cpu",
                            training_config: dict[str, Any] | None = None,
                            processed_path: str = "data/processed/learning_states.npz") -> TrafficPredictor:
    """Train using chronological traffic-feature windows from Step 2."""
    config = training_config or {}
    traffic = np.load(processed_path)["states"][:, 192:224]
    window = min(3, len(traffic) - 1)
    if window < 1:
        raise ValueError("At least two processed states are required for traffic prediction")
    histories = np.stack([traffic[index - window:index] for index in range(window, len(traffic))])
    targets = traffic[window:]
    target_device = resolve_device(device)
    model = TrafficPredictor().to(target_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config.get("learning_rate", 1e-3)))
    history_tensor = torch.tensor(histories, dtype=torch.float32, device=target_device)
    target_tensor = torch.tensor(targets, dtype=torch.float32, device=target_device)
    for _ in range(epochs):
        loss = nn.functional.mse_loss(model(history_tensor), target_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    model.eval()
    torch.save({"state_dict": model.state_dict()}, default_model_path("traffic_predictor"))
    return model