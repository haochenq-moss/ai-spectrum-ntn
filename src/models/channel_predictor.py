"""LSTM predictor for the channel-encoding portion of network states."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn

from .training_utils import default_model_path, resolve_device


class ChannelPredictor(nn.Module):
    """Predict the next 128-dimensional channel encoding from a state history."""

    def __init__(self, input_dim: int = 128, hidden_dim: int = 128, num_layers: int = 2) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.output = nn.Linear(hidden_dim, input_dim)

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        sequence, _ = self.lstm(history)
        return self.output(sequence[:, -1])


def train_predictor(epochs: int = 100, device: str = "cpu",
                    training_config: dict[str, Any] | None = None,
                    processed_path: str = "data/processed/learning_states.npz") -> ChannelPredictor:
    """Train an LSTM with chronological channel-encoding windows."""
    config = training_config or {}
    states = np.load(processed_path)["states"][:, 64:192]
    window = min(3, len(states) - 1)
    if window < 1:
        raise ValueError("At least two processed states are required for channel prediction")
    histories = np.stack([states[index - window:index] for index in range(window, len(states))])
    targets = states[window:]
    target_device = resolve_device(device)
    model = ChannelPredictor().to(target_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config.get("learning_rate", 1e-3)))
    history_tensor = torch.tensor(histories, dtype=torch.float32, device=target_device)
    target_tensor = torch.tensor(targets, dtype=torch.float32, device=target_device)
    for _ in range(epochs):
        loss = nn.functional.mse_loss(model(history_tensor), target_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    model.eval()
    torch.save({"state_dict": model.state_dict()}, default_model_path("channel_predictor"))
    return model