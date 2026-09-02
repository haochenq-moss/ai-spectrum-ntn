"""Monte-Carlo dropout uncertainty estimator for allocation policies."""

from __future__ import annotations

import torch
from torch import nn


class BayesianPolicyNetwork(nn.Module):
    """Estimate allocation mean and epistemic uncertainty via MC dropout."""

    def __init__(self, state_dim: int = 256, action_dim: int = 512, dropout_rate: float = 0.1) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(), nn.Dropout(dropout_rate),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(dropout_rate),
            nn.Linear(128, action_dim),
        )

    def forward(self, state: torch.Tensor, num_samples: int = 32) -> tuple[torch.Tensor, torch.Tensor]:
        was_training = self.training
        self.train()
        samples = torch.stack([torch.softmax(self.network(state), dim=-1) for _ in range(num_samples)])
        self.train(was_training)
        return samples.mean(dim=0), samples.std(dim=0)