from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


class QNetwork(nn.Module):
    """Simple MLP that predicts Q-values for all discrete actions."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_sizes: Sequence[int] = (128, 128),
    ):
        super().__init__()
        if obs_dim <= 0:
            raise ValueError("obs_dim must be positive")
        if action_dim <= 0:
            raise ValueError("action_dim must be positive")

        layers: list[nn.Module] = []
        in_dim = obs_dim
        for hidden in hidden_sizes:
            if hidden <= 0:
                raise ValueError("hidden sizes must be positive")
            layers.append(nn.Linear(in_dim, hidden))
            layers.append(nn.ReLU())
            in_dim = hidden
        layers.append(nn.Linear(in_dim, action_dim))

        self.model = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        return self.model(obs)

