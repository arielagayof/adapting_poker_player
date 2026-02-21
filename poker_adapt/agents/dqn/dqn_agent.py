from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from poker_adapt.agents.dqn.q_network import QNetwork
from poker_adapt.agents.dqn.replay_buffer import ReplayBuffer


def legal_action_mask(action_dim: int, legal_action_ids: list[int]) -> np.ndarray:
    mask = np.zeros((action_dim,), dtype=np.float32)
    for action_id in legal_action_ids:
        if 0 <= action_id < action_dim:
            mask[action_id] = 1.0
    return mask


def masked_argmax(q_values: torch.Tensor, legal_mask: torch.Tensor) -> torch.Tensor:
    """
    Argmax over legal actions only.

    q_values shape: [batch, action_dim]
    legal_mask shape: [batch, action_dim] (0/1 or bool)
    """
    if q_values.dim() != 2:
        raise ValueError("q_values must be 2D [batch, action_dim]")
    if legal_mask.shape != q_values.shape:
        raise ValueError("legal_mask shape must match q_values")

    legal_bool = legal_mask.bool()
    masked_q = q_values.masked_fill(~legal_bool, -1e9)
    best_actions = torch.argmax(masked_q, dim=1)
    no_legal = legal_bool.sum(dim=1) == 0
    return torch.where(no_legal, torch.zeros_like(best_actions), best_actions)


def masked_max(q_values: torch.Tensor, legal_mask: torch.Tensor) -> torch.Tensor:
    if q_values.dim() != 2:
        raise ValueError("q_values must be 2D [batch, action_dim]")
    if legal_mask.shape != q_values.shape:
        raise ValueError("legal_mask shape must match q_values")

    legal_bool = legal_mask.bool()
    masked_q = q_values.masked_fill(~legal_bool, -1e9)
    best = torch.max(masked_q, dim=1).values
    no_legal = legal_bool.sum(dim=1) == 0
    return torch.where(no_legal, torch.zeros_like(best), best)


@dataclass(frozen=True)
class DQNConfig:
    gamma: float = 0.99
    lr: float = 1e-3
    batch_size: int = 64
    buffer_size: int = 50_000
    learning_starts: int = 1_000
    target_update_interval: int = 200
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 5_000
    grad_clip_norm: float = 10.0
    hidden_sizes: tuple[int, int] = (128, 128)


class DQNAgent:
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        config: DQNConfig | None = None,
        device: str = "cpu",
        seed: int = 42,
    ):
        if config is None:
            config = DQNConfig()
        self.config = config

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = torch.device(device)

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.online_net = QNetwork(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_sizes=config.hidden_sizes,
        ).to(self.device)
        self.target_net = QNetwork(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_sizes=config.hidden_sizes,
        ).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=config.lr)
        self.replay = ReplayBuffer(
            capacity=config.buffer_size,
            obs_dim=obs_dim,
            action_dim=action_dim,
        )
        self._rng = random.Random(seed)
        self.train_steps = 0

    def epsilon_by_step(self, step: int) -> float:
        c = self.config
        if c.epsilon_decay_steps <= 0:
            return c.epsilon_end
        t = min(1.0, step / c.epsilon_decay_steps)
        return c.epsilon_start + t * (c.epsilon_end - c.epsilon_start)

    def select_action(self, obs: np.ndarray, legal_action_ids: list[int], epsilon: float) -> int:
        if not legal_action_ids:
            raise ValueError("legal_action_ids cannot be empty")

        legal_ids = [a for a in legal_action_ids if 0 <= a < self.action_dim]
        if not legal_ids:
            raise ValueError("no legal action ids within action space")

        if self._rng.random() < epsilon:
            return self._rng.choice(legal_ids)

        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.online_net(obs_tensor)
            legal_mask = torch.as_tensor(
                legal_action_mask(self.action_dim, legal_ids),
                dtype=torch.float32,
                device=self.device,
            ).unsqueeze(0)
            action = masked_argmax(q_values, legal_mask)[0].item()
        return int(action)

    def store_transition(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
        next_legal_action_ids: list[int],
    ) -> None:
        next_mask = legal_action_mask(self.action_dim, next_legal_action_ids)
        self.replay.add(
            obs=obs.astype(np.float32),
            action=int(action),
            reward=float(reward),
            next_obs=next_obs.astype(np.float32),
            done=bool(done),
            next_legal_mask=next_mask,
        )

    def update(self) -> float | None:
        if len(self.replay) < max(self.config.batch_size, self.config.learning_starts):
            return None

        batch = self.replay.sample(self.config.batch_size)
        obs = torch.as_tensor(batch["obs"], dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(batch["actions"], dtype=torch.long, device=self.device)
        rewards = torch.as_tensor(batch["rewards"], dtype=torch.float32, device=self.device)
        next_obs = torch.as_tensor(batch["next_obs"], dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(batch["dones"], dtype=torch.float32, device=self.device)
        next_legal_masks = torch.as_tensor(
            batch["next_legal_masks"], dtype=torch.float32, device=self.device
        )

        q_values = self.online_net(obs)
        q_selected = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q_values = self.target_net(next_obs)
            next_q_max = masked_max(next_q_values, next_legal_masks)
            targets = rewards + (1.0 - dones) * self.config.gamma * next_q_max

        loss = F.mse_loss(q_selected, targets)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_net.parameters(), self.config.grad_clip_norm)
        self.optimizer.step()

        self.train_steps += 1
        if self.train_steps % self.config.target_update_interval == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

        return float(loss.item())

    def save(self, path: str) -> None:
        payload = {
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
            "model_state_dict": self.online_net.state_dict(),
            "config": self.config.__dict__,
        }
        torch.save(payload, path)

    @classmethod
    def from_checkpoint(cls, path: str, device: str = "cpu") -> DQNAgent:
        checkpoint = torch.load(path, map_location=device)
        config_dict = dict(checkpoint["config"])
        config_dict["hidden_sizes"] = tuple(config_dict["hidden_sizes"])
        config = DQNConfig(**config_dict)
        agent = cls(
            obs_dim=int(checkpoint["obs_dim"]),
            action_dim=int(checkpoint["action_dim"]),
            config=config,
            device=device,
        )
        agent.online_net.load_state_dict(checkpoint["model_state_dict"])
        agent.target_net.load_state_dict(checkpoint["model_state_dict"])
        return agent
