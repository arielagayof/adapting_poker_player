from __future__ import annotations

import numpy as np


class ReplayBuffer:
    """Fixed-size replay buffer storing numpy arrays for DQN updates."""

    def __init__(self, capacity: int, obs_dim: int, action_dim: int):
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        if obs_dim <= 0:
            raise ValueError("obs_dim must be positive")
        if action_dim <= 0:
            raise ValueError("action_dim must be positive")

        self.capacity = capacity
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        self._obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self._actions = np.zeros((capacity,), dtype=np.int64)
        self._rewards = np.zeros((capacity,), dtype=np.float32)
        self._next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self._dones = np.zeros((capacity,), dtype=np.float32)
        self._next_legal_masks = np.zeros((capacity, action_dim), dtype=np.float32)

        self._size = 0
        self._ptr = 0

    def __len__(self) -> int:
        return self._size

    def add(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
        next_legal_mask: np.ndarray,
    ) -> None:
        self._obs[self._ptr] = obs
        self._actions[self._ptr] = action
        self._rewards[self._ptr] = reward
        self._next_obs[self._ptr] = next_obs
        self._dones[self._ptr] = float(done)
        self._next_legal_masks[self._ptr] = next_legal_mask

        self._ptr = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(
        self,
        batch_size: int,
        rng: np.random.Generator | None = None,
    ) -> dict[str, np.ndarray]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self._size < batch_size:
            raise ValueError("not enough samples in replay buffer")

        if rng is None:
            rng = np.random.default_rng()
        idxs = rng.integers(0, self._size, size=batch_size)
        return {
            "obs": self._obs[idxs],
            "actions": self._actions[idxs],
            "rewards": self._rewards[idxs],
            "next_obs": self._next_obs[idxs],
            "dones": self._dones[idxs],
            "next_legal_masks": self._next_legal_masks[idxs],
        }
