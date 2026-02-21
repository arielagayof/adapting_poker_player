import numpy as np

from poker_adapt.agents.dqn.replay_buffer import ReplayBuffer


def test_replay_buffer_add_and_sample_shapes():
    buf = ReplayBuffer(capacity=10, obs_dim=4, action_dim=5)

    for i in range(6):
        obs = np.full((4,), i, dtype=np.float32)
        next_obs = np.full((4,), i + 1, dtype=np.float32)
        next_mask = np.array([1, 0, 1, 0, 1], dtype=np.float32)
        buf.add(
            obs=obs,
            action=i % 5,
            reward=float(i),
            next_obs=next_obs,
            done=(i % 2 == 0),
            next_legal_mask=next_mask,
        )

    assert len(buf) == 6
    batch = buf.sample(batch_size=4, rng=np.random.default_rng(0))
    assert batch["obs"].shape == (4, 4)
    assert batch["actions"].shape == (4,)
    assert batch["next_legal_masks"].shape == (4, 5)

