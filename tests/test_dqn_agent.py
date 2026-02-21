import numpy as np
import torch

from poker_adapt.agents.dqn.dqn_agent import DQNAgent, DQNConfig


def _small_config() -> DQNConfig:
    return DQNConfig(
        gamma=0.99,
        lr=1e-3,
        batch_size=4,
        buffer_size=100,
        learning_starts=4,
        target_update_interval=2,
        epsilon_start=0.0,
        epsilon_end=0.0,
        epsilon_decay_steps=1,
        hidden_sizes=(16, 16),
    )


def test_select_action_with_masking_returns_legal_best_action():
    agent = DQNAgent(obs_dim=3, action_dim=5, config=_small_config(), seed=7)
    with torch.no_grad():
        for p in agent.online_net.parameters():
            p.zero_()
        final_linear = agent.online_net.model[-1]
        final_linear.bias[:] = torch.tensor([0.0, 5.0, 1.0, 4.0, 3.0])

    obs = np.array([0.1, -0.2, 0.3], dtype=np.float32)
    action = agent.select_action(obs, legal_action_ids=[0, 2, 4], epsilon=0.0)
    assert action == 4  # best among legal actions


def test_update_returns_loss_after_learning_starts():
    agent = DQNAgent(obs_dim=3, action_dim=5, config=_small_config(), seed=11)

    for i in range(8):
        obs = np.array([i, i + 1, i + 2], dtype=np.float32)
        next_obs = obs + 0.5
        agent.store_transition(
            obs=obs,
            action=i % 5,
            reward=1.0 if i % 3 == 0 else 0.0,
            next_obs=next_obs,
            done=(i % 4 == 0),
            next_legal_action_ids=[0, 1, 2],
        )

    loss = agent.update()
    assert loss is not None
    assert np.isfinite(loss)


def test_save_and_reload_checkpoint(tmp_path):
    agent = DQNAgent(obs_dim=3, action_dim=5, config=_small_config(), seed=19)
    model_path = tmp_path / "model.pt"
    agent.save(str(model_path))

    loaded = DQNAgent.from_checkpoint(str(model_path))
    assert loaded.obs_dim == 3
    assert loaded.action_dim == 5
