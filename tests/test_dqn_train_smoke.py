from pathlib import Path

from poker_adapt.agents.dqn.train import run_training


def test_dqn_train_smoke_creates_model_file(tmp_path: Path):
    model_path = tmp_path / "dqn_vs_tp.pt"
    summary = run_training(
        opponent="TP",
        steps=80,
        seed=42,
        save_path=str(model_path),
        hidden_sizes=(32, 32),
        batch_size=16,
        buffer_size=500,
        learning_starts=16,
        target_update_interval=20,
        epsilon_decay_steps=100,
        log_every=200,  # suppress intermediate logs in short test
    )

    assert model_path.exists()
    assert summary["steps"] == 80
    assert summary["buffer_size"] > 0

