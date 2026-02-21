from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import rlcard

from poker_adapt.agents.dqn.dqn_agent import DQNAgent, DQNConfig
from poker_adapt.env.runner import make_opponent
from poker_adapt.opponents.action_map import build_action_map


def _legal_action_ids(state) -> list[int]:
    legal = state.get("legal_actions")
    return list(legal.keys()) if hasattr(legal, "keys") else list(legal)


def _obs_vector(state) -> np.ndarray:
    obs = np.asarray(state["obs"], dtype=np.float32)
    return obs.reshape(-1)


def _advance_opponent_turns(env, state, player_id: int, opponent):
    while not env.is_over() and player_id != 0:
        action = opponent.step(state)
        state, player_id = env.step(action)
    return state, player_id


def _resolve_save_path(opponent: str, save_path: str | None) -> Path:
    if save_path:
        return Path(save_path)
    return Path("models") / f"dqn_vs_{opponent.upper()}.pt"


def run_training(
    *,
    opponent: str,
    steps: int = 5_000,
    seed: int = 42,
    save_path: str | None = None,
    device: str = "cpu",
    hidden_sizes: tuple[int, int] = (128, 128),
    gamma: float = 0.99,
    lr: float = 1e-3,
    batch_size: int = 64,
    buffer_size: int = 50_000,
    learning_starts: int = 500,
    target_update_interval: int = 200,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.05,
    epsilon_decay_steps: int = 5_000,
    log_every: int = 500,
) -> dict:
    if steps <= 0:
        raise ValueError("steps must be positive")
    if log_every <= 0:
        raise ValueError("log_every must be positive")

    env = rlcard.make("no-limit-holdem")
    if hasattr(env, "set_seed"):
        try:
            env.set_seed(seed)
        except Exception:
            pass

    action_map = build_action_map(env)
    action_dim = max(action_map.id_to_name) + 1
    opponent_agent = make_opponent(opponent, action_map, seed=seed + 1)

    state, player_id = env.reset()
    state, player_id = _advance_opponent_turns(env, state, player_id, opponent_agent)
    if env.is_over():
        state, player_id = env.reset()
        state, player_id = _advance_opponent_turns(env, state, player_id, opponent_agent)

    obs_dim = _obs_vector(state).shape[0]
    config = DQNConfig(
        gamma=gamma,
        lr=lr,
        batch_size=batch_size,
        buffer_size=buffer_size,
        learning_starts=learning_starts,
        target_update_interval=target_update_interval,
        epsilon_start=epsilon_start,
        epsilon_end=epsilon_end,
        epsilon_decay_steps=epsilon_decay_steps,
        hidden_sizes=hidden_sizes,
    )
    agent = DQNAgent(
        obs_dim=obs_dim,
        action_dim=action_dim,
        config=config,
        device=device,
        seed=seed,
    )

    total_steps = 0
    total_hands = 0
    losses: list[float] = []
    recent_returns: list[float] = []

    while total_steps < steps:
        state, player_id = env.reset()
        state, player_id = _advance_opponent_turns(env, state, player_id, opponent_agent)
        if env.is_over():
            total_hands += 1
            continue

        episode_return = 0.0
        while not env.is_over() and total_steps < steps:
            obs = _obs_vector(state)
            legal_ids = _legal_action_ids(state)
            epsilon = agent.epsilon_by_step(total_steps)
            action = agent.select_action(obs, legal_ids, epsilon=epsilon)

            next_state, next_player = env.step(action)
            next_state, next_player = _advance_opponent_turns(
                env, next_state, next_player, opponent_agent
            )

            if env.is_over():
                reward = float(env.get_payoffs()[0])
                done = True
                next_obs = np.zeros((obs_dim,), dtype=np.float32)
                next_legal_ids: list[int] = []
            else:
                reward = 0.0
                done = False
                next_obs = _obs_vector(next_state)
                next_legal_ids = _legal_action_ids(next_state)

            agent.store_transition(
                obs=obs,
                action=action,
                reward=reward,
                next_obs=next_obs,
                done=done,
                next_legal_action_ids=next_legal_ids,
            )
            loss = agent.update()
            if loss is not None:
                losses.append(loss)

            episode_return += reward
            total_steps += 1

            if total_steps % log_every == 0 or total_steps == steps:
                avg_loss = float(np.mean(losses[-100:])) if losses else float("nan")
                avg_return = float(np.mean(recent_returns[-20:])) if recent_returns else 0.0
                print(
                    f"step={total_steps}/{steps} eps={epsilon:.3f} "
                    f"buffer={len(agent.replay)} avg_loss={avg_loss:.4f} "
                    f"recent_return={avg_return:.4f}"
                )

            state = next_state
            player_id = next_player

        total_hands += 1
        recent_returns.append(episode_return)

    out_path = _resolve_save_path(opponent, save_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    agent.save(str(out_path))
    print(f"saved_model={out_path}")

    return {
        "steps": total_steps,
        "hands": total_hands,
        "buffer_size": len(agent.replay),
        "save_path": str(out_path),
        "avg_recent_return": float(np.mean(recent_returns[-20:])) if recent_returns else 0.0,
    }


def _parse_hidden_sizes(text: str) -> tuple[int, int]:
    parts = [p.strip() for p in text.split(",") if p.strip()]
    if len(parts) != 2:
        raise ValueError("hidden_sizes must be provided as two comma-separated ints, e.g. 128,128")
    return int(parts[0]), int(parts[1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--opponent", required=True, choices=["TP", "LAG", "CS"])
    parser.add_argument("--steps", type=int, default=5_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-path", type=str, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--hidden-sizes", type=str, default="128,128")
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--buffer-size", type=int, default=50_000)
    parser.add_argument("--learning-starts", type=int, default=500)
    parser.add_argument("--target-update-interval", type=int, default=200)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--epsilon-decay-steps", type=int, default=5_000)
    parser.add_argument("--log-every", type=int, default=500)
    args = parser.parse_args()

    run_training(
        opponent=args.opponent,
        steps=args.steps,
        seed=args.seed,
        save_path=args.save_path,
        device=args.device,
        hidden_sizes=_parse_hidden_sizes(args.hidden_sizes),
        gamma=args.gamma,
        lr=args.lr,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        learning_starts=args.learning_starts,
        target_update_interval=args.target_update_interval,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay_steps=args.epsilon_decay_steps,
        log_every=args.log_every,
    )


if __name__ == "__main__":
    main()
