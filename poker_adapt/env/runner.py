from __future__ import annotations

import argparse
from collections import Counter

import rlcard

from poker_adapt.opponents.action_map import build_action_map
from poker_adapt.opponents.probe_aggressive import ProbeAggressiveAgent
from poker_adapt.opponents.random_legal import RandomLegalAgent
from poker_adapt.opponents.scripted import (
    CallingStationAgent,
    LooseAggressiveAgent,
    TightPassiveAgent,
)


def _format_counter(counter: Counter[str], top_n: int = 20) -> str:
    total = sum(counter.values())
    if total == 0:
        return "no actions recorded"
    items = counter.most_common(top_n)
    parts = [f"{k}={v} ({v / total:.1%})" for k, v in items]
    return ", ".join(parts)


def make_opponent(name: str, action_map, seed: int):
    name = name.upper()
    if name == "TP":
        return TightPassiveAgent(action_map, seed=seed)
    if name == "LAG":
        return LooseAggressiveAgent(action_map, seed=seed)
    if name == "CS":
        return CallingStationAgent(action_map, seed=seed)
    raise ValueError("Unknown opponent. Use one of: TP, LAG, CS")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--opponent", required=True, choices=["TP", "LAG", "CS"])
    parser.add_argument("--hands", type=int, default=200)
    parser.add_argument("--learner", choices=["random", "probe"], default="probe")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    env = rlcard.make("no-limit-holdem")
    if getattr(env, "num_players", 2) != 2:
        raise RuntimeError(f"Expected Heads-Up (2 players) but got num_players={env.num_players}")

    if hasattr(env, "set_seed"):
        try:
            env.set_seed(args.seed)
        except Exception:
            pass

    action_map = build_action_map(env)

    if args.learner == "probe":
        learner = ProbeAggressiveAgent(action_map, seed=args.seed)
    else:
        learner = RandomLegalAgent(action_map, seed=args.seed)

    opponent = make_opponent(args.opponent, action_map, seed=args.seed + 1)

    env.set_agents([learner, opponent])

    total_payoff = 0.0
    for _ in range(args.hands):
        trajectories, payoffs = env.run(is_training=False)
        total_payoff += float(payoffs[0])

    avg_payoff = total_payoff / args.hands

    print(f"env=no-limit-holdem hands={args.hands} seed={args.seed}")
    print(f"learner={learner.__class__.__name__} opponent={args.opponent}")
    print(f"avg_payoff_per_hand(player0)={avg_payoff:.4f}")
    print("\nOpponent action distribution:")
    print(_format_counter(opponent.action_counter))

    print("\nOpponent facing_bet frequency (heuristic):")
    print(_format_counter(opponent.facing_bet_counter))

    print("\nLearner action distribution:")
    print(_format_counter(learner.action_counter))


if __name__ == "__main__":
    main()
