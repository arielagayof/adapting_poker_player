from __future__ import annotations

import argparse
from collections import Counter

import rlcard

from poker_adapt.opponent_modeling.classifier import StyleClassifier
from poker_adapt.opponent_modeling.router import ModelRouter
from poker_adapt.opponent_modeling.stats import OpponentStats
from poker_adapt.opponent_modeling.trajectory import events_from_player_trajectory
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


def run_session(
    *,
    opponent_name: str,
    hands: int = 200,
    learner_name: str = "probe",
    seed: int = 42,
    track_stats: bool = False,
    print_every: int = 20,
    stats_window: int = 100,
    min_hands: int = 20,
    verbose: bool = True,
):
    if hands <= 0:
        raise ValueError("hands must be positive")
    if print_every <= 0:
        raise ValueError("print_every must be positive")

    opponent_name = opponent_name.upper()
    env = rlcard.make("no-limit-holdem")
    if getattr(env, "num_players", 2) != 2:
        raise RuntimeError(f"Expected Heads-Up (2 players) but got num_players={env.num_players}")

    if hasattr(env, "set_seed"):
        try:
            env.set_seed(seed)
        except Exception:
            pass

    action_map = build_action_map(env)

    if learner_name == "probe":
        learner = ProbeAggressiveAgent(action_map, seed=seed)
    else:
        learner = RandomLegalAgent(action_map, seed=seed)

    opponent = make_opponent(opponent_name, action_map, seed=seed + 1)

    env.set_agents([learner, opponent])

    stats = OpponentStats(window=stats_window) if track_stats else None
    classifier = StyleClassifier(min_hands=min_hands) if track_stats else None
    router = ModelRouter() if track_stats else None
    last_predicted = None
    last_routed = None

    total_payoff = 0.0
    for hand_idx in range(1, hands + 1):
        trajectories, payoffs = env.run(is_training=False)
        total_payoff += float(payoffs[0])

        if stats is not None and classifier is not None and router is not None:
            stats.begin_hand()
            for event in events_from_player_trajectory(trajectories[1]):
                stats.record(event)
            stats.end_hand()

            predicted = classifier.classify(stats)
            routed = router.update(predicted)
            last_predicted = predicted
            last_routed = routed

            if verbose and (hand_idx % print_every == 0 or hand_idx == hands):
                print(
                    "[track] "
                    f"hand={hand_idx} est={predicted.value} routed={routed.value} "
                    f"vpip={stats.vpip_rate:.3f} pfr={stats.pfr_rate:.3f} "
                    f"post_agg={stats.postflop_raises_per_hand:.3f} "
                    f"ftr={stats.fold_to_raise_rate:.3f} n={stats.num_hands}"
                )

    avg_payoff = total_payoff / hands

    if verbose:
        print(f"env=no-limit-holdem hands={hands} seed={seed}")
        print(f"learner={learner.__class__.__name__} opponent={opponent_name}")
        print(f"avg_payoff_per_hand(player0)={avg_payoff:.4f}")
        print("\nOpponent action distribution:")
        print(_format_counter(opponent.action_counter))

        print("\nOpponent facing_bet frequency:")
        print(_format_counter(opponent.facing_bet_counter))

        print("\nLearner action distribution:")
        print(_format_counter(learner.action_counter))

        if stats is not None and last_predicted is not None and last_routed is not None:
            print("\nTracking final snapshot:")
            print(
                f"estimated_style={last_predicted.value} "
                f"routed_style={last_routed.value} "
                f"vpip={stats.vpip_rate:.3f} pfr={stats.pfr_rate:.3f} "
                f"post_agg={stats.postflop_raises_per_hand:.3f} "
                f"ftr={stats.fold_to_raise_rate:.3f}"
            )

    return {
        "avg_payoff": avg_payoff,
        "opponent_action_counter": opponent.action_counter,
        "opponent_facing_bet_counter": opponent.facing_bet_counter,
        "learner_action_counter": learner.action_counter,
        "stats": stats,
        "predicted_style": last_predicted,
        "routed_style": last_routed,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--opponent", required=True, choices=["TP", "LAG", "CS"])
    parser.add_argument("--hands", type=int, default=200)
    parser.add_argument("--learner", choices=["random", "probe"], default="probe")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--track-stats", action="store_true")
    parser.add_argument("--print-every", type=int, default=20)
    parser.add_argument("--stats-window", type=int, default=100)
    parser.add_argument("--min-hands", type=int, default=20)
    args = parser.parse_args()

    run_session(
        opponent_name=args.opponent,
        hands=args.hands,
        learner_name=args.learner,
        seed=args.seed,
        track_stats=args.track_stats,
        print_every=args.print_every,
        stats_window=args.stats_window,
        min_hands=args.min_hands,
        verbose=True,
    )


if __name__ == "__main__":
    main()
