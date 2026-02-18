import random
import sys

import rlcard


def _legal_action_ids(state):
    """Return a list of legal action ids from RLCard state."""
    legal = state.get("legal_actions")
    if legal is None:
        raise KeyError("state has no 'legal_actions' key")

    # RLCard often uses a dict/OrderedDict mapping action_id -> action_info
    if hasattr(legal, "keys"):
        return list(legal.keys())

    # Fallback: sometimes it can be a list/iterable of ids
    return list(legal)


def _action_to_str(env, action_id):
    """Best-effort conversion of an action id to a readable string."""
    if hasattr(env, "get_action_str"):
        try:
            return env.get_action_str(action_id)
        except Exception:
            pass
    return str(action_id)


def main(seed=42, max_steps=200):
    # Create HU No-Limit Hold'em env (action abstraction is built-in in RLCard)
    env = rlcard.make("no-limit-holdem")

    # Seed if supported (different RLCard versions expose this differently)
    if hasattr(env, "set_seed"):
        try:
            env.set_seed(seed)
        except Exception:
            pass

    state, player_id = env.reset()
    num_players = getattr(env, "num_players", "unknown")
    print(f"env=no-limit-holdem num_players={num_players} seed={seed}")

    step = 0
    while not env.is_over():
        step += 1
        legal_ids = _legal_action_ids(state)

        readable_legal = [(aid, _action_to_str(env, aid)) for aid in legal_ids]
        print(f"\nstep={step} player_id={player_id}")
        print(f"legal_actions={readable_legal}")

        action = random.choice(legal_ids)
        print(f"chosen_action={action} ({_action_to_str(env, action)})")

        state, player_id = env.step(action)

        if step >= max_steps:
            print("\nReached max_steps; stopping early (hand may be unfinished).")
            break

    if env.is_over():
        payoffs = env.get_payoffs()
        print(f"\nHAND OVER. payoffs={payoffs}")
    else:
        print("\nHAND NOT OVER (stopped early).")


if __name__ == "__main__":
    seed = 42
    if len(sys.argv) >= 2:
        seed = int(sys.argv[1])
    main(seed=seed)
