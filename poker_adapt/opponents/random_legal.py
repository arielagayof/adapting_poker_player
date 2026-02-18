from __future__ import annotations

import random
from collections import Counter

from poker_adapt.opponents.action_map import ActionMap


class RandomLegalAgent:
    """Agent that picks a uniformly random legal action."""

    def __init__(self, action_map: ActionMap, seed: int | None = None):
        self.action_map = action_map
        self.rng = random.Random(seed)
        self.use_raw = False  # RLCard Agent API requirement
        self.action_counter: Counter[str] = Counter()

    def step(self, state) -> int:
        legal = state.get("legal_actions")
        legal_ids = list(legal.keys()) if hasattr(legal, "keys") else list(legal)
        action_id = self.rng.choice(legal_ids)

        name = self.action_map.id_to_name.get(action_id, str(action_id))
        self.action_counter[name] += 1
        return action_id

    def eval_step(self, state):
        action_id = self.step(state)
        return action_id, {}
