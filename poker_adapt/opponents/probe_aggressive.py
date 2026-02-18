from __future__ import annotations

import random
from collections import Counter

from poker_adapt.opponents.action_map import ActionMap


class ProbeAggressiveAgent:
    """
    A simple aggressive probe agent to put opponents under pressure.
    Used for sanity-checking scripted opponent behavior (TP/CS).
    """

    def __init__(self, action_map: ActionMap, seed: int | None = None):
        self.action_map = action_map
        self.rng = random.Random(seed)
        self.use_raw = False  # RLCard Agent API requirement
        self.action_counter: Counter[str] = Counter()

    def step(self, state) -> int:
        legal = state.get("legal_actions")
        legal_ids = list(legal.keys()) if hasattr(legal, "keys") else list(legal)

        def id_for(name: str) -> int | None:
            for aid in legal_ids:
                if self.action_map.id_to_name.get(aid, str(aid)) == name:
                    return aid
            return None

        # Prefer raises to force opponent responses
        if (aid := id_for("raise_pot")) is not None and self.rng.random() < 0.70:
            self._record(aid)
            return aid
        if (aid := id_for("raise_half_pot")) is not None and self.rng.random() < 0.20:
            self._record(aid)
            return aid
        if (aid := id_for("all_in")) is not None and self.rng.random() < 0.05:
            self._record(aid)
            return aid

        for fallback in ["call", "check", "fold"]:
            if (aid := id_for(fallback)) is not None:
                self._record(aid)
                return aid

        action_id = self.rng.choice(legal_ids)
        self._record(action_id)
        return action_id

    def _record(self, action_id: int) -> None:
        name = self.action_map.id_to_name.get(action_id, str(action_id))
        self.action_counter[name] += 1

    def eval_step(self, state):
        action_id = self.step(state)
        return action_id, {}
