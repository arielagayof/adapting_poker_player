from __future__ import annotations

import random
from collections import Counter

from poker_adapt.opponents.action_map import ActionMap


def _is_facing_bet(state) -> bool:
    raw_obs = state.get("raw_obs")
    if isinstance(raw_obs, dict):
        stakes = raw_obs.get("stakes")
        current_player = raw_obs.get("current_player")
        if isinstance(stakes, (list, tuple)) and isinstance(current_player, int):
            if 0 <= current_player < len(stakes):
                return stakes[current_player] > min(stakes)
    return False


def _display_action_name(raw_name: str, facing_bet: bool) -> str:
    if raw_name == "check_call":
        return "call" if facing_bet else "check"
    return raw_name


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
        facing_bet = _is_facing_bet(state)

        def id_for(name: str) -> int | None:
            for aid in legal_ids:
                if self.action_map.id_to_name.get(aid, str(aid)) == name:
                    return aid
            return None

        # Prefer raises to force opponent responses
        if (aid := id_for("raise_pot")) is not None and self.rng.random() < 0.70:
            self._record(aid, facing_bet)
            return aid
        if (aid := id_for("raise_half_pot")) is not None and self.rng.random() < 0.20:
            self._record(aid, facing_bet)
            return aid
        if (aid := id_for("all_in")) is not None and self.rng.random() < 0.05:
            self._record(aid, facing_bet)
            return aid

        for fallback in ["call", "check_call", "check", "fold"]:
            if (aid := id_for(fallback)) is not None:
                self._record(aid, facing_bet)
                return aid

        action_id = self.rng.choice(legal_ids)
        self._record(action_id, facing_bet)
        return action_id

    def _record(self, action_id: int, facing_bet: bool) -> None:
        raw_name = self.action_map.id_to_name.get(action_id, str(action_id))
        name = _display_action_name(raw_name, facing_bet)
        self.action_counter[name] += 1

    def eval_step(self, state):
        action_id = self.step(state)
        return action_id, {}
