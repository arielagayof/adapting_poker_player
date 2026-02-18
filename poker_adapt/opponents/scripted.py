from __future__ import annotations

import random
from collections import Counter

from poker_adapt.opponents.action_map import ActionMap


def _legal_action_ids(state) -> list[int]:
    legal = state.get("legal_actions")
    if hasattr(legal, "keys"):
        return list(legal.keys())
    return list(legal)


def _legal_names(action_map: ActionMap, legal_ids: list[int]) -> set[str]:
    return {action_map.id_to_name.get(aid, str(aid)) for aid in legal_ids}


def _pick_first_legal(
    action_map: ActionMap, legal_ids: list[int], preferred_names: list[str]
) -> int | None:
    legal_name_set = _legal_names(action_map, legal_ids)
    for name in preferred_names:
        if name in legal_name_set:
            return action_map.name_to_id[name]
    return None


def _is_facing_bet(legal_name_set: set[str]) -> bool:
    # Heuristic: when "call" is legal but "check" is not, you're likely facing a bet/raise.
    return ("call" in legal_name_set) and ("check" not in legal_name_set)


class ScriptedBaseAgent:
    """Base class for simple scripted opponents."""

    def __init__(self, action_map: ActionMap, seed: int | None = None):
        self.action_map = action_map
        self.rng = random.Random(seed)
        self.use_raw = False  # RLCard Agent API requirement
        self.action_counter: Counter[str] = Counter()
        self.facing_bet_counter: Counter[str] = Counter()  # keys: "yes"/"no"

    def _record(self, action_id: int, facing_bet: bool) -> None:
        name = self.action_map.id_to_name.get(action_id, str(action_id))
        self.action_counter[name] += 1
        self.facing_bet_counter["yes" if facing_bet else "no"] += 1

    def step(self, state) -> int:
        raise NotImplementedError

    def eval_step(self, state):
        action_id = self.step(state)
        return action_id, {}


class TightPassiveAgent(ScriptedBaseAgent):
    """
    Tight-Passive: folds more vs pressure, checks a lot, rarely raises.
    """

    def step(self, state) -> int:
        legal_ids = _legal_action_ids(state)
        legal_set = _legal_names(self.action_map, legal_ids)
        facing_bet = _is_facing_bet(legal_set)

        action_id: int | None = None

        if facing_bet:
            # Prefer folding vs bets, sometimes call.
            if "fold" in legal_set and self.rng.random() < 0.70:
                action_id = self.action_map.name_to_id["fold"]
            elif "call" in legal_set and self.rng.random() < 0.80:
                action_id = self.action_map.name_to_id["call"]
            else:
                action_id = _pick_first_legal(self.action_map, legal_ids, ["fold", "call", "check"])
        else:
            # Mostly check, very rarely raise.
            if "check" in legal_set and self.rng.random() < 0.85:
                action_id = self.action_map.name_to_id["check"]
            elif "raise_half_pot" in legal_set and self.rng.random() < 0.12:
                action_id = self.action_map.name_to_id["raise_half_pot"]
            elif "raise_pot" in legal_set and self.rng.random() < 0.03:
                action_id = self.action_map.name_to_id["raise_pot"]
            else:
                action_id = _pick_first_legal(
                    self.action_map, legal_ids, ["check", "call", "raise_half_pot", "raise_pot"]
                )

        if action_id is None:
            action_id = self.rng.choice(legal_ids)

        self._record(action_id, facing_bet)
        return action_id


class LooseAggressiveAgent(ScriptedBaseAgent):
    """
    Loose-Aggressive: raises frequently when possible.
    """

    def step(self, state) -> int:
        legal_ids = _legal_action_ids(state)
        legal_set = _legal_names(self.action_map, legal_ids)
        facing_bet = _is_facing_bet(legal_set)

        action_id: int | None = None

        # Prefer aggressive actions in a prioritized way.
        r = self.rng.random()
        if "raise_pot" in legal_set and r < 0.50:
            action_id = self.action_map.name_to_id["raise_pot"]
        elif "raise_half_pot" in legal_set and r < 0.85:
            action_id = self.action_map.name_to_id["raise_half_pot"]
        elif "all_in" in legal_set and r < 0.95:
            action_id = self.action_map.name_to_id["all_in"]
        else:
            action_id = _pick_first_legal(
                self.action_map,
                legal_ids,
                ["call", "check", "raise_half_pot", "raise_pot", "all_in", "fold"],
            )

        if action_id is None:
            action_id = self.rng.choice(legal_ids)

        self._record(action_id, facing_bet)
        return action_id


class CallingStationAgent(ScriptedBaseAgent):
    """
    Calling Station: calls a lot vs bets, rarely folds, rarely raises.
    """

    def step(self, state) -> int:
        legal_ids = _legal_action_ids(state)
        legal_set = _legal_names(self.action_map, legal_ids)
        facing_bet = _is_facing_bet(legal_set)

        action_id: int | None = None

        if facing_bet:
            if "call" in legal_set and self.rng.random() < 0.90:
                action_id = self.action_map.name_to_id["call"]
            elif "raise_half_pot" in legal_set and self.rng.random() < 0.05:
                action_id = self.action_map.name_to_id["raise_half_pot"]
            else:
                action_id = _pick_first_legal(
                    self.action_map, legal_ids, ["call", "check", "raise_half_pot", "fold"]
                )
        else:
            if "check" in legal_set and self.rng.random() < 0.90:
                action_id = self.action_map.name_to_id["check"]
            else:
                action_id = _pick_first_legal(
                    self.action_map, legal_ids, ["check", "call", "raise_half_pot"]
                )

        if action_id is None:
            action_id = self.rng.choice(legal_ids)

        self._record(action_id, facing_bet)
        return action_id
