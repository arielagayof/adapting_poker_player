from __future__ import annotations

import random
from collections import Counter

from poker_adapt.opponents.action_map import ActionMap


def _legal_action_ids(state) -> list[int]:
    legal = state.get("legal_actions")
    if hasattr(legal, "keys"):
        return list(legal.keys())
    return list(legal)


def _id_for_name(action_map: ActionMap, legal_ids: list[int], name: str) -> int | None:
    for aid in legal_ids:
        if action_map.id_to_name.get(aid, str(aid)) == name:
            return aid
    return None


def _id_for_call(action_map: ActionMap, legal_ids: list[int]) -> int | None:
    # Some RLCard setups merge check/call into a single CHECK_CALL action.
    return (
        _id_for_name(action_map, legal_ids, "call")
        or _id_for_name(action_map, legal_ids, "check_call")
        or _id_for_name(action_map, legal_ids, "check")
    )


def _id_for_check(action_map: ActionMap, legal_ids: list[int]) -> int | None:
    return (
        _id_for_name(action_map, legal_ids, "check")
        or _id_for_name(action_map, legal_ids, "check_call")
        or _id_for_name(action_map, legal_ids, "call")
    )


def _legal_names(action_map: ActionMap, legal_ids: list[int]) -> set[str]:
    return {action_map.id_to_name.get(aid, str(aid)) for aid in legal_ids}


def _pick_first_legal(
    action_map: ActionMap, legal_ids: list[int], preferred_names: list[str]
) -> int | None:
    for name in preferred_names:
        aid = _id_for_name(action_map, legal_ids, name)
        if aid is not None:
            return aid
    return None


def _is_facing_bet(state, legal_names: set[str]) -> bool:
    """
    Determine whether acting player is facing a bet/raise.

    RLCard no-limit-holdem exposes this in raw_obs.stakes:
    if current player's stake is below the table max, they are facing action.
    """
    raw_obs = state.get("raw_obs")
    if isinstance(raw_obs, dict):
        stakes = raw_obs.get("stakes")
        current_player = raw_obs.get("current_player")
        if isinstance(stakes, (list, tuple)) and isinstance(current_player, int):
            if 0 <= current_player < len(stakes):
                return stakes[current_player] > min(stakes)

    # Fallback heuristic for synthetic tests without raw_obs.
    if "call" in legal_names and "check" not in legal_names:
        return True
    return False


def _display_action_name(raw_name: str, facing_bet: bool) -> str:
    if raw_name == "check_call":
        return "call" if facing_bet else "check"
    return raw_name


class ScriptedBaseAgent:
    """Base class for simple scripted opponents."""

    def __init__(self, action_map: ActionMap, seed: int | None = None):
        self.action_map = action_map
        self.rng = random.Random(seed)
        self.use_raw = False  # RLCard Agent API requirement
        self.action_counter: Counter[str] = Counter()
        self.facing_bet_counter: Counter[str] = Counter()  # keys: "yes"/"no"

    def _record(self, action_id: int, facing_bet: bool) -> None:
        raw_name = self.action_map.id_to_name.get(action_id, str(action_id))
        name = _display_action_name(raw_name, facing_bet)
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
        facing_bet = _is_facing_bet(state, legal_set)

        action_id: int | None = None

        if facing_bet:
            if "fold" in legal_set and self.rng.random() < 0.70:
                action_id = _id_for_name(self.action_map, legal_ids, "fold")
            elif self.rng.random() < 0.80:
                action_id = _id_for_call(self.action_map, legal_ids)
            else:
                action_id = _pick_first_legal(
                    self.action_map, legal_ids, ["fold", "call", "check_call", "check"]
                )
        else:
            if self.rng.random() < 0.85:
                action_id = _id_for_check(self.action_map, legal_ids)
            elif "raise_half_pot" in legal_set and self.rng.random() < 0.12:
                action_id = _id_for_name(self.action_map, legal_ids, "raise_half_pot")
            elif "raise_pot" in legal_set and self.rng.random() < 0.03:
                action_id = _id_for_name(self.action_map, legal_ids, "raise_pot")
            else:
                action_id = _pick_first_legal(
                    self.action_map,
                    legal_ids,
                    ["check", "check_call", "call", "raise_half_pot", "raise_pot"],
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

        facing_bet = _is_facing_bet(state, legal_set)

        action_id: int | None = None

        r = self.rng.random()
        if "raise_pot" in legal_set and r < 0.50:
            action_id = _id_for_name(self.action_map, legal_ids, "raise_pot")
        elif "raise_half_pot" in legal_set and r < 0.85:
            action_id = _id_for_name(self.action_map, legal_ids, "raise_half_pot")
        elif "all_in" in legal_set and r < 0.95:
            action_id = _id_for_name(self.action_map, legal_ids, "all_in")
        else:
            action_id = _pick_first_legal(
                self.action_map,
                legal_ids,
                ["call", "check_call", "check", "raise_half_pot", "raise_pot", "all_in", "fold"],
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

        facing_bet = _is_facing_bet(state, legal_set)

        action_id: int | None = None

        if facing_bet:
            if self.rng.random() < 0.90:
                action_id = _id_for_call(self.action_map, legal_ids)
            elif "raise_half_pot" in legal_set and self.rng.random() < 0.05:
                action_id = _id_for_name(self.action_map, legal_ids, "raise_half_pot")
            else:
                action_id = _pick_first_legal(
                    self.action_map,
                    legal_ids,
                    ["call", "check_call", "check", "raise_half_pot", "fold"],
                )
        else:
            if self.rng.random() < 0.90:
                action_id = _id_for_check(self.action_map, legal_ids)
            else:
                action_id = _pick_first_legal(
                    self.action_map, legal_ids, ["check", "check_call", "call", "raise_half_pot"]
                )

        if action_id is None:
            action_id = self.rng.choice(legal_ids)

        self._record(action_id, facing_bet)
        return action_id
