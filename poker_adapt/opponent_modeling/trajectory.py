from __future__ import annotations

from collections.abc import Iterable

from poker_adapt.opponent_modeling.stats import ActionEvent

_FALLBACK_ACTION_NAMES = {
    0: "fold",
    1: "check_call",
    2: "raise_half_pot",
    3: "raise_pot",
    4: "all_in",
}


def _normalize_token(value) -> str:
    text = str(value)
    if "." in text:
        text = text.split(".")[-1]
    return text.lower()


def stage_from_state(state: dict) -> str:
    raw_obs = state.get("raw_obs")
    if isinstance(raw_obs, dict):
        stage = raw_obs.get("stage")
        if stage is not None:
            return _normalize_token(stage).upper()
    return "PREFLOP"


def facing_raise_from_state(state: dict) -> bool:
    raw_obs = state.get("raw_obs")
    if not isinstance(raw_obs, dict):
        return False

    stakes = raw_obs.get("stakes")
    current_player = raw_obs.get("current_player")
    if isinstance(stakes, (list, tuple)) and isinstance(current_player, int):
        if 0 <= current_player < len(stakes):
            return stakes[current_player] > min(stakes)
    return False


def _raw_action_name_from_state(state: dict, action_id: int) -> str:
    raw_legal = state.get("raw_legal_actions")
    if isinstance(raw_legal, Iterable):
        for action in raw_legal:
            value = getattr(action, "value", None)
            if value == action_id:
                name = getattr(action, "name", None)
                if name:
                    return _normalize_token(name)
                return _normalize_token(action)

    return _FALLBACK_ACTION_NAMES.get(action_id, str(action_id))


def canonical_action_name(raw_name: str, facing_raise: bool) -> str:
    if raw_name == "check_call":
        return "call" if facing_raise else "check"
    return raw_name


def events_from_player_trajectory(player_trajectory: list[object]) -> list[ActionEvent]:
    """
    Convert a single player's RLCard trajectory into ActionEvent objects.

    RLCard trajectory format is [state0, action0, state1, action1, ..., stateN].
    """
    events: list[ActionEvent] = []
    for idx in range(0, len(player_trajectory) - 1, 2):
        state = player_trajectory[idx]
        action = player_trajectory[idx + 1]

        if not isinstance(state, dict):
            continue
        if not isinstance(action, int):
            continue

        facing_raise = facing_raise_from_state(state)
        street = stage_from_state(state)

        raw_action_name = _raw_action_name_from_state(state, action)
        action_name = canonical_action_name(raw_action_name, facing_raise)

        events.append(
            ActionEvent(
                street=street,
                action=action_name,
                facing_raise=facing_raise,
            )
        )

    return events
