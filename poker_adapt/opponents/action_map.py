from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


def _normalize_action_name(name: str) -> str:
    text = name.strip()
    if "." in text:
        text = text.split(".")[-1]
    return text.lower()


def _infer_from_raw_legal_actions(env) -> dict[int, str]:
    """
    Infer action-id mapping from RLCard's raw legal action enums.

    In no-limit-holdem, env often does not expose get_action_str/action_num.
    raw_legal_actions contains Enum values (e.g. Action.CHECK_CALL) with
    stable integer .value ids.
    """
    try:
        state, _ = env.reset()
    except Exception:
        return {}

    raw = state.get("raw_legal_actions")
    if not raw:
        return {}

    first = raw[0]
    if isinstance(first, Enum):
        enum_cls = type(first)
        return {int(member.value): _normalize_action_name(member.name) for member in enum_cls}

    inferred: dict[int, str] = {}
    for item in raw:
        value = getattr(item, "value", None)
        if value is None:
            continue
        inferred[int(value)] = _normalize_action_name(str(item))
    return inferred

DEFAULT_ACTION_NAMES = [
    "fold",
    "check_call",
    "raise_half_pot",
    "raise_pot",
    "all_in",
]


@dataclass(frozen=True)
class ActionMap:
    id_to_name: dict[int, str]
    name_to_id: dict[str, int]


def build_action_map(env) -> ActionMap:
    """
    Build mapping between RLCard action ids and readable action names.
    Tries env.get_action_str when available; falls back to a default list.
    """
    id_to_name = _infer_from_raw_legal_actions(env)

    if not id_to_name:
        action_num = getattr(env, "action_num", None)
        if action_num is None:
            # Conservative fallback for no-limit-holdem discrete abstraction.
            action_num = len(DEFAULT_ACTION_NAMES)

        if hasattr(env, "get_action_str"):
            for aid in range(action_num):
                try:
                    id_to_name[aid] = _normalize_action_name(env.get_action_str(aid))
                except Exception:
                    id_to_name[aid] = (
                        DEFAULT_ACTION_NAMES[aid] if aid < len(DEFAULT_ACTION_NAMES) else str(aid)
                    )
        else:
            for aid in range(action_num):
                id_to_name[aid] = (
                    DEFAULT_ACTION_NAMES[aid] if aid < len(DEFAULT_ACTION_NAMES) else str(aid)
                )

    name_to_id = {name: aid for aid, name in id_to_name.items()}
    return ActionMap(id_to_name=id_to_name, name_to_id=name_to_id)
