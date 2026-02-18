from __future__ import annotations

from dataclasses import dataclass

DEFAULT_ACTION_NAMES = [
    "fold",
    "check",
    "call",
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
    id_to_name: dict[int, str] = {}

    action_num = getattr(env, "action_num", None)
    if action_num is None:
        # Fallback if env doesn't expose action_num
        action_num = len(DEFAULT_ACTION_NAMES)

    if hasattr(env, "get_action_str"):
        for aid in range(action_num):
            try:
                id_to_name[aid] = env.get_action_str(aid)
            except Exception:
                # Fallback for any unexpected errors
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
