from poker_adapt.opponent_modeling.trajectory import events_from_player_trajectory


class _DummyAction:
    def __init__(self, name: str, value: int):
        self.name = name
        self.value = value

    def __str__(self):
        return f"Action.{self.name}"


def _raw_actions():
    return [
        _DummyAction("FOLD", 0),
        _DummyAction("CHECK_CALL", 1),
        _DummyAction("RAISE_HALF_POT", 2),
        _DummyAction("RAISE_POT", 3),
        _DummyAction("ALL_IN", 4),
    ]


def test_events_from_player_trajectory_maps_check_call_to_call_or_check():
    trajectory = [
        {
            "raw_obs": {"stage": "Stage.PREFLOP", "stakes": [98, 99], "current_player": 1},
            "raw_legal_actions": _raw_actions(),
        },
        1,  # CHECK_CALL while facing bet -> call
        {
            "raw_obs": {"stage": "Stage.FLOP", "stakes": [40, 40], "current_player": 1},
            "raw_legal_actions": _raw_actions(),
        },
        1,  # CHECK_CALL while not facing bet -> check
        {
            "raw_obs": {"stage": "Stage.TURN", "stakes": [30, 30], "current_player": 1},
            "raw_legal_actions": _raw_actions(),
        },
    ]

    events = events_from_player_trajectory(trajectory)
    assert len(events) == 2

    assert events[0].street == "PREFLOP"
    assert events[0].facing_raise is True
    assert events[0].action == "call"

    assert events[1].street == "FLOP"
    assert events[1].facing_raise is False
    assert events[1].action == "check"


def test_events_from_player_trajectory_maps_raises():
    trajectory = [
        {
            "raw_obs": {"stage": "Stage.RIVER", "stakes": [10, 10], "current_player": 0},
            "raw_legal_actions": _raw_actions(),
        },
        3,  # RAISE_POT
        {
            "raw_obs": {"stage": "Stage.RIVER", "stakes": [0, 0], "current_player": 1},
            "raw_legal_actions": _raw_actions(),
        },
    ]

    events = events_from_player_trajectory(trajectory)
    assert len(events) == 1
    assert events[0].street == "RIVER"
    assert events[0].action == "raise_pot"
