from poker_adapt.opponents.action_map import ActionMap
from poker_adapt.opponents.scripted import (
    CallingStationAgent,
    LooseAggressiveAgent,
    TightPassiveAgent,
)


def _dummy_action_map():
    id_to_name = {
        0: "fold",
        1: "check",
        2: "call",
        3: "raise_half_pot",
        4: "raise_pot",
        5: "all_in",
    }
    name_to_id = {v: k for k, v in id_to_name.items()}
    return ActionMap(id_to_name=id_to_name, name_to_id=name_to_id)


def test_scripted_agents_choose_legal_actions():
    action_map = _dummy_action_map()
    agents = [
        TightPassiveAgent(action_map, seed=1),
        LooseAggressiveAgent(action_map, seed=2),
        CallingStationAgent(action_map, seed=3),
    ]

    # A few representative legal sets
    legal_sets = [
        {0: None, 2: None, 3: None, 4: None, 5: None},  # facing bet
        {1: None, 3: None, 4: None, 5: None},  # not facing bet (check is legal)
        {0: None, 2: None},  # very limited
    ]

    for legal in legal_sets:
        state = {"legal_actions": legal}
        for agent in agents:
            action = agent.step(state)
            assert action in legal
