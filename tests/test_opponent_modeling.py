from poker_adapt.opponent_modeling.classifier import StyleClassifier
from poker_adapt.opponent_modeling.router import ModelRouter
from poker_adapt.opponent_modeling.stats import ActionEvent, OpponentStats, OpponentStyle


def test_stats_vpip_and_pfr_preflop_only():
    stats = OpponentStats(window=10)

    # Hand 1: preflop check -> no vpip, no pfr
    stats.begin_hand()
    stats.record(ActionEvent(street="PREFLOP", action="check"))
    stats.end_hand()

    # Hand 2: preflop call -> vpip yes, pfr no
    stats.begin_hand()
    stats.record(ActionEvent(street="PREFLOP", action="call"))
    stats.end_hand()

    # Hand 3: preflop raise -> vpip yes, pfr yes
    stats.begin_hand()
    stats.record(ActionEvent(street="PREFLOP", action="raise_pot"))
    stats.end_hand()

    assert stats.vpip_rate == 2 / 3
    assert stats.pfr_rate == 1 / 3


def test_stats_postflop_aggression_counts_raises():
    stats = OpponentStats(window=10)

    stats.begin_hand()
    stats.record(ActionEvent(street="FLOP", action="raise_pot"))
    stats.record(ActionEvent(street="TURN", action="call"))
    stats.record(ActionEvent(street="RIVER", action="raise_half_pot"))
    stats.end_hand()

    assert stats.postflop_raises_total == 2
    assert stats.postflop_raises_per_hand == 2.0


def test_stats_fold_to_raise_all_streets():
    stats = OpponentStats(window=10)

    stats.begin_hand()
    stats.record(ActionEvent(street="PREFLOP", action="fold", facing_raise=True))
    stats.record(ActionEvent(street="FLOP", action="call", facing_raise=True))
    stats.end_hand()

    assert stats.folds_to_raise_total == 1
    assert stats.fold_to_raise_rate == 0.5


def test_stats_window_trimming():
    stats = OpponentStats(window=2)

    # Hand 1: vpip yes
    stats.begin_hand()
    stats.record(ActionEvent(street="PREFLOP", action="call"))
    stats.end_hand()

    # Hand 2: vpip no
    stats.begin_hand()
    stats.record(ActionEvent(street="PREFLOP", action="check"))
    stats.end_hand()

    # Hand 3: vpip yes (now window keeps only hands 2+3)
    stats.begin_hand()
    stats.record(ActionEvent(street="PREFLOP", action="raise_pot"))
    stats.end_hand()

    assert stats.num_hands == 2
    assert stats.vpip_rate == 1 / 2  # only hand 3 counts in window


def test_classifier_basic_labels():
    stats = OpponentStats(window=50)

    # Make it easy: require only 1 hand for classification in this test
    clf = StyleClassifier(min_hands=1)

    # LAG: high PFR
    stats.begin_hand()
    stats.record(ActionEvent(street="PREFLOP", action="raise_pot"))
    stats.record(ActionEvent(street="FLOP", action="raise_pot"))
    stats.end_hand()
    assert clf.classify(stats) == OpponentStyle.LAG

    # Reset stats for CS-like: high VPIP, low PFR, low fold-to-raise
    stats = OpponentStats(window=50)
    stats.begin_hand()
    stats.record(ActionEvent(street="PREFLOP", action="call"))
    stats.record(ActionEvent(street="FLOP", action="call", facing_raise=True))
    stats.end_hand()
    assert clf.classify(stats) in {OpponentStyle.CS, OpponentStyle.UNKNOWN}

    # Reset stats for TP-like: low VPIP + folds to raises
    stats = OpponentStats(window=50)
    stats.begin_hand()
    stats.record(ActionEvent(street="PREFLOP", action="fold", facing_raise=True))
    stats.end_hand()
    assert clf.classify(stats) in {OpponentStyle.TP, OpponentStyle.UNKNOWN}


def test_model_router_switch_after_two_consecutive():
    r = ModelRouter()

    assert r.current_style == OpponentStyle.UNKNOWN

    r.update(OpponentStyle.TP)
    assert r.current_style == OpponentStyle.UNKNOWN  # not yet

    r.update(OpponentStyle.TP)
    assert r.current_style == OpponentStyle.TP  # switched

    r.update(OpponentStyle.CS)
    assert r.current_style == OpponentStyle.TP  # not yet

    r.update(OpponentStyle.CS)
    assert r.current_style == OpponentStyle.CS  # switched
