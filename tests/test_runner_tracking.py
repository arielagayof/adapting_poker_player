from poker_adapt.env.runner import run_session
from poker_adapt.opponent_modeling.stats import OpponentStyle


def test_tracking_tp_style_converges_and_metrics_look_tight_passive():
    result = run_session(
        opponent_name="TP",
        hands=80,
        learner_name="probe",
        seed=42,
        track_stats=True,
        print_every=80,
        verbose=False,
    )
    stats = result["stats"]

    assert result["routed_style"] == OpponentStyle.TP
    assert stats is not None
    assert stats.vpip_rate <= 0.35
    assert stats.pfr_rate <= 0.12
    assert stats.fold_to_raise_rate >= 0.50


def test_tracking_cs_style_converges_and_metrics_look_calling_station():
    result = run_session(
        opponent_name="CS",
        hands=80,
        learner_name="probe",
        seed=42,
        track_stats=True,
        print_every=80,
        verbose=False,
    )
    stats = result["stats"]

    assert result["routed_style"] == OpponentStyle.CS
    assert stats is not None
    assert stats.vpip_rate >= 0.45
    assert stats.pfr_rate <= 0.20
    assert stats.fold_to_raise_rate <= 0.30


def test_tracking_lag_style_converges_and_metrics_look_aggressive():
    result = run_session(
        opponent_name="LAG",
        hands=80,
        learner_name="probe",
        seed=42,
        track_stats=True,
        print_every=80,
        verbose=False,
    )
    stats = result["stats"]

    assert result["routed_style"] == OpponentStyle.LAG
    assert stats is not None
    assert stats.pfr_rate >= 0.25
