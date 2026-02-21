from __future__ import annotations

from dataclasses import dataclass

from poker_adapt.opponent_modeling.stats import OpponentStats, OpponentStyle


@dataclass(frozen=True)
class StyleClassifier:
    min_hands: int = 20

    def classify(self, stats: OpponentStats) -> OpponentStyle:
        if stats.num_hands < self.min_hands:
            return OpponentStyle.UNKNOWN

        vpip = stats.vpip_rate
        pfr = stats.pfr_rate
        ftr = stats.fold_to_raise_rate
        post_agg = stats.postflop_raises_per_hand

        # Loose-Aggressive: raises a lot preflop and/or postflop
        if pfr >= 0.25 or post_agg >= 0.60:
            return OpponentStyle.LAG

        # Tight-Passive: low vpip/pfr, folds to raises often
        if vpip <= 0.30 and pfr <= 0.12 and ftr >= 0.50:
            return OpponentStyle.TP

        # Calling Station: high vpip, low pfr, doesn't fold to raises much
        if vpip >= 0.45 and pfr <= 0.20 and ftr <= 0.30:
            return OpponentStyle.CS

        return OpponentStyle.UNKNOWN
