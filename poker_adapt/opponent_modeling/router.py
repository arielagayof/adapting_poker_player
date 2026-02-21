from __future__ import annotations

from dataclasses import dataclass

from poker_adapt.opponent_modeling.stats import OpponentStyle


@dataclass
class ModelRouter:
    current_style: OpponentStyle = OpponentStyle.UNKNOWN
    _last_pred: OpponentStyle = OpponentStyle.UNKNOWN
    _streak: int = 0

    def update(self, predicted: OpponentStyle) -> OpponentStyle:
        if predicted == OpponentStyle.UNKNOWN:
            self._last_pred = OpponentStyle.UNKNOWN
            self._streak = 0
            return self.current_style

        if predicted == self._last_pred:
            self._streak += 1
        else:
            self._last_pred = predicted
            self._streak = 1

        if self._streak >= 2 and predicted != self.current_style:
            self.current_style = predicted

        return self.current_style
