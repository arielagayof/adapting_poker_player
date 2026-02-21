from __future__ import annotations

import collections
from dataclasses import dataclass
from enum import StrEnum


class OpponentStyle(StrEnum):
    TP = "TP"
    CS = "CS"
    LAG = "LAG"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class ActionEvent:
    street: str  # "PREFLOP" / "FLOP" / "TURN" / "RIVER"
    action: str  # "fold" / "check" / "call" / "raise_*" / "all_in"
    facing_raise: bool = False


_RAISE_ACTIONS = {"raise_half_pot", "raise_pot", "all_in"}
_POSTFLOP_STREETS = {"FLOP", "TURN", "RIVER"}
_VPIP_ACTIONS = {"call"} | _RAISE_ACTIONS


class OpponentStats:
    """
    Rolling-window opponent statistics computed from per-action events.
    Tests can feed synthetic events without RLCard.
    """

    def __init__(self, window: int = 100):
        if window <= 0:
            raise ValueError("window must be positive")
        self.window = window

        self._hands: collections.deque[dict[str, int]] = collections.deque()
        self._in_hand = False

        # Current-hand accumulators
        self._hand_vpip = 0
        self._hand_pfr = 0
        self._hand_postflop_raises = 0
        self._hand_facing_raise_opps = 0
        self._hand_folds_to_raise = 0

        # Rolling sums
        self._vpip_hands = 0
        self._pfr_hands = 0
        self._postflop_raises_total = 0
        self._facing_raise_opps_total = 0
        self._folds_to_raise_total = 0

    def begin_hand(self) -> None:
        self._in_hand = True
        self._hand_vpip = 0
        self._hand_pfr = 0
        self._hand_postflop_raises = 0
        self._hand_facing_raise_opps = 0
        self._hand_folds_to_raise = 0

    def record(self, event: ActionEvent) -> None:
        if not self._in_hand:
            raise RuntimeError("begin_hand() must be called before record()")

        street = event.street.upper()
        action = event.action

        if street == "PREFLOP":
            if action in _VPIP_ACTIONS:
                self._hand_vpip = 1
            if action in _RAISE_ACTIONS:
                self._hand_pfr = 1

        if street in _POSTFLOP_STREETS and action in _RAISE_ACTIONS:
            self._hand_postflop_raises += 1

        if event.facing_raise:
            self._hand_facing_raise_opps += 1
            if action == "fold":
                self._hand_folds_to_raise += 1

    def end_hand(self) -> None:
        if not self._in_hand:
            raise RuntimeError("begin_hand() must be called before end_hand()")

        summary = {
            "vpip": self._hand_vpip,
            "pfr": self._hand_pfr,
            "postflop_raises": self._hand_postflop_raises,
            "facing_raise_opps": self._hand_facing_raise_opps,
            "folds_to_raise": self._hand_folds_to_raise,
        }

        self._hands.append(summary)

        self._vpip_hands += summary["vpip"]
        self._pfr_hands += summary["pfr"]
        self._postflop_raises_total += summary["postflop_raises"]
        self._facing_raise_opps_total += summary["facing_raise_opps"]
        self._folds_to_raise_total += summary["folds_to_raise"]

        while len(self._hands) > self.window:
            old = self._hands.popleft()
            self._vpip_hands -= old["vpip"]
            self._pfr_hands -= old["pfr"]
            self._postflop_raises_total -= old["postflop_raises"]
            self._facing_raise_opps_total -= old["facing_raise_opps"]
            self._folds_to_raise_total -= old["folds_to_raise"]

        self._in_hand = False

    @property
    def num_hands(self) -> int:
        return len(self._hands)

    @property
    def vpip_rate(self) -> float:
        return self._vpip_hands / self.num_hands if self.num_hands else 0.0

    @property
    def pfr_rate(self) -> float:
        return self._pfr_hands / self.num_hands if self.num_hands else 0.0

    @property
    def postflop_raises_total(self) -> int:
        return self._postflop_raises_total

    @property
    def postflop_raises_per_hand(self) -> float:
        return self._postflop_raises_total / self.num_hands if self.num_hands else 0.0

    @property
    def facing_raise_opps_total(self) -> int:
        return self._facing_raise_opps_total

    @property
    def folds_to_raise_total(self) -> int:
        return self._folds_to_raise_total

    @property
    def fold_to_raise_rate(self) -> float:
        denom = self._facing_raise_opps_total
        return self._folds_to_raise_total / denom if denom else 0.0
