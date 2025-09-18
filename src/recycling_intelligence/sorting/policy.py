"""Policy definitions for mapping detections to mechanical actions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

from ..core.types import MaterialCategory, SortingAction


@dataclass
class PolicyRule:
    category: MaterialCategory
    lane: str
    actuator: str
    priority: int
    weight: float = 1.0
    fallback_lane: str | None = None
    contamination_threshold: float = 0.6


class SortingPolicy:
    """Simple policy engine using priority ordering."""

    def __init__(self, rules: Iterable[PolicyRule]) -> None:
        self._rules: Dict[MaterialCategory, PolicyRule] = {rule.category: rule for rule in rules}

    def actions_for(self, category: MaterialCategory, *, contamination_probability: float = 0.0) -> List[SortingAction]:
        if category not in self._rules:
            rule = PolicyRule(
                category=MaterialCategory.UNKNOWN,
                lane="reject",
                actuator="air_jet",
                priority=1,
                weight=0.8,
                fallback_lane="reject",
                contamination_threshold=0.3,
            )
        else:
            rule = self._rules[category]
        actions = [
            SortingAction(
                target_lane=rule.lane,
                actuator=rule.actuator,
                priority=rule.priority,
                probability=0.95 if category == rule.category else 0.5,
                policy_weight=rule.weight,
            )
        ]
        if (
            rule.fallback_lane
            and contamination_probability >= rule.contamination_threshold
            and rule.fallback_lane != rule.lane
        ):
            actions.append(
                SortingAction(
                    target_lane=rule.fallback_lane,
                    actuator="reject_gate",
                    priority=max(1, rule.priority - 1),
                    probability=0.6,
                    policy_weight=rule.weight * 0.75,
                )
            )
        return actions

    @classmethod
    def default(cls) -> "SortingPolicy":
        return cls(
            rules=[
                PolicyRule(MaterialCategory.PLASTIC, "lane_plastic", "air_jet", 3, weight=1.1, fallback_lane="lane_textile"),
                PolicyRule(MaterialCategory.GLASS, "lane_glass", "mechanical", 4, weight=1.2, fallback_lane="lane_metal"),
                PolicyRule(MaterialCategory.METAL, "lane_metal", "electrostatic", 5, weight=1.3, fallback_lane="lane_glass"),
                PolicyRule(MaterialCategory.PAPER, "lane_paper", "air_jet", 2, weight=1.0, fallback_lane="lane_organic"),
                PolicyRule(MaterialCategory.ORGANIC, "lane_organic", "ballistic", 1, weight=0.9, fallback_lane="lane_reject"),
                PolicyRule(MaterialCategory.E_WASTE, "lane_e_waste", "pusher", 5, weight=1.4, fallback_lane="lane_reject"),
                PolicyRule(MaterialCategory.TEXTILE, "lane_textile", "pusher", 2, weight=0.95, fallback_lane="lane_plastic"),
            ]
        )
