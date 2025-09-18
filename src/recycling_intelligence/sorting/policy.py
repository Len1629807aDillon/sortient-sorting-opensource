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


class SortingPolicy:
    """Simple policy engine using priority ordering."""

    def __init__(self, rules: Iterable[PolicyRule]) -> None:
        self._rules: Dict[MaterialCategory, PolicyRule] = {rule.category: rule for rule in rules}

    def actions_for(self, category: MaterialCategory) -> List[SortingAction]:
        if category not in self._rules:
            rule = PolicyRule(category=MaterialCategory.UNKNOWN, lane="reject", actuator="air_jet", priority=1)
        else:
            rule = self._rules[category]
        return [
            SortingAction(
                target_lane=rule.lane,
                actuator=rule.actuator,
                priority=rule.priority,
                probability=0.95 if category == rule.category else 0.5,
            )
        ]

    @classmethod
    def default(cls) -> "SortingPolicy":
        return cls(
            rules=[
                PolicyRule(MaterialCategory.PLASTIC, "lane_plastic", "air_jet", 3),
                PolicyRule(MaterialCategory.GLASS, "lane_glass", "mechanical", 4),
                PolicyRule(MaterialCategory.METAL, "lane_metal", "electrostatic", 5),
                PolicyRule(MaterialCategory.PAPER, "lane_paper", "air_jet", 2),
                PolicyRule(MaterialCategory.ORGANIC, "lane_organic", "ballistic", 1),
                PolicyRule(MaterialCategory.E_WASTE, "lane_e_waste", "pusher", 5),
                PolicyRule(MaterialCategory.TEXTILE, "lane_textile", "pusher", 2),
            ]
        )
