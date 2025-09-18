"""Decision engine aligning AI predictions with facility policies."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Dict, Iterable, List

from ..core.types import FacilityConfiguration, MaterialDetection, SortingAction, SortingDecision
from .actuators import ActuatorBank, default_actuator_bank
from .policy import SortingPolicy


@dataclass
class DecisionContext:
    detection: MaterialDetection
    candidate_actions: List[SortingAction]
    facility: FacilityConfiguration


class SortingDecisionEngine:
    """Combines AI predictions with rules, optimization, and actuator control."""

    def __init__(
        self,
        facility: FacilityConfiguration,
        policy: SortingPolicy | None = None,
        actuators: ActuatorBank | None = None,
    ) -> None:
        self.facility = facility
        self.policy = policy or SortingPolicy.default()
        self.actuators = actuators or default_actuator_bank()

    def decide(self, detection: MaterialDetection) -> SortingDecision:
        policy_actions = self.policy.actions_for(detection.category)
        scoring_trace = []
        scored_actions: List[SortingAction] = []
        for action in policy_actions:
            lane_category = self.facility.lanes.get(action.target_lane, detection.category)
            score = action.probability * detection.confidence
            if lane_category == detection.category:
                score *= 1.2
            if detection.contamination in ("high", "medium"):
                score *= 0.85
            scoring_trace.append(
                f"Action {action.actuator} -> {action.target_lane} scored {score:.3f}"
            )
            scored_actions.append(
                SortingAction(
                    target_lane=action.target_lane,
                    actuator=action.actuator,
                    priority=action.priority,
                    probability=min(1.0, score),
                )
            )
        return SortingDecision(detection=detection, actions=scored_actions, reasoning_trace=scoring_trace)

    async def execute(self, decision: SortingDecision) -> None:
        best = decision.best_action()
        direction = 1.0 if "lane" in best.target_lane else 0.5
        self.actuators.trigger(best.actuator, direction)
        await asyncio.sleep(0.001)
        self.actuators.reset(best.actuator)
