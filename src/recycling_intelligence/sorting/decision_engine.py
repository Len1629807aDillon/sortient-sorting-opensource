"""Decision engine aligning AI predictions with facility policies."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping

from ..core.types import (
    ContaminationLevel,
    FacilityConfiguration,
    MaterialDetection,
    SortingAction,
    SortingDecision,
)
from .actuators import ActuatorBank, default_actuator_bank
from .policy import SortingPolicy


@dataclass
class DecisionContext:
    detection: MaterialDetection
    candidate_actions: List[SortingAction]
    facility: FacilityConfiguration
    stage_latencies: Mapping[str, float]
    remaining_budget_ms: float
    total_budget_ms: float


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

    def decide(
        self,
        detection: MaterialDetection,
        *,
        stage_latencies: Mapping[str, float] | None = None,
        remaining_budget_ms: float | None = None,
        total_budget_ms: float | None = None,
    ) -> SortingDecision:
        """Evaluate candidate actions and capture detailed reasoning traces."""

        stage_latencies = dict(stage_latencies or {})
        remaining_budget = remaining_budget_ms if remaining_budget_ms is not None else 0.0
        total_budget = total_budget_ms if total_budget_ms is not None else sum(stage_latencies.values())

        contamination_penalty = self._contamination_penalty(detection.contamination)
        policy_actions = self.policy.actions_for(
            detection.category,
            contamination_probability=detection.metadata.get("contamination_probability", 0.0),
        )
        context = DecisionContext(
            detection=detection,
            candidate_actions=policy_actions,
            facility=self.facility,
            stage_latencies=stage_latencies,
            remaining_budget_ms=remaining_budget,
            total_budget_ms=total_budget,
        )
        scoring_trace: List[str] = []
        scored_actions: List[SortingAction] = []
        policy_weights: Dict[str, float] = {}

        for action in context.candidate_actions:
            lane_category = context.facility.lanes.get(action.target_lane, detection.category)
            lane_alignment = 1.25 if lane_category == detection.category else 0.8
            lane_score = detection.confidence * action.probability * lane_alignment * contamination_penalty
            priority_weight = 1.0 + (action.priority / 10.0)
            combined_probability = max(0.0, min(1.0, lane_score * priority_weight * action.policy_weight))

            policy_weights[action.target_lane] = priority_weight * action.policy_weight

            scored_actions.append(
                SortingAction(
                    target_lane=action.target_lane,
                    actuator=action.actuator,
                    priority=action.priority,
                    probability=combined_probability,
                    policy_weight=action.policy_weight,
                    lane_score=lane_score,
                )
            )

            scoring_trace.append(
                (
                    f"Lane {action.target_lane} | suitability={lane_score:.3f} (conf={detection.confidence:.2f}, "
                    f"policy_prob={action.probability:.2f}, contamination_penalty={contamination_penalty:.2f}, "
                    f"alignment={lane_alignment:.2f}) -> weighted={combined_probability:.3f}; "
                    f"priority_weight={priority_weight:.2f}; actuator={action.actuator}"
                )
            )

        if context.remaining_budget_ms:
            scoring_trace.append(
                (
                    f"Latency budget snapshot | total={context.total_budget_ms:.2f}ms, "
                    f"consumed={context.total_budget_ms - context.remaining_budget_ms:.2f}ms, "
                    f"remaining={context.remaining_budget_ms:.2f}ms"
                )
            )
        if context.stage_latencies:
            ordered = ", ".join(
                f"{stage}:{latency:.2f}ms" for stage, latency in context.stage_latencies.items()
            )
            scoring_trace.append(f"Stage latency breakdown (pre-decision): {ordered}")

        decision = SortingDecision(
            detection=detection,
            actions=scored_actions,
            reasoning_trace=scoring_trace,
            latency_breakdown=dict(context.stage_latencies),
            policy_weights=policy_weights,
            total_latency_budget_ms=context.total_budget_ms,
        )
        return decision

    async def execute(self, decision: SortingDecision) -> None:
        best = decision.best_action()
        direction = 1.0 if "lane" in best.target_lane else 0.5
        self.actuators.trigger(best.actuator, direction)
        await asyncio.sleep(0.001)
        self.actuators.reset(best.actuator)

    @staticmethod
    def _contamination_penalty(level: ContaminationLevel) -> float:
        return {
            ContaminationLevel.NONE: 1.0,
            ContaminationLevel.LOW: 0.94,
            ContaminationLevel.MEDIUM: 0.75,
            ContaminationLevel.HIGH: 0.45,
        }[level]
