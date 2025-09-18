"""System orchestration for the Recycling Intelligence platform."""

from __future__ import annotations

import asyncio
import math
import random
from dataclasses import dataclass, field
from typing import Iterable, List, Sequence

from .metrics import ConfusionMatrix, LatencyBudget, RollingMetric, ThroughputTracker
from .types import (
    DEFAULT_FEATURE_SPACE,
    FacilityConfiguration,
    MaterialCategory,
    MaterialDetection,
    SortingDecision,
)
from ..data.generator import SyntheticWasteStream
from ..ml.network import DeepWasteSorter
from ..ml.training import Trainer
from ..sorting.decision_engine import SortingDecisionEngine
from ..analytics.monitoring import MonitoringHub


@dataclass
class PipelineStage:
    """Represents a logical stage inside the processing pipeline."""

    name: str
    latency_budget_ms: float
    metric: RollingMetric = field(default_factory=lambda: RollingMetric(window=120))


class RecyclingIntelligenceSystem:
    """High level orchestrator joining machine learning and mechanical systems."""

    def __init__(
        self,
        facility: FacilityConfiguration | None = None,
        sorter: DeepWasteSorter | None = None,
        decision_engine: SortingDecisionEngine | None = None,
        monitoring: MonitoringHub | None = None,
    ) -> None:
        self.facility = facility or FacilityConfiguration(
            lanes={
                "lane_plastic": MaterialCategory.PLASTIC,
                "lane_glass": MaterialCategory.GLASS,
                "lane_metal": MaterialCategory.METAL,
                "lane_paper": MaterialCategory.PAPER,
                "lane_organic": MaterialCategory.ORGANIC,
                "lane_e_waste": MaterialCategory.E_WASTE,
                "lane_textile": MaterialCategory.TEXTILE,
                "lane_reject": MaterialCategory.UNKNOWN,
            },
            max_throughput_per_min=3000,
            feature_space=DEFAULT_FEATURE_SPACE,
        )
        self.sorter = sorter or DeepWasteSorter.from_feature_space(self.facility.feature_space)
        self.decision_engine = decision_engine or SortingDecisionEngine(self.facility)
        self.monitoring = monitoring or MonitoringHub()
        self.stages: Sequence[PipelineStage] = (
            PipelineStage("sensing", 4.0),
            PipelineStage("inference", 6.0),
            PipelineStage("decision", 3.0),
            PipelineStage("actuation", 7.0),
        )
        self.throughput = ThroughputTracker(horizon=60)
        self.confusion = ConfusionMatrix(tuple(MaterialCategory))
        self.latency_budget = LatencyBudget(budget_ms=sum(stage.latency_budget_ms for stage in self.stages))

    def train(self, dataset: Iterable[tuple[list[float], MaterialCategory]], epochs: int = 3) -> None:
        trainer = Trainer(self.sorter)
        trainer.train(dataset, epochs=epochs)

    async def run_stream(self, stream: SyntheticWasteStream, duration_seconds: float = 5.0) -> None:
        """Run the processing pipeline on a synthetic waste stream."""

        start_time = stream.current_time
        processed = 0
        while stream.current_time - start_time < duration_seconds:
            sample = stream.next_item()
            latency_budget = self.latency_budget
            latency_budget.reset()
            detection_latency = self._simulate_stage_latency(self.stages[0])
            features = sample.features
            self.monitoring.log_feature_vector(features)
            latency_budget.consume(detection_latency)
            stage_latencies = {"sensing": detection_latency}

            predictions = self.sorter.predict(features)
            inference_latency = self._simulate_stage_latency(self.stages[1])
            latency_budget.consume(inference_latency)
            stage_latencies["inference"] = inference_latency
            detection = MaterialDetection(
                material_id=sample.identifier,
                category=predictions.category,
                confidence=predictions.confidence,
                contamination=predictions.contamination,
                features=features,
                metadata={
                    "throughput": stream.items_per_minute,
                    "stage_sensing_ms": detection_latency,
                    "stage_inference_ms": inference_latency,
                    "contamination_probability": sample.contamination_probability,
                },
            )
            self.confusion.update(predictions.category, sample.category)

            decision = self.decision_engine.decide(
                detection,
                stage_latencies=stage_latencies,
                remaining_budget_ms=latency_budget.remaining(),
                total_budget_ms=self.latency_budget.budget_ms,
            )
            decision_latency = self._simulate_stage_latency(self.stages[2])
            latency_budget.consume(decision_latency)
            decision.record_stage_latency("decision", decision_latency)

            await self.decision_engine.execute(decision)
            actuation_latency = self._simulate_stage_latency(self.stages[3])
            latency_budget.consume(actuation_latency)
            decision.record_stage_latency("actuation", actuation_latency)

            processed += 1
            self.throughput.update(stream.items_per_minute)
            self.monitoring.emit_detection(detection)
            self.monitoring.emit_decision(decision, remaining_budget_ms=latency_budget.remaining())
            await asyncio.sleep(max(0.0, stream.time_increment))

        self.monitoring.emit_summary(
            accuracy=self.confusion.accuracy(),
            throughput=self.throughput.average(),
            processed=processed,
        )

    def _simulate_stage_latency(self, stage: PipelineStage) -> float:
        """Simulate latency contributions with micro-variability."""

        jitter = random.uniform(-0.2, 0.2) * stage.latency_budget_ms
        latency = max(0.1, stage.latency_budget_ms + jitter)
        stage.metric.update(latency)
        return latency

    def evaluate(self) -> dict[str, float]:
        """Return evaluation metrics for reporting."""

        return {
            "accuracy": self.confusion.accuracy(),
            "throughput_avg": self.throughput.average(),
            "latency_budget_ms": self.latency_budget.budget_ms,
            "stage_latency_mean": {
                stage.name: stage.metric.mean() for stage in self.stages
            },
        }
