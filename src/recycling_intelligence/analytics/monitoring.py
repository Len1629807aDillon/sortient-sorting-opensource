"""Real-time monitoring instrumentation."""

from __future__ import annotations

import statistics
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Iterable, List

from ..core.metrics import RollingMetric
from ..core.types import ContaminationLevel, MaterialDetection, SortingDecision


@dataclass
class MonitoringEvent:
    event_type: str
    payload: Dict[str, float]


class MonitoringHub:
    """Captures and aggregates monitoring events across the system."""

    def __init__(self, window: int = 128) -> None:
        self.window = window
        self.events: Deque[MonitoringEvent] = deque(maxlen=window)
        self.feature_vectors: Deque[List[float]] = deque(maxlen=window)
        self.detection_metric = RollingMetric(window=window)
        self.confidence_metric = RollingMetric(window=window)
        self.contamination_metric = RollingMetric(window=window)
        self.throughput_metric = RollingMetric(window=window)
        self.latency_metrics: Dict[str, RollingMetric] = {}

    def log_feature_vector(self, features: List[float]) -> None:
        self.feature_vectors.append(features)

    def emit_detection(self, detection: MaterialDetection) -> None:
        contamination_value = self._contamination_to_numeric(detection.contamination)
        self.events.append(
            MonitoringEvent(
                event_type="detection",
                payload={
                    "confidence": detection.confidence,
                    "contamination": contamination_value,
                    "throughput": detection.metadata.get("throughput", 0.0),
                },
            )
        )
        self.detection_metric.update(detection.confidence)
        self.confidence_metric.update(detection.confidence)
        self.contamination_metric.update(contamination_value)
        throughput = detection.metadata.get("throughput")
        if throughput is not None:
            self.throughput_metric.update(float(throughput))
        for stage_key in ("stage_sensing_ms", "stage_inference_ms"):
            if stage_key in detection.metadata:
                self._record_latency(stage_key.replace("stage_", "").replace("_ms", ""), float(detection.metadata[stage_key]))

    def emit_decision(self, decision: SortingDecision, remaining_budget_ms: float) -> None:
        for stage, latency in decision.latency_breakdown.items():
            self._record_latency(stage, latency)
        self.events.append(
            MonitoringEvent(
                event_type="decision",
                payload={
                    "remaining_latency": remaining_budget_ms,
                    "probability": decision.best_action().probability,
                    "selected_lane": decision.best_action().target_lane,
                    "latency_profile": decision.latency_breakdown,
                },
            )
        )

    def emit_summary(self, accuracy: float, throughput: float, processed: int) -> None:
        self.events.append(
            MonitoringEvent(
                event_type="summary",
                payload={"accuracy": accuracy, "throughput": throughput, "processed": float(processed)},
            )
        )

    def feature_statistics(self) -> Dict[str, float]:
        if not self.feature_vectors:
            return {"mean": 0.0, "variance": 0.0}
        flattened = [value for vector in self.feature_vectors for value in vector]
        return {
            "mean": statistics.fmean(flattened),
            "variance": statistics.pvariance(flattened),
        }

    def event_history(self) -> List[MonitoringEvent]:
        return list(self.events)

    def latency_profile(self) -> Dict[str, float]:
        return {stage: metric.mean() for stage, metric in self.latency_metrics.items()}

    def throughput_average(self) -> float:
        return self.throughput_metric.mean()

    def _record_latency(self, stage: str, latency_ms: float) -> None:
        metric = self.latency_metrics.setdefault(stage, RollingMetric(window=self.window))
        metric.update(latency_ms)

    @staticmethod
    def _contamination_to_numeric(level: ContaminationLevel) -> float:
        return {
            ContaminationLevel.NONE: 0.0,
            ContaminationLevel.LOW: 0.25,
            ContaminationLevel.MEDIUM: 0.6,
            ContaminationLevel.HIGH: 0.9,
        }[level]
