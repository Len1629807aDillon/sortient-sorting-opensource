"""Real-time monitoring instrumentation."""

from __future__ import annotations

import statistics
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Iterable, List

from ..core.metrics import RollingMetric
from ..core.types import MaterialDetection, SortingDecision


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

    def log_feature_vector(self, features: List[float]) -> None:
        self.feature_vectors.append(features)

    def emit_detection(self, detection: MaterialDetection) -> None:
        self.events.append(
            MonitoringEvent(
                event_type="detection",
                payload={"confidence": detection.confidence, "contamination": float(detection.metadata.get("contamination", 0.0))},
            )
        )
        self.detection_metric.update(detection.confidence)

    def emit_decision(self, decision: SortingDecision, remaining_budget_ms: float) -> None:
        self.events.append(
            MonitoringEvent(
                event_type="decision",
                payload={"remaining_latency": remaining_budget_ms, "probability": decision.best_action().probability},
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
