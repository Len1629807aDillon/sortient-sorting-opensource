"""Metric utilities for monitoring detection and sorting performance."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Tuple

from .types import MaterialCategory


@dataclass
class RollingMetric:
    """Maintains a rolling window statistic for latency or accuracy."""

    window: int
    values: List[float] = field(default_factory=list)

    def update(self, value: float) -> None:
        self.values.append(value)
        if len(self.values) > self.window:
            self.values.pop(0)

    def mean(self) -> float:
        if not self.values:
            return 0.0
        return sum(self.values) / len(self.values)

    def best(self) -> float:
        return max(self.values) if self.values else 0.0

    def worst(self) -> float:
        return min(self.values) if self.values else 0.0


@dataclass
class ConfusionMatrix:
    """Confusion matrix for multi-class classification."""

    categories: Tuple[MaterialCategory, ...]
    counts: Dict[Tuple[MaterialCategory, MaterialCategory], int] = field(default_factory=dict)

    def update(self, predicted: MaterialCategory, actual: MaterialCategory) -> None:
        key = (predicted, actual)
        self.counts[key] = self.counts.get(key, 0) + 1

    def accuracy(self) -> float:
        total = sum(self.counts.values())
        if total == 0:
            return 0.0
        correct = sum(
            count
            for (predicted, actual), count in self.counts.items()
            if predicted == actual
        )
        return correct / total

    def per_class_precision(self) -> Dict[MaterialCategory, float]:
        precision: Dict[MaterialCategory, float] = {}
        for category in self.categories:
            tp = self.counts.get((category, category), 0)
            predicted_total = sum(
                count for (predicted, _), count in self.counts.items() if predicted == category
            )
            precision[category] = tp / predicted_total if predicted_total else 0.0
        return precision

    def per_class_recall(self) -> Dict[MaterialCategory, float]:
        recall: Dict[MaterialCategory, float] = {}
        for category in self.categories:
            tp = self.counts.get((category, category), 0)
            actual_total = sum(
                count for (_, actual), count in self.counts.items() if actual == category
            )
            recall[category] = tp / actual_total if actual_total else 0.0
        return recall


@dataclass
class ThroughputTracker:
    """Monitors throughput metrics such as items per minute."""

    horizon: int
    processed: List[int] = field(default_factory=list)

    def update(self, items: int) -> None:
        self.processed.append(items)
        if len(self.processed) > self.horizon:
            self.processed.pop(0)

    def average(self) -> float:
        if not self.processed:
            return 0.0
        return sum(self.processed) / len(self.processed)

    def best_interval(self) -> int:
        return max(self.processed) if self.processed else 0


@dataclass
class LatencyBudget:
    """Tracks how much latency budget remains for a pipeline stage."""

    budget_ms: float
    consumed_ms: float = 0.0

    def consume(self, latency_ms: float) -> None:
        self.consumed_ms += latency_ms

    def remaining(self) -> float:
        return max(self.budget_ms - self.consumed_ms, 0.0)

    def reset(self) -> None:
        self.consumed_ms = 0.0
