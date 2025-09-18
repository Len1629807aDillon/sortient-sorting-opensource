"""Synthetic data generation for recycling streams."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Iterable, Iterator, List, Sequence

from ..core.types import ContaminationLevel, MaterialCategory


@dataclass
class WasteSample:
    """Represents a synthetic piece of waste moving across the conveyor."""

    identifier: str
    category: MaterialCategory
    features: List[float]
    contamination_probability: float


class SyntheticWasteStream:
    """Generates realistic waste data for experimentation and testing."""

    def __init__(
        self,
        categories: Sequence[MaterialCategory] = tuple(MaterialCategory),
        seed: int | None = None,
        items_per_minute: int = 1800,
        contamination_bias: float = 0.1,
        feature_dim: int = 128,
    ) -> None:
        self.categories = list(categories)
        self.random = random.Random(seed)
        self.items_per_minute = items_per_minute
        self.contamination_bias = contamination_bias
        self.feature_dim = feature_dim
        self.current_time = 0.0
        self.time_increment = 60.0 / max(1, items_per_minute)
        self._counter = 0

    def _generate_feature_vector(self, category: MaterialCategory) -> List[float]:
        base_frequency = self.categories.index(category) + 1
        vector = [
            math.sin(base_frequency * 0.1 * idx) + self.random.uniform(-0.1, 0.1)
            for idx in range(self.feature_dim)
        ]
        for idx in range(0, self.feature_dim, 16):
            vector[idx] += base_frequency * 0.05
        return vector

    def _contamination_probability(self, category: MaterialCategory) -> float:
        base = self.random.betavariate(2.0, 12.0)
        adjustment = 0.0
        if category == MaterialCategory.ORGANIC:
            adjustment = 0.15
        elif category == MaterialCategory.PAPER:
            adjustment = 0.05
        return min(1.0, max(0.0, base * (1 + self.contamination_bias + adjustment)))

    def next_item(self) -> WasteSample:
        category = self.random.choices(
            population=self.categories,
            weights=self._category_weights(),
            k=1,
        )[0]
        features = self._generate_feature_vector(category)
        contamination_probability = self._contamination_probability(category)
        identifier = f"item_{self._counter}"
        self._counter += 1
        self.current_time += self.time_increment
        return WasteSample(
            identifier=identifier,
            category=category,
            features=features,
            contamination_probability=contamination_probability,
        )

    def samples(self, count: int) -> List[WasteSample]:
        return [self.next_item() for _ in range(count)]

    def stream(self, duration_seconds: float) -> Iterator[WasteSample]:
        start = self.current_time
        while self.current_time - start < duration_seconds:
            yield self.next_item()

    def _category_weights(self) -> List[float]:
        weights = []
        for category in self.categories:
            if category == MaterialCategory.UNKNOWN:
                weights.append(0.01)
            elif category == MaterialCategory.E_WASTE:
                weights.append(0.05)
            else:
                weights.append(1.0)
        return weights


def generate_training_dataset(
    stream: SyntheticWasteStream,
    samples: int,
) -> Iterable[tuple[list[float], MaterialCategory]]:
    """Generate a labelled dataset from the synthetic stream."""

    for _ in range(samples):
        sample = stream.next_item()
        yield sample.features, sample.category
