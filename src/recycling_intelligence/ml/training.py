"""Training utilities for the DeepWasteSorter network."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterable, List

from ..core.metrics import RollingMetric
from ..core.types import MaterialCategory
from .losses import categorical_cross_entropy
from .network import DeepWasteSorter


@dataclass
class TrainingStatistics:
    """Records loss and accuracy metrics across epochs."""

    loss_history: List[float]
    accuracy_history: List[float]


class Trainer:
    def __init__(self, model: DeepWasteSorter, learning_rate: float = 1e-3) -> None:
        self.model = model
        self.learning_rate = learning_rate
        self.loss_metric = RollingMetric(window=50)

    def train(
        self,
        dataset: Iterable[tuple[list[float], MaterialCategory]],
        epochs: int = 3,
    ) -> TrainingStatistics:
        dataset = list(dataset)
        if not dataset:
            raise ValueError("Dataset must contain at least one sample")
        loss_history: List[float] = []
        accuracy_history: List[float] = []
        for _ in range(epochs):
            random.shuffle(dataset)
            correct = 0
            total_loss = 0.0
            for features, target in dataset:
                logits, _, _ = self.model.forward(features)
                loss, grad_logits = categorical_cross_entropy(logits, target, self.model.categories)
                predicted_index = max(range(len(logits)), key=logits.__getitem__)
                if self.model.categories[predicted_index] == target:
                    correct += 1
                self.model.backward(grad_logits, self.learning_rate)
                total_loss += loss
            accuracy = correct / len(dataset)
            loss_history.append(total_loss / len(dataset))
            accuracy_history.append(accuracy)
            self.loss_metric.update(loss_history[-1])
        return TrainingStatistics(loss_history=loss_history, accuracy_history=accuracy_history)
