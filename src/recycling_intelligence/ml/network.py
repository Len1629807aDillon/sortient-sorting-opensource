"""Neural network architecture for material recognition."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

from ..core.types import (
    ContaminationLevel,
    MaterialCategory,
    MaterialFeatureSpace,
    contamination_from_probability,
)
from .activations import softmax
from .layers import DenseLayer


@dataclass
class NetworkPrediction:
    category: MaterialCategory
    confidence: float
    contamination: ContaminationLevel


@dataclass
class DeepWasteSorter:
    """Configurable deep network for multi-modal material sorting."""

    feature_space: MaterialFeatureSpace
    hidden_layers: List[DenseLayer] = field(default_factory=list)
    output_layer: DenseLayer | None = None
    contamination_layer: DenseLayer | None = None
    categories: List[MaterialCategory] = field(
        default_factory=lambda: [
            MaterialCategory.PLASTIC,
            MaterialCategory.GLASS,
            MaterialCategory.METAL,
            MaterialCategory.PAPER,
            MaterialCategory.ORGANIC,
            MaterialCategory.E_WASTE,
            MaterialCategory.TEXTILE,
            MaterialCategory.UNKNOWN,
        ]
    )

    def __post_init__(self) -> None:
        if not self.hidden_layers:
            dims = [self.feature_space.dimensionality, 256, 128, 64]
            self.hidden_layers = [
                DenseLayer(dims[i], dims[i + 1], activation="gelu") for i in range(len(dims) - 1)
            ]
        if self.output_layer is None:
            self.output_layer = DenseLayer(self.hidden_layers[-1].output_dim, len(self.categories), activation="relu")
        if self.contamination_layer is None:
            self.contamination_layer = DenseLayer(self.hidden_layers[-1].output_dim, 1, activation="relu")

    @classmethod
    def from_feature_space(cls, feature_space: MaterialFeatureSpace) -> "DeepWasteSorter":
        return cls(feature_space=feature_space)

    def forward(self, inputs: List[float]) -> tuple[List[float], List[float], List[List[float]]]:
        activations = inputs
        hidden_states: List[List[float]] = []
        for layer in self.hidden_layers:
            activations = layer.forward(activations)
            hidden_states.append(activations)
        logits = self.output_layer.forward(activations)
        contamination_logits = self.contamination_layer.forward(activations)
        return logits, contamination_logits, hidden_states

    def predict(self, inputs: List[float]) -> NetworkPrediction:
        logits, contamination_logits, _ = self.forward(inputs)
        probabilities = softmax(logits)
        best_index = max(range(len(probabilities)), key=probabilities.__getitem__)
        category = self.categories[best_index]
        confidence = probabilities[best_index]
        contamination_probability = 1 / (1 + pow(2.71828, -contamination_logits[0]))
        contamination = contamination_from_probability(contamination_probability)
        return NetworkPrediction(category=category, confidence=confidence, contamination=contamination)

    def parameters(self) -> List[tuple[List[List[float]], List[float]]]:
        params = [layer.parameters() for layer in self.hidden_layers]
        if self.output_layer:
            params.append(self.output_layer.parameters())
        if self.contamination_layer:
            params.append(self.contamination_layer.parameters())
        return params

    def backward(self, grad_logits: List[float], learning_rate: float) -> None:
        grad = self.output_layer.backward(grad_logits, learning_rate)
        for layer in reversed(self.hidden_layers):
            grad = layer.backward(grad, learning_rate)
        # Contamination layer is trained with unsupervised regularization pushing towards low contamination
        contamination_grad = [0.01]  # encourages lower contamination scores
        self.contamination_layer.backward(contamination_grad, learning_rate)
