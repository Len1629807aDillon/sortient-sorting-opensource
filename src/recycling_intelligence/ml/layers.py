"""Neural network layers implemented with NumPy-like operations."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, List, Tuple

from .activations import ACTIVATIONS, ACTIVATION_DERIVATIVES, relu


@dataclass
class DenseLayer:
    """Fully connected layer with optional activation."""

    input_dim: int
    output_dim: int
    activation: str = "relu"

    def __post_init__(self) -> None:
        limit = (6 / (self.input_dim + self.output_dim)) ** 0.5
        self.weights = [
            [random.uniform(-limit, limit) for _ in range(self.input_dim)]
            for _ in range(self.output_dim)
        ]
        self.bias = [0.0 for _ in range(self.output_dim)]
        self.activation_fn: Callable[[float], float] = ACTIVATIONS.get(self.activation, relu)
        self.activation_derivative: Callable[[float], float] = ACTIVATION_DERIVATIVES.get(
            self.activation, lambda _: 1.0
        )

    def forward(self, inputs: List[float]) -> List[float]:
        self.last_inputs = inputs
        self.last_linear = []
        outputs: List[float] = []
        for idx, row in enumerate(self.weights):
            z = sum(w * i for w, i in zip(row, inputs)) + self.bias[idx]
            self.last_linear.append(z)
            outputs.append(self.activation_fn(z))
        self.last_outputs = outputs
        return outputs

    def backward(self, grad_output: List[float], learning_rate: float) -> List[float]:
        grad_input = [0.0 for _ in range(self.input_dim)]
        for i, row in enumerate(self.weights):
            activation_grad = self.activation_derivative(self.last_linear[i])
            grad = grad_output[i] * activation_grad
            for j in range(self.input_dim):
                grad_input[j] += grad * row[j]
                row[j] -= learning_rate * grad * self.last_inputs[j]
            self.bias[i] -= learning_rate * grad
        return grad_input

    def parameters(self) -> Tuple[List[List[float]], List[float]]:
        return self.weights, self.bias
