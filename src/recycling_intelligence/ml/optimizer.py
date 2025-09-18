"""Adaptive optimizers for training the neural network."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Sequence, Tuple


@dataclass
class AdamLikeState:
    m: List[List[float]]
    v: List[List[float]]
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8
    step: int = 0


def initialize_state(parameters: Sequence[Tuple[List[List[float]], List[float]]]) -> AdamLikeState:
    m = [[0.0 for _ in layer_weights] for weights, _ in parameters for layer_weights in weights]
    v = [[0.0 for _ in layer_weights] for weights, _ in parameters for layer_weights in weights]
    return AdamLikeState(m=m, v=v)


@dataclass
class AdamOptimizer:
    """Minimal Adam optimizer implementation for dense layers."""

    learning_rate: float = 1e-3
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8
    state: AdamLikeState | None = None

    def step(
        self,
        parameters: Sequence[Tuple[List[List[float]], List[float]]],
        gradients: Sequence[Tuple[List[List[float]], List[float]]],
    ) -> None:
        if self.state is None:
            self.state = initialize_state(parameters)
        self.state.step += 1
        m_index = 0
        for (weights, bias), (grad_weights, grad_bias) in zip(parameters, gradients):
            for i, (row, grad_row) in enumerate(zip(weights, grad_weights)):
                for j, (w, g) in enumerate(zip(row, grad_row)):
                    m = self.state.m[m_index][j] = (
                        self.beta1 * self.state.m[m_index][j] + (1 - self.beta1) * g
                    )
                    v = self.state.v[m_index][j] = (
                        self.beta2 * self.state.v[m_index][j] + (1 - self.beta2) * (g ** 2)
                    )
                    m_hat = m / (1 - self.beta1 ** self.state.step)
                    v_hat = v / (1 - self.beta2 ** self.state.step)
                    row[j] -= self.learning_rate * m_hat / (v_hat ** 0.5 + self.epsilon)
                m_index += 1
            for i, (b, g) in enumerate(zip(bias, grad_bias)):
                bias[i] -= self.learning_rate * g
