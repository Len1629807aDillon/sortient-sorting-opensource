"""Activation functions used by the neural network."""

from __future__ import annotations

import math
from typing import Callable, List


def relu(x: float) -> float:
    return x if x > 0 else 0.0


def relu_derivative(x: float) -> float:
    return 1.0 if x > 0 else 0.0


def gelu(x: float) -> float:
    return 0.5 * x * (1.0 + math.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x ** 3)))


def gelu_derivative(x: float) -> float:
    c = math.sqrt(2.0 / math.pi)
    tanh_arg = c * (x + 0.044715 * x ** 3)
    tanh_val = math.tanh(tanh_arg)
    sech_sq = 1.0 - tanh_val ** 2
    return 0.5 * (1.0 + tanh_val) + 0.5 * x * sech_sq * c * (1 + 3 * 0.044715 * x ** 2)


def softmax(vector: List[float]) -> List[float]:
    max_value = max(vector)
    exps = [math.exp(v - max_value) for v in vector]
    total = sum(exps)
    return [exp_v / total for exp_v in exps]


ACTIVATIONS: dict[str, Callable[[float], float]] = {
    "relu": relu,
    "gelu": gelu,
}

ACTIVATION_DERIVATIVES: dict[str, Callable[[float], float]] = {
    "relu": relu_derivative,
    "gelu": gelu_derivative,
}
