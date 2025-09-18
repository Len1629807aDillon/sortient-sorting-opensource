"""Loss functions for multi-class classification."""

from __future__ import annotations

import math
from typing import List

from ..core.types import MaterialCategory
from .activations import softmax


def categorical_cross_entropy(
    logits: List[float],
    target: MaterialCategory,
    categories: List[MaterialCategory],
) -> tuple[float, List[float]]:
    probabilities = softmax(logits)
    target_index = categories.index(target)
    loss = -math.log(max(probabilities[target_index], 1e-9))
    grad = probabilities[:]
    grad[target_index] -= 1.0
    return loss, grad
