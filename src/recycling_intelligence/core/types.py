"""Type definitions for the Recycling Intelligence platform."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Iterable, List, Mapping, Optional


class MaterialCategory(str, Enum):
    """Supported high-level material categories."""

    PLASTIC = "plastic"
    GLASS = "glass"
    METAL = "metal"
    PAPER = "paper"
    ORGANIC = "organic"
    E_WASTE = "e_waste"
    TEXTILE = "textile"
    UNKNOWN = "unknown"


class ContaminationLevel(str, Enum):
    """Contamination level classifications based on probability estimates."""

    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass(frozen=True)
class MaterialFeatureSpace:
    """Describes the dimensionality and semantic meaning of feature vectors."""

    dimensionality: int
    spectral_bands: int
    spatial_res: int
    sensor_fusion: List[str]


@dataclass
class MaterialDetection:
    """Represents a single detection result produced by the neural network."""

    material_id: str
    category: MaterialCategory
    confidence: float
    contamination: ContaminationLevel
    features: List[float] = field(default_factory=list)
    metadata: Dict[str, float] = field(default_factory=dict)

    def summary(self) -> str:
        return (
            f"Material {self.material_id} -> {self.category} "
            f"(confidence={self.confidence:.3f}, contamination={self.contamination})"
        )


@dataclass
class SortingAction:
    """Represents a mechanical action executed by the sorting subsystem."""

    target_lane: str
    actuator: str
    priority: int
    probability: float
    policy_weight: float = 1.0
    lane_score: float = 0.0


@dataclass
class SortingDecision:
    """Result of the decision engine combining AI predictions and policies."""

    detection: MaterialDetection
    actions: List[SortingAction]
    reasoning_trace: List[str] = field(default_factory=list)
    latency_breakdown: Dict[str, float] = field(default_factory=dict)
    policy_weights: Dict[str, float] = field(default_factory=dict)
    total_latency_budget_ms: float = 0.0

    def best_action(self) -> SortingAction:
        if not self.actions:
            raise ValueError("No actions available for decision")
        return max(self.actions, key=lambda action: action.probability)

    def record_stage_latency(self, stage: str, latency_ms: float) -> None:
        """Attach latency metrics for downstream analytics."""

        self.latency_breakdown[stage] = latency_ms

    def total_latency(self) -> float:
        return sum(self.latency_breakdown.values())


@dataclass
class EdgeNodeTelemetry:
    """Telemetry data produced by an edge compute node."""

    node_id: str
    latency_ms: float
    throughput_items_per_min: float
    utilization: float
    buffer_level: float


@dataclass
class FacilityConfiguration:
    """Configuration describing the capabilities of a recycling facility."""

    lanes: Mapping[str, MaterialCategory]
    max_throughput_per_min: int
    feature_space: MaterialFeatureSpace


@dataclass
class TrainingBatch:
    """Container for batched samples used in training."""

    features: List[List[float]]
    labels: List[MaterialCategory]

    def __post_init__(self) -> None:
        if len(self.features) != len(self.labels):
            raise ValueError("Number of feature vectors must match labels")


def category_distribution(labels: Iterable[MaterialCategory]) -> Dict[MaterialCategory, float]:
    """Compute normalized category distribution for diagnostic purposes."""

    counts: Dict[MaterialCategory, int] = {category: 0 for category in MaterialCategory}
    total = 0
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
        total += 1
    if total == 0:
        return {category: 0.0 for category in MaterialCategory}
    return {category: count / total for category, count in counts.items()}


def contamination_from_probability(probability: float) -> ContaminationLevel:
    """Derive contamination level from contamination probability."""

    if probability < 0.05:
        return ContaminationLevel.NONE
    if probability < 0.2:
        return ContaminationLevel.LOW
    if probability < 0.5:
        return ContaminationLevel.MEDIUM
    return ContaminationLevel.HIGH


DEFAULT_FEATURE_SPACE = MaterialFeatureSpace(
    dimensionality=128,
    spectral_bands=16,
    spatial_res=64,
    sensor_fusion=["rgb", "nir", "xrf", "tof"],
)
