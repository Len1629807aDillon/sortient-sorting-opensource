"""Adaptive scheduler for coordinating edge nodes."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

from .edge_node import EdgeNode, EdgeTask


@dataclass
class SchedulingDecision:
    node_id: str
    task_id: str
    predicted_latency_ms: float


@dataclass
class EdgeScheduler:
    """Distributes inference workloads across edge nodes with latency awareness."""

    nodes: Dict[str, EdgeNode]
    smoothing_factor: float = 0.6
    latency_threshold_ms: float = 12.0
    history: List[SchedulingDecision] = field(default_factory=list)

    def register_node(self, node: EdgeNode) -> None:
        self.nodes[node.node_id] = node

    def best_node(self) -> EdgeNode:
        return max(self.nodes.values(), key=lambda node: node.capacity_score())

    async def schedule(self, task: EdgeTask) -> SchedulingDecision:
        node = self.best_node()
        predicted_latency = max(1.0, node.capacity_score() * node.latency_budget_ms)
        await node.submit(task)
        decision = SchedulingDecision(node_id=node.node_id, task_id=task.identifier, predicted_latency_ms=predicted_latency)
        self.history.append(decision)
        if len(self.history) > 1000:
            self.history.pop(0)
        return decision

    async def rebalance(self) -> None:
        for node in self.nodes.values():
            telemetry = node.telemetry()
            if telemetry.latency_ms > self.latency_threshold_ms:
                await asyncio.sleep(0.001)

    def utilization_snapshot(self) -> Dict[str, float]:
        return {node_id: node.telemetry().utilization for node_id, node in self.nodes.items()}
