"""Adaptive scheduler for coordinating edge nodes."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

from .edge_node import EdgeNode, EdgeTask
from ..core.types import EdgeNodeTelemetry


@dataclass
class SchedulingDecision:
    node_id: str
    task_id: str
    predicted_latency_ms: float
    queue_level: float
    utilization: float


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
        scored_nodes = [self._score_node(node, task) for node in self.nodes.values()]
        scored_nodes.sort(key=lambda entry: entry[0], reverse=True)
        _, node, predicted_latency, telemetry = scored_nodes[0]
        await node.submit(task)
        decision = SchedulingDecision(
            node_id=node.node_id,
            task_id=task.identifier,
            predicted_latency_ms=predicted_latency,
            queue_level=telemetry.buffer_level,
            utilization=telemetry.utilization,
        )
        self.history.append(decision)
        if len(self.history) > 1000:
            self.history.pop(0)
        return decision

    async def rebalance(self) -> None:
        overloaded: List[EdgeNode] = []
        for node in self.nodes.values():
            telemetry = node.telemetry()
            if telemetry.latency_ms > self.latency_threshold_ms or telemetry.buffer_level > 0.6:
                overloaded.append(node)
        if not overloaded:
            return

        underloaded = [node for node in self.nodes.values() if node not in overloaded]
        underloaded.sort(key=lambda node: node.capacity_score(), reverse=True)

        for node in overloaded:
            spill = max(0, node.queue_depth() // 2)
            tasks = node.steal_tasks(spill)
            for task in tasks:
                assigned = False
                for target in underloaded:
                    try:
                        await target.submit(task)
                        assigned = True
                        break
                    except RuntimeError:
                        continue
                if not assigned:
                    node.accept_tasks_front([task])
            await asyncio.sleep(0)

    def utilization_snapshot(self) -> Dict[str, float]:
        return {node_id: node.telemetry().utilization for node_id, node in self.nodes.items()}

    def _score_node(self, node: EdgeNode, task: EdgeTask) -> tuple[float, EdgeNode, float, EdgeNodeTelemetry]:
        telemetry = node.telemetry()
        queue_factor = 1.0 - telemetry.buffer_level
        util_factor = 1.0 - telemetry.utilization
        latency_headroom = max(0.0, self.latency_threshold_ms - telemetry.latency_ms)
        predicted_latency = telemetry.latency_ms + task.estimated_latency_ms * (1 + telemetry.buffer_level)
        normalized_headroom = latency_headroom / max(1.0, self.latency_threshold_ms)
        score = 0.5 * queue_factor + 0.3 * util_factor + 0.2 * normalized_headroom
        if predicted_latency > self.latency_threshold_ms:
            overflow_ratio = (predicted_latency - self.latency_threshold_ms) / max(1.0, self.latency_threshold_ms)
            score -= overflow_ratio * 0.5
        score += node.capacity_score() * 0.1
        return score, node, predicted_latency, telemetry
