"""Edge compute node abstractions for low-latency processing."""

from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Deque, Optional
from collections import deque

from ..core.metrics import LatencyBudget
from ..core.types import EdgeNodeTelemetry


@dataclass
class EdgeTask:
    """Represents a unit of work scheduled on an edge node."""

    identifier: str
    coro_factory: Callable[[], Awaitable[float]]
    estimated_latency_ms: float


@dataclass
class EdgeNode:
    """High-performance compute node optimized for recycling workloads."""

    node_id: str
    max_queue: int = 32
    latency_budget_ms: float = 10.0
    _queue: Deque[EdgeTask] = field(default_factory=deque)
    _processing: bool = False
    _latency_budget: LatencyBudget = field(default_factory=lambda: LatencyBudget(10.0))
    utilization: float = 0.0

    async def submit(self, task: EdgeTask) -> float:
        if len(self._queue) >= self.max_queue:
            raise RuntimeError(f"Node {self.node_id} queue overflow")
        self._queue.append(task)
        if not self._processing:
            asyncio.create_task(self._process_queue())
        return task.estimated_latency_ms

    async def _process_queue(self) -> None:
        self._processing = True
        while self._queue:
            task = self._queue.popleft()
            self._latency_budget.reset()
            self._latency_budget.consume(task.estimated_latency_ms)
            start = asyncio.get_event_loop().time()
            await task.coro_factory()
            duration = (asyncio.get_event_loop().time() - start) * 1000
            self.utilization = min(1.0, self.utilization * 0.8 + duration / (self.latency_budget_ms * 1.5))
            await asyncio.sleep(0)
        self._processing = False

    def telemetry(self) -> EdgeNodeTelemetry:
        queue_fill = len(self._queue) / self.max_queue
        throughput = max(0.0, (1 - queue_fill) * 2400)
        return EdgeNodeTelemetry(
            node_id=self.node_id,
            latency_ms=self._latency_budget.consumed_ms,
            throughput_items_per_min=throughput,
            utilization=self.utilization,
            buffer_level=queue_fill,
        )

    def capacity_score(self) -> float:
        score = (1.0 - len(self._queue) / self.max_queue) * (1.0 - self.utilization)
        return max(0.0, min(score, 1.0))


async def simulate_task(duration_ms: float) -> float:
    await asyncio.sleep(duration_ms / 1000.0)
    return duration_ms + random.uniform(-0.2, 0.2) * duration_ms
