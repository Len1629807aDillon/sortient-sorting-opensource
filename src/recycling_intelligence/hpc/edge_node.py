"""Edge compute node abstractions for low-latency processing."""

from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Deque, Iterable, List, Optional
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
    _latency_history: Deque[float] = field(default_factory=lambda: deque(maxlen=64))

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
            loop = asyncio.get_event_loop()
            start = loop.time()
            actual_latency = await task.coro_factory()
            if actual_latency is None:
                actual_latency = (loop.time() - start) * 1000
            duration = float(actual_latency)
            self._latency_budget.consume(duration)
            self._latency_history.append(duration)
            self.utilization = min(
                1.0,
                self.utilization * 0.75 + duration / max(1.0, self.latency_budget_ms * 1.5),
            )
            await asyncio.sleep(0)
        self._processing = False

    def telemetry(self) -> EdgeNodeTelemetry:
        queue_fill = len(self._queue) / self.max_queue
        throughput = max(0.0, (1 - queue_fill) * 2400)
        latency = self._latency_history[-1] if self._latency_history else self._latency_budget.consumed_ms
        return EdgeNodeTelemetry(
            node_id=self.node_id,
            latency_ms=latency,
            throughput_items_per_min=throughput,
            utilization=self.utilization,
            buffer_level=queue_fill,
        )

    def capacity_score(self) -> float:
        queue_factor = 1.0 - len(self._queue) / max(1, self.max_queue)
        util_factor = 1.0 - self.utilization
        latency_penalty = 0.0
        if self._latency_history:
            avg_latency = sum(self._latency_history) / len(self._latency_history)
            latency_penalty = min(1.0, avg_latency / max(1.0, self.latency_budget_ms))
        score = 0.6 * queue_factor + 0.4 * util_factor
        score *= 1.0 - 0.5 * latency_penalty
        return max(0.0, min(score, 1.0))

    def queue_depth(self) -> int:
        return len(self._queue)

    def steal_tasks(self, count: int) -> List[EdgeTask]:
        stolen: List[EdgeTask] = []
        while count > 0 and self._queue:
            stolen.append(self._queue.pop())
            count -= 1
        stolen.reverse()
        return stolen

    def accept_tasks_front(self, tasks: Iterable[EdgeTask]) -> None:
        incoming = list(tasks)
        if not incoming:
            return
        if len(incoming) + len(self._queue) > self.max_queue:
            raise RuntimeError(f"Node {self.node_id} cannot accept {len(incoming)} tasks (queue overflow)")
        for task in reversed(incoming):
            self._queue.appendleft(task)
        if not self._processing:
            asyncio.create_task(self._process_queue())


async def simulate_task(duration_ms: float) -> float:
    await asyncio.sleep(duration_ms / 1000.0)
    jitter = random.uniform(-0.15, 0.15) * duration_ms
    return max(0.1, duration_ms + jitter)
