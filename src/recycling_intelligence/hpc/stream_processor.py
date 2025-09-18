"""Stream processor combining scheduler and edge tasks."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Awaitable, Callable, Iterable, List

from .edge_node import EdgeNode, EdgeTask
from .scheduler import EdgeScheduler, SchedulingDecision


@dataclass
class StreamProcessor:
    """Coordinates asynchronous processing of material detections."""

    scheduler: EdgeScheduler

    async def process(self, tasks: Iterable[EdgeTask]) -> List[SchedulingDecision]:
        decisions: List[SchedulingDecision] = []
        for task in tasks:
            decision = await self.scheduler.schedule(task)
            decisions.append(decision)
        await self.scheduler.rebalance()
        return decisions

    @classmethod
    def with_nodes(cls, node_ids: Iterable[str]) -> "StreamProcessor":
        nodes = {node_id: EdgeNode(node_id=node_id) for node_id in node_ids}
        scheduler = EdgeScheduler(nodes=nodes)
        return cls(scheduler=scheduler)

    async def infer(self, count: int, workload_factory: Callable[[int], Awaitable[float]]) -> List[SchedulingDecision]:
        tasks = [
            EdgeTask(identifier=f"task_{i}", coro_factory=lambda i=i: workload_factory(i), estimated_latency_ms=4.0)
            for i in range(count)
        ]
        return await self.process(tasks)
