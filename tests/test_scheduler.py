import asyncio

from recycling_intelligence.hpc.edge_node import simulate_task
from recycling_intelligence.hpc.stream_processor import StreamProcessor


def test_stream_processor_infer():
    processor = StreamProcessor.with_nodes(["edge_a", "edge_b"])

    async def workload(index: int):
        return await simulate_task(2.0 + index)

    decisions = asyncio.run(processor.infer(5, workload))
    assert len(decisions) == 5
    assert all(decision.predicted_latency_ms >= 1.0 for decision in decisions)
    assert all(0.0 <= decision.queue_level <= 1.0 for decision in decisions)
