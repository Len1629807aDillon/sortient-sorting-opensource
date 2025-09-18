import asyncio

from recycling_intelligence.hpc.edge_node import simulate_task
from recycling_intelligence.hpc.stream_processor import StreamProcessor


def test_stream_processor_infer():
    processor = StreamProcessor.with_nodes(["edge_a", "edge_b"])

    async def workload(index: int):
        return await simulate_task(2.0 + index)

    latencies = asyncio.run(processor.infer(5, workload))
    assert len(latencies) == 5
    assert all(latency >= 1.0 for latency in latencies)
