import asyncio

from recycling_intelligence.core.pipeline import RecyclingIntelligenceSystem
from recycling_intelligence.data.generator import SyntheticWasteStream, generate_training_dataset


def test_pipeline_runs_event_loop():
    system = RecyclingIntelligenceSystem()
    stream = SyntheticWasteStream(seed=42, items_per_minute=600)
    dataset = generate_training_dataset(stream, samples=32)
    system.train(dataset, epochs=1)

    async def run():
        await system.run_stream(stream, duration_seconds=0.2)
        return system.evaluate()

    metrics = asyncio.run(run())
    assert metrics["latency_budget_ms"] > 0
    assert metrics["throughput_avg"] >= 0
