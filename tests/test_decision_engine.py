import asyncio

from recycling_intelligence.core.types import ContaminationLevel, MaterialCategory, MaterialDetection
from recycling_intelligence.core.pipeline import RecyclingIntelligenceSystem


def test_decision_engine_selects_policy_lane():
    system = RecyclingIntelligenceSystem()
    detection = MaterialDetection(
        material_id="test",
        category=MaterialCategory.PLASTIC,
        confidence=0.95,
        contamination=ContaminationLevel.LOW,
        features=[0.1] * system.facility.feature_space.dimensionality,
        metadata={"contamination_probability": 0.2},
    )
    decision = system.decision_engine.decide(
        detection,
        stage_latencies={"sensing": 3.2, "inference": 4.1},
        remaining_budget_ms=12.7,
        total_budget_ms=20.0,
    )
    decision.record_stage_latency("decision", 2.0)
    decision.record_stage_latency("actuation", 3.0)
    assert decision.best_action().target_lane == "lane_plastic"
    assert any("suitability" in trace for trace in decision.reasoning_trace)
    assert "latency" in decision.reasoning_trace[-1]
    assert "decision" in decision.latency_breakdown
    assert decision.latency_breakdown["decision"] == 2.0

    async def execute():
        await system.decision_engine.execute(decision)

    asyncio.run(execute())
