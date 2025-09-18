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
    )
    decision = system.decision_engine.decide(detection)
    assert decision.best_action().target_lane == "lane_plastic"

    async def execute():
        await system.decision_engine.execute(decision)

    asyncio.run(execute())
