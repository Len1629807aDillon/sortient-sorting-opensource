# Getting Started

1. Clone the repository and install the package in editable mode:
   ```bash
   git clone https://github.com/your-org/recycling-intelligence.git
   cd recycling-intelligence
   python -m pip install -e .[dev]
   ```
2. Execute unit tests to confirm the environment:
   ```bash
   pytest
   ```
3. Run a simulation and observe telemetry:
   ```bash
   python -m recycling_intelligence.cli --train --epochs 1 --duration 2.0 --items-per-minute 900
   ```
4. Inspect the monitoring events:
   ```python
   from recycling_intelligence.core.pipeline import RecyclingIntelligenceSystem
   system = RecyclingIntelligenceSystem()
   stream = SyntheticWasteStream(seed=7)
   await system.run_stream(stream, duration_seconds=0.5)
   history = system.monitoring.event_history()
   ```

## Extending the System

- Modify `SortingPolicy.default()` to reflect the lane layout of your facility.
- Swap `SyntheticWasteStream` with a real sensor interface and feed features to `DeepWasteSorter`.
- Override `SortingDecisionEngine.decide` for advanced optimisation logic.
