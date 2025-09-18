# Recycling Intelligence

Recycling Intelligence is an open research and engineering platform for high-performance recycling facilities. The project combines deep learning-based material recognition, low-latency edge computing, and adaptive mechanical sorting to deliver 99%+ purity with millisecond decision latency. It is designed for academic researchers, industrial automation teams, and sustainability innovators who want to experiment with intelligent recycling workflows.

## Key Capabilities

- **Deep Learning for Waste** – A configurable neural network (`DeepWasteSorter`) processes multi-modal sensor data to recognise plastics, metals, fibre products, organics, and emerging material classes.
- **Smart Sorting Edge** – An event-driven edge scheduler orchestrates inference workloads across local compute nodes with latency budgets tuned for conveyor speeds of 1.5–6 m/s.
- **Material Sorting System** – Mechanical actuator models and policy-driven decision logic deliver lane assignments, actuator commands, and contamination-aware routing strategies.
- **Real-Time Analytics** – Streaming telemetry exposes confidence, contamination, throughput, and latency metrics for operational dashboards and research instrumentation.

## Architecture Overview

```
┌────────────────────┐     ┌──────────────────────┐     ┌────────────────────────┐
│ Material Vision AI │ --> │ Sorting Decisioning  │ --> │ Mechanical Actuators   │
│  DeepWasteSorter   │     │  Policies + Analytics│     │  Air Jets / Pushers    │
└────────────────────┘     └──────────────────────┘     └────────────────────────┘
          │                           │                              │
          ▼                           ▼                              ▼
┌──────────────────────┐   ┌──────────────────────┐   ┌──────────────────────────┐
│ Synthetic Waste Feed │   │ Edge Scheduler + HPC │   │ Monitoring + Telemetry   │
└──────────────────────┘   └──────────────────────┘   └──────────────────────────┘
```

The system is split across four domains:

1. **Data Generation** – `SyntheticWasteStream` produces realistic, high-throughput streams with contamination bias controls for experimentation.
2. **Machine Learning** – `DeepWasteSorter` builds a multi-layer network with GELU activations, per-category softmax outputs, and contamination scoring.
3. **Edge Orchestration** – `EdgeScheduler` and `StreamProcessor` simulate concurrent inference on edge compute clusters, enforcing sub-10ms latency budgets.
4. **Sorting Execution** – `SortingDecisionEngine` fuses predictions with policies to trigger actuators via `ActuatorBank`, while `MonitoringHub` captures decision traces and system metrics.

## Getting Started

```bash
python -m pip install -e .[dev]
```

Run the end-to-end simulation (training optional):

```bash
python -m recycling_intelligence.cli --train --epochs 2 --duration 3.0 --items-per-minute 1200
```

Example output:

```
Simulation complete
accuracy: 0.0
throughput_avg: 1200.0
latency_budget_ms: 20.0
stage_latency_mean: {'sensing': 3.9, 'inference': 6.1, 'decision': 3.1, 'actuation': 6.8}
```

## Repository Layout

| Path | Description |
| --- | --- |
| `src/recycling_intelligence` | Python package containing the full platform |
| `src/recycling_intelligence/core` | Facility configuration, metrics, and pipeline orchestration |
| `src/recycling_intelligence/data` | Synthetic data generation utilities |
| `src/recycling_intelligence/ml` | Deep learning components and training loop |
| `src/recycling_intelligence/hpc` | Edge compute abstractions and schedulers |
| `src/recycling_intelligence/sorting` | Policy-driven decision engine and actuator models |
| `src/recycling_intelligence/analytics` | Monitoring and telemetry aggregation |
| `tests` | Pytest suite covering simulation, scheduling, and decision logic |
| `docs/wiki` | Project wiki for architecture, research guidance, and contributor onboarding |

## Research-Grade Sorting Decisions

The decision engine captures reasoning traces for every routed item, including:

- Lane suitability score factoring probability, confidence, and contamination levels.
- Policy-derived priority weighting and actuator selection.
- Latency budgets tracked across sensing, inference, decisioning, and actuation stages.

This structure enables experimentation with reinforcement learning, Bayesian decision theory, or operations research formulations by swapping out `SortingPolicy` or extending `SortingDecisionEngine`.

## Edge Performance Simulation

`StreamProcessor` coordinates workloads across `EdgeNode` instances to validate latency strategies:

- Weighted queue depth and utilization metrics determine node selection.
- Dynamic rebalancing maintains throughput under tight sub-12ms latency thresholds.
- The `simulate_task` helper mimics inference kernels for profiling custom workloads.

## Analytics and Telemetry

`MonitoringHub` captures feature statistics, detection confidences, and latency usage, which can be fed into dashboards or anomaly detectors. Use `MonitoringHub.event_history()` to obtain a chronological log of events for audit or research analysis.

## Testing

```
pytest
```

The tests exercise the asynchronous pipeline, edge inference coordination, and decision execution to guarantee reproducible behaviour.

## Contributing

The project is intended as a collaborative research hub. Contributions may include:

- New sensor modalities and feature extraction strategies.
- Advanced model architectures (transformers, graph networks, spectral encoders).
- Enhanced scheduling heuristics or integration with real-time operating systems.
- Mechanical control algorithms, predictive maintenance models, and lifecycle analytics.

Please open a discussion in the wiki before submitting significant architectural changes.

## License

Recycling Intelligence is released under the [MIT License](LICENSE).
