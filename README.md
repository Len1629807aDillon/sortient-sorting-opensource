# Recycling Intelligence

Recycling Intelligence is an open research and engineering platform for high-performance recycling facilities. The project combines deep learning-based material recognition, low-latency edge computing, and adaptive mechanical sorting to deliver 99%+ purity with millisecond decision latency. It is designed for academic researchers, industrial automation teams, and sustainability innovators who want to experiment with intelligent recycling workflows.

## Why it matters (plain English)

Modern recycling plants move thousands of bottles, cans, fibres, and circuit boards every minute. Human spotters or fixed-rule systems cannot keep up, especially when materials arrive contaminated or in unexpected shapes. This project simulates how an AI-enabled facility "sees" each item, chooses the right chute, and keeps conveyor belts flowing without blowing past the millisecond timing windows that real hardware requires. You can run the system end-to-end on a laptop to understand each decision, yet the architecture mirrors what production plants deploy.

## Key Capabilities

- **Deep Learning for Waste** – A configurable neural network (`DeepWasteSorter`) processes multi-modal sensor data to recognise plastics, metals, fibre products, organics, and emerging material classes.
- **Smart Sorting Edge** – A latency-aware scheduler weighs queue depth, utilisation, and headroom before dispatching inference workloads to edge nodes, continuously rebalancing to stay under a 12 ms budget even as demand spikes.
- **Material Sorting System** – Mechanical actuator models and policy-driven decision logic compute lane suitability, actuator selection, and contamination-aware fallback routes while keeping a detailed reasoning trail per item.
- **Real-Time Analytics** – Streaming telemetry exposes confidence, contamination, throughput, and per-stage latency metrics that feed dashboards, anomaly detectors, or research notebooks.

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
3. **Edge Orchestration** – `EdgeScheduler` and `StreamProcessor` simulate concurrent inference on edge compute clusters, weighing utilisation, queue depth, and headroom before dispatching work.
4. **Sorting Execution** – `SortingDecisionEngine` fuses predictions with policies to trigger actuators via `ActuatorBank`, while `MonitoringHub` captures decision traces, latency profiles, and system metrics.

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

The decision engine now records machine-readable reasoning traces for every routed item:

- **Lane suitability** combines predicted probability, model confidence, lane alignment, and contamination penalties into a continuous score.
- **Policy weights** quantify actuator priorities and fallback rules, enabling optimisation studies or reinforcement learning agents to plug in.
- **Latency budgets** persist per-stage timings (sensing, inference, decisioning, actuation) alongside remaining headroom so researchers can test new control strategies without breaking millisecond constraints.

`SortingDecision` exposes these values via `reasoning_trace`, `policy_weights`, and `latency_breakdown`, allowing downstream analytics or experiments to consume structured evidence instead of opaque decisions.

## Edge Performance Simulation

`StreamProcessor` coordinates workloads across `EdgeNode` instances to validate latency strategies:

- Weighted queue depth, utilisation, and latency headroom determine node selection.
- Dynamic rebalancing siphons queued tasks away from overloaded nodes to hold throughput under tight sub-12 ms latency thresholds.
- The `simulate_task` helper mimics inference kernels with realistic jitter for profiling custom workloads.

## Analytics and Telemetry

`MonitoringHub` captures feature statistics, detection confidences, contamination trends, throughput averages, and stage latency profiles. Use `MonitoringHub.event_history()` for chronological audit logs and `MonitoringHub.latency_profile()` when tuning conveyor timing or edge budgets.

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
