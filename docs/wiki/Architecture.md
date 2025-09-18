# Architecture

Recycling Intelligence is structured around four cooperating subsystems.

## 1. Material Intelligence Layer

- `DeepWasteSorter` neural network with configurable depth and feature space awareness.
- Supports new sensor embeddings by adjusting `MaterialFeatureSpace`.
- Contamination head predicts probability of soiling to influence downstream routing.

## 2. Edge Acceleration Layer

- `EdgeNode` models compute cells located along the conveyor.
- `EdgeScheduler` applies capacity-aware selection to maintain sub-12ms latency.
- `StreamProcessor` ingests inference batches and rebalances nodes as telemetry drifts.

## 3. Sorting Execution Layer

- `SortingPolicy` defines target lanes, actuators, and priorities per material.
- `SortingDecisionEngine` builds reasoning traces to justify action selection.
- `ActuatorBank` exposes mechanical primitives: air jets, pushers, electrostatics, ballistic separation.

## 4. Insight Layer

- `MonitoringHub` aggregates confidence, contamination, and latency metrics.
- Rolling statistics support quality dashboards and anomaly detection research.

### Data Flow

1. Synthetic waste items (or future sensor integrations) generate feature vectors.
2. `DeepWasteSorter` infers material category and contamination probability.
3. `SortingDecisionEngine` calculates lane assignments while respecting facility policies.
4. Actuator commands execute while `MonitoringHub` records telemetry for auditing and analytics.

Each stage exposes extension points for experimentation, from plugging in advanced machine learning models to integrating real PLC control pathways.
