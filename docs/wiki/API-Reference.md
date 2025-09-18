# API Reference

## recycling_intelligence.core

### `RecyclingIntelligenceSystem`
- `train(dataset, epochs=3)` – Optimises the `DeepWasteSorter` model.
- `run_stream(stream, duration_seconds)` – Runs the asynchronous sensing → inference → decision pipeline.
- `evaluate()` – Returns accuracy, throughput, and latency statistics.

## recycling_intelligence.data

### `SyntheticWasteStream`
- Parameters: `items_per_minute`, `contamination_bias`, `feature_dim`.
- Methods: `next_item()`, `samples(count)`, `stream(duration_seconds)`.

### `generate_training_dataset(stream, samples)`
- Produces (feature, label) tuples for supervised training.

## recycling_intelligence.ml

### `DeepWasteSorter`
- `forward(features)` – Returns class logits, contamination logits, and hidden activations.
- `predict(features)` – Produces a structured prediction with category, confidence, contamination level.
- `backward(grad_logits, learning_rate)` – Performs backpropagation on the model.

### `Trainer`
- `train(dataset, epochs)` – High-level training loop with rolling loss metrics.

## recycling_intelligence.hpc

### `EdgeNode`
- `submit(task)` – Enqueues an asynchronous workload.
- `telemetry()` – Reports latency, utilisation, throughput, and buffer levels.

### `EdgeScheduler`
- `schedule(task)` – Assigns tasks to the most capable node.
- `rebalance()` – Lightweight fairness routine for congested nodes.

### `StreamProcessor`
- `process(tasks)` – Schedules a batch of tasks.
- `infer(count, workload_factory)` – Convenience helper for inference benchmarking.

## recycling_intelligence.sorting

### `SortingDecisionEngine`
- `decide(detection)` – Generates scored actions and reasoning trace.
- `execute(decision)` – Simulates actuator control with latency awareness.

### `SortingPolicy`
- `default()` – Returns baseline rules for the demo facility.

### `ActuatorBank`
- `register(actuator_id, cycle_time_ms)` – Adds mechanical components.
- `trigger(actuator_id, direction)` – Activates an actuator for a decision cycle.
