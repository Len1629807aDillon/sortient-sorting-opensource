"""Command line interface for running simulations."""

from __future__ import annotations

import argparse
import asyncio
from typing import Any

from .core.pipeline import RecyclingIntelligenceSystem
from .data.generator import SyntheticWasteStream, generate_training_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Recycling Intelligence facility simulator")
    parser.add_argument("--train", action="store_true", help="Train the neural network before running")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs for training")
    parser.add_argument("--duration", type=float, default=5.0, help="Simulation duration in seconds")
    parser.add_argument("--items-per-minute", type=int, default=1800, help="Synthetic stream throughput")
    return parser


async def run_simulation(args: argparse.Namespace) -> None:
    system = RecyclingIntelligenceSystem()
    stream = SyntheticWasteStream(items_per_minute=args.items_per_minute)
    if args.train:
        dataset = generate_training_dataset(stream, samples=256)
        system.train(dataset, epochs=args.epochs)
    await system.run_stream(stream, duration_seconds=args.duration)
    metrics = system.evaluate()
    print("Simulation complete")
    for key, value in metrics.items():
        print(f"{key}: {value}")


def main(argv: list[str] | None = None) -> Any:
    parser = build_parser()
    args = parser.parse_args(argv)
    asyncio.run(run_simulation(args))


if __name__ == "__main__":  # pragma: no cover
    main()
