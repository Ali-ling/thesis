"""Inspect action probabilities and coverage risk for a saved arena policy."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from train_ppo_arena import ArenaActorCritic


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect a finite-arena PPO policy.")
    parser.add_argument("checkpoint", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state_count = int(checkpoint["state_count"])
    action_count = int(checkpoint["action_count"])
    hidden_size = int(checkpoint["config"]["hidden_size"])
    model = ArenaActorCritic(state_count, action_count, hidden_size)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    with torch.no_grad():
        distribution, values = model.distribution_and_value(
            torch.eye(state_count, dtype=torch.float32)
        )
        probabilities = distribution.probs.numpy()

    print("state probabilities values")
    for state in range(state_count):
        formatted = " ".join(f"{value:.4f}" for value in probabilities[state])
        print(f"S{state:<4} {formatted}  {float(values[state]):.4f}")
    minimum = float(np.min(probabilities, axis=1).min())
    coverage_per_1000 = minimum * 1_000
    print(f"minimum_action_probability={minimum:.6f}")
    print(f"expected_visits_per_action_at_H1000={coverage_per_1000:.2f}")


if __name__ == "__main__":
    main()
