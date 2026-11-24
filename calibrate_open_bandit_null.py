"""Calibrate the click-rate tie tolerance using within-window placebo splits."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from analyze_open_bandit_decision import decide


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Open Bandit placebo calibration.")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/open_bandit/application_v2/aggregated.npz"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/open_bandit_application_v2/null_calibration.csv"),
    )
    parser.add_argument("--window-blocks", type=int, default=48)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20271103)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = np.load(args.data)
    counts = data["counts"]
    clicks = data["clicks"]
    windows = counts.shape[0] // args.window_blocks
    counts = counts[: windows * args.window_blocks].reshape(
        windows,
        args.window_blocks,
        counts.shape[1],
        counts.shape[2],
    )
    clicks = clicks[: windows * args.window_blocks].reshape(
        windows,
        args.window_blocks,
        clicks.shape[1],
        clicks.shape[2],
    )
    rng = np.random.default_rng(args.seed)
    tolerances = [
        0.00001,
        0.00002,
        0.00005,
        0.0001,
        0.0002,
        0.0005,
        0.001,
        0.002,
        0.005,
    ]
    rows: list[dict[str, object]] = []
    for tolerance in tolerances:
        impacts: list[float] = []
        unresolved_values: list[float] = []
        for window in range(windows):
            for _ in range(args.repeats):
                permutation = rng.permutation(args.window_blocks)
                half = args.window_blocks // 2
                first = permutation[:half]
                second = permutation[half:]
                impact, unresolved, _, _ = decide(
                    counts,
                    clicks,
                    counts[window, first].sum(axis=0),
                    clicks[window, first].sum(axis=0),
                    counts[window, second].sum(axis=0),
                    clicks[window, second].sum(axis=0),
                    prior_strength=50.0,
                    min_exposures=20,
                    tie_tolerance=tolerance,
                )
                impacts.append(impact)
                unresolved_values.append(unresolved)
        impacts_array = np.asarray(impacts)
        unresolved_array = np.asarray(unresolved_values)
        rows.append(
            {
                "tie_tolerance": tolerance,
                "placebo_runs": len(impacts),
                "mean_impact": float(impacts_array.mean()),
                "impact_p95": float(np.quantile(impacts_array, 0.95)),
                "high_impact_rate": float((impacts_array >= 0.55).mean()),
                "mean_unresolved_mass": float(unresolved_array.mean()),
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(row)
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
