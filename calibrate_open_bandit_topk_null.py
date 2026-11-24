"""Placebo calibration for Top-K Open Bandit decision changes."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from open_bandit_topk import fit_two_window_rates, topk_from_rates


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Top-K placebo calibration.")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/open_bandit/application_v2/aggregated.npz"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/open_bandit_application_v2/top3_null_calibration.csv"),
    )
    parser.add_argument("--window-blocks", type=int, default=48)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20271104)
    parser.add_argument(
        "--estimator",
        choices=["glm", "empirical"],
        default="glm",
    )
    parser.add_argument("--prior-strength", type=float, default=50.0)
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
    margins = [0.0, 0.00001, 0.00002, 0.00005, 0.0001, 0.0002, 0.0005, 0.001]
    collected = {margin: [] for margin in margins}
    unresolved = {margin: [] for margin in margins}
    half = args.window_blocks // 2
    for window in range(windows):
        for _ in range(args.repeats):
            permutation = rng.permutation(args.window_blocks)
            first = permutation[:half]
            second = permutation[half:]
            first_counts = counts[window, first].sum(axis=0)
            first_clicks = clicks[window, first].sum(axis=0)
            second_counts = counts[window, second].sum(axis=0)
            second_clicks = clicks[window, second].sum(axis=0)
            if args.estimator == "empirical":
                def rates(counts_value, clicks_value):
                    action_counts = counts_value.sum(axis=0)
                    action_clicks = clicks_value.sum(axis=0)
                    prior = (action_clicks + 1.0) / (action_counts + 2.0)
                    return (
                        clicks_value
                        + args.prior_strength * prior[None, :]
                    ) / (counts_value + args.prior_strength)

                ref_rates = rates(first_counts, first_clicks)
                recent_rates = rates(second_counts, second_clicks)
            else:
                ref_rates, recent_rates = fit_two_window_rates(
                    first_counts,
                    first_clicks,
                    second_counts,
                    second_clicks,
                )
            for margin in margins:
                result = topk_from_rates(
                    ref_rates,
                    recent_rates,
                    first_counts,
                    second_counts,
                    top_k=args.top_k,
                    min_exposures=20,
                    margin=margin,
                )
                collected[margin].append(result.impact)
                unresolved[margin].append(result.unresolved_mass)
    rows: list[dict[str, object]] = []
    for margin in margins:
        impacts = np.asarray(collected[margin])
        unresolved_values = np.asarray(unresolved[margin])
        for unresolved_limit in [0.20, 0.30, 0.40, 0.50]:
            high_rule = (impacts >= 0.55) & (
                unresolved_values <= unresolved_limit
            )
            rows.append(
                {
                    "margin": margin,
                    "unresolved_limit": unresolved_limit,
                    "placebo_runs": len(impacts),
                    "mean_impact": float(impacts.mean()),
                    "impact_p95": float(np.quantile(impacts, 0.95)),
                    "high_impact_rate": float(high_rule.mean()),
                    "mean_unresolved_mass": float(unresolved_values.mean()),
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
