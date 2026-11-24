"""Decision-relevant change analysis on aggregated Open Bandit logs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Open Bandit decision analysis.")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/open_bandit/application/aggregated.npz"),
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("data/open_bandit/application/metadata.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/open_bandit_application/decision_windows.csv"),
    )
    parser.add_argument("--window-blocks", type=int, default=48)
    parser.add_argument("--prior-strength", type=float, default=50.0)
    parser.add_argument("--min-exposures", type=int, default=20)
    parser.add_argument("--tie-tolerance", type=float, default=0.0005)
    parser.add_argument("--delta0", type=float, default=0.20)
    parser.add_argument("--delta1", type=float, default=0.55)
    parser.add_argument("--max-unresolved", type=float, default=0.20)
    parser.add_argument("--bootstrap", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20271102)
    return parser.parse_args()


def decide(
    counts: np.ndarray,
    clicks: np.ndarray,
    reference_counts: np.ndarray,
    reference_clicks: np.ndarray,
    recent_counts: np.ndarray,
    recent_clicks: np.ndarray,
    *,
    prior_strength: float,
    min_exposures: int,
    tie_tolerance: float,
) -> tuple[float, float, int, int]:
    def action_values(window_counts: np.ndarray, window_clicks: np.ndarray) -> np.ndarray:
        action_counts = window_counts.sum(axis=0)
        action_clicks = window_clicks.sum(axis=0)
        prior = (action_clicks + 1.0) / (action_counts + 2.0)
        return (window_clicks + prior_strength * prior[None, :]) / (
            window_counts + prior_strength
        )

    ref_values = action_values(reference_counts, reference_clicks)
    recent_values = action_values(recent_counts, recent_clicks)
    ref_supported = reference_counts >= min_exposures
    recent_supported = recent_counts >= min_exposures
    supported = ref_supported & recent_supported
    comparable = supported.sum(axis=1) >= 2
    ref_masked = np.where(supported, ref_values, -np.inf)
    recent_masked = np.where(supported, recent_values, -np.inf)
    ref_actions = np.argmax(ref_masked, axis=1)
    recent_actions = np.argmax(recent_masked, axis=1)
    ref_top = np.take_along_axis(ref_masked, ref_actions[:, None], axis=1)[:, 0]
    recent_top = np.take_along_axis(
        recent_masked, recent_actions[:, None], axis=1
    )[:, 0]
    ref_second = np.partition(ref_masked, -2, axis=1)[:, -2]
    recent_second = np.partition(recent_masked, -2, axis=1)[:, -2]
    tied = (ref_top - ref_second <= tie_tolerance) | (
        recent_top - recent_second <= tie_tolerance
    )
    comparable &= ~tied
    flips = comparable & (ref_actions != recent_actions)
    context_occupancy = recent_counts.sum(axis=1).astype(float)
    context_occupancy /= context_occupancy.sum()
    impact = float(np.dot(context_occupancy, flips.astype(float)))
    unresolved_mass = float(
        np.dot(context_occupancy, (~comparable).astype(float))
    )
    return impact, unresolved_mass, int(flips.sum()), int((~comparable).sum())


def main() -> None:
    args = parse_args()
    data = np.load(args.data)
    counts = data["counts"]
    clicks = data["clicks"]
    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    total_blocks = counts.shape[0]
    window_count = total_blocks // args.window_blocks
    counts = counts[: window_count * args.window_blocks].reshape(
        window_count,
        args.window_blocks,
        counts.shape[1],
        counts.shape[2],
    )
    clicks = clicks[: window_count * args.window_blocks].reshape(
        window_count,
        args.window_blocks,
        clicks.shape[1],
        clicks.shape[2],
    )
    rng = np.random.default_rng(args.seed)
    rows: list[dict[str, object]] = []
    for transition in range(1, window_count):
        ref_counts = counts[transition - 1].sum(axis=0)
        ref_clicks = clicks[transition - 1].sum(axis=0)
        recent_counts = counts[transition].sum(axis=0)
        recent_clicks = clicks[transition].sum(axis=0)
        impact, unresolved, flips, unresolved_contexts = decide(
            counts,
            clicks,
            ref_counts,
            ref_clicks,
            recent_counts,
            recent_clicks,
            prior_strength=args.prior_strength,
            min_exposures=args.min_exposures,
            tie_tolerance=args.tie_tolerance,
        )
        bootstrap_values = []
        for _ in range(args.bootstrap):
            ref_indices = rng.integers(0, args.window_blocks, args.window_blocks)
            recent_indices = rng.integers(0, args.window_blocks, args.window_blocks)
            replicate, _, _, _ = decide(
                counts,
                clicks,
                counts[transition - 1, ref_indices].sum(axis=0),
                clicks[transition - 1, ref_indices].sum(axis=0),
                counts[transition, recent_indices].sum(axis=0),
                clicks[transition, recent_indices].sum(axis=0),
                prior_strength=args.prior_strength,
                min_exposures=args.min_exposures,
                tie_tolerance=args.tie_tolerance,
            )
            bootstrap_values.append(replicate)
        lower = float(np.quantile(bootstrap_values, 0.025))
        upper = float(np.quantile(bootstrap_values, 0.975))
        if unresolved <= args.max_unresolved and lower >= args.delta1:
            evidence = "high"
        elif unresolved <= args.max_unresolved and upper <= args.delta0:
            evidence = "low"
        else:
            evidence = "uncertain"
        ref_rate = float(ref_clicks.sum() / max(1, ref_counts.sum()))
        recent_rate = float(recent_clicks.sum() / max(1, recent_counts.sum()))
        standard_error = np.sqrt(
            ref_rate * (1.0 - ref_rate) / max(1, ref_counts.sum())
            + recent_rate * (1.0 - recent_rate) / max(1, recent_counts.sum())
        )
        rows.append(
            {
                "transition": transition,
                "reference_window": transition - 1,
                "recent_window": transition,
                "reference_click_rate": ref_rate,
                "recent_click_rate": recent_rate,
                "click_rate_difference": recent_rate - ref_rate,
                "standardized_click_shift": (recent_rate - ref_rate)
                / max(standard_error, 1e-12),
                "decision_impact": impact,
                "ci_lower": lower,
                "ci_upper": upper,
                "evidence_level": evidence,
                "unresolved_mass": unresolved,
                "changed_contexts": flips,
                "unresolved_contexts": unresolved_contexts,
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"windows={window_count} transitions={len(rows)} output={args.output}")
    for row in rows:
        print(
            f"t={row['transition']:02d} I={row['decision_impact']:.3f} "
            f"CI=[{row['ci_lower']:.3f},{row['ci_upper']:.3f}] "
            f"{row['evidence_level']} unresolved={row['unresolved_mass']:.3f}"
        )


if __name__ == "__main__":
    main()
