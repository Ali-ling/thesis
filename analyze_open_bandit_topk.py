"""Estimate Top-K recommendation-set changes on Open Bandit windows."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from open_bandit_topk import empirical_topk_change, topk_change


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Open Bandit Top-K analysis.")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/open_bandit/application_v2/aggregated.npz"),
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("data/open_bandit/application_v2/metadata.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/open_bandit_application_v2/top3_windows.csv"),
    )
    parser.add_argument("--window-blocks", type=int, default=48)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--margin", type=float, default=0.0001)
    parser.add_argument("--min-exposures", type=int, default=20)
    parser.add_argument("--delta0", type=float, default=0.20)
    parser.add_argument("--delta1", type=float, default=0.55)
    parser.add_argument("--max-unresolved", type=float, default=0.40)
    parser.add_argument("--bootstrap", type=int, default=200)
    parser.add_argument(
        "--estimator",
        choices=["glm", "empirical"],
        default="empirical",
    )
    parser.add_argument("--prior-strength", type=float, default=50.0)
    parser.add_argument("--seed", type=int, default=20271105)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = np.load(args.data)
    counts = data["counts"]
    clicks = data["clicks"]
    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
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
    rows: list[dict[str, object]] = []
    for transition in range(1, windows):
        ref_counts = counts[transition - 1].sum(axis=0)
        ref_clicks = clicks[transition - 1].sum(axis=0)
        recent_counts = counts[transition].sum(axis=0)
        recent_clicks = clicks[transition].sum(axis=0)
        def estimate(
            reference_counts: np.ndarray,
            reference_clicks: np.ndarray,
            recent_counts: np.ndarray,
            recent_clicks: np.ndarray,
        ):
            if args.estimator == "empirical":
                return empirical_topk_change(
                    reference_counts,
                    reference_clicks,
                    recent_counts,
                    recent_clicks,
                    top_k=args.top_k,
                    min_exposures=args.min_exposures,
                    margin=args.margin,
                    prior_strength=args.prior_strength,
                )
            return topk_change(
                reference_counts,
                reference_clicks,
                recent_counts,
                recent_clicks,
                top_k=args.top_k,
                min_exposures=args.min_exposures,
                margin=args.margin,
            )

        point = estimate(
            ref_counts,
            ref_clicks,
            recent_counts,
            recent_clicks,
        )
        bootstrap_values: list[float] = []
        for _ in range(args.bootstrap):
            ref_indices = rng.integers(0, args.window_blocks, args.window_blocks)
            recent_indices = rng.integers(0, args.window_blocks, args.window_blocks)
            try:
                replicate = estimate(
                    counts[transition - 1, ref_indices].sum(axis=0),
                    clicks[transition - 1, ref_indices].sum(axis=0),
                    counts[transition, recent_indices].sum(axis=0),
                    clicks[transition, recent_indices].sum(axis=0),
                )
            except Exception:
                continue
            bootstrap_values.append(replicate.impact)
        lower = float(np.quantile(bootstrap_values, 0.025))
        upper = float(np.quantile(bootstrap_values, 0.975))
        if (
            point.unresolved_mass <= args.max_unresolved
            and lower >= args.delta1
        ):
            evidence = "high"
        elif (
            point.unresolved_mass <= args.max_unresolved
            and upper <= args.delta0
        ):
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
                "topk_impact": point.impact,
                "ci_lower": lower,
                "ci_upper": upper,
                "evidence_level": evidence,
                "unresolved_mass": point.unresolved_mass,
                "changed_contexts": point.changed_contexts,
                "unresolved_contexts": point.unresolved_contexts,
                "valid_bootstrap_runs": len(bootstrap_values),
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"windows={windows} transitions={len(rows)} output={args.output}")
    for row in rows:
        print(
            f"t={row['transition']:02d} I={row['topk_impact']:.3f} "
            f"CI=[{row['ci_lower']:.3f},{row['ci_upper']:.3f}] "
            f"{row['evidence_level']} unresolved={row['unresolved_mass']:.3f}"
        )


if __name__ == "__main__":
    main()
