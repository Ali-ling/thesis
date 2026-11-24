"""Create thesis-ready summaries for the formal matched PPO experiment."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from run_arena_experiments import write_csv


NUMERIC_FIELDS = [
    "pre_change_value",
    "post_change_min_value",
    "final_value",
    "final_minus_pre_change",
    "drop_from_pre_change",
    "trigger_latency",
    "recovery_delay",
    "trigger_events",
    "strong_updates",
    "mean_epochs_post_change",
]


def read_rows(path: Path) -> list[dict[str, object]]:
    with path.open("r", newline="", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def optional_float(value: object) -> float | None:
    if value in {"", None}:
        return None
    return float(value)


def group_summary(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in rows:
        groups.setdefault((str(row["scenario"]), str(row["method"])), []).append(row)

    output: list[dict[str, object]] = []
    for (scenario, method), group in sorted(groups.items()):
        summary: dict[str, object] = {
            "scenario": scenario,
            "method": method,
            "runs": len(group),
        }
        for field in NUMERIC_FIELDS:
            values = [
                value
                for row in group
                if (value := optional_float(row[field])) is not None
            ]
            summary[f"{field}_mean"] = float(np.mean(values)) if values else ""
            summary[f"{field}_std"] = (
                float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            ) if values else ""

        recovery_values = [
            value
            for row in group
            if (value := optional_float(row["recovery_timesteps"])) is not None
        ]
        summary["recovery_count"] = len(recovery_values)
        summary["recovery_rate"] = len(recovery_values) / len(group)
        summary["trigger_count"] = sum(
            optional_float(row["trigger_events"]) not in {None, 0.0}
            for row in group
        )
        summary["trigger_rate"] = summary["trigger_count"] / len(group)
        output.append(summary)
    return output


def paired_comparison(
    rows: list[dict[str, object]],
    *,
    scenario: str,
    left: str,
    right: str,
    samples: int = 10_000,
    seed: int = 20271101,
) -> dict[str, object]:
    left_rows = {
        int(row["seed"]): row
        for row in rows
        if row["scenario"] == scenario and row["method"] == left
    }
    right_rows = {
        int(row["seed"]): row
        for row in rows
        if row["scenario"] == scenario and row["method"] == right
    }
    seeds = sorted(set(left_rows) & set(right_rows))
    final_differences = np.asarray(
        [
            float(left_rows[item]["final_value"])
            - float(right_rows[item]["final_value"])
            for item in seeds
        ],
        dtype=np.float64,
    )
    recovery_differences = np.asarray(
        [
            float(left_rows[item]["recovery_delay"])
            - float(right_rows[item]["recovery_delay"])
            for item in seeds
            if left_rows[item]["recovery_delay"] not in {"", None}
            and right_rows[item]["recovery_delay"] not in {"", None}
        ],
        dtype=np.float64,
    )
    rng = np.random.default_rng(seed)
    if final_differences.size:
        bootstrap_means = np.asarray(
            [
                np.mean(
                    rng.choice(
                        final_differences,
                        size=final_differences.size,
                        replace=True,
                    )
                )
                for _ in range(samples)
            ]
        )
        lower = float(np.quantile(bootstrap_means, 0.025))
        upper = float(np.quantile(bootstrap_means, 0.975))
    else:
        lower = upper = float("nan")
    return {
        "scenario": scenario,
        "comparison": f"{left}_minus_{right}",
        "paired_runs": len(seeds),
        "final_difference_mean": float(np.mean(final_differences))
        if final_differences.size
        else "",
        "final_difference_std": float(np.std(final_differences, ddof=1))
        if final_differences.size > 1
        else 0.0,
        "final_difference_ci_lower": lower,
        "final_difference_ci_upper": upper,
        "paired_recovery_runs": int(recovery_differences.size),
        "recovery_delay_difference_mean": float(np.mean(recovery_differences))
        if recovery_differences.size
        else "",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize formal experiment results.")
    parser.add_argument(
        "--runs",
        type=Path,
        default=Path("results/formal_v3/pilot_runs.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/formal_v3/analysis"),
    )
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_rows(args.runs)
    scenarios = sorted({str(row["scenario"]) for row in rows})
    comparisons = [
        paired_comparison(
            rows,
            scenario=scenario,
            left="decision",
            right="fixed",
            samples=args.bootstrap_samples,
        )
        for scenario in scenarios
    ]
    summaries = group_summary(rows)
    write_csv(args.output_dir / "group_summary.csv", summaries)
    write_csv(args.output_dir / "decision_vs_fixed.csv", comparisons)
    print("scenario method final_mean final_std recovery_rate trigger_rate strong_updates")
    for row in summaries:
        print(
            f"{row['scenario']:>16} {row['method']:>10} "
            f"{row['final_value_mean']:>10.3f} {row['final_value_std']:>9.3f} "
            f"{row['recovery_rate']:>13.2f} {row['trigger_rate']:>12.2f} "
            f"{row['strong_updates_mean']:>14.2f}"
        )
    print(f"\nsummary={args.output_dir / 'group_summary.csv'}")
    print(f"paired={args.output_dir / 'decision_vs_fixed.csv'}")


if __name__ == "__main__":
    main()
