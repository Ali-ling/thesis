"""Combine and analyze the fixed, decision, and oracle ablations."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from analyze_formal_results import group_summary, paired_comparison
from run_arena_experiments import write_csv


def read_rows(path: Path) -> list[dict[str, object]]:
    with path.open("r", newline="", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze update ablation.")
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("results/update_ablation"),
    )
    parser.add_argument(
        "--oracle-dir",
        type=Path,
        default=Path("results/update_ablation_oracle_v2"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/update_ablation/analysis"),
    )
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    for path in sorted(args.base_dir.glob("seed_*/ablation_runs.csv")):
        rows.extend(
            row for row in read_rows(path) if row["method"] != "oracle"
        )
    for path in sorted(args.oracle_dir.glob("seed_*/ablation_runs.csv")):
        rows.extend(read_rows(path))
    rows.sort(
        key=lambda row: (
            int(row["seed"]),
            str(row["scenario"]),
            str(row["method"]),
        )
    )
    summary = group_summary(rows)
    write_csv(args.output_dir / "ablation_runs.csv", rows)
    write_csv(args.output_dir / "ablation_summary.csv", summary)

    hindsight: list[dict[str, object]] = []
    for scenario in sorted({str(row["scenario"]) for row in rows}):
        fixed_rows = [
            row
            for row in summary
            if row["scenario"] == scenario
            and row["method"] in {"fixed3", "fixed6", "fixed9"}
        ]
        best = max(fixed_rows, key=lambda row: float(row["final_value_mean"]))
        decision = next(
            row
            for row in summary
            if row["scenario"] == scenario and row["method"] == "decision"
        )
        hindsight.append(
            {
                "scenario": scenario,
                "selected_fixed_strategy": best["method"],
                "best_fixed_final_mean": best["final_value_mean"],
                "decision_final_mean": decision["final_value_mean"],
                "decision_minus_best_fixed": float(decision["final_value_mean"])
                - float(best["final_value_mean"]),
                "best_fixed_recovery_delay": best["recovery_delay_mean"],
            }
        )
    write_csv(args.output_dir / "hindsight_oracle.csv", hindsight)

    comparison_specs = [
        ("fixed3", "fixed6"),
        ("fixed9", "fixed6"),
        ("decision", "fixed6"),
        ("oracle", "fixed6"),
        ("oracle", "decision"),
    ]
    comparisons = [
        paired_comparison(
            rows,
            scenario=scenario,
            left=left,
            right=right,
            samples=10_000,
        )
        for scenario in sorted({str(row["scenario"]) for row in rows})
        for left, right in comparison_specs
    ]
    write_csv(args.output_dir / "ablation_comparisons.csv", comparisons)

    print("scenario method final_mean final_std recovery_delay strong_updates")
    for row in summary:
        print(
            f"{row['scenario']:>16} {row['method']:>10} "
            f"{row['final_value_mean']:>10.3f} {row['final_value_std']:>9.3f} "
            f"{row['recovery_delay_mean']:>14} "
            f"{row['strong_updates_mean']:>14.2f}"
        )
    print(f"summary={args.output_dir / 'ablation_summary.csv'}")


if __name__ == "__main__":
    main()
