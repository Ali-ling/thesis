"""Post-hoc threshold sensitivity from saved formal detection logs.

This analysis does not retrain PPO and must not be used to rewrite the primary
pre-registered result. It shows the recall-specificity trade-off implied by
different high-impact thresholds and unresolved-mass limits.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from run_arena_experiments import write_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Threshold sensitivity analysis.")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/formal_v3"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/formal_v3/analysis/threshold_sensitivity.csv"),
    )
    parser.add_argument("--change-step", type=int, default=2_000)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def main() -> None:
    args = parse_args()
    runs: list[dict[str, object]] = []
    for path in sorted(args.results_dir.glob("seed_*/seed_*_decision.csv")):
        rows = [
            row
            for row in read_rows(path)
            if row["detection_ran"] == "1"
            and int(row["timesteps"]) >= args.change_step
        ]
        if not rows:
            continue
        runs.append(
            {
                "scenario": "_".join(path.stem.split("_")[2:-1]),
                "rows": rows,
            }
        )

    output: list[dict[str, object]] = []
    for delta1 in [0.45, 0.50, 0.55, 0.60, 0.65]:
        for unresolved_limit in [0.20, 0.25, 0.30, 0.35]:
            for scenario in sorted({str(run["scenario"]) for run in runs}):
                scenario_runs = [
                    run for run in runs if str(run["scenario"]) == scenario
                ]
                triggered_latencies: list[int] = []
                for run in scenario_runs:
                    first = next(
                        (
                            row
                            for row in run["rows"]
                            if float(row["ci_lower"]) >= delta1
                            and float(row["unresolved_mass"]) <= unresolved_limit
                        ),
                        None,
                    )
                    if first is not None:
                        triggered_latencies.append(
                            int(first["timesteps"]) - args.change_step
                        )
                output.append(
                    {
                        "delta1": delta1,
                        "unresolved_limit": unresolved_limit,
                        "scenario": scenario,
                        "runs": len(scenario_runs),
                        "triggered_runs": len(triggered_latencies),
                        "trigger_rate": len(triggered_latencies)
                        / max(1, len(scenario_runs)),
                        "mean_latency": sum(triggered_latencies)
                        / len(triggered_latencies)
                        if triggered_latencies
                        else "",
                    }
                )
    write_csv(args.output, output)
    print(f"rows={len(output)} output={args.output}")


if __name__ == "__main__":
    main()
