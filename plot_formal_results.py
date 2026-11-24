"""Generate thesis-ready figures for the formal experiment."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


METHODS = ["fixed", "periodic", "drift", "decision"]
COLORS = {
    "fixed": "#4e79a7",
    "periodic": "#f28e2b",
    "drift": "#59a14f",
    "decision": "#e15759",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def to_float(value: str) -> float:
    return float("nan") if value in {"", None} else float(value)


def plot_final_values(
    summary: list[dict[str, str]],
    path: Path,
) -> None:
    scenarios = sorted({row["scenario"] for row in summary})
    width = 0.18
    positions = np.arange(len(scenarios))
    figure, axis = plt.subplots(figsize=(12, 5.5))
    for index, method in enumerate(METHODS):
        values = [
            to_float(
                next(
                    row["final_value_mean"]
                    for row in summary
                    if row["scenario"] == scenario and row["method"] == method
                )
            )
            for scenario in scenarios
        ]
        errors = [
            to_float(
                next(
                    row["final_value_std"]
                    for row in summary
                    if row["scenario"] == scenario and row["method"] == method
                )
            )
            for scenario in scenarios
        ]
        axis.bar(
            positions + (index - 1.5) * width,
            values,
            width,
            yerr=errors,
            capsize=2,
            label=method,
            color=COLORS[method],
        )
    axis.set_xticks(positions, scenarios, rotation=15)
    axis.set_ylabel("Final exact policy value")
    axis.set_title("Final policy value across matched seeds (mean ± std)")
    axis.legend(ncol=4)
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=300)
    plt.close(figure)


def plot_costs(
    summary: list[dict[str, str]],
    path: Path,
) -> None:
    scenarios = sorted({row["scenario"] for row in summary})
    positions = np.arange(len(scenarios))
    width = 0.18
    figure, (trigger_axis, epoch_axis) = plt.subplots(
        2,
        1,
        figsize=(12, 8),
        sharex=True,
    )
    for index, method in enumerate(METHODS):
        trigger_rates = [
            to_float(
                next(
                    row["trigger_rate"]
                    for row in summary
                    if row["scenario"] == scenario and row["method"] == method
                )
            )
            for scenario in scenarios
        ]
        strong_updates = [
            to_float(
                next(
                    row["strong_updates_mean"]
                    for row in summary
                    if row["scenario"] == scenario and row["method"] == method
                )
            )
            for scenario in scenarios
        ]
        trigger_axis.bar(
            positions + (index - 1.5) * width,
            trigger_rates,
            width,
            label=method,
            color=COLORS[method],
        )
        epoch_axis.bar(
            positions + (index - 1.5) * width,
            strong_updates,
            width,
            label=method,
            color=COLORS[method],
        )
    trigger_axis.set_ylabel("Trigger rate")
    trigger_axis.set_ylim(0, 1.05)
    trigger_axis.grid(axis="y", alpha=0.25)
    trigger_axis.legend(ncol=4)
    epoch_axis.set_ylabel("Mean strong updates")
    epoch_axis.set_xticks(positions, scenarios, rotation=15)
    epoch_axis.grid(axis="y", alpha=0.25)
    figure.suptitle("Trigger frequency and optimization cost")
    figure.tight_layout()
    figure.savefig(path, dpi=300)
    plt.close(figure)


def plot_paired_differences(
    paired: list[dict[str, str]],
    path: Path,
) -> None:
    scenarios = [row["scenario"] for row in paired]
    estimates = np.asarray(
        [to_float(row["final_difference_mean"]) for row in paired]
    )
    lower = np.asarray(
        [to_float(row["final_difference_ci_lower"]) for row in paired]
    )
    upper = np.asarray(
        [to_float(row["final_difference_ci_upper"]) for row in paired]
    )
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.errorbar(
        estimates,
        np.arange(len(scenarios)),
        xerr=np.vstack([estimates - lower, upper - estimates]),
        fmt="o",
        color="#e15759",
        capsize=4,
    )
    axis.axvline(0.0, color="black", linewidth=1)
    axis.set_yticks(np.arange(len(scenarios)), scenarios)
    axis.set_xlabel("Decision minus fixed final value (paired bootstrap 95% CI)")
    axis.set_title("Decision-evidence effect against fixed PPO")
    axis.grid(axis="x", alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=300)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot formal experiment results.")
    parser.add_argument(
        "--analysis-dir",
        type=Path,
        default=Path("results/formal_v3/analysis"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/formal_v3/figures"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = read_csv(args.analysis_dir / "group_summary.csv")
    paired = read_csv(args.analysis_dir / "decision_vs_fixed.csv")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_final_values(summary, args.output_dir / "final_policy_value.png")
    plot_costs(summary, args.output_dir / "trigger_and_cost.png")
    plot_paired_differences(
        paired,
        args.output_dir / "decision_vs_fixed_effect.png",
    )
    print(f"figures={args.output_dir}")


if __name__ == "__main__":
    main()
