"""Merge isolated per-seed pilot outputs into one run table and summary."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from run_arena_experiments import aggregate_summary, write_csv


def read_csv(path: Path) -> list[dict[str, object]]:
    with path.open("r", newline="", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge isolated pilot outputs.")
    parser.add_argument("input_dirs", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows: list[dict[str, object]] = []
    for directory in args.input_dirs:
        path = directory / "pilot_runs.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        rows.extend(read_csv(path))
    rows.sort(
        key=lambda row: (
            int(row["seed"]),
            str(row["scenario"]),
            str(row["method"]),
        )
    )
    summary = aggregate_summary(rows)
    write_csv(args.output_dir / "pilot_runs.csv", rows)
    write_csv(args.output_dir / "pilot_summary.csv", summary)
    print(f"runs={len(rows)} aggregate_rows={len(summary)}")
    print(f"summary={args.output_dir / 'pilot_summary.csv'}")


if __name__ == "__main__":
    main()
