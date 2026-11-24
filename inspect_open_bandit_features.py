"""Inspect Open Bandit user-feature cardinalities from the full CSV."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect Open Bandit features.")
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("data/open_bandit/full/bts_all.csv"),
    )
    parser.add_argument("--rows", type=int, default=200_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    columns = [f"user_feature_{index}" for index in range(4)]
    values = {column: Counter() for column in columns}
    affinity_summary: dict[str, tuple[float, float]] = {}
    with args.csv.open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        for index, row in enumerate(reader):
            if index >= args.rows:
                break
            for column in columns:
                values[column][row[column]] += 1
            for column in [name for name in reader.fieldnames or [] if name.startswith("user-item_affinity_")]:
                value = float(row[column])
                previous = affinity_summary.get(column)
                affinity_summary[column] = (
                    min(value, previous[0]) if previous else value,
                    max(value, previous[1]) if previous else value,
                )
    for column in columns:
        print(column, "unique=", len(values[column]))
        print(" top=", values[column].most_common(15))
    changed_affinities = [
        (column, bounds)
        for column, bounds in affinity_summary.items()
        if bounds[0] != bounds[1]
    ]
    print("affinity_columns_with_variation=", len(changed_affinities))
    print("affinity_examples=", changed_affinities[:10])


if __name__ == "__main__":
    main()
