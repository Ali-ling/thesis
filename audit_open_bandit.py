"""Audit downloaded Open Bandit sample files before formal analysis."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path


def find_column(fieldnames: list[str], candidates: list[str]) -> str:
    for name in candidates:
        if name in fieldnames:
            return name
    raise KeyError(f"none of {candidates} found in {fieldnames}")


def audit_log(path: Path) -> dict[str, object]:
    with path.open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        assert reader.fieldnames is not None
        timestamp_column = find_column(reader.fieldnames, ["timestamp", "time"])
        action_column = find_column(reader.fieldnames, ["item_id", "action"])
        click_column = find_column(reader.fieldnames, ["click", "is_click"])
        position_column = find_column(reader.fieldnames, ["position"])
        propensity_column = find_column(
            reader.fieldnames,
            ["propensity_score", "action_prob", "propensity"],
        )
        actions: Counter[int] = Counter()
        positions: Counter[int] = Counter()
        clicks = 0
        propensities: list[float] = []
        timestamps: list[str] = []
        rows = 0
        for row in reader:
            rows += 1
            actions[int(float(row[action_column]))] += 1
            positions[int(float(row[position_column]))] += 1
            clicks += int(float(row[click_column]))
            propensities.append(float(row[propensity_column]))
            timestamps.append(row[timestamp_column])
    return {
        "path": str(path),
        "rows": rows,
        "columns": reader.fieldnames,
        "timestamp_min": min(timestamps),
        "timestamp_max": max(timestamps),
        "unique_timestamps": len(set(timestamps)),
        "unique_actions": len(actions),
        "top_actions": actions.most_common(10),
        "positions": dict(sorted(positions.items())),
        "clicks": clicks,
        "click_rate": clicks / max(1, rows),
        "propensity_min": min(propensities),
        "propensity_max": max(propensities),
        "propensity_mean": sum(propensities) / max(1, len(propensities)),
        "zero_propensities": sum(value <= 0 for value in propensities),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit Open Bandit sample.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data/open_bandit/sample"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for name in ["bts_all.csv", "random_all.csv"]:
        result = audit_log(args.root / name)
        print(f"\n[{name}]")
        for key, value in result.items():
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
