"""Stream-audit the full Open Bandit BTS/All CSV without loading it in memory."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit full BTS/All data.")
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("data/open_bandit/full/bts_all.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/open_bandit/full/bts_all_audit.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    actions: Counter[int] = Counter()
    positions: Counter[int] = Counter()
    dates: Counter[str] = Counter()
    clicks = 0
    rows = 0
    propensity_sum = 0.0
    propensity_square_sum = 0.0
    propensity_min = math.inf
    propensity_max = -math.inf
    zero_propensities = 0
    timestamp_min = None
    timestamp_max = None
    with args.csv.open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        for row in reader:
            rows += 1
            action = int(float(row["item_id"]))
            position = int(float(row["position"]))
            propensity = float(row["propensity_score"])
            click = int(float(row["click"]))
            timestamp = row["timestamp"]
            actions[action] += 1
            positions[position] += 1
            dates[timestamp[:10]] += 1
            clicks += click
            propensity_sum += propensity
            propensity_square_sum += propensity * propensity
            propensity_min = min(propensity_min, propensity)
            propensity_max = max(propensity_max, propensity)
            zero_propensities += int(propensity <= 0.0)
            timestamp_min = timestamp if timestamp_min is None else min(timestamp_min, timestamp)
            timestamp_max = timestamp if timestamp_max is None else max(timestamp_max, timestamp)
            if rows % 1_000_000 == 0:
                print(f"rows={rows}", flush=True)
    mean_propensity = propensity_sum / max(1, rows)
    propensity_std = math.sqrt(
        max(0.0, propensity_square_sum / max(1, rows) - mean_propensity**2)
    )
    result = {
        "path": str(args.csv),
        "rows": rows,
        "columns": reader.fieldnames,
        "timestamp_min": timestamp_min,
        "timestamp_max": timestamp_max,
        "dates": dict(sorted(dates.items())),
        "unique_actions": len(actions),
        "top_actions": actions.most_common(20),
        "positions": dict(sorted(positions.items())),
        "clicks": clicks,
        "click_rate": clicks / max(1, rows),
        "propensity_min": propensity_min,
        "propensity_max": propensity_max,
        "propensity_mean": mean_propensity,
        "propensity_std": propensity_std,
        "zero_propensities": zero_propensities,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
