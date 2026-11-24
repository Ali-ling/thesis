"""Build a compact Open Bandit application dataset for change detection.

The output aggregates position-1 logged bandit data into:

* 15-minute temporal blocks;
* 16 calibration-period user-context clusters;
* top-20 actions plus an ``other`` action.

Only counts, clicks, and propensity sums are stored. The original 6 GB CSV is
not loaded into memory.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans


FEATURES = [f"user_feature_{index}" for index in range(4)] + ["position"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Open Bandit application data.")
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("data/open_bandit/full/bts_all.csv"),
    )
    parser.add_argument(
        "--audit",
        type=Path,
        default=Path("data/open_bandit/full/bts_all_audit.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/open_bandit/application"),
    )
    parser.add_argument("--context-clusters", type=int, default=16)
    parser.add_argument("--top-actions", type=int, default=20)
    parser.add_argument(
        "--item-context",
        type=Path,
        default=Path("data/open_bandit/full/item_context.csv"),
    )
    parser.add_argument("--action-clusters", type=int, default=0)
    parser.add_argument(
        "--positions",
        choices=["1", "2", "3", "all"],
        default="all",
    )
    parser.add_argument("--fit-sample", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=20271101)
    return parser.parse_args()


def timestamp_parts(value: str) -> tuple[str, int]:
    return value[:10], int(value[11:13]) * 4 + int(value[14:16]) // 15


def feature_maps(
    csv_path: Path,
    first_date: str,
    positions: str,
) -> dict[str, dict[str, int]]:
    maps = {feature: {} for feature in FEATURES}
    with csv_path.open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row["timestamp"][:10] != first_date:
                break
            if positions != "all" and row["position"] != positions:
                continue
            for feature in FEATURES:
                value = row[feature]
                if value not in maps[feature]:
                    maps[feature][value] = len(maps[feature]) + 1
    return maps


def one_hot_size(maps: dict[str, dict[str, int]]) -> int:
    return sum(len(mapping) + 1 for mapping in maps.values())


def encode_features(
    row: dict[str, str],
    maps: dict[str, dict[str, int]],
) -> np.ndarray:
    vector = np.zeros(one_hot_size(maps), dtype=np.float32)
    offset = 0
    for feature in FEATURES:
        index = maps[feature].get(row[feature], 0)
        vector[offset + index] = 1.0
        offset += len(maps[feature]) + 1
    return vector


def fit_context_map(
    csv_path: Path,
    first_date: str,
    maps: dict[str, dict[str, int]],
    positions: str,
    clusters: int,
    sample_limit: int,
    seed: int,
) -> dict[tuple[str, ...], int]:
    samples: list[np.ndarray] = []
    step = 1
    with csv_path.open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        rows_seen = 0
        for row in reader:
            if row["timestamp"][:10] != first_date:
                break
            if positions != "all" and row["position"] != positions:
                continue
            rows_seen += 1
            if rows_seen % step != 0:
                continue
            samples.append(encode_features(row, maps))
            if len(samples) >= sample_limit:
                break
    matrix = np.stack(samples)
    model = MiniBatchKMeans(
        n_clusters=clusters,
        random_state=seed,
        n_init=10,
        batch_size=4096,
        max_iter=100,
    )
    model.fit(matrix)
    combinations = list(
        itertools.product(*[list(mapping) + ["__UNK__"] for mapping in maps.values()])
    )
    combo_matrix = np.zeros(
        (len(combinations), one_hot_size(maps)),
        dtype=np.float32,
    )
    for index, combination in enumerate(combinations):
        offset = 0
        for feature, value in zip(FEATURES, combination):
            mapping = maps[feature]
            code = 0 if value == "__UNK__" else mapping[value]
            combo_matrix[index, offset + code] = 1.0
            offset += len(mapping) + 1
    labels = model.predict(combo_matrix)
    return dict(zip(combinations, labels.astype(int).tolist()))


def cluster_items(
    path: Path,
    clusters: int,
    seed: int,
) -> tuple[dict[int, int], np.ndarray, list[dict[str, str]]]:
    with path.open("r", newline="", encoding="utf-8-sig") as file:
        rows = list(csv.DictReader(file))
    numeric = np.asarray(
        [float(row["item_feature_0"]) for row in rows],
        dtype=float,
    )
    numeric = (numeric - numeric.mean()) / max(numeric.std(), 1e-12)
    categorical_values = {
        feature: sorted({row[feature] for row in rows})
        for feature in ["item_feature_1", "item_feature_2", "item_feature_3"]
    }
    width = 1 + sum(len(values) for values in categorical_values.values())
    matrix = np.zeros((len(rows), width), dtype=float)
    matrix[:, 0] = numeric
    offset = 1
    for feature, values in categorical_values.items():
        mapping = {value: index for index, value in enumerate(values)}
        for row_index, row in enumerate(rows):
            matrix[row_index, offset + mapping[row[feature]]] = 1.0
        offset += len(values)
    model = KMeans(n_clusters=clusters, n_init=20, random_state=seed)
    labels = model.fit_predict(matrix)
    item_to_cluster = {
        int(float(row["item_id"])): int(label)
        for row, label in zip(rows, labels)
    }
    return item_to_cluster, labels, rows


def main() -> None:
    args = parse_args()
    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    top_action_ids = [int(action) for action, _ in audit["top_actions"][: args.top_actions]]
    action_to_index = {action: index for index, action in enumerate(top_action_ids)}
    if args.action_clusters > 0:
        item_to_action, item_labels, _ = cluster_items(
            args.item_context,
            args.action_clusters,
            args.seed,
        )
        action_count = args.action_clusters
        other_index = None
    else:
        item_to_action = {}
        item_labels = np.asarray([])
        action_count = len(top_action_ids) + 1
        other_index = len(top_action_ids)
    dates = sorted(audit["dates"])
    date_to_index = {date: index for index, date in enumerate(dates)}
    first_date = dates[0]
    maps = feature_maps(args.csv, first_date, args.positions)
    context_map = fit_context_map(
        args.csv,
        first_date,
        maps,
        args.positions,
        args.context_clusters,
        args.fit_sample,
        args.seed,
    )

    block_count = len(dates) * 96
    counts = np.zeros(
        (block_count, args.context_clusters, action_count),
        dtype=np.int32,
    )
    clicks = np.zeros_like(counts)
    propensity_sums = np.zeros(
        (block_count, args.context_clusters, action_count),
        dtype=np.float64,
    )
    with args.csv.open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        for index, row in enumerate(reader, start=1):
            if args.positions != "all" and row["position"] != args.positions:
                continue
            date, time_block = timestamp_parts(row["timestamp"])
            block = date_to_index[date] * 96 + time_block
            combination = tuple(row[feature] for feature in FEATURES)
            context = context_map.get(combination, 0)
            action_id = int(float(row["item_id"]))
            if args.action_clusters > 0:
                action = item_to_action[action_id]
            else:
                action = action_to_index.get(action_id, int(other_index))
            counts[block, context, action] += 1
            clicks[block, context, action] += int(float(row["click"]))
            propensity_sums[block, context, action] += float(row["propensity_score"])
            if index % 1_000_000 == 0:
                print(f"read_rows={index}", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_dir / "aggregated.npz",
        counts=counts,
        clicks=clicks,
        propensity_sums=propensity_sums,
    )
    metadata = {
        "dates": dates,
        "blocks_per_day": 96,
        "block_minutes": 15,
        "window_blocks": 48,
        "top_action_ids": top_action_ids,
        "other_action_index": other_index,
        "action_clusters": args.action_clusters,
        "item_action_map": {
            str(item): action for item, action in sorted(item_to_action.items())
        },
        "item_cluster_labels": item_labels.astype(int).tolist(),
        "positions": args.positions,
        "context_clusters": args.context_clusters,
        "feature_maps": maps,
        "counts": counts.tolist(),
        "clicks": clicks.tolist(),
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(
            {key: value for key, value in metadata.items() if key not in {"counts", "clicks"}},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"output={args.output_dir / 'aggregated.npz'}")
    print(f"exposures={int(counts.sum())} clicks={int(clicks.sum())}")


if __name__ == "__main__":
    main()
