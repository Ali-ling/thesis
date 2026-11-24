"""Inspect action support in aggregated Open Bandit application data."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect application support.")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/open_bandit/application_v2/aggregated.npz"),
    )
    parser.add_argument("--window-blocks", type=int, default=48)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = np.load(args.data)
    counts = data["counts"]
    windows = counts.shape[0] // args.window_blocks
    counts = counts[: windows * args.window_blocks].reshape(
        windows,
        args.window_blocks,
        counts.shape[1],
        counts.shape[2],
    ).sum(axis=1)
    for action in range(counts.shape[2]):
        context_counts = counts[:, :, action].sum(axis=0)
        window_counts = counts[:, :, action].sum(axis=1)
        print(
            f"action={action:02d} total={int(counts[:, :, action].sum())} "
            f"context_min={int(context_counts.min())} "
            f"context_max={int(context_counts.max())} "
            f"window_min={int(window_counts.min())} "
            f"window_max={int(window_counts.max())}"
        )


if __name__ == "__main__":
    main()
