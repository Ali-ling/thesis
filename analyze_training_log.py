"""Summarize one Stage 3 training log with recovery and trigger metrics."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from run_arena_experiments import summarize_run


def read_log(path: Path) -> list[dict[str, object]]:
    with path.open("r", newline="", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize a training CSV.")
    parser.add_argument("log_path", type=Path)
    parser.add_argument("--change-step", type=int, required=True)
    parser.add_argument("--strong-epochs", type=int, default=9)
    parser.add_argument("--recovery-fraction", type=float, default=0.95)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = summarize_run(
        read_log(args.log_path),
        change_step=args.change_step,
        strong_epochs=args.strong_epochs,
        recovery_fraction=args.recovery_fraction,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
