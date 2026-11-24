"""Calibrate the stable-environment PPO pretraining budget.

The script trains one stable PPO run per seed for the maximum requested
budget, then reports exact policy value at several earlier milestones. This
avoids selecting a budget from a single final value without seeing whether the
policy has plateaued.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from arena_env import NonStationaryArenaEnv
from train_ppo_arena import ArenaPPOConfig, ArenaPPOTrainer, save_checkpoint


def row_at_or_after(rows: list[dict[str, object]], timesteps: int) -> dict[str, object]:
    for row in rows:
        if int(row["timesteps"]) >= timesteps:
            return row
    return rows[-1]


def summarize_budget_rows(
    rows: list[dict[str, object]],
    *,
    seed: int,
    optimal_value: float,
    milestones: list[int],
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for milestone in milestones:
        row = row_at_or_after(rows, milestone)
        value = float(row["exact_policy_value"])
        output.append(
            {
                "seed": seed,
                "requested_timesteps": milestone,
                "actual_timesteps": int(row["timesteps"]),
                "exact_policy_value": value,
                "optimal_value": optimal_value,
                "optimality_ratio": value / optimal_value,
                "optimality_gap": optimal_value - value,
                "entropy": float(row["entropy"]),
            }
        )
    return output


def aggregate(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    milestones = sorted({int(row["requested_timesteps"]) for row in rows})
    output: list[dict[str, object]] = []
    for milestone in milestones:
        group = [row for row in rows if int(row["requested_timesteps"]) == milestone]
        output.append(
            {
                "timesteps": milestone,
                "runs": len(group),
                "mean_exact_policy_value": float(
                    np.mean([float(row["exact_policy_value"]) for row in group])
                ),
                "std_exact_policy_value": float(
                    np.std([float(row["exact_policy_value"]) for row in group])
                ),
                "mean_optimality_ratio": float(
                    np.mean([float(row["optimality_ratio"]) for row in group])
                ),
                "mean_optimality_gap": float(
                    np.mean([float(row["optimality_gap"]) for row in group])
                ),
                "mean_entropy": float(
                    np.mean([float(row["entropy"]) for row in group])
                ),
            }
        )
    return output


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty budget calibration")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate stable PPO budget.")
    parser.add_argument("--timesteps", type=int, default=16_000)
    parser.add_argument("--milestones", nargs="+", type=int, default=[2_000, 4_000, 8_000, 12_000, 16_000])
    parser.add_argument("--seed-start", type=int, default=20270101)
    parser.add_argument("--seed-count", type=int, default=3)
    parser.add_argument("--rollout-steps", type=int, default=256)
    parser.add_argument("--update-epochs", type=int, default=6)
    parser.add_argument("--minibatch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--entropy-coef", type=float, default=0.02)
    parser.add_argument("--behavior-epsilon", type=float, default=0.0)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
    parser.add_argument("--output-dir", type=Path, default=Path("results/budget_calibration"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.timesteps <= 0 or args.seed_count <= 0:
        raise ValueError("timesteps and seed_count must be positive")
    milestones = sorted(set(args.milestones + [args.timesteps]))
    if any(value <= 0 for value in milestones):
        raise ValueError("all milestones must be positive")

    env = NonStationaryArenaEnv(scenario="stable", emit_oracle_info=False)
    initial_state = int(env.protocol["initial_state"])
    optimal_value = float(max(env.true_q_values(0)[initial_state]))
    all_rows: list[dict[str, object]] = []
    for seed in range(args.seed_start, args.seed_start + args.seed_count):
        config = ArenaPPOConfig(
            timesteps=args.timesteps,
            rollout_steps=args.rollout_steps,
            update_epochs=args.update_epochs,
            base_update_epochs=3,
            strong_update_epochs=9,
            minibatch_size=args.minibatch_size,
            learning_rate=args.learning_rate,
            entropy_coef=args.entropy_coef,
            behavior_epsilon=args.behavior_epsilon,
            hidden_size=args.hidden_size,
            seed=seed,
            change_step=args.timesteps + 1,
            monitor_window=1_000,
            device=args.device,
        )
        trainer = ArenaPPOTrainer(
            config,
            scenario="stable",
            schedule_mode="fixed",
        )
        trainer.train()
        seed_dir = args.output_dir / f"seed_{seed}"
        trainer.write_log(seed_dir / "training.csv")
        save_checkpoint(
            seed_dir / "pretrained.pt",
            trainer.model,
            trainer.optimizer,
            config,
            "stable",
            trainer.timesteps,
        )
        seed_rows = summarize_budget_rows(
            trainer.rows,
            seed=seed,
            optimal_value=optimal_value,
            milestones=milestones,
        )
        all_rows.extend(seed_rows)
        final = seed_rows[-1]
        print(
            f"seed={seed} value={final['exact_policy_value']:.4f} "
            f"ratio={final['optimality_ratio']:.4f}"
        )

    aggregate_rows = aggregate(all_rows)
    write_csv(args.output_dir / "budget_runs.csv", all_rows)
    write_csv(args.output_dir / "budget_summary.csv", aggregate_rows)
    print("\ntimesteps mean_value mean_ratio mean_gap std_value")
    for row in aggregate_rows:
        print(
            f"{row['timesteps']:>9} "
            f"{row['mean_exact_policy_value']:>10.4f} "
            f"{row['mean_optimality_ratio']:>10.4f} "
            f"{row['mean_optimality_gap']:>8.4f} "
            f"{row['std_exact_policy_value']:>9.4f}"
        )
    print(f"\nsummary={args.output_dir / 'budget_summary.csv'}")


if __name__ == "__main__":
    main()
