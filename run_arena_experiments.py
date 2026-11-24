"""Run matched PPO adaptation arms from a shared pretrained checkpoint.

The orchestrator currently covers the Stage 3 pilot:

    fixed, periodic, CUSUM drift, and decision-evidence scheduling

Each seed is pretrained once in the stable environment. All four methods for
that seed then load the same checkpoint and receive the same initial policy.
"""

from __future__ import annotations

import argparse
import csv
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import torch

from train_ppo_arena import (
    ArenaPPOConfig,
    ArenaPPOTrainer,
    SCENARIOS,
    save_checkpoint,
)


METHODS = ["fixed", "periodic", "drift", "decision", "oracle"]


def summarize_run(
    rows: list[dict[str, object]],
    *,
    change_step: int,
    strong_epochs: int,
    recovery_fraction: float = 0.95,
) -> dict[str, object]:
    """Calculate policy, trigger, recovery, and update-cost summaries."""
    if not rows:
        raise ValueError("cannot summarize an empty run")
    base_rows = [row for row in rows if row["phase"] == "base"]
    updated_rows = [
        row for row in rows if row["phase"] in {"transition", "updated"}
    ]
    reference_value = (
        float(base_rows[-1]["exact_policy_value"])
        if base_rows
        else float(rows[0]["exact_policy_value"])
    )
    post_values = [float(row["exact_policy_value"]) for row in updated_rows]
    minimum_value = min(post_values) if post_values else reference_value
    final_value = float(rows[-1]["exact_policy_value"])

    trigger_rows = [
        row for row in updated_rows if int(row["controller_triggered"]) == 1
    ]
    first_trigger = trigger_rows[0] if trigger_rows else None
    trigger_timesteps = (
        int(first_trigger["timesteps"]) if first_trigger is not None else None
    )
    trigger_latency = (
        max(0, trigger_timesteps - change_step)
        if trigger_timesteps is not None
        else None
    )

    recovery_target = reference_value * recovery_fraction
    recovered = next(
        (
            row
            for row in updated_rows
            if float(row["exact_policy_value"]) >= recovery_target
        ),
        None,
    )
    recovery_timesteps = int(recovered["timesteps"]) if recovered else None
    return {
        "pre_change_value": reference_value,
        "post_change_min_value": minimum_value,
        "final_value": final_value,
        "final_minus_pre_change": final_value - reference_value,
        "drop_from_pre_change": reference_value - minimum_value,
        "recovery_target": recovery_target,
        "recovery_timesteps": recovery_timesteps
        if recovery_timesteps is not None
        else "",
        "recovery_delay": recovery_timesteps - change_step
        if recovery_timesteps is not None
        else "",
        "trigger_timesteps": trigger_timesteps
        if trigger_timesteps is not None
        else "",
        "trigger_latency": trigger_latency if trigger_latency is not None else "",
        "trigger_events": len(trigger_rows),
        "strong_updates": sum(
            int(row["ppo_epochs"]) == strong_epochs for row in rows
        ),
        "mean_epochs_post_change": float(
            np.mean([int(row["ppo_epochs"]) for row in updated_rows])
        )
        if updated_rows
        else 0.0,
        "detections": sum(int(row["detection_ran"]) for row in rows),
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty CSV")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate_summary(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in rows:
        key = (str(row["scenario"]), str(row["method"]))
        groups.setdefault(key, []).append(row)

    numeric_fields = [
        "pre_change_value",
        "post_change_min_value",
        "final_value",
        "final_minus_pre_change",
        "drop_from_pre_change",
        "trigger_latency",
        "trigger_events",
        "strong_updates",
        "mean_epochs_post_change",
        "detections",
    ]
    output: list[dict[str, object]] = []
    for (scenario, method), group in sorted(groups.items()):
        aggregate: dict[str, object] = {
            "scenario": scenario,
            "method": method,
            "runs": len(group),
        }
        for field in numeric_fields:
            values = [
                float(row[field])
                for row in group
                if row[field] not in {"", None}
            ]
            aggregate[f"mean_{field}"] = float(np.mean(values)) if values else ""
        output.append(aggregate)
    return output


def build_online_config(
    args: argparse.Namespace,
    *,
    timesteps: Optional[int] = None,
) -> ArenaPPOConfig:
    return ArenaPPOConfig(
        timesteps=args.online_timesteps if timesteps is None else timesteps,
        rollout_steps=args.rollout_steps,
        update_epochs=args.update_epochs,
        base_update_epochs=args.base_epochs,
        strong_update_epochs=args.strong_epochs,
        minibatch_size=args.minibatch_size,
        learning_rate=args.learning_rate,
        entropy_coef=args.entropy_coef,
        behavior_epsilon=args.behavior_epsilon,
        hidden_size=args.hidden_size,
        seed=args.seed,
        change_step=args.change_step,
        monitor_window=args.monitor_window,
        monitor_interval=args.monitor_interval,
        monitor_min_visits=args.monitor_min_visits,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_block_length=args.bootstrap_block_length,
        periodic_interval=args.periodic_interval,
        cusum_threshold=args.cusum_threshold,
        cusum_allowance=args.cusum_allowance,
        cusum_warmup=args.cusum_warmup,
        cusum_baseline_rate=args.cusum_baseline_rate,
        device=args.device,
    )


def read_log(path: Path) -> list[dict[str, object]]:
    with path.open("r", newline="", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def _run_one_seed(
    seed: int,
    scenarios: list[str],
    methods: list[str],
    args: argparse.Namespace,
) -> list[dict[str, object]]:
    requested_threads = int(getattr(args, "torch_threads", 0))
    torch.set_num_threads(
        requested_threads
        if requested_threads > 0
        else max(1, (os.cpu_count() or 1) // max(1, int(getattr(args, "workers", 1))))
    )
    output_dir = Path(args.output_dir)
    source_dir = (
        Path(args.checkpoint_source_dir)
        if getattr(args, "checkpoint_source_dir", None)
        else None
    )
    pretrain_config = build_online_config(
        args,
        timesteps=args.pretrain_timesteps,
    )
    pretrain_config = replace(
        pretrain_config,
        seed=seed,
        change_step=args.pretrain_timesteps + 1,
    )
    pretrain_path = (
        source_dir / f"seed_{seed}_pretrained.pt"
        if source_dir is not None
        else output_dir / f"seed_{seed}_pretrained.pt"
    )
    pretrain_log = (
        source_dir / f"seed_{seed}_pretrained.csv"
        if source_dir is not None
        else output_dir / f"seed_{seed}_pretrained.csv"
    )
    resume = bool(getattr(args, "resume", False))
    reuse_pretrain = (resume or source_dir is not None) and pretrain_path.exists()
    if not (reuse_pretrain and pretrain_log.exists()):
        pretrainer = ArenaPPOTrainer(
            pretrain_config,
            scenario="stable",
            schedule_mode="fixed",
        )
        pretrainer.train()
        pretrainer.write_log(pretrain_log)
        save_checkpoint(
            pretrain_path,
            pretrainer.model,
            pretrainer.optimizer,
            pretrain_config,
            "stable",
            pretrainer.timesteps,
        )
        final_value = float(pretrainer.rows[-1]["exact_policy_value"])
    else:
        final_value = float(read_log(pretrain_log)[-1]["exact_policy_value"])
    print(
        f"pretrain seed={seed} timesteps={args.pretrain_timesteps} "
        f"value={final_value:.4f}"
    )

    detailed: list[dict[str, object]] = []
    for scenario in scenarios:
        for method in methods:
            prefix = f"seed_{seed}_{scenario}_{method}"
            log_path = output_dir / f"{prefix}.csv"
            checkpoint_path = output_dir / f"{prefix}.pt"
            trajectory_path = output_dir / f"{prefix}.trajectory.csv.gz"
            save_trajectories = bool(getattr(args, "save_trajectories", False))
            if (
                resume
                and log_path.exists()
                and checkpoint_path.exists()
                and (not save_trajectories or trajectory_path.exists())
            ):
                summary = summarize_run(
                    read_log(log_path),
                    change_step=args.change_step,
                    strong_epochs=args.strong_epochs,
                )
            else:
                config = replace(
                    pretrain_config,
                    timesteps=args.online_timesteps,
                    change_step=args.change_step,
                )
                trainer = ArenaPPOTrainer(
                    config,
                    scenario=scenario,
                    schedule_mode=method,
                    checkpoint_input=pretrain_path,
                )
                trainer.train()
                trainer.write_log(log_path)
                if save_trajectories:
                    trainer.write_trajectory(trajectory_path)
                save_checkpoint(
                    checkpoint_path,
                    trainer.model,
                    trainer.optimizer,
                    config,
                    scenario,
                    trainer.timesteps,
                )
                summary = summarize_run(
                    trainer.rows,
                    change_step=args.change_step,
                    strong_epochs=args.strong_epochs,
                )
            detailed.append(
                {
                    "seed": seed,
                    "scenario": scenario,
                    "method": method,
                    **summary,
                }
            )
            print(
                f"adapt seed={seed} scenario={scenario} method={method} "
                f"final={summary['final_value']:.4f} "
                f"trigger={summary['trigger_latency']} "
                f"strong={summary['strong_updates']}"
            )
    return detailed


def run_experiments(
    *,
    seeds: Iterable[int],
    scenarios: Iterable[str],
    methods: Iterable[str],
    args: argparse.Namespace,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_list = list(seeds)
    scenario_list = list(scenarios)
    method_list = list(methods)
    detailed: list[dict[str, object]] = []
    workers = max(1, int(getattr(args, "workers", 1)))
    if workers > 1 and len(seed_list) > 1:
        try:
            with ProcessPoolExecutor(max_workers=min(workers, len(seed_list))) as pool:
                futures = [
                    pool.submit(
                        _run_one_seed,
                        seed,
                        scenario_list,
                        method_list,
                        args,
                    )
                    for seed in seed_list
                ]
                for future in as_completed(futures):
                    detailed.extend(future.result())
        except (OSError, PermissionError) as error:
            print(
                f"parallel_workers unavailable ({error}); "
                "falling back to sequential execution"
            )
            for seed in seed_list:
                detailed.extend(
                    _run_one_seed(seed, scenario_list, method_list, args)
                )
    else:
        for seed in seed_list:
            detailed.extend(
                _run_one_seed(seed, scenario_list, method_list, args)
            )
    detailed.sort(key=lambda row: (int(row["seed"]), str(row["scenario"]), str(row["method"])))

    aggregate = aggregate_summary(detailed)
    write_csv(output_dir / "pilot_runs.csv", detailed)
    write_csv(output_dir / "pilot_summary.csv", aggregate)
    return detailed, aggregate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run matched PPO adaptation baselines on the finite arena."
    )
    parser.add_argument("--scenarios", nargs="+", choices=SCENARIOS, default=["stable", "local_flip", "global_flip"])
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=METHODS)
    parser.add_argument("--seed-start", type=int, default=20261201)
    parser.add_argument("--seed-count", type=int, default=3)
    parser.add_argument("--pretrain-timesteps", type=int, default=8_000)
    parser.add_argument("--online-timesteps", type=int, default=6_000)
    parser.add_argument("--rollout-steps", type=int, default=256)
    parser.add_argument("--update-epochs", type=int, default=6)
    parser.add_argument("--base-epochs", type=int, default=3)
    parser.add_argument("--strong-epochs", type=int, default=9)
    parser.add_argument("--minibatch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--entropy-coef", type=float, default=0.02)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--behavior-epsilon", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=20261201, help=argparse.SUPPRESS)
    parser.add_argument("--change-step", type=int, default=2_000)
    parser.add_argument("--monitor-window", type=int, default=1_500)
    parser.add_argument("--monitor-interval", type=int, default=256)
    parser.add_argument("--monitor-min-visits", type=int, default=2)
    parser.add_argument("--bootstrap-samples", type=int, default=60)
    parser.add_argument("--bootstrap-block-length", type=int, default=10)
    parser.add_argument("--periodic-interval", type=int, default=1_024)
    parser.add_argument("--cusum-threshold", type=float, default=1.0)
    parser.add_argument("--cusum-allowance", type=float, default=0.1)
    parser.add_argument("--cusum-warmup", type=int, default=4)
    parser.add_argument("--cusum-baseline-rate", type=float, default=0.05)
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--torch-threads", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--checkpoint-source-dir", type=Path, default=None)
    parser.add_argument("--save-trajectories", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("results/stage3_pilot"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seeds = range(args.seed_start, args.seed_start + args.seed_count)
    detailed, aggregate = run_experiments(
        seeds=seeds,
        scenarios=args.scenarios,
        methods=args.methods,
        args=args,
    )
    print(f"\nruns={len(detailed)} aggregate_rows={len(aggregate)}")
    print(f"detail={args.output_dir / 'pilot_runs.csv'}")
    print(f"summary={args.output_dir / 'pilot_summary.csv'}")


if __name__ == "__main__":
    main()
