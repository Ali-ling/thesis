"""Run fixed update-strength, decision, and oracle schedule ablations."""

from __future__ import annotations

import argparse
from pathlib import Path

from run_arena_experiments import aggregate_summary, read_log, summarize_run, write_csv
from train_ppo_arena import (
    ArenaPPOConfig,
    ArenaPPOTrainer,
    SCENARIOS,
    save_checkpoint,
)


STRATEGIES = {
    "fixed3": ("fixed", 3),
    "fixed6": ("fixed", 6),
    "fixed9": ("fixed", 9),
    "decision": ("decision", 6),
    "oracle": ("oracle", 6),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Update-strength ablation.")
    parser.add_argument("--scenarios", nargs="+", choices=SCENARIOS, default=["stable", "local_flip", "global_flip"])
    parser.add_argument("--strategies", nargs="+", choices=list(STRATEGIES), default=list(STRATEGIES))
    parser.add_argument("--seed-start", type=int, default=20270501)
    parser.add_argument("--seed-count", type=int, default=3)
    parser.add_argument("--pretrain-source", type=Path, default=Path("results/formal_v3"))
    parser.add_argument("--online-timesteps", type=int, default=16_000)
    parser.add_argument("--change-step", type=int, default=2_000)
    parser.add_argument("--rollout-steps", type=int, default=256)
    parser.add_argument("--minibatch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--entropy-coef", type=float, default=0.02)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--monitor-window", type=int, default=1_500)
    parser.add_argument("--monitor-interval", type=int, default=256)
    parser.add_argument("--monitor-min-visits", type=int, default=2)
    parser.add_argument("--bootstrap-samples", type=int, default=30)
    parser.add_argument("--bootstrap-block-length", type=int, default=10)
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
    parser.add_argument("--torch-threads", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("results/update_ablation"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.torch_threads > 0:
        import torch

        torch.set_num_threads(args.torch_threads)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for seed in range(args.seed_start, args.seed_start + args.seed_count):
        checkpoint = (
            args.pretrain_source / f"seed_{seed}" / f"seed_{seed}_pretrained.pt"
        )
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)
        for scenario in args.scenarios:
            for strategy in args.strategies:
                schedule_mode, epochs = STRATEGIES[strategy]
                config = ArenaPPOConfig(
                    timesteps=args.online_timesteps,
                    rollout_steps=args.rollout_steps,
                    update_epochs=epochs,
                    base_update_epochs=3,
                    strong_update_epochs=9,
                    minibatch_size=args.minibatch_size,
                    learning_rate=args.learning_rate,
                    entropy_coef=args.entropy_coef,
                    behavior_epsilon=0.0,
                    hidden_size=args.hidden_size,
                    seed=seed,
                    change_step=args.change_step,
                    monitor_window=args.monitor_window,
                    monitor_interval=args.monitor_interval,
                    monitor_min_visits=args.monitor_min_visits,
                    bootstrap_samples=args.bootstrap_samples,
                    bootstrap_block_length=args.bootstrap_block_length,
                    device=args.device,
                )
                prefix = f"seed_{seed}_{scenario}_{strategy}"
                log_path = args.output_dir / f"{prefix}.csv"
                checkpoint_path = args.output_dir / f"{prefix}.pt"
                if args.resume and log_path.exists() and checkpoint_path.exists():
                    summary = summarize_run(
                        read_log(log_path),
                        change_step=args.change_step,
                        strong_epochs=9,
                    )
                else:
                    trainer = ArenaPPOTrainer(
                        config,
                        scenario=scenario,
                        schedule_mode=schedule_mode,
                        checkpoint_input=checkpoint,
                    )
                    trainer.train()
                    trainer.write_log(log_path)
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
                        strong_epochs=9,
                    )
                rows.append(
                    {
                        "seed": seed,
                        "scenario": scenario,
                        "method": strategy,
                        **summary,
                    }
                )
                print(
                    f"seed={seed} scenario={scenario} strategy={strategy} "
                    f"final={summary['final_value']:.4f} "
                    f"strong={summary['strong_updates']} "
                    f"trigger={summary['trigger_latency']}"
                )
    summary = aggregate_summary(rows)
    write_csv(args.output_dir / "ablation_runs.csv", rows)
    write_csv(args.output_dir / "ablation_summary.csv", summary)
    print(f"runs={len(rows)} summary={args.output_dir / 'ablation_summary.csv'}")


if __name__ == "__main__":
    main()
