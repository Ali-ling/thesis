"""Tests for matched Stage 3 experiment orchestration and summaries."""

from __future__ import annotations

import unittest
from argparse import Namespace
from pathlib import Path

from run_arena_experiments import (
    aggregate_summary,
    run_experiments,
    summarize_run,
)


class ArenaExperimentOrchestratorTest(unittest.TestCase):
    def test_summary_calculates_trigger_and_recovery_metrics(self) -> None:
        rows = [
            {
                "phase": "base",
                "timesteps": 1000,
                "exact_policy_value": 10.0,
                "controller_triggered": 0,
                "ppo_epochs": 6,
                "detection_ran": 0,
            },
            {
                "phase": "updated",
                "timesteps": 2000,
                "exact_policy_value": 8.0,
                "controller_triggered": 0,
                "ppo_epochs": 6,
                "detection_ran": 1,
            },
            {
                "phase": "updated",
                "timesteps": 3000,
                "exact_policy_value": 9.6,
                "controller_triggered": 1,
                "ppo_epochs": 9,
                "detection_ran": 1,
            },
        ]

        summary = summarize_run(
            rows,
            change_step=1500,
            strong_epochs=9,
        )

        self.assertAlmostEqual(summary["pre_change_value"], 10.0)
        self.assertAlmostEqual(summary["post_change_min_value"], 8.0)
        self.assertEqual(summary["trigger_timesteps"], 3000)
        self.assertEqual(summary["trigger_latency"], 1500)
        self.assertEqual(summary["recovery_timesteps"], 3000)
        self.assertEqual(summary["strong_updates"], 1)

        aggregate = aggregate_summary(
            [
                {"scenario": "local_flip", "method": "fixed", **summary},
            ]
        )
        self.assertEqual(aggregate[0]["runs"], 1)

    def test_orchestrator_runs_matched_methods_from_shared_checkpoint(self) -> None:
        args = Namespace(
            seed=41,
            pretrain_timesteps=16,
            online_timesteps=32,
            rollout_steps=8,
            update_epochs=1,
            base_epochs=1,
            strong_epochs=2,
            minibatch_size=8,
            learning_rate=3e-4,
            entropy_coef=0.01,
            behavior_epsilon=0.0,
            hidden_size=16,
            change_step=16,
            monitor_window=8,
            monitor_interval=8,
            monitor_min_visits=1,
            bootstrap_samples=5,
            bootstrap_block_length=2,
            periodic_interval=8,
            cusum_threshold=1.0,
            cusum_allowance=0.1,
            cusum_warmup=0,
            cusum_baseline_rate=0.01,
            device="cpu",
            output_dir=Path("results/test_stage3_orchestrator"),
        )
        detailed, aggregate = run_experiments(
            seeds=[41],
            scenarios=["local_flip"],
            methods=["fixed", "decision"],
            args=args,
        )

        self.assertEqual(len(detailed), 2)
        self.assertEqual(len(aggregate), 2)
        self.assertTrue((args.output_dir / "pilot_runs.csv").exists())
        self.assertTrue((args.output_dir / "pilot_summary.csv").exists())
        self.assertTrue((args.output_dir / "seed_41_pretrained.pt").exists())


if __name__ == "__main__":
    unittest.main()
