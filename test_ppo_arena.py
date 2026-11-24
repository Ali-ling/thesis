"""Tests for the finite-state PPO integration."""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import torch

from train_ppo_arena import (
    ArenaActorCritic,
    ArenaPPOConfig,
    ArenaPPOTrainer,
    compute_gae,
    exact_policy_value,
    save_checkpoint,
)


class ArenaPPOTest(unittest.TestCase):
    def test_gae_resets_incomplete_path_at_episode_boundary(self) -> None:
        advantages, returns = compute_gae(
            rewards=np.array([1.0, 1.0], dtype=np.float32),
            values=np.array([0.0, 0.0], dtype=np.float32),
            dones=np.array([0.0, 1.0], dtype=np.float32),
            last_value=0.0,
            gamma=0.9,
            gae_lambda=0.5,
        )

        self.assertAlmostEqual(float(advantages[1]), 1.0, places=6)
        self.assertAlmostEqual(float(advantages[0]), 1.45, places=6)
        self.assertAlmostEqual(float(returns[0]), 1.45, places=6)

    def test_exact_policy_value_for_stationary_single_state(self) -> None:
        model = ArenaActorCritic(state_count=1, action_count=2, hidden_size=8)
        rewards = np.array([[1.0, 1.0]], dtype=np.float64)
        transitions = np.array([[[1.0], [1.0]]], dtype=np.float64)

        value = exact_policy_value(
            model,
            rewards,
            transitions,
            gamma=0.9,
            initial_state=0,
            device=torch.device("cpu"),
        )

        self.assertAlmostEqual(value, 10.0, places=4)

    def test_short_training_runs_and_writes_auditable_outputs(self) -> None:
        config = ArenaPPOConfig(
            timesteps=32,
            rollout_steps=8,
            update_epochs=1,
            base_update_epochs=1,
            strong_update_epochs=2,
            minibatch_size=8,
            hidden_size=16,
            seed=19,
            change_step=16,
            monitor_window=8,
            monitor_interval=8,
            monitor_min_visits=1,
            bootstrap_samples=10,
            bootstrap_block_length=2,
        )
        trainer = ArenaPPOTrainer(
            config,
            scenario="local_flip",
            schedule_mode="decision",
        )
        trainer.train()

        self.assertEqual(trainer.timesteps, 32)
        self.assertEqual(len(trainer.rows), 4)
        self.assertTrue(any(int(row["detection_ran"]) == 1 for row in trainer.rows))
        self.assertTrue(
            any(isinstance(row["evidence_level"], str) and row["evidence_level"] for row in trainer.rows)
        )

        root = Path("results/test_ppo_arena")
        trainer.write_log(root / "train.csv")
        trainer.write_trajectory(root / "trajectory.csv.gz")
        save_checkpoint(
            root / "model.pt",
            trainer.model,
            trainer.optimizer,
            config,
            "local_flip",
            trainer.timesteps,
        )
        self.assertTrue((root / "train.csv").exists())
        self.assertTrue((root / "trajectory.csv.gz").exists())
        self.assertTrue((root / "model.pt").exists())

    def test_fixed_schedule_is_reproducible(self) -> None:
        config = ArenaPPOConfig(
            timesteps=16,
            rollout_steps=8,
            update_epochs=1,
            base_update_epochs=1,
            strong_update_epochs=2,
            minibatch_size=8,
            hidden_size=16,
            seed=29,
            monitor_window=8,
            monitor_interval=8,
            monitor_min_visits=1,
            bootstrap_samples=5,
            bootstrap_block_length=2,
        )
        left = ArenaPPOTrainer(
            config,
            scenario="stable",
            schedule_mode="fixed",
        )
        left.train()
        right = ArenaPPOTrainer(
            config,
            scenario="stable",
            schedule_mode="fixed",
        )
        right.train()

        for left_parameter, right_parameter in zip(
            left.model.parameters(),
            right.model.parameters(),
        ):
            self.assertTrue(torch.equal(left_parameter, right_parameter))

    def test_oracle_schedule_logs_true_impact_without_crashing(self) -> None:
        config = ArenaPPOConfig(
            timesteps=16,
            rollout_steps=8,
            update_epochs=1,
            base_update_epochs=1,
            strong_update_epochs=2,
            minibatch_size=8,
            hidden_size=16,
            seed=37,
            change_step=8,
            monitor_window=8,
            monitor_interval=8,
            monitor_min_visits=1,
            bootstrap_samples=5,
            bootstrap_block_length=2,
        )
        trainer = ArenaPPOTrainer(
            config,
            scenario="stable",
            schedule_mode="oracle",
        )
        trainer.train()

        self.assertEqual(len(trainer.rows), 2)
        self.assertTrue(
            all(row["oracle_impact"] != "" for row in trainer.rows)
        )
        self.assertTrue(
            all(row["oracle_level"] in {"low", "uncertain", "high"} for row in trainer.rows)
        )


if __name__ == "__main__":
    unittest.main()
