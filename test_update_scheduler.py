"""Unit tests for evidence-driven PPO update scheduling."""

from __future__ import annotations

import unittest

from decision_detector import DecisionChangeDetector, DetectorConfig, TransitionRecord
from update_scheduler import (
    CusumDriftController,
    EvidenceDrivenScheduler,
    PeriodicUpdateController,
    UpdateSchedule,
)


class EvidenceDrivenSchedulerTest(unittest.TestCase):
    def test_disabled_scheduler_is_fixed_ppo(self) -> None:
        scheduler = EvidenceDrivenScheduler(
            UpdateSchedule(3, 6, 9),
            enabled=False,
            fixed_epochs=6,
        )
        self.assertEqual(scheduler.epochs_for(None), 6)

    def test_evidence_levels_map_to_epochs(self) -> None:
        config = DetectorConfig(
            n_states=2,
            n_actions=2,
            gamma=0.0,
            min_visits_per_state_action=2,
            tie_tolerance=0.01,
            bootstrap_samples=30,
            bootstrap_block_length=3,
            random_seed=5,
        )
        records = []
        flipped_records = []
        for _ in range(8):
            for state in range(2):
                for action in range(2):
                    records.append(
                        TransitionRecord(
                            state=state,
                            action=action,
                            reward=float(action == state),
                            next_state=state,
                        )
                    )
                    reward = float(action == (1 - state))
                    flipped_records.append(
                        TransitionRecord(
                            state=state,
                            action=action,
                            reward=reward,
                            next_state=state,
                        )
                    )
        low_result = DecisionChangeDetector(config).detect(records, records)
        self.assertEqual(low_result.evidence_level, "low")

        high_result = DecisionChangeDetector(config).detect(records, flipped_records)
        self.assertEqual(high_result.evidence_level, "high")

        scheduler = EvidenceDrivenScheduler(UpdateSchedule(3, 6, 9))
        self.assertEqual(scheduler.epochs_for(low_result), 3)
        self.assertEqual(scheduler.epochs_for(high_result), 9)
        self.assertEqual(scheduler.epochs_for(None), 6)
        self.assertEqual(scheduler.epochs_for_level("low"), 3)
        self.assertEqual(scheduler.epochs_for_level("uncertain"), 6)
        self.assertEqual(scheduler.epochs_for_level("high"), 9)

    def test_schedule_order_is_enforced(self) -> None:
        with self.assertRaises(ValueError):
            UpdateSchedule(base_epochs=6, standard_epochs=3, strong_epochs=9)

    def test_periodic_controller_triggers_once_per_interval(self) -> None:
        controller = PeriodicUpdateController(
            interval=100,
            schedule=UpdateSchedule(3, 6, 9),
        )

        self.assertEqual(controller.epochs_for(100), 9)
        self.assertEqual(controller.epochs_for(199), 6)
        self.assertEqual(controller.epochs_for(200), 9)

    def test_cusum_detects_positive_and_negative_residual_shocks(self) -> None:
        positive = CusumDriftController(
            threshold=1.0,
            allowance=0.1,
            warmup_steps=0,
            baseline_rate=0.0,
            schedule=UpdateSchedule(3, 6, 9),
        )
        self.assertFalse(positive.update(0.0))
        self.assertTrue(positive.update(1.2))
        self.assertEqual(positive.epochs_for(True), 9)
        self.assertEqual(positive.trigger_count, 1)

        negative = CusumDriftController(
            threshold=1.0,
            allowance=0.1,
            warmup_steps=0,
            baseline_rate=0.0,
            schedule=UpdateSchedule(3, 6, 9),
        )
        self.assertFalse(negative.update(0.0))
        self.assertTrue(negative.update(-1.2))

        decrease_only = CusumDriftController(
            threshold=1.0,
            allowance=0.1,
            warmup_steps=0,
            baseline_rate=0.0,
            detect_positive=False,
            detect_negative=True,
            schedule=UpdateSchedule(3, 6, 9),
        )
        self.assertFalse(decrease_only.update(0.0))
        self.assertFalse(decrease_only.update(1.2))
        self.assertTrue(decrease_only.update(-1.2))


if __name__ == "__main__":
    unittest.main()
