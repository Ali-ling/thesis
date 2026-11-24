"""Unit tests for the decision-relevant change detector."""

from __future__ import annotations

import unittest

import numpy as np

from decision_detector import (
    DecisionChangeDetector,
    DetectorConfig,
    TransitionRecord,
)


def repeating_records(
    *,
    states: list[int],
    next_states: dict[int, int],
    rewards: list[list[float]],
    repeats: int,
) -> list[TransitionRecord]:
    records: list[TransitionRecord] = []
    for state in states:
        for action in range(len(rewards[state])):
            for _ in range(repeats):
                records.append(
                    TransitionRecord(
                        state=state,
                        action=action,
                        reward=rewards[state][action],
                        next_state=next_states[state],
                    )
                )
    return records


class DecisionChangeDetectorTest(unittest.TestCase):
    def test_detects_local_reward_flip_with_occupancy_weighting(self) -> None:
        config = DetectorConfig(
            n_states=2,
            n_actions=2,
            gamma=0.0,
            min_visits_per_state_action=3,
            tie_tolerance=0.01,
            bootstrap_samples=50,
            bootstrap_block_length=4,
            random_seed=7,
        )
        reference = repeating_records(
            states=[0, 1],
            next_states={0: 0, 1: 1},
            rewards=[[2.0, 1.0], [0.0, 1.0]],
            repeats=10,
        )
        recent = repeating_records(
            states=[0] * 8 + [1] * 2,
            next_states={0: 0, 1: 1},
            rewards=[[1.0, 2.0], [0.0, 1.0]],
            repeats=5,
        )

        result = DecisionChangeDetector(config).detect(reference, recent)

        self.assertEqual(result.confirmed_flip_states, [0])
        self.assertAlmostEqual(result.impact, 0.8, places=7)
        self.assertAlmostEqual(result.unresolved_mass, 0.0, places=7)
        self.assertEqual(result.evidence_level, "high")

    def test_reward_shift_without_action_flip_is_low_evidence(self) -> None:
        config = DetectorConfig(
            n_states=2,
            n_actions=2,
            gamma=0.0,
            min_visits_per_state_action=3,
            tie_tolerance=0.01,
            bootstrap_samples=50,
            bootstrap_block_length=5,
            random_seed=11,
        )
        rewards = [[2.0, 1.0], [1.0, 2.0]]
        reference = repeating_records(
            states=[0, 1],
            next_states={0: 0, 1: 1},
            rewards=rewards,
            repeats=10,
        )
        shifted = [[value + 3.0 for value in row] for row in rewards]
        recent = repeating_records(
            states=[0, 1],
            next_states={0: 0, 1: 1},
            rewards=shifted,
            repeats=10,
        )

        result = DecisionChangeDetector(config).detect(reference, recent)

        self.assertEqual(result.confirmed_flip_states, [])
        self.assertEqual(result.impact, 0.0)
        self.assertEqual(result.evidence_level, "low")

    def test_undercovered_visited_state_is_not_called_unchanged(self) -> None:
        config = DetectorConfig(
            n_states=2,
            n_actions=2,
            gamma=0.0,
            min_visits_per_state_action=5,
            bootstrap_samples=30,
            bootstrap_block_length=5,
            random_seed=17,
        )
        rewards = [[2.0, 1.0], [1.0, 2.0]]
        reference = repeating_records(
            states=[0, 1],
            next_states={0: 0, 1: 1},
            rewards=rewards,
            repeats=10,
        )
        recent = [
            TransitionRecord(state=0, action=0, reward=1.0, next_state=0),
            TransitionRecord(state=1, action=1, reward=2.0, next_state=1),
        ]

        result = DecisionChangeDetector(config).detect(reference, recent)

        self.assertIn(0, result.unresolved_states)
        self.assertGreater(result.unresolved_mass, 0.0)
        self.assertEqual(result.evidence_level, "uncertain")

    def test_bootstrap_interval_covers_point_estimate(self) -> None:
        config = DetectorConfig(
            n_states=2,
            n_actions=2,
            gamma=0.0,
            min_visits_per_state_action=2,
            bootstrap_samples=40,
            bootstrap_block_length=3,
            random_seed=23,
        )
        reference = repeating_records(
            states=[0, 1],
            next_states={0: 1, 1: 0},
            rewards=[[1.0, 2.0], [2.0, 1.0]],
            repeats=8,
        )
        recent = repeating_records(
            states=[0, 1],
            next_states={0: 1, 1: 0},
            rewards=[[2.0, 1.0], [2.0, 1.0]],
            repeats=8,
        )

        result = DecisionChangeDetector(config).detect(reference, recent)

        self.assertLessEqual(result.impact_ci_lower, result.impact)
        self.assertGreaterEqual(result.impact_ci_upper, result.impact)
        self.assertTrue(np.isfinite(result.bootstrap_impacts).all())


if __name__ == "__main__":
    unittest.main()
