"""Tests for stable PPO budget calibration summaries."""

from __future__ import annotations

import unittest

from calibrate_ppo_budget import aggregate, row_at_or_after, summarize_budget_rows


class BudgetCalibrationTest(unittest.TestCase):
    def test_computes_optimality_ratio_at_milestones(self) -> None:
        rows = [
            {"timesteps": 256, "exact_policy_value": 20.0, "entropy": 0.5},
            {"timesteps": 512, "exact_policy_value": 25.0, "entropy": 0.4},
            {"timesteps": 768, "exact_policy_value": 30.0, "entropy": 0.3},
        ]
        summary = summarize_budget_rows(
            rows,
            seed=1,
            optimal_value=32.0,
            milestones=[400, 700, 900],
        )

        self.assertEqual(row_at_or_after(rows, 700)["timesteps"], 768)
        self.assertAlmostEqual(float(summary[0]["optimality_ratio"]), 25.0 / 32.0)
        self.assertEqual(summary[-1]["actual_timesteps"], 768)

    def test_aggregates_multiple_seeds(self) -> None:
        rows = [
            {
                "requested_timesteps": 100,
                "exact_policy_value": 20.0,
                "optimality_ratio": 0.8,
                "optimality_gap": 5.0,
                "entropy": 0.4,
            },
            {
                "requested_timesteps": 100,
                "exact_policy_value": 22.0,
                "optimality_ratio": 0.88,
                "optimality_gap": 3.0,
                "entropy": 0.2,
            },
        ]

        aggregate_rows = aggregate(rows)

        self.assertEqual(aggregate_rows[0]["runs"], 2)
        self.assertAlmostEqual(aggregate_rows[0]["mean_exact_policy_value"], 21.0)
        self.assertAlmostEqual(aggregate_rows[0]["mean_optimality_ratio"], 0.84)


if __name__ == "__main__":
    unittest.main()
