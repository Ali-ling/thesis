"""Tests for formal experiment aggregation."""

from __future__ import annotations

import unittest

from analyze_formal_results import group_summary, paired_comparison


def row(
    seed: int,
    method: str,
    *,
    final: float,
    recovery: object,
    trigger: float,
) -> dict[str, object]:
    return {
        "seed": str(seed),
        "scenario": "global_flip",
        "method": method,
        "pre_change_value": "28",
        "post_change_min_value": "20",
        "final_value": str(final),
        "final_minus_pre_change": str(final - 28),
        "drop_from_pre_change": "8",
        "trigger_latency": "100" if trigger else "",
        "recovery_timesteps": "7000" if recovery not in {"", None} else "",
        "recovery_delay": recovery,
        "trigger_events": str(trigger),
        "strong_updates": str(trigger),
        "mean_epochs_post_change": "4" if trigger else "6",
    }


class FormalAnalysisTest(unittest.TestCase):
    def test_group_summary_counts_recovery_and_trigger_rates(self) -> None:
        rows = [
            row(1, "decision", final=27.0, recovery="5000", trigger=1),
            row(2, "decision", final=26.0, recovery="", trigger=0),
        ]
        summary = group_summary(rows)[0]

        self.assertEqual(summary["runs"], 2)
        self.assertAlmostEqual(summary["final_value_mean"], 26.5)
        self.assertAlmostEqual(summary["recovery_rate"], 0.5)
        self.assertAlmostEqual(summary["trigger_rate"], 0.5)

    def test_paired_comparison_aligns_seeds(self) -> None:
        rows = [
            row(1, "decision", final=27.0, recovery="5000", trigger=1),
            row(1, "fixed", final=26.0, recovery="6000", trigger=0),
            row(2, "decision", final=26.5, recovery="", trigger=0),
            row(2, "fixed", final=26.0, recovery="", trigger=0),
        ]
        comparison = paired_comparison(
            rows,
            scenario="global_flip",
            left="decision",
            right="fixed",
            samples=100,
        )

        self.assertEqual(comparison["paired_runs"], 2)
        self.assertAlmostEqual(comparison["final_difference_mean"], 0.75)
        self.assertEqual(comparison["paired_recovery_runs"], 1)
        self.assertAlmostEqual(comparison["recovery_delay_difference_mean"], -1000.0)


if __name__ == "__main__":
    unittest.main()
