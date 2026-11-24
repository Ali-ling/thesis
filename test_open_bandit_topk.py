"""Tests for Open Bandit Top-K change estimation."""

from __future__ import annotations

import unittest

import numpy as np

from open_bandit_topk import topk_change


class OpenBanditTopKTest(unittest.TestCase):
    def test_detects_complete_topk_replacement(self) -> None:
        reference_counts = np.full((2, 4), 1000, dtype=np.int64)
        recent_counts = np.full((2, 4), 1000, dtype=np.int64)
        reference_clicks = np.tile(np.array([100, 90, 10, 5]), (2, 1))
        recent_clicks = np.tile(np.array([10, 5, 100, 90]), (2, 1))

        result = topk_change(
            reference_counts,
            reference_clicks,
            recent_counts,
            recent_clicks,
            top_k=2,
            min_exposures=20,
            margin=0.0,
        )

        self.assertAlmostEqual(result.impact, 1.0)
        self.assertEqual(result.changed_contexts, 2)
        self.assertEqual(result.unresolved_contexts, 0)

    def test_identical_windows_have_zero_impact(self) -> None:
        counts = np.full((2, 4), 1000, dtype=np.int64)
        clicks = np.tile(np.array([100, 90, 10, 5]), (2, 1))
        result = topk_change(
            counts,
            clicks,
            counts,
            clicks,
            top_k=2,
            min_exposures=20,
            margin=0.0001,
        )
        self.assertAlmostEqual(result.impact, 0.0)
        self.assertEqual(result.unresolved_mass, 0.0)


if __name__ == "__main__":
    unittest.main()
