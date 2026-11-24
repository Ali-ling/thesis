"""Offline validation of the decision detector on all arena scenarios.

Example:

    python validate_detector.py --window-sizes 50 100 200 --seeds 5

The behavior policy used for generation is uniform random exploration by
default. This isolates detector quality from the additional coverage problem
of a learned policy. A future PPO integration should use the same detector on
the policy's rollout stream with an epsilon floor.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from arena_env import NonStationaryArenaEnv, greedy_actions
from decision_detector import (
    DecisionChangeDetector,
    DetectorConfig,
    TransitionRecord,
)


SCENARIOS = [
    "stable",
    "reward_shift",
    "transition_shift",
    "global_flip",
    "local_flip",
    "gradual_flip",
]


@dataclass(frozen=True)
class Truth:
    reference_actions: list[int]
    recent_actions: list[int]
    flip_mask: np.ndarray
    decided_mask: np.ndarray
    occupancy: np.ndarray
    impact: float


@dataclass(frozen=True)
class ValidationRow:
    scenario: str
    window_size: int
    seed: int
    true_impact: float
    estimated_impact: float
    impact_abs_error: float
    ci_lower: float
    ci_upper: float
    evidence_level: str
    true_evidence_level: str
    evidence_matches_truth: int
    true_flip_states: str
    estimated_flip_states: str
    unresolved_states: str
    false_positive_flips: int
    false_negative_flips: int
    true_positives: int
    reference_coverage: float
    recent_coverage: float
    unresolved_mass: float
    valid_bootstrap_samples: int


def generate_records(
    scenario: str,
    window_size: int,
    seed: int,
) -> tuple[NonStationaryArenaEnv, list[TransitionRecord], list[TransitionRecord]]:
    """Generate adjacent reference and recent windows around one change point."""
    env = NonStationaryArenaEnv(
        scenario=scenario,
        change_step=window_size,
        seed=seed,
    )
    state, _ = env.reset(
        seed=seed,
        options={"hard_reset": True, "initial_state": env.protocol["initial_state"]},
    )
    rng = random.Random(seed)
    records: list[TransitionRecord] = []
    for _ in range(2 * window_size):
        action = rng.randrange(env.n_actions)
        next_state, reward, _, truncated, _ = env.step(action)
        records.append(
            TransitionRecord(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
            )
        )
        if truncated:
            state, _ = env.reset()
        else:
            state = next_state
    return env, records[:window_size], records[window_size:]


def calculate_truth(
    env: NonStationaryArenaEnv,
    window_size: int,
    recent_records: list[TransitionRecord],
    tie_tolerance: float,
) -> Truth:
    """Compute the exact decision truth at the two window endpoints."""
    q_reference = np.asarray(env.true_q_values(window_size - 1), dtype=np.float64)
    q_recent = np.asarray(env.true_q_values(2 * window_size - 1), dtype=np.float64)
    reference_actions = np.asarray(
        greedy_actions(q_reference, tie_tolerance),
        dtype=int,
    )
    recent_actions = np.asarray(
        greedy_actions(q_recent, tie_tolerance),
        dtype=int,
    )
    ref_gap = np.max(q_reference, axis=1) - np.min(q_reference, axis=1)
    recent_gap = np.max(q_recent, axis=1) - np.min(q_recent, axis=1)
    decided = (ref_gap > tie_tolerance) & (recent_gap > tie_tolerance)
    flips = decided & (reference_actions != recent_actions)
    state_counts = np.bincount(
        [record.state for record in recent_records],
        minlength=env.n_states,
    ).astype(np.float64)
    occupancy = state_counts / float(np.sum(state_counts))
    impact = float(np.dot(occupancy, flips.astype(np.float64)))
    return Truth(
        reference_actions=reference_actions.tolist(),
        recent_actions=recent_actions.tolist(),
        flip_mask=flips,
        decided_mask=decided,
        occupancy=occupancy,
        impact=impact,
    )


def truth_level(impact: float, delta0: float, delta1: float) -> str:
    if impact <= delta0:
        return "low"
    if impact >= delta1:
        return "high"
    return "uncertain"


def validate_one(
    scenario: str,
    window_size: int,
    seed: int,
    config: DetectorConfig,
) -> ValidationRow:
    env, reference, recent = generate_records(scenario, window_size, seed)
    truth = calculate_truth(env, window_size, recent, config.tie_tolerance)
    result = DecisionChangeDetector(config).detect(reference, recent)

    estimated_flips = np.asarray(result.confirmed_flip_mask, dtype=bool)
    resolved = ~np.asarray(result.unresolved_mask, dtype=bool)
    truth_decided = np.asarray(truth.decided_mask, dtype=bool)
    evaluated = truth_decided & resolved
    true_positive = int(np.sum(estimated_flips & truth.flip_mask))
    false_positive = int(np.sum(estimated_flips & ~truth.flip_mask & truth_decided))
    false_negative = int(np.sum(~estimated_flips & truth.flip_mask & evaluated))
    expected_level = truth_level(truth.impact, config.delta0, config.delta1)
    return ValidationRow(
        scenario=scenario,
        window_size=window_size,
        seed=seed,
        true_impact=truth.impact,
        estimated_impact=result.impact,
        impact_abs_error=abs(result.impact - truth.impact),
        ci_lower=result.impact_ci_lower,
        ci_upper=result.impact_ci_upper,
        evidence_level=result.evidence_level,
        true_evidence_level=expected_level,
        evidence_matches_truth=int(result.evidence_level == expected_level),
        true_flip_states=json.dumps(np.flatnonzero(truth.flip_mask).tolist()),
        estimated_flip_states=json.dumps(result.confirmed_flip_states),
        unresolved_states=json.dumps(result.unresolved_states),
        false_positive_flips=false_positive,
        false_negative_flips=false_negative,
        true_positives=true_positive,
        reference_coverage=result.reference_coverage,
        recent_coverage=result.recent_coverage,
        unresolved_mass=result.unresolved_mass,
        valid_bootstrap_samples=result.valid_bootstrap_samples,
    )


def summarize_rows(rows: list[ValidationRow]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, int], list[ValidationRow]] = defaultdict(list)
    for row in rows:
        grouped[(row.scenario, row.window_size)].append(row)

    summary: list[dict[str, object]] = []
    for (scenario, window_size), group in sorted(grouped.items()):
        true_levels = [row.true_evidence_level for row in group]
        predicted_levels = [row.evidence_level for row in group]
        summary.append(
            {
                "scenario": scenario,
                "window_size": window_size,
                "runs": len(group),
                "mean_true_impact": float(np.mean([row.true_impact for row in group])),
                "mean_estimated_impact": float(
                    np.mean([row.estimated_impact for row in group])
                ),
                "impact_mae": float(np.mean([row.impact_abs_error for row in group])),
                "mean_ci_lower": float(np.mean([row.ci_lower for row in group])),
                "mean_ci_upper": float(np.mean([row.ci_upper for row in group])),
                "truth_low_rate": true_levels.count("low") / len(group),
                "truth_high_rate": true_levels.count("high") / len(group),
                "predicted_low_rate": predicted_levels.count("low") / len(group),
                "predicted_high_rate": predicted_levels.count("high") / len(group),
                "predicted_uncertain_rate": predicted_levels.count("uncertain")
                / len(group),
                "false_high_rate": sum(
                    row.true_evidence_level == "low"
                    and row.evidence_level == "high"
                    for row in group
                )
                / len(group),
                "missed_high_rate": sum(
                    row.true_evidence_level == "high"
                    and row.evidence_level != "high"
                    for row in group
                )
                / len(group),
                "evidence_match_rate": float(
                    np.mean([row.evidence_matches_truth for row in group])
                ),
                "mean_reference_coverage": float(
                    np.mean([row.reference_coverage for row in group])
                ),
                "mean_recent_coverage": float(
                    np.mean([row.recent_coverage for row in group])
                ),
                "mean_unresolved_mass": float(
                    np.mean([row.unresolved_mass for row in group])
                ),
            }
        )
    return summary


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def print_summary(summary: list[dict[str, object]]) -> None:
    headers = [
        "scenario",
        "H",
        "MAE",
        "trueI",
        "predI",
        "low",
        "high",
        "unc",
        "falseHi",
        "missHi",
        "covRef",
        "covRec",
    ]
    print(" ".join(f"{header:>12}" for header in headers))
    for item in summary:
        values = [
            item["scenario"],
            item["window_size"],
            f"{item['impact_mae']:.3f}",
            f"{item['mean_true_impact']:.3f}",
            f"{item['mean_estimated_impact']:.3f}",
            f"{item['predicted_low_rate']:.2f}",
            f"{item['predicted_high_rate']:.2f}",
            f"{item['predicted_uncertain_rate']:.2f}",
            f"{item['false_high_rate']:.2f}",
            f"{item['missed_high_rate']:.2f}",
            f"{item['mean_reference_coverage']:.2f}",
            f"{item['mean_recent_coverage']:.2f}",
        ]
        print(" ".join(f"{str(value):>12}" for value in values))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate decision-relevant change detection on the finite arena."
    )
    parser.add_argument("--scenarios", nargs="+", choices=SCENARIOS, default=SCENARIOS)
    parser.add_argument("--window-sizes", nargs="+", type=int, default=[50, 100, 200])
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--seed-start", type=int, default=20260916)
    parser.add_argument("--bootstrap-samples", type=int, default=100)
    parser.add_argument("--bootstrap-block-length", type=int, default=10)
    parser.add_argument("--min-visits", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, default=Path("results/detector_validation"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if any(size <= 1 for size in args.window_sizes):
        raise ValueError("each window size must be greater than one")
    if args.seeds <= 0:
        raise ValueError("--seeds must be positive")

    rows: list[ValidationRow] = []
    for scenario in args.scenarios:
        for window_size in args.window_sizes:
            for offset in range(args.seeds):
                seed = args.seed_start + offset
                config = DetectorConfig(
                    min_visits_per_state_action=args.min_visits,
                    bootstrap_samples=args.bootstrap_samples,
                    bootstrap_block_length=args.bootstrap_block_length,
                    random_seed=seed,
                )
                rows.append(validate_one(scenario, window_size, seed, config))

    row_dicts = [asdict(row) for row in rows]
    summary = summarize_rows(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "detector_validation_runs.csv", row_dicts)
    write_csv(args.output_dir / "detector_validation_summary.csv", summary)
    print_summary(summary)
    print(f"\nruns={len(rows)}")
    print(f"detail={args.output_dir / 'detector_validation_runs.csv'}")
    print(f"summary={args.output_dir / 'detector_validation_summary.csv'}")


if __name__ == "__main__":
    main()
