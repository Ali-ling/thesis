"""Known-truth semi-synthetic experiments on real Open Bandit exposure data."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

from open_bandit_topk import empirical_topk_change, topk_change


SCENARIOS = [
    "stable",
    "click_shift",
    "high_exposure_local",
    "low_exposure_local",
    "global_flip",
    "gradual_flip",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Open Bandit semi-synthetic study.")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/open_bandit/application_v2/aggregated.npz"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/open_bandit_semisynthetic"),
    )
    parser.add_argument("--window-blocks", type=int, default=48)
    parser.add_argument("--change-window", type=int, default=7)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--margin", type=float, default=0.0001)
    parser.add_argument("--min-exposures", type=int, default=20)
    parser.add_argument("--delta0", type=float, default=0.20)
    parser.add_argument("--delta1", type=float, default=0.55)
    parser.add_argument("--max-unresolved", type=float, default=0.40)
    parser.add_argument("--bootstrap", type=int, default=50)
    parser.add_argument("--effect-size", type=float, default=0.01)
    parser.add_argument(
        "--estimator",
        choices=["glm", "empirical", "posterior"],
        default="empirical",
    )
    parser.add_argument("--posterior-delta0", type=float, default=0.05)
    parser.add_argument("--posterior-delta1", type=float, default=0.20)
    parser.add_argument("--prior-strength", type=float, default=50.0)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--seed-start", type=int, default=20271201)
    return parser.parse_args()


def flip_topk_actions(
    probabilities: np.ndarray,
    contexts: list[int],
    top_k: int,
    effect_size: float,
) -> np.ndarray:
    target = probabilities.copy()
    for context in contexts:
        order = np.argsort(probabilities[context])
        old_top = order[-top_k:]
        new_top = order[:top_k]
        target[context, old_top] = np.maximum(
            0.0,
            probabilities[context, old_top] - effect_size,
        )
        target[context, new_top] = probabilities[context, new_top] + effect_size
    return target


def topk_sets(probabilities: np.ndarray, top_k: int) -> list[set[int]]:
    return [
        set(np.argsort(row)[::-1][:top_k].tolist())
        for row in probabilities
    ]


def true_impact(
    baseline: np.ndarray,
    current: np.ndarray,
    occupancy: np.ndarray,
    top_k: int,
) -> float:
    base_sets = topk_sets(baseline, top_k)
    current_sets = topk_sets(current, top_k)
    distances = np.asarray(
        [
            1.0 - len(left & right) / max(1, len(left | right))
            for left, right in zip(base_sets, current_sets)
        ]
    )
    return float(np.dot(occupancy, distances))


def posterior_topk_impact(
    reference_counts: np.ndarray,
    reference_clicks: np.ndarray,
    recent_counts: np.ndarray,
    recent_clicks: np.ndarray,
    *,
    top_k: int,
    min_exposures: int,
    prior_strength: float,
    bootstrap: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    def membership(
        blocks_counts: np.ndarray,
        blocks_clicks: np.ndarray,
    ) -> np.ndarray:
        context_count, action_count = blocks_counts.shape[1:]
        frequency = np.zeros((context_count, action_count), dtype=float)
        for _ in range(bootstrap):
            indices = rng.integers(0, blocks_counts.shape[0], blocks_counts.shape[0])
            sampled_counts = blocks_counts[indices].sum(axis=0)
            sampled_clicks = blocks_clicks[indices].sum(axis=0)
            action_counts = sampled_counts.sum(axis=0)
            action_clicks = sampled_clicks.sum(axis=0)
            prior = (action_clicks + 1.0) / (action_counts + 2.0)
            rates = (sampled_clicks + prior_strength * prior[None, :]) / (
                sampled_counts + prior_strength
            )
            for context in range(context_count):
                supported = np.flatnonzero(sampled_counts[context] >= min_exposures)
                if supported.size < top_k:
                    continue
                order = supported[np.argsort(rates[context, supported])[::-1]]
                frequency[context, order[:top_k]] += 1.0
        return frequency / bootstrap

    ref_membership = membership(reference_counts, reference_clicks)
    recent_membership = membership(recent_counts, recent_clicks)
    ref_totals = reference_counts.sum(axis=0)
    recent_totals = recent_counts.sum(axis=0)
    context_support = np.all(
        (ref_totals >= min_exposures) & (recent_totals >= min_exposures),
        axis=1,
    )
    divergence = (
        0.5
        * np.abs(ref_membership - recent_membership).sum(axis=1)
        / top_k
    )
    occupancy = recent_totals.sum(axis=1).astype(float)
    occupancy /= occupancy.sum()
    impact = float(np.dot(occupancy, divergence * context_support))
    unresolved_mass = float(np.dot(occupancy, (~context_support).astype(float)))
    return impact, unresolved_mass


def build_scenario(
    scenario: str,
    baseline: np.ndarray,
    high_contexts: list[int],
    low_contexts: list[int],
    window_count: int,
    change_window: int,
    top_k: int,
    effect_size: float,
) -> np.ndarray:
    probabilities = np.repeat(baseline[None, :, :], window_count, axis=0)
    if scenario == "stable":
        return probabilities
    if scenario == "click_shift":
        probabilities += 0.0015
        return np.clip(probabilities, 0.0, 0.10)
    if scenario == "high_exposure_local":
        target = flip_topk_actions(
            baseline,
            high_contexts,
            top_k,
            effect_size,
        )
    elif scenario == "low_exposure_local":
        target = flip_topk_actions(
            baseline,
            low_contexts,
            top_k,
            effect_size,
        )
    elif scenario in {"global_flip", "gradual_flip"}:
        target = flip_topk_actions(
            baseline,
            list(range(baseline.shape[0])),
            top_k,
            effect_size,
        )
    else:
        raise ValueError(scenario)
    for window in range(change_window, window_count):
        if scenario == "gradual_flip":
            progress = (window - change_window + 1) / (window_count - change_window)
            probabilities[window] = baseline + progress * (target - baseline)
        else:
            probabilities[window] = target
    return np.clip(probabilities, 0.0, 0.10)


def main() -> None:
    args = parse_args()
    data = np.load(args.data)
    real_counts = data["counts"]
    real_clicks = data["clicks"]
    windows = real_counts.shape[0] // args.window_blocks
    counts = real_counts[: windows * args.window_blocks].reshape(
        windows,
        args.window_blocks,
        real_counts.shape[1],
        real_counts.shape[2],
    )
    clicks = real_clicks[: windows * args.window_blocks].reshape(
        windows,
        args.window_blocks,
        real_clicks.shape[1],
        real_clicks.shape[2],
    )
    total_counts = counts.sum(axis=(0, 1))
    total_clicks = clicks.sum(axis=(0, 1))
    baseline = (total_clicks + 1.0) / (total_counts + 2.0)
    context_occupancy = total_counts.sum(axis=1).astype(float)
    context_occupancy /= context_occupancy.sum()
    high_order = np.argsort(context_occupancy)[::-1]
    high_contexts: list[int] = []
    cumulative = 0.0
    for context in high_order:
        high_contexts.append(int(context))
        cumulative += float(context_occupancy[context])
        if cumulative >= 0.60:
            break
    low_order = np.argsort(context_occupancy)
    low_contexts: list[int] = []
    cumulative = 0.0
    for context in low_order:
        low_contexts.append(int(context))
        cumulative += float(context_occupancy[context])
        if cumulative >= 0.20:
            break

    rows: list[dict[str, object]] = []
    for seed in range(args.seed_start, args.seed_start + args.seeds):
        rng = np.random.default_rng(seed)
        for scenario in SCENARIOS:
            scenario_probabilities = build_scenario(
                scenario,
                baseline,
                high_contexts,
                low_contexts,
                windows,
                args.change_window,
                args.top_k,
                args.effect_size,
            )
            simulated = np.zeros_like(counts, dtype=np.int32)
            for window in range(windows):
                simulated[window] = rng.binomial(
                    counts[window],
                    scenario_probabilities[window],
                )
            for transition in range(1, windows):
                ref_counts = counts[transition - 1].sum(axis=0)
                recent_counts = counts[transition].sum(axis=0)
                estimator = (
                    empirical_topk_change
                    if args.estimator == "empirical"
                    else topk_change
                )

                def estimate(
                    reference_counts: np.ndarray,
                    reference_clicks: np.ndarray,
                    recent_counts: np.ndarray,
                    recent_clicks: np.ndarray,
                ):
                    if args.estimator == "empirical":
                        return empirical_topk_change(
                            reference_counts,
                            reference_clicks,
                            recent_counts,
                            recent_clicks,
                            top_k=args.top_k,
                            min_exposures=args.min_exposures,
                            margin=args.margin,
                            prior_strength=args.prior_strength,
                        )
                    return topk_change(
                        reference_counts,
                        reference_clicks,
                        recent_counts,
                        recent_clicks,
                        top_k=args.top_k,
                        min_exposures=args.min_exposures,
                        margin=args.margin,
                    )

                if args.estimator == "posterior":
                    estimated_impact, unresolved_mass = posterior_topk_impact(
                        counts[transition - 1],
                        simulated[transition - 1],
                        counts[transition],
                        simulated[transition],
                        top_k=args.top_k,
                        min_exposures=args.min_exposures,
                        prior_strength=args.prior_strength,
                        bootstrap=args.bootstrap,
                        rng=rng,
                    )
                    changed_contexts = ""
                    lower = upper = estimated_impact
                    delta0 = args.posterior_delta0
                    delta1 = args.posterior_delta1
                else:
                    point = estimate(
                        ref_counts,
                        simulated[transition - 1].sum(axis=0),
                        recent_counts,
                        simulated[transition].sum(axis=0),
                    )
                    bootstrap_values = []
                    for _ in range(args.bootstrap):
                        ref_indices = rng.integers(
                            0,
                            args.window_blocks,
                            args.window_blocks,
                        )
                        recent_indices = rng.integers(
                            0,
                            args.window_blocks,
                            args.window_blocks,
                        )
                        replicate = estimate(
                            counts[transition - 1, ref_indices].sum(axis=0),
                            simulated[transition - 1, ref_indices].sum(axis=0),
                            counts[transition, recent_indices].sum(axis=0),
                            simulated[transition, recent_indices].sum(axis=0),
                        )
                        bootstrap_values.append(replicate.impact)
                    estimated_impact = point.impact
                    unresolved_mass = point.unresolved_mass
                    changed_contexts = point.changed_contexts
                    lower = float(np.quantile(bootstrap_values, 0.025))
                    upper = float(np.quantile(bootstrap_values, 0.975))
                    delta0 = args.delta0
                    delta1 = args.delta1
                evidence = (
                    "high"
                    if unresolved_mass <= args.max_unresolved
                    and lower >= delta1
                    else "low"
                    if unresolved_mass <= args.max_unresolved
                    and upper <= delta0
                    else "uncertain"
                )
                occupancy = recent_counts.sum(axis=1).astype(float)
                occupancy /= occupancy.sum()
                truth = true_impact(
                    scenario_probabilities[transition - 1],
                    scenario_probabilities[transition],
                    occupancy,
                    args.top_k,
                )
                rows.append(
                    {
                        "seed": seed,
                        "scenario": scenario,
                        "transition": transition,
                        "after_change": int(transition >= args.change_window),
                        "true_impact": truth,
                        "estimated_impact": estimated_impact,
                        "ci_lower": lower,
                        "ci_upper": upper,
                        "unresolved_mass": unresolved_mass,
                        "evidence_level": evidence,
                        "changed_contexts": changed_contexts,
                    }
                )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    detail_path = args.output_dir / "semi_synthetic_runs.csv"
    with detail_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    grouped: dict[tuple[str, int], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["scenario"]), int(row["seed"]))].append(row)
    summary: list[dict[str, object]] = []
    for scenario in SCENARIOS:
        for seed in range(args.seed_start, args.seed_start + args.seeds):
            events = grouped[(scenario, seed)]
            post = [event for event in events if event["after_change"]]
            true_high = [event for event in post if event["true_impact"] >= args.delta1]
            detected = [event for event in post if event["evidence_level"] == "high"]
            false_high = [
                event
                for event in post
                if event["evidence_level"] == "high"
                and event["true_impact"] < args.delta1
            ]
            summary.append(
                {
                    "scenario": scenario,
                    "seed": seed,
                    "post_change_events": len(post),
                    "true_high_events": len(true_high),
                    "detected_high_events": len(detected),
                    "false_high_events": len(false_high),
                    "first_detection_transition": detected[0]["transition"]
                    if detected
                    else "",
                    "first_true_high_transition": true_high[0]["transition"]
                    if true_high
                    else "",
                    "mean_true_impact": float(
                        np.mean([event["true_impact"] for event in post])
                    ),
                    "mean_estimated_impact": float(
                        np.mean([event["estimated_impact"] for event in post])
                    ),
                }
            )
    summary_path = args.output_dir / "semi_synthetic_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    print(
        f"high_contexts={high_contexts} low_contexts={low_contexts} "
        f"detail={detail_path} summary={summary_path}"
    )


if __name__ == "__main__":
    main()
