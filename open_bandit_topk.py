"""Top-K action-set change estimation for aggregated Open Bandit data."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TopKResult:
    impact: float
    unresolved_mass: float
    changed_contexts: int
    unresolved_contexts: int
    reference_sets: np.ndarray
    recent_sets: np.ndarray


def _design(context_count: int, action_count: int) -> np.ndarray:
    context_action = context_count * action_count
    columns = context_action + action_count
    reference = np.zeros((context_action, columns), dtype=float)
    recent = np.zeros_like(reference)
    for context in range(context_count):
        for action in range(action_count):
            row = context * action_count + action
            reference[row, row] = 1.0
            recent[row, row] = 1.0
            recent[row, context_action + action] = 1.0
    return np.vstack([reference, recent])


def fit_two_window_rates(
    reference_counts: np.ndarray,
    reference_clicks: np.ndarray,
    recent_counts: np.ndarray,
    recent_clicks: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    context_count, action_count = reference_counts.shape
    design = _design(context_count, action_count)
    counts = np.concatenate([reference_counts.reshape(-1), recent_counts.reshape(-1)])
    clicks = np.concatenate([reference_clicks.reshape(-1), recent_clicks.reshape(-1)])
    beta = np.zeros(design.shape[1], dtype=float)
    for _ in range(50):
        linear = design @ beta
        probability = 1.0 / (1.0 + np.exp(-np.clip(linear, -30.0, 30.0)))
        probability = np.clip(probability, 1e-8, 1.0 - 1e-8)
        variance = np.maximum(counts * probability * (1.0 - probability), 1e-12)
        weights = counts * probability * (1.0 - probability)
        working = np.where(
            counts > 0,
            linear + (clicks - counts * probability) / variance,
            linear,
        )
        weighted_design = design * weights[:, None]
        system = design.T @ weighted_design + np.eye(design.shape[1]) * 1e-6
        rhs = design.T @ (weights * working)
        updated = np.linalg.solve(system, rhs)
        if np.max(np.abs(updated - beta)) < 1e-8:
            beta = updated
            break
        beta = updated
    linear = design @ beta
    probabilities = 1.0 / (1.0 + np.exp(-np.clip(linear, -30.0, 30.0)))
    split = context_count * action_count
    reference_rates = probabilities[:split].reshape(
        context_count,
        action_count,
    )
    recent_rates = probabilities[split:].reshape(
        context_count,
        action_count,
    )
    return reference_rates, recent_rates


def topk_from_rates(
    reference_rates: np.ndarray,
    recent_rates: np.ndarray,
    reference_counts: np.ndarray,
    recent_counts: np.ndarray,
    *,
    top_k: int,
    min_exposures: int,
    margin: float,
) -> TopKResult:
    support = (reference_counts >= min_exposures) & (
        recent_counts >= min_exposures
    )
    context_count = reference_counts.shape[0]
    reference_sets = np.full((context_count, top_k), -1, dtype=int)
    recent_sets = np.full((context_count, top_k), -1, dtype=int)
    changed = np.zeros(context_count, dtype=bool)
    comparable = np.zeros(context_count, dtype=bool)
    for context in range(context_count):
        supported_actions = np.flatnonzero(support[context])
        if supported_actions.size < top_k:
            continue
        ref_order = supported_actions[
            np.argsort(reference_rates[context, supported_actions])[::-1]
        ]
        recent_order = supported_actions[
            np.argsort(recent_rates[context, supported_actions])[::-1]
        ]
        ref_gap = (
            reference_rates[context, ref_order[top_k - 1]]
            - reference_rates[context, ref_order[top_k]]
            if supported_actions.size > top_k
            else np.inf
        )
        recent_gap = (
            recent_rates[context, recent_order[top_k - 1]]
            - recent_rates[context, recent_order[top_k]]
            if supported_actions.size > top_k
            else np.inf
        )
        if ref_gap <= margin or recent_gap <= margin:
            continue
        reference_sets[context] = ref_order[:top_k]
        recent_sets[context] = recent_order[:top_k]
        comparable[context] = True
        changed[context] = not np.array_equal(
            reference_sets[context],
            recent_sets[context],
        )
    context_totals = recent_counts.sum(axis=1).astype(float)
    occupancy = context_totals / context_totals.sum()
    distance = np.zeros(context_count, dtype=float)
    for context in range(context_count):
        if not comparable[context]:
            continue
        intersection = np.intersect1d(
            reference_sets[context],
            recent_sets[context],
        ).size
        distance[context] = 1.0 - intersection / (2 * top_k - intersection)
    impact = float(np.dot(occupancy, distance))
    unresolved_mass = float(np.dot(occupancy, (~comparable).astype(float)))
    return TopKResult(
        impact=impact,
        unresolved_mass=unresolved_mass,
        changed_contexts=int(changed.sum()),
        unresolved_contexts=int((~comparable).sum()),
        reference_sets=reference_sets,
        recent_sets=recent_sets,
    )


def topk_change(
    reference_counts: np.ndarray,
    reference_clicks: np.ndarray,
    recent_counts: np.ndarray,
    recent_clicks: np.ndarray,
    *,
    top_k: int,
    min_exposures: int,
    margin: float,
) -> TopKResult:
    reference_rates, recent_rates = fit_two_window_rates(
        reference_counts,
        reference_clicks,
        recent_counts,
        recent_clicks,
    )
    return topk_from_rates(
        reference_rates,
        recent_rates,
        reference_counts,
        recent_counts,
        top_k=top_k,
        min_exposures=min_exposures,
        margin=margin,
    )


def empirical_topk_change(
    reference_counts: np.ndarray,
    reference_clicks: np.ndarray,
    recent_counts: np.ndarray,
    recent_clicks: np.ndarray,
    *,
    top_k: int,
    min_exposures: int,
    margin: float,
    prior_strength: float = 50.0,
) -> TopKResult:
    def rates(counts: np.ndarray, clicks: np.ndarray) -> np.ndarray:
        action_counts = counts.sum(axis=0)
        action_clicks = clicks.sum(axis=0)
        prior = (action_clicks + 1.0) / (action_counts + 2.0)
        return (clicks + prior_strength * prior[None, :]) / (
            counts + prior_strength
        )

    reference_rates = rates(reference_counts, reference_clicks)
    recent_rates = rates(recent_counts, recent_clicks)
    return topk_from_rates(
        reference_rates,
        recent_rates,
        reference_counts,
        recent_counts,
        top_k=top_k,
        min_exposures=min_exposures,
        margin=margin,
    )
