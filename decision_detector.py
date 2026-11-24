"""Decision-relevant change detection for the finite-state arena.

The detector estimates the environment model from two adjacent trajectory
windows. It does not read the true reward or transition model. The true model
in ``arena_env.py`` is only used by the evaluation script to score the result.

Main pipeline:

    trajectories -> empirical R/P -> value iteration -> action flip mask
                 -> state-occupancy weighted impact I_t
                 -> moving-block bootstrap confidence interval
                 -> low / uncertain / high evidence
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class TransitionRecord:
    """One complete transition used by the online monitor."""

    state: int
    action: int
    reward: float
    next_state: int


@dataclass(frozen=True)
class DetectorConfig:
    """Pre-registered detector parameters.

    ``min_visits_per_state_action`` controls whether a state-action value can
    be compared. States below this coverage level are never silently treated
    as unchanged.
    """

    n_states: int = 8
    n_actions: int = 2
    gamma: float = 0.92
    min_visits_per_state_action: int = 5
    transition_smoothing: float = 0.05
    tie_tolerance: float = 0.08
    delta0: float = 0.20
    delta1: float = 0.55
    confidence_level: float = 0.95
    bootstrap_samples: int = 200
    bootstrap_block_length: int = 10
    max_unresolved_mass: float = 0.20
    min_valid_bootstrap_samples: int = 20
    random_seed: int = 20260916
    bellman_solver_max_iterations: int = 100

    def __post_init__(self) -> None:
        if self.n_states <= 0 or self.n_actions <= 0:
            raise ValueError("n_states and n_actions must be positive")
        if not 0.0 <= self.gamma < 1.0:
            raise ValueError("gamma must be in [0, 1)")
        if self.min_visits_per_state_action < 1:
            raise ValueError("min_visits_per_state_action must be positive")
        if self.transition_smoothing <= 0.0:
            raise ValueError("transition_smoothing must be positive")
        if self.tie_tolerance < 0.0:
            raise ValueError("tie_tolerance cannot be negative")
        if not 0.0 <= self.delta0 < self.delta1 <= 1.0:
            raise ValueError("expected 0 <= delta0 < delta1 <= 1")
        if not 0.0 < self.confidence_level < 1.0:
            raise ValueError("confidence_level must be in (0, 1)")
        if self.bootstrap_samples < 1:
            raise ValueError("bootstrap_samples must be positive")
        if self.bootstrap_block_length < 1:
            raise ValueError("bootstrap_block_length must be positive")
        if not 0.0 <= self.max_unresolved_mass <= 1.0:
            raise ValueError("max_unresolved_mass must be in [0, 1]")


@dataclass(frozen=True)
class ModelEstimate:
    """Empirical model and the resulting estimated optimal action values."""

    rewards: np.ndarray
    transitions: np.ndarray
    state_action_counts: np.ndarray
    q_values: np.ndarray
    action_values: np.ndarray
    state_ready: np.ndarray
    tie_states: np.ndarray
    row_counts_valid: bool


@dataclass(frozen=True)
class DetectionResult:
    """One online detection decision and all quantities needed for auditing."""

    q_reference: np.ndarray
    q_recent: np.ndarray
    reference_actions: np.ndarray
    recent_actions: np.ndarray
    confirmed_flip_mask: np.ndarray
    unresolved_mask: np.ndarray
    tie_mask: np.ndarray
    state_occupancy: np.ndarray
    impact: float
    impact_ci_lower: float
    impact_ci_upper: float
    unresolved_mass: float
    evidence_level: str
    reference_coverage: float
    recent_coverage: float
    valid_bootstrap_samples: int
    bootstrap_impacts: np.ndarray
    config: DetectorConfig

    @property
    def confirmed_flip_states(self) -> list[int]:
        return np.flatnonzero(self.confirmed_flip_mask).astype(int).tolist()

    @property
    def unresolved_states(self) -> list[int]:
        return np.flatnonzero(self.unresolved_mask).astype(int).tolist()

    @property
    def bootstrap_valid(self) -> bool:
        return self.valid_bootstrap_samples >= self.config.min_valid_bootstrap_samples


def _as_record_array(
    records: Iterable[TransitionRecord],
    config: DetectorConfig,
) -> np.ndarray:
    rows: list[tuple[int, int, float, int]] = []
    for index, record in enumerate(records):
        state = int(record.state)
        action = int(record.action)
        next_state = int(record.next_state)
        if not 0 <= state < config.n_states:
            raise ValueError(f"record {index}: state {state} is out of range")
        if not 0 <= action < config.n_actions:
            raise ValueError(f"record {index}: action {action} is out of range")
        if not 0 <= next_state < config.n_states:
            raise ValueError(f"record {index}: next_state {next_state} is out of range")
        reward = float(record.reward)
        if not np.isfinite(reward):
            raise ValueError(f"record {index}: reward must be finite")
        rows.append((state, action, reward, next_state))
    return np.asarray(rows, dtype=np.float64).reshape((-1, 4))


def estimate_empirical_model(
    records: Sequence[TransitionRecord] | np.ndarray,
    config: DetectorConfig,
) -> ModelEstimate:
    """Estimate R and P from records and solve the empirical Bellman optimum."""
    data = records if isinstance(records, np.ndarray) else _as_record_array(records, config)
    data = np.asarray(data, dtype=np.float64).reshape((-1, 4))

    reward_sum = np.zeros((config.n_states, config.n_actions), dtype=np.float64)
    state_action_counts = np.zeros((config.n_states, config.n_actions), dtype=np.float64)
    transition_counts = np.zeros(
        (config.n_states, config.n_actions, config.n_states),
        dtype=np.float64,
    )
    states = data[:, 0].astype(int)
    actions = data[:, 1].astype(int)
    measured_rewards = data[:, 2]
    next_states = data[:, 3].astype(int)
    np.add.at(reward_sum, (states, actions), measured_rewards)
    np.add.at(state_action_counts, (states, actions), 1.0)
    np.add.at(transition_counts, (states, actions, next_states), 1.0)

    rewards = np.zeros_like(reward_sum)
    transitions = np.zeros_like(transition_counts)
    for state in range(config.n_states):
        for action in range(config.n_actions):
            count = state_action_counts[state, action]
            if count > 0.0:
                rewards[state, action] = reward_sum[state, action] / count
                transitions[state, action] = (
                    transition_counts[state, action] + config.transition_smoothing
                ) / (count + config.transition_smoothing * config.n_states)
            else:
                # An unseen row is retained for numerical stability but the
                # state is marked unresolved and cannot produce a confirmed flip.
                transitions[state, action] = 1.0 / config.n_states

    # Solve the empirical Bellman optimum exactly with policy iteration. This
    # is an efficient equivalent of value iteration for the 8-state MDP and
    # avoids making bootstrap runtime depend on a very small epsilon.
    state_indices = np.arange(config.n_states)
    policy = np.zeros(config.n_states, dtype=int)
    q_values = np.zeros((config.n_states, config.n_actions), dtype=np.float64)
    identity = np.eye(config.n_states, dtype=np.float64)
    for _ in range(config.bellman_solver_max_iterations):
        policy_rewards = rewards[state_indices, policy]
        policy_transitions = transitions[state_indices, policy, :]
        values = np.linalg.solve(
            identity - config.gamma * policy_transitions,
            policy_rewards,
        )
        q_values = rewards + config.gamma * np.einsum("san,n->sa", transitions, values)
        improved_policy = np.argmax(q_values, axis=1).astype(int)
        if np.array_equal(improved_policy, policy):
            break
        policy = improved_policy
    else:
        raise RuntimeError("empirical Bellman solver did not converge")

    action_values = np.argmax(q_values, axis=1).astype(int)
    q_gaps = np.max(q_values, axis=1) - np.min(q_values, axis=1)
    tie_states = q_gaps <= config.tie_tolerance
    state_ready = np.all(
        state_action_counts >= config.min_visits_per_state_action,
        axis=1,
    )
    return ModelEstimate(
        rewards=rewards,
        transitions=transitions,
        state_action_counts=state_action_counts,
        q_values=q_values,
        action_values=action_values,
        state_ready=state_ready,
        tie_states=tie_states,
        row_counts_valid=True,
    )


def state_occupancy(records: Sequence[TransitionRecord] | np.ndarray, config: DetectorConfig) -> np.ndarray:
    """Return the empirical state-occupancy distribution in a window."""
    data = records if isinstance(records, np.ndarray) else _as_record_array(records, config)
    data = np.asarray(data, dtype=np.float64).reshape((-1, 4))
    if data.shape[0] == 0:
        return np.full(config.n_states, 1.0 / config.n_states, dtype=np.float64)
    counts = np.bincount(data[:, 0].astype(int), minlength=config.n_states).astype(np.float64)
    return counts / float(np.sum(counts))


def moving_block_indices(
    sample_count: int,
    block_length: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample contiguous trajectory blocks until ``sample_count`` indices exist."""
    if sample_count <= 0:
        return np.empty(0, dtype=int)
    length = min(block_length, sample_count)
    block_count = int(np.ceil(sample_count / length))
    starts = rng.integers(0, sample_count - length + 1, size=block_count)
    indices = np.concatenate(
        [np.arange(start, start + length, dtype=int) for start in starts]
    )
    return indices[:sample_count]


class DecisionChangeDetector:
    """Estimate action flips and decision impact from two adjacent windows."""

    def __init__(self, config: Optional[DetectorConfig] = None) -> None:
        self.config = config or DetectorConfig()

    def detect(
        self,
        reference_records: Sequence[TransitionRecord],
        recent_records: Sequence[TransitionRecord],
        *,
        run_bootstrap: bool = True,
    ) -> DetectionResult:
        if len(reference_records) == 0 or len(recent_records) == 0:
            raise ValueError("both windows must contain at least one transition")

        ref_data = _as_record_array(reference_records, self.config)
        recent_data = _as_record_array(recent_records, self.config)
        point = self._detect_from_arrays(ref_data, recent_data)

        bootstrap_impacts = np.empty(0, dtype=np.float64)
        if run_bootstrap:
            bootstrap_impacts = self._bootstrap(
                ref_data,
                recent_data,
                point.impact,
            )
        lower, upper = self._confidence_bounds(
            point.impact,
            bootstrap_impacts,
        )
        evidence = self._classify_evidence(
            impact=point.impact,
            lower=lower,
            upper=upper,
            unresolved_mass=point.unresolved_mass,
            valid_bootstrap_samples=int(bootstrap_impacts.size),
        )
        return DetectionResult(
            q_reference=point.q_reference,
            q_recent=point.q_recent,
            reference_actions=point.reference_actions,
            recent_actions=point.recent_actions,
            confirmed_flip_mask=point.confirmed_flip_mask,
            unresolved_mask=point.unresolved_mask,
            tie_mask=point.tie_mask,
            state_occupancy=point.state_occupancy,
            impact=point.impact,
            impact_ci_lower=lower,
            impact_ci_upper=upper,
            unresolved_mass=point.unresolved_mass,
            evidence_level=evidence,
            reference_coverage=point.reference_coverage,
            recent_coverage=point.recent_coverage,
            valid_bootstrap_samples=int(bootstrap_impacts.size),
            bootstrap_impacts=bootstrap_impacts,
            config=self.config,
        )

    def _detect_from_arrays(self, reference: np.ndarray, recent: np.ndarray) -> DetectionResult:
        reference_estimate = estimate_empirical_model(reference, self.config)
        recent_estimate = estimate_empirical_model(recent, self.config)
        determined = reference_estimate.state_ready & recent_estimate.state_ready
        tie_mask = determined & (
            reference_estimate.tie_states | recent_estimate.tie_states
        )
        comparable = determined & ~tie_mask
        confirmed_flip = comparable & (
            reference_estimate.action_values != recent_estimate.action_values
        )
        unresolved = ~comparable
        occupancy = state_occupancy(recent, self.config)
        impact = float(np.dot(occupancy, confirmed_flip.astype(np.float64)))
        unresolved_mass = float(np.dot(occupancy, unresolved.astype(np.float64)))
        return DetectionResult(
            q_reference=reference_estimate.q_values,
            q_recent=recent_estimate.q_values,
            reference_actions=reference_estimate.action_values,
            recent_actions=recent_estimate.action_values,
            confirmed_flip_mask=confirmed_flip,
            unresolved_mask=unresolved,
            tie_mask=tie_mask,
            state_occupancy=occupancy,
            impact=impact,
            impact_ci_lower=impact,
            impact_ci_upper=impact,
            unresolved_mass=unresolved_mass,
            evidence_level="uncertain",
            reference_coverage=float(np.mean(reference_estimate.state_ready)),
            recent_coverage=float(np.mean(recent_estimate.state_ready)),
            valid_bootstrap_samples=0,
            bootstrap_impacts=np.empty(0, dtype=np.float64),
            config=self.config,
        )

    def _bootstrap(
        self,
        reference: np.ndarray,
        recent: np.ndarray,
        point_impact: float,
    ) -> np.ndarray:
        rng = np.random.default_rng(self.config.random_seed)
        impacts: list[float] = []
        for _ in range(self.config.bootstrap_samples):
            ref_indices = moving_block_indices(
                reference.shape[0],
                self.config.bootstrap_block_length,
                rng,
            )
            recent_indices = moving_block_indices(
                recent.shape[0],
                self.config.bootstrap_block_length,
                rng,
            )
            replicate = self._detect_from_arrays(
                reference[ref_indices],
                recent[recent_indices],
            )
            impacts.append(replicate.impact)

        if not impacts:
            impacts.append(point_impact)
        return np.asarray(impacts, dtype=np.float64)

    def _confidence_bounds(self, point: float, samples: np.ndarray) -> tuple[float, float]:
        if samples.size == 0:
            return point, point
        alpha = (1.0 - self.config.confidence_level) / 2.0
        lower = float(np.quantile(samples, alpha))
        upper = float(np.quantile(samples, 1.0 - alpha))
        return min(lower, point), max(upper, point)

    def _classify_evidence(
        self,
        *,
        impact: float,
        lower: float,
        upper: float,
        unresolved_mass: float,
        valid_bootstrap_samples: int,
    ) -> str:
        if unresolved_mass > self.config.max_unresolved_mass:
            return "uncertain"
        if valid_bootstrap_samples < self.config.min_valid_bootstrap_samples:
            return "uncertain"
        if upper <= self.config.delta0:
            return "low"
        if lower >= self.config.delta1:
            return "high"
        # Guard against a degenerate interval caused by zero bootstrap variance.
        if impact <= self.config.delta0 and unresolved_mass == 0.0:
            return "low"
        if impact >= self.config.delta1 and unresolved_mass == 0.0:
            return "high"
        return "uncertain"


def record_from_env_step(
    state: int,
    action: int,
    reward: float,
    next_state: int,
) -> TransitionRecord:
    """Convenience adapter from ``NonStationaryArenaEnv.step`` values."""
    return TransitionRecord(
        state=int(state),
        action=int(action),
        reward=float(reward),
        next_state=int(next_state),
    )
