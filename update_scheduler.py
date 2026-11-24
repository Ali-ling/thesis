"""Map decision evidence to the number of PPO update epochs.

The first implementation deliberately changes only ``update_epochs``. The
rollout batch, clipping, GAE, optimizer, and minibatch size remain identical
across evidence levels so that the scheduling effect is easier to attribute.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from decision_detector import DetectionResult


@dataclass(frozen=True)
class UpdateSchedule:
    base_epochs: int = 3
    standard_epochs: int = 6
    strong_epochs: int = 9

    def __post_init__(self) -> None:
        if min(self.base_epochs, self.standard_epochs, self.strong_epochs) < 1:
            raise ValueError("all update epoch counts must be positive")
        if not self.base_epochs <= self.standard_epochs <= self.strong_epochs:
            raise ValueError("expected base_epochs <= standard_epochs <= strong_epochs")


class EvidenceDrivenScheduler:
    """Select update strength from a detector result.

    ``enabled=False`` reproduces a fixed PPO schedule and is intended as a
    regression check before comparing full experimental methods.
    """

    def __init__(
        self,
        schedule: Optional[UpdateSchedule] = None,
        *,
        enabled: bool = True,
        fixed_epochs: Optional[int] = None,
    ) -> None:
        self.schedule = schedule or UpdateSchedule()
        self.enabled = enabled
        self.fixed_epochs = fixed_epochs if fixed_epochs is not None else self.schedule.standard_epochs
        if self.fixed_epochs < 1:
            raise ValueError("fixed_epochs must be positive")

    def epochs_for(self, result: Optional[DetectionResult]) -> int:
        if result is None:
            return self.fixed_epochs
        return self.epochs_for_level(result.evidence_level)

    def epochs_for_level(self, evidence_level: Optional[str]) -> int:
        if not self.enabled or evidence_level is None:
            return self.fixed_epochs
        if evidence_level == "low":
            return self.schedule.base_epochs
        if evidence_level == "high":
            return self.schedule.strong_epochs
        return self.schedule.standard_epochs


class PeriodicUpdateController:
    """Trigger strong updates once whenever a fixed time bucket is crossed."""

    def __init__(
        self,
        interval: int,
        schedule: Optional[UpdateSchedule] = None,
    ) -> None:
        if interval <= 0:
            raise ValueError("periodic interval must be positive")
        self.interval = int(interval)
        self.schedule = schedule or UpdateSchedule()
        self.last_bucket = -1

    def epochs_for(self, timesteps: int) -> int:
        bucket = int(timesteps) // self.interval
        if bucket > self.last_bucket:
            self.last_bucket = bucket
            return self.schedule.strong_epochs
        return self.schedule.standard_epochs


class CusumDriftController:
    """Two-sided CUSUM baseline over one-step TD residuals.

    The detector triggers strong adaptation when either the positive or
    negative cumulative residual exceeds ``threshold``. After a trigger both
    statistics reset to zero.
    """

    def __init__(
        self,
        *,
        threshold: float = 1.0,
        allowance: float = 0.1,
        warmup_steps: int = 4,
        baseline_rate: float = 0.05,
        detect_positive: bool = True,
        detect_negative: bool = True,
        schedule: Optional[UpdateSchedule] = None,
    ) -> None:
        if threshold <= 0.0:
            raise ValueError("CUSUM threshold must be positive")
        if allowance < 0.0:
            raise ValueError("CUSUM allowance cannot be negative")
        if warmup_steps < 0:
            raise ValueError("CUSUM warmup_steps cannot be negative")
        if not 0.0 <= baseline_rate < 1.0:
            raise ValueError("CUSUM baseline_rate must be in [0, 1)")
        self.threshold = float(threshold)
        self.allowance = float(allowance)
        self.warmup_steps = int(warmup_steps)
        self.baseline_rate = float(baseline_rate)
        self.detect_positive = bool(detect_positive)
        self.detect_negative = bool(detect_negative)
        if not self.detect_positive and not self.detect_negative:
            raise ValueError("at least one CUSUM direction must be enabled")
        self.schedule = schedule or UpdateSchedule()
        self.mean_residual = 0.0
        self.residual_variance = 1.0
        self.samples_seen = 0
        self.positive_statistic = 0.0
        self.negative_statistic = 0.0
        self.trigger_count = 0

    def update(self, residual: float) -> bool:
        value = float(residual)
        if self.samples_seen == 0:
            self.mean_residual = value
        else:
            difference = value - self.mean_residual
            self.mean_residual += self.baseline_rate * difference
            self.residual_variance = (1.0 - self.baseline_rate) * (
                self.residual_variance
                + self.baseline_rate * difference * difference
            )
        self.samples_seen += 1
        standard_deviation = max(0.05, self.residual_variance**0.5)
        centered_residual = (value - self.mean_residual) / standard_deviation
        if self.detect_positive:
            self.positive_statistic = max(
                0.0,
                self.positive_statistic + centered_residual - self.allowance,
            )
        if self.detect_negative:
            self.negative_statistic = max(
                0.0,
                self.negative_statistic - centered_residual - self.allowance,
            )
        if self.samples_seen <= self.warmup_steps:
            return False
        triggered = (
            self.positive_statistic >= self.threshold
            or self.negative_statistic >= self.threshold
        )
        if triggered:
            self.positive_statistic = 0.0
            self.negative_statistic = 0.0
            self.trigger_count += 1
        return triggered

    def epochs_for(self, triggered: bool) -> int:
        return (
            self.schedule.strong_epochs
            if triggered
            else self.schedule.standard_epochs
        )

