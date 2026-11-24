"""Standard PPO with online decision-evidence scheduling on the 8-state arena.

The main experiment separates two phases:

1. ``pretrain`` trains one stable-environment checkpoint.
2. ``adapt`` starts from that checkpoint, changes the environment, continues
   collecting rollouts, and optionally schedules PPO update epochs from the
   decision detector.

Examples:

    python train_ppo_arena.py --mode pretrain --timesteps 20000 \
        --checkpoint results/arena_ppo_pretrained.pt

    python train_ppo_arena.py --mode adapt --scenario local_flip \
        --load-checkpoint results/arena_ppo_pretrained.pt \
        --schedule decision --timesteps 20000 \
        --checkpoint results/arena_ppo_local_flip_decision.pt
"""

from __future__ import annotations

import argparse
import csv
import gzip
import random
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Deque, Dict, Optional

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from arena_env import NonStationaryArenaEnv
from decision_detector import (
    DecisionChangeDetector,
    DetectionResult,
    DetectorConfig,
    TransitionRecord,
)
from update_scheduler import (
    CusumDriftController,
    EvidenceDrivenScheduler,
    PeriodicUpdateController,
    UpdateSchedule,
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
class ArenaPPOConfig:
    timesteps: int = 20_000
    rollout_steps: int = 256
    update_epochs: int = 6
    base_update_epochs: int = 3
    strong_update_epochs: int = 9
    minibatch_size: int = 64
    learning_rate: float = 3e-4
    gamma: float = 0.92
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.02
    behavior_epsilon: float = 0.0
    max_grad_norm: float = 0.5
    hidden_size: int = 64
    seed: int = 20260916
    change_step: int = 4_000
    monitor_window: int = 1_500
    monitor_interval: int = 256
    monitor_min_visits: int = 2
    bootstrap_samples: int = 60
    bootstrap_block_length: int = 10
    periodic_interval: int = 1_024
    cusum_threshold: float = 1.0
    cusum_allowance: float = 0.1
    cusum_warmup: int = 4
    cusum_baseline_rate: float = 0.05
    device: str = "cpu"

    def __post_init__(self) -> None:
        positive = {
            "timesteps": self.timesteps,
            "rollout_steps": self.rollout_steps,
            "update_epochs": self.update_epochs,
            "base_update_epochs": self.base_update_epochs,
            "strong_update_epochs": self.strong_update_epochs,
            "minibatch_size": self.minibatch_size,
            "hidden_size": self.hidden_size,
            "monitor_window": self.monitor_window,
            "monitor_interval": self.monitor_interval,
            "monitor_min_visits": self.monitor_min_visits,
            "bootstrap_samples": self.bootstrap_samples,
            "bootstrap_block_length": self.bootstrap_block_length,
            "periodic_interval": self.periodic_interval,
        }
        invalid = [name for name, value in positive.items() if value <= 0]
        if invalid:
            raise ValueError(f"parameters must be positive: {', '.join(invalid)}")
        if not 0.0 <= self.gamma < 1.0:
            raise ValueError("gamma must be in [0, 1)")
        if not 0.0 <= self.gae_lambda < 1.0:
            raise ValueError("gae_lambda must be in [0, 1)")
        if not 0.0 <= self.behavior_epsilon < 1.0:
            raise ValueError("behavior_epsilon must be in [0, 1)")
        if not (
            self.base_update_epochs
            <= self.update_epochs
            <= self.strong_update_epochs
        ):
            raise ValueError(
                "expected base_update_epochs <= update_epochs <= strong_update_epochs"
            )
        if self.cusum_threshold <= 0.0 or self.cusum_allowance < 0.0:
            raise ValueError("CUSUM threshold must be positive and allowance non-negative")
        if self.cusum_warmup < 0 or not 0.0 <= self.cusum_baseline_rate < 1.0:
            raise ValueError("CUSUM warmup and baseline rate are invalid")


@dataclass
class RolloutBatch:
    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    log_probabilities: np.ndarray
    values: np.ndarray
    advantages: np.ndarray
    returns: np.ndarray

    @classmethod
    def empty(cls, steps: int, observation_dim: int) -> "RolloutBatch":
        return cls(
            observations=np.zeros((steps, observation_dim), dtype=np.float32),
            actions=np.zeros(steps, dtype=np.int64),
            rewards=np.zeros(steps, dtype=np.float32),
            dones=np.zeros(steps, dtype=np.float32),
            log_probabilities=np.zeros(steps, dtype=np.float32),
            values=np.zeros(steps, dtype=np.float32),
            advantages=np.zeros(steps, dtype=np.float32),
            returns=np.zeros(steps, dtype=np.float32),
        )


class ArenaActorCritic(nn.Module):
    def __init__(self, state_count: int, action_count: int, hidden_size: int = 64) -> None:
        super().__init__()
        self.state_count = state_count
        self.action_count = action_count
        self.backbone = nn.Sequential(
            nn.Linear(state_count, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
        )
        self.actor = nn.Linear(hidden_size, action_count)
        self.critic = nn.Linear(hidden_size, 1)
        self._initialize()

    def _initialize(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2.0))
                nn.init.constant_(module.bias, 0.0)
        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)

    def distribution_and_value(
        self,
        observations: torch.Tensor,
    ) -> tuple[Categorical, torch.Tensor]:
        hidden = self.backbone(observations)
        return Categorical(logits=self.actor(hidden)), self.critic(hidden).squeeze(-1)

    def act(
        self,
        observations: torch.Tensor,
        *,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        distribution, values = self.distribution_and_value(observations)
        actions = (
            torch.argmax(distribution.logits, dim=-1)
            if deterministic
            else distribution.sample()
        )
        return actions, distribution.log_prob(actions), values


def encode_state(state: int, state_count: int) -> np.ndarray:
    observation = np.zeros(state_count, dtype=np.float32)
    observation[int(state)] = 1.0
    return observation


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    last_value: float,
    gamma: float,
    gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return GAE advantages and value targets for one trajectory stream."""
    advantages = np.zeros_like(rewards, dtype=np.float32)
    next_advantage = 0.0
    for step in reversed(range(len(rewards))):
        next_value = float(last_value) if step == len(rewards) - 1 else float(values[step + 1])
        next_nonterminal = 1.0 if step == len(rewards) - 1 else 1.0 - float(dones[step])
        delta = (
            float(rewards[step])
            + gamma * next_value * next_nonterminal
            - float(values[step])
        )
        next_advantage = delta + gamma * gae_lambda * next_nonterminal * next_advantage
        advantages[step] = next_advantage
    return advantages, advantages + values


def ppo_update(
    model: ArenaActorCritic,
    optimizer: torch.optim.Optimizer,
    batch: RolloutBatch,
    *,
    epochs: int,
    minibatch_size: int,
    clip_range: float,
    value_coef: float,
    entropy_coef: float,
    max_grad_norm: float,
    device: torch.device,
    rng: np.random.Generator,
) -> Dict[str, float]:
    advantages = (batch.advantages - batch.advantages.mean()) / (
        batch.advantages.std() + 1e-8
    )
    sample_count = len(batch.rewards)
    size = min(minibatch_size, sample_count)
    metrics: Dict[str, list[float]] = {
        "policy_loss": [],
        "value_loss": [],
        "entropy": [],
        "approx_kl": [],
        "clip_fraction": [],
    }

    for _ in range(epochs):
        indices = rng.permutation(sample_count)
        for start in range(0, sample_count, size):
            selected = indices[start : start + size]
            observations = torch.as_tensor(
                batch.observations[selected],
                dtype=torch.float32,
                device=device,
            )
            actions = torch.as_tensor(
                batch.actions[selected],
                dtype=torch.long,
                device=device,
            )
            old_log_probabilities = torch.as_tensor(
                batch.log_probabilities[selected],
                dtype=torch.float32,
                device=device,
            )
            selected_advantages = torch.as_tensor(
                advantages[selected],
                dtype=torch.float32,
                device=device,
            )
            returns = torch.as_tensor(
                batch.returns[selected],
                dtype=torch.float32,
                device=device,
            )

            distribution, values = model.distribution_and_value(observations)
            log_probabilities = distribution.log_prob(actions)
            ratio = torch.exp(log_probabilities - old_log_probabilities)
            unclipped = ratio * selected_advantages
            clipped = (
                torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range)
                * selected_advantages
            )
            policy_loss = -torch.min(unclipped, clipped).mean()
            value_loss = 0.5 * (returns - values).pow(2).mean()
            entropy = distribution.entropy().mean()
            loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

            metrics["policy_loss"].append(float(policy_loss.detach().cpu()))
            metrics["value_loss"].append(float(value_loss.detach().cpu()))
            metrics["entropy"].append(float(entropy.detach().cpu()))
            metrics["approx_kl"].append(
                float((old_log_probabilities - log_probabilities).mean().detach().cpu())
            )
            metrics["clip_fraction"].append(
                float(((ratio - 1.0).abs() > clip_range).float().mean().detach().cpu())
            )
    return {key: float(np.mean(values)) for key, values in metrics.items()}


def exact_policy_value(
    model: ArenaActorCritic,
    rewards: np.ndarray,
    transitions: np.ndarray,
    gamma: float,
    initial_state: int,
    device: torch.device,
) -> float:
    """Evaluate the stochastic policy exactly through its induced Markov chain."""
    state_count = rewards.shape[0]
    observations = np.eye(state_count, dtype=np.float32)
    model.eval()
    with torch.no_grad():
        distribution, _ = model.distribution_and_value(
            torch.as_tensor(observations, device=device)
        )
        action_probabilities = distribution.probs.detach().cpu().numpy().astype(np.float64)
    model.train()
    policy_rewards = np.sum(action_probabilities * rewards, axis=1)
    policy_transitions = np.einsum(
        "sa,san->sn",
        action_probabilities,
        transitions,
    )
    values = np.linalg.solve(
        np.eye(state_count, dtype=np.float64) - gamma * policy_transitions,
        policy_rewards,
    )
    return float(values[int(initial_state)])


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        # The main finite-state experiment is intentionally CPU-first.
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but this PyTorch build has no CUDA support; "
            "install a CUDA-enabled PyTorch build or use --device cpu"
        )
    return torch.device(requested)


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_checkpoint(
    path: Path,
    model: ArenaActorCritic,
    optimizer: torch.optim.Optimizer,
    config: ArenaPPOConfig,
    scenario: str,
    timesteps: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": asdict(config),
            "scenario": scenario,
            "timesteps": timesteps,
            "state_count": model.state_count,
            "action_count": model.action_count,
        },
        path,
    )


class ArenaPPOTrainer:
    def __init__(
        self,
        config: ArenaPPOConfig,
        *,
        scenario: str,
        schedule_mode: str,
        checkpoint_input: Optional[Path] = None,
        update_schedule: Optional[UpdateSchedule] = None,
    ) -> None:
        if scenario not in SCENARIOS:
            raise ValueError(f"unknown scenario: {scenario}")
        if schedule_mode not in {"fixed", "periodic", "drift", "decision", "oracle"}:
            raise ValueError(
                "schedule_mode must be 'fixed', 'periodic', 'drift', 'decision', or 'oracle'"
            )

        self.config = config
        self.schedule_mode = schedule_mode
        self.device = resolve_device(config.device)
        set_global_seed(config.seed)
        self.env = NonStationaryArenaEnv(
            scenario=scenario,
            change_step=config.change_step,
            seed=config.seed,
            emit_oracle_info=False,
        )
        self.state, _ = self.env.reset(
            seed=config.seed,
            options={"hard_reset": True},
        )
        self.model = ArenaActorCritic(
            self.env.n_states,
            self.env.n_actions,
            config.hidden_size,
        ).to(self.device)
        if checkpoint_input is not None:
            checkpoint = torch.load(
                checkpoint_input,
                map_location=self.device,
                weights_only=False,
            )
            if int(checkpoint["state_count"]) != self.env.n_states:
                raise ValueError("checkpoint state count does not match the arena")
            if int(checkpoint["action_count"]) != self.env.n_actions:
                raise ValueError("checkpoint action count does not match the arena")
            self.model.load_state_dict(checkpoint["model"])
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.learning_rate,
            eps=1e-5,
        )
        self.rng = np.random.default_rng(config.seed + 1)
        self.monitor: Deque[TransitionRecord] = deque(
            maxlen=2 * config.monitor_window
        )
        detector_config = DetectorConfig(
            n_states=self.env.n_states,
            n_actions=self.env.n_actions,
            gamma=self.env.gamma,
            min_visits_per_state_action=config.monitor_min_visits,
            tie_tolerance=self.env.tie_tolerance,
            bootstrap_samples=config.bootstrap_samples,
            bootstrap_block_length=min(
                config.bootstrap_block_length,
                config.monitor_window,
            ),
            random_seed=config.seed,
        )
        self.detector = DecisionChangeDetector(detector_config)
        self.update_schedule = update_schedule or UpdateSchedule(
            base_epochs=config.base_update_epochs,
            standard_epochs=config.update_epochs,
            strong_epochs=config.strong_update_epochs,
        )
        self.scheduler = EvidenceDrivenScheduler(
            self.update_schedule,
            enabled=schedule_mode in {"decision", "oracle"},
            fixed_epochs=config.update_epochs,
        )
        self.periodic_controller = PeriodicUpdateController(
            config.periodic_interval,
            self.update_schedule,
        )
        self.cusum_controller = CusumDriftController(
            threshold=config.cusum_threshold,
            allowance=config.cusum_allowance,
            warmup_steps=config.cusum_warmup,
            baseline_rate=config.cusum_baseline_rate,
            detect_positive=False,
            detect_negative=True,
            schedule=self.update_schedule,
        )
        self.last_detection: Optional[DetectionResult] = None
        self.last_controller_event = ""
        self.last_controller_triggered = False
        self.last_oracle_impact = 0.0
        self.last_oracle_level = "low"
        self.last_oracle_value = 0.0
        self.oracle_reference_value = 0.0
        self.episode_returns: Deque[float] = deque(maxlen=20)
        self.rows: list[dict[str, object]] = []
        self.trajectory_rows: list[dict[str, object]] = []
        self.timesteps = 0
        self.steps_since_detection = config.monitor_interval
        self.rollout_cusum_triggers = 0
        self.last_cusum_mean_residual = 0.0
        if schedule_mode == "oracle":
            reference_rewards, reference_transitions = self.env.model_at(
                max(0, self.env.change_step - 1)
            )
            self.oracle_reference_value = exact_policy_value(
                self.model,
                np.asarray(reference_rewards, dtype=np.float64),
                np.asarray(reference_transitions, dtype=np.float64),
                self.env.gamma,
                int(self.env.protocol["initial_state"]),
                self.device,
            )

    def train(self) -> None:
        update_index = 0
        while self.timesteps < self.config.timesteps:
            steps = min(
                self.config.rollout_steps,
                self.config.timesteps - self.timesteps,
            )
            batch, last_value = self._collect_rollout(steps)
            if self.schedule_mode == "drift":
                self._maybe_measure_reward_drift()
            if self.schedule_mode == "oracle":
                self._update_oracle_decision()
            batch.advantages, batch.returns = compute_gae(
                batch.rewards,
                batch.values,
                batch.dones,
                last_value,
                self.config.gamma,
                self.config.gae_lambda,
            )
            self.steps_since_detection += steps
            detection_ran = self._maybe_detect()
            next_timesteps = self.timesteps + steps
            epochs, controller_event, controller_triggered = self._select_epochs(
                next_timesteps,
                detection_ran,
            )
            metrics = ppo_update(
                self.model,
                self.optimizer,
                batch,
                epochs=epochs,
                minibatch_size=self.config.minibatch_size,
                clip_range=self.config.clip_range,
                value_coef=self.config.value_coef,
                entropy_coef=self.config.entropy_coef,
                max_grad_norm=self.config.max_grad_norm,
                device=self.device,
                rng=self.rng,
            )
            update_index += 1
            self.timesteps = next_timesteps
            self._log_row(
                update_index,
                int(detection_ran),
                epochs,
                metrics,
                controller_event,
                controller_triggered,
            )
            self.rollout_cusum_triggers = 0

    def _select_epochs(
        self,
        next_timesteps: int,
        detection_ran: bool,
    ) -> tuple[int, str, bool]:
        if self.schedule_mode == "fixed":
            return self.config.update_epochs, "", False
        if self.schedule_mode == "periodic":
            epochs = self.periodic_controller.epochs_for(next_timesteps)
            triggered = epochs == self.update_schedule.strong_epochs
            return epochs, "periodic" if triggered else "", triggered
        if self.schedule_mode == "drift":
            triggered = self.rollout_cusum_triggers > 0
            return (
                self.cusum_controller.epochs_for(triggered),
                "cusum" if triggered else "",
                triggered,
            )
        if self.schedule_mode == "oracle":
            epochs = self.scheduler.epochs_for_level(self.last_oracle_level)
            triggered = self.last_oracle_level == "high"
            return epochs, "oracle_high" if triggered else self.last_oracle_level, triggered

        epochs = self.scheduler.epochs_for(self.last_detection)
        if not detection_ran or self.last_detection is None:
            return epochs, "", False
        if self.last_detection.evidence_level == "high":
            return epochs, "decision_high", True
        return epochs, self.last_detection.evidence_level, False

    def _collect_rollout(self, steps: int) -> tuple[RolloutBatch, float]:
        batch = RolloutBatch.empty(steps, self.env.n_states)
        last_done = False
        for index in range(steps):
            observation = encode_state(self.state, self.env.n_states)
            tensor = torch.as_tensor(
                observation[None],
                dtype=torch.float32,
                device=self.device,
            )
            with torch.no_grad():
                distribution, value_tensor = self.model.distribution_and_value(tensor)
                target_probabilities = distribution.probs
            behavior_probabilities = (
                (1.0 - self.config.behavior_epsilon) * target_probabilities
                + self.config.behavior_epsilon / self.env.n_actions
            )
            action_tensor = torch.multinomial(
                behavior_probabilities,
                num_samples=1,
            ).squeeze(-1)
            log_probability_tensor = torch.log(
                behavior_probabilities[0, action_tensor].clamp_min(1e-12)
            )
            action = int(action_tensor.item())
            state_before = self.state
            environment_step = self.env.global_step
            next_state, reward, terminated, truncated, info = self.env.step(action)
            done = bool(terminated or truncated)
            self.trajectory_rows.append(
                {
                    "timestep": environment_step,
                    "state": state_before,
                    "action": action,
                    "reward": float(reward),
                    "next_state": next_state,
                    "terminated": int(terminated),
                    "truncated": int(truncated),
                    "phase": info["phase"],
                    "episode_index": info["episode_index"],
                }
            )
            batch.observations[index] = observation
            batch.actions[index] = action
            batch.rewards[index] = float(reward)
            batch.dones[index] = float(done)
            batch.log_probabilities[index] = float(log_probability_tensor.item())
            batch.values[index] = float(value_tensor.item())
            self.monitor.append(
                TransitionRecord(
                    state=state_before,
                    action=action,
                    reward=float(reward),
                    next_state=next_state,
                )
            )
            last_done = done
            if done:
                self.episode_returns.append(float(info["episode_return"]))
                self.state, _ = self.env.reset()
            else:
                self.state = next_state

        final_observation = torch.as_tensor(
            encode_state(self.state, self.env.n_states)[None],
            dtype=torch.float32,
            device=self.device,
        )
        with torch.no_grad():
            _, final_value = self.model.distribution_and_value(final_observation)
        return batch, 0.0 if last_done else float(final_value.item())

    def _maybe_measure_reward_drift(self) -> None:
        if self.steps_since_detection < self.config.monitor_interval:
            return
        if len(self.monitor) < 2 * self.config.monitor_window:
            return
        self.steps_since_detection = 0
        records = list(self.monitor)
        reference_rewards = np.asarray(
            [record.reward for record in records[: self.config.monitor_window]],
            dtype=np.float64,
        )
        recent_rewards = np.asarray(
            [record.reward for record in records[self.config.monitor_window :]],
            dtype=np.float64,
        )
        mean_difference = float(recent_rewards.mean() - reference_rewards.mean())
        standard_error = float(
            np.sqrt(
                recent_rewards.var(ddof=1) / len(recent_rewards)
                + reference_rewards.var(ddof=1) / len(reference_rewards)
            )
        )
        self.last_cusum_mean_residual = mean_difference / max(standard_error, 0.02)
        if self.cusum_controller.update(self.last_cusum_mean_residual):
            self.rollout_cusum_triggers = 1

    def _maybe_detect(self) -> bool:
        if self.schedule_mode != "decision":
            return False
        if self.steps_since_detection < self.config.monitor_interval:
            return False
        if len(self.monitor) < 2 * self.config.monitor_window:
            return False
        self.steps_since_detection = 0
        records = list(self.monitor)
        self.last_detection = self.detector.detect(
            records[: self.config.monitor_window],
            records[self.config.monitor_window :],
            run_bootstrap=True,
        )
        return True

    def _update_oracle_decision(self) -> None:
        current_time = max(0, self.env.global_step - 1)
        rewards, transitions = self.env.model_at(current_time)
        self.last_oracle_value = exact_policy_value(
            self.model,
            np.asarray(rewards, dtype=np.float64),
            np.asarray(transitions, dtype=np.float64),
            self.env.gamma,
            int(self.env.protocol["initial_state"]),
            self.device,
        )
        retention = (
            self.last_oracle_value / self.oracle_reference_value
            if self.oracle_reference_value > 0.0
            else 1.0
        )
        self.last_oracle_impact = max(0.0, 1.0 - retention)
        if self.last_oracle_impact >= 0.05:
            self.last_oracle_level = "high"
        elif self.last_oracle_impact >= 0.01:
            self.last_oracle_level = "uncertain"
        else:
            self.last_oracle_level = "low"

    def _log_row(
        self,
        update: int,
        detection_ran: int,
        epochs: int,
        metrics: Dict[str, float],
        controller_event: str,
        controller_triggered: bool,
    ) -> None:
        rewards, transitions = self.env.model_at()
        exact_value = exact_policy_value(
            self.model,
            np.asarray(rewards, dtype=np.float64),
            np.asarray(transitions, dtype=np.float64),
            self.env.gamma,
            int(self.env.protocol["initial_state"]),
            self.device,
        )
        detection = self.last_detection
        row = {
            "update": update,
            "timesteps": self.timesteps,
            "environment_step": self.env.global_step,
            "phase": self.env.phase_at(),
            "schedule_mode": self.schedule_mode,
            "mean_episode_return": float(np.mean(self.episode_returns))
            if self.episode_returns
            else 0.0,
            "exact_policy_value": exact_value,
            "detection_ran": detection_ran,
            "controller_event": controller_event,
            "controller_triggered": int(controller_triggered),
            "cusum_positive": self.cusum_controller.positive_statistic,
            "cusum_negative": self.cusum_controller.negative_statistic,
            "cusum_trigger_count": self.cusum_controller.trigger_count,
            "reward_drift_effect": self.last_cusum_mean_residual,
            "impact": detection.impact if detection_ran and detection else "",
            "ci_lower": detection.impact_ci_lower
            if detection_ran and detection
            else "",
            "ci_upper": detection.impact_ci_upper
            if detection_ran and detection
            else "",
            "evidence_level": detection.evidence_level
            if detection_ran and detection
            else "",
            "unresolved_mass": detection.unresolved_mass
            if detection_ran and detection
            else "",
            "oracle_impact": self.last_oracle_impact
            if self.schedule_mode == "oracle"
            else "",
            "oracle_level": self.last_oracle_level
            if self.schedule_mode == "oracle"
            else "",
            "oracle_value": self.last_oracle_value
            if self.schedule_mode == "oracle"
            else "",
            "ppo_epochs": epochs,
            **metrics,
        }
        self.rows.append(row)

    def write_log(self, path: Path) -> None:
        if not self.rows:
            raise ValueError("no training rows to write")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8-sig") as file:
            writer = csv.DictWriter(file, fieldnames=list(self.rows[0]))
            writer.writeheader()
            writer.writerows(self.rows)

    def write_trajectory(self, path: Path) -> None:
        if not self.trajectory_rows:
            raise ValueError("no trajectory rows to write")
        path.parent.mkdir(parents=True, exist_ok=True)
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "wt", newline="", encoding="utf-8-sig") as file:
            writer = csv.DictWriter(file, fieldnames=list(self.trajectory_rows[0]))
            writer.writeheader()
            writer.writerows(self.trajectory_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train PPO on the finite arena with optional evidence scheduling."
    )
    parser.add_argument("--mode", choices=["pretrain", "adapt"], default="adapt")
    parser.add_argument("--scenario", choices=SCENARIOS, default="local_flip")
    parser.add_argument(
        "--schedule",
        choices=["fixed", "periodic", "drift", "decision", "oracle"],
        default="decision",
    )
    parser.add_argument("--timesteps", type=int, default=20_000)
    parser.add_argument("--rollout-steps", type=int, default=256)
    parser.add_argument("--update-epochs", type=int, default=6)
    parser.add_argument("--base-epochs", type=int, default=3)
    parser.add_argument("--strong-epochs", type=int, default=9)
    parser.add_argument("--minibatch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--entropy-coef", type=float, default=0.02)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--behavior-epsilon", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=20260916)
    parser.add_argument("--change-step", type=int, default=4_000)
    parser.add_argument("--monitor-window", type=int, default=1_500)
    parser.add_argument("--monitor-interval", type=int, default=256)
    parser.add_argument("--monitor-min-visits", type=int, default=2)
    parser.add_argument("--bootstrap-samples", type=int, default=60)
    parser.add_argument("--bootstrap-block-length", type=int, default=10)
    parser.add_argument("--periodic-interval", type=int, default=1_024)
    parser.add_argument("--cusum-threshold", type=float, default=1.0)
    parser.add_argument("--cusum-allowance", type=float, default=0.1)
    parser.add_argument("--cusum-warmup", type=int, default=4)
    parser.add_argument("--cusum-baseline-rate", type=float, default=0.05)
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
    parser.add_argument("--load-checkpoint", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--log-path", type=Path, default=None)
    parser.add_argument("--trajectory-path", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scenario = "stable" if args.mode == "pretrain" else args.scenario
    schedule_mode = "fixed" if args.mode == "pretrain" else args.schedule
    checkpoint = args.checkpoint or Path(
        "results/arena_ppo_pretrained.pt"
        if args.mode == "pretrain"
        else f"results/arena_ppo_{scenario}_{schedule_mode}.pt"
    )
    log_path = args.log_path or Path(
        "results/arena_ppo_pretrained.csv"
        if args.mode == "pretrain"
        else f"results/arena_ppo_{scenario}_{schedule_mode}.csv"
    )
    config = ArenaPPOConfig(
        timesteps=args.timesteps,
        rollout_steps=args.rollout_steps,
        update_epochs=args.update_epochs,
        base_update_epochs=args.base_epochs,
        strong_update_epochs=args.strong_epochs,
        minibatch_size=args.minibatch_size,
        learning_rate=args.learning_rate,
        entropy_coef=args.entropy_coef,
        behavior_epsilon=args.behavior_epsilon,
        hidden_size=args.hidden_size,
        seed=args.seed,
        change_step=args.change_step,
        monitor_window=args.monitor_window,
        monitor_interval=args.monitor_interval,
        monitor_min_visits=args.monitor_min_visits,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_block_length=args.bootstrap_block_length,
        periodic_interval=args.periodic_interval,
        cusum_threshold=args.cusum_threshold,
        cusum_allowance=args.cusum_allowance,
        cusum_warmup=args.cusum_warmup,
        cusum_baseline_rate=args.cusum_baseline_rate,
        device=args.device,
    )
    trainer = ArenaPPOTrainer(
        config,
        scenario=scenario,
        schedule_mode=schedule_mode,
        checkpoint_input=args.load_checkpoint,
    )
    trainer.train()
    trainer.write_log(log_path)
    if args.trajectory_path is not None:
        trainer.write_trajectory(args.trajectory_path)
    save_checkpoint(
        checkpoint,
        trainer.model,
        trainer.optimizer,
        config,
        scenario,
        trainer.timesteps,
    )
    final = trainer.rows[-1]
    final_event = (
        str(final["controller_event"])
        or str(final["evidence_level"])
        or "none"
    )
    print(
        f"mode={args.mode} scenario={scenario} schedule={schedule_mode} "
        f"timesteps={trainer.timesteps} exact_value={final['exact_policy_value']:.4f} "
        f"last_event={final_event}"
    )
    print(f"checkpoint={checkpoint}")
    print(f"log={log_path}")


if __name__ == "__main__":
    main()
