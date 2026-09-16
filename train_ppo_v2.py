"""在星环守卫 II 上训练 PPO agent。

最小可运行示例：

    python3 train_ppo_v2.py \
        --scenario mixed_shift \
        --total-timesteps 200000 \
        --change-step 4000 \
        --checkpoint results/ppo_v2_mixed_shift.pt

训练策略只接收 observation、reward 和 done，不读取环境的 oracle_snapshot。
脚本不依赖 Gymnasium，便于后续迁移到 GPU 或 Stable-Baselines3。
"""

from __future__ import annotations

import argparse
import csv
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from advanced_arena_env import DeepSpaceExpeditionEnv


SCENARIOS = ["stable", "storm_surge", "route_rewrite", "resource_decay", "mixed_shift", "gradual_update"]


@dataclass
class PPOConfig:
    total_timesteps: int = 200_000
    num_envs: int = 8
    rollout_steps: int = 256
    update_epochs: int = 6
    minibatch_size: int = 256
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.015
    max_grad_norm: float = 0.5
    hidden_size: int = 128
    seed: int = 20260916
    change_step: Optional[int] = 4000
    device: str = "auto"


class ActorCritic(nn.Module):
    def __init__(self, observation_dim: int, action_dim: int, hidden_size: int = 128) -> None:
        super().__init__()
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.backbone = nn.Sequential(
            nn.Linear(observation_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
        )
        self.actor = nn.Linear(hidden_size, action_dim)
        self.critic = nn.Linear(hidden_size, 1)
        self._initialize()

    def _initialize(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0.0)
        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)

    def distribution_and_value(self, observations: torch.Tensor) -> Tuple[Categorical, torch.Tensor]:
        hidden = self.backbone(observations)
        distribution = Categorical(logits=self.actor(hidden))
        value = self.critic(hidden).squeeze(-1)
        return distribution, value

    def act(self, observations: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        distribution, value = self.distribution_and_value(observations)
        action = torch.argmax(distribution.logits, dim=-1) if deterministic else distribution.sample()
        log_probability = distribution.log_prob(action)
        return action, log_probability, value


class RolloutBuffer:
    def __init__(self, steps: int, envs: int, observation_dim: int) -> None:
        self.observations = np.zeros((steps, envs, observation_dim), dtype=np.float32)
        self.actions = np.zeros((steps, envs), dtype=np.int64)
        self.rewards = np.zeros((steps, envs), dtype=np.float32)
        self.dones = np.zeros((steps, envs), dtype=np.float32)
        self.log_probabilities = np.zeros((steps, envs), dtype=np.float32)
        self.values = np.zeros((steps, envs), dtype=np.float32)
        self.advantages = np.zeros((steps, envs), dtype=np.float32)
        self.returns = np.zeros((steps, envs), dtype=np.float32)

    def add(
        self,
        step: int,
        observations: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        dones: np.ndarray,
        log_probabilities: np.ndarray,
        values: np.ndarray,
    ) -> None:
        self.observations[step] = observations
        self.actions[step] = actions
        self.rewards[step] = rewards
        self.dones[step] = dones
        self.log_probabilities[step] = log_probabilities
        self.values[step] = values

    def finish(self, last_values: np.ndarray, last_dones: np.ndarray, gamma: float, gae_lambda: float) -> None:
        last_advantage = np.zeros_like(last_values)
        for step in reversed(range(self.rewards.shape[0])):
            if step == self.rewards.shape[0] - 1:
                next_values = last_values
                next_nonterminal = 1.0 - last_dones
            else:
                next_values = self.values[step + 1]
                next_nonterminal = 1.0 - self.dones[step]
            delta = self.rewards[step] + gamma * next_values * next_nonterminal - self.values[step]
            last_advantage = delta + gamma * gae_lambda * next_nonterminal * last_advantage
            self.advantages[step] = last_advantage
        self.returns = self.advantages + self.values

    def flattened(self) -> Dict[str, np.ndarray]:
        return {
            "observations": self.observations.reshape(-1, self.observations.shape[-1]),
            "actions": self.actions.reshape(-1),
            "log_probabilities": self.log_probabilities.reshape(-1),
            "advantages": self.advantages.reshape(-1),
            "returns": self.returns.reshape(-1),
            "values": self.values.reshape(-1),
        }


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def make_envs(config: PPOConfig, scenario: str) -> List[DeepSpaceExpeditionEnv]:
    environments = []
    for index in range(config.num_envs):
        environments.append(
            DeepSpaceExpeditionEnv(
                scenario=scenario,
                change_step=config.change_step,
                seed=config.seed + index * 9973,
            )
        )
    return environments


def ppo_update(
    model: ActorCritic,
    optimizer: torch.optim.Optimizer,
    data: Dict[str, np.ndarray],
    config: PPOConfig,
    device: torch.device,
) -> Dict[str, float]:
    advantages = data["advantages"]
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    total_samples = data["observations"].shape[0]
    minibatch_size = min(config.minibatch_size, total_samples)
    indices = np.arange(total_samples)
    metrics: Dict[str, List[float]] = {"policy_loss": [], "value_loss": [], "entropy": [], "approx_kl": [], "clip_fraction": []}

    for _ in range(config.update_epochs):
        np.random.shuffle(indices)
        for start in range(0, total_samples, minibatch_size):
            batch_indices = indices[start : start + minibatch_size]
            observations = torch.as_tensor(data["observations"][batch_indices], dtype=torch.float32, device=device)
            actions = torch.as_tensor(data["actions"][batch_indices], dtype=torch.long, device=device)
            old_log_probabilities = torch.as_tensor(data["log_probabilities"][batch_indices], dtype=torch.float32, device=device)
            batch_advantages = torch.as_tensor(advantages[batch_indices], dtype=torch.float32, device=device)
            returns = torch.as_tensor(data["returns"][batch_indices], dtype=torch.float32, device=device)

            distribution, values = model.distribution_and_value(observations)
            log_probabilities = distribution.log_prob(actions)
            entropy = distribution.entropy().mean()
            ratio = torch.exp(log_probabilities - old_log_probabilities)
            unclipped = ratio * batch_advantages
            clipped = torch.clamp(ratio, 1.0 - config.clip_range, 1.0 + config.clip_range) * batch_advantages
            policy_loss = -torch.min(unclipped, clipped).mean()
            value_loss = 0.5 * (returns - values).pow(2).mean()
            loss = policy_loss + config.value_coef * value_loss - config.entropy_coef * entropy

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()

            approx_kl = (old_log_probabilities - log_probabilities).mean().detach().cpu().item()
            clip_fraction = ((ratio - 1.0).abs() > config.clip_range).float().mean().detach().cpu().item()
            metrics["policy_loss"].append(policy_loss.detach().cpu().item())
            metrics["value_loss"].append(value_loss.detach().cpu().item())
            metrics["entropy"].append(entropy.detach().cpu().item())
            metrics["approx_kl"].append(approx_kl)
            metrics["clip_fraction"].append(clip_fraction)

    return {key: float(np.mean(values)) for key, values in metrics.items()}


def evaluate_policy(
    model: ActorCritic,
    scenario: str,
    episodes: int,
    seed: int,
    change_step: Optional[int],
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    environment = DeepSpaceExpeditionEnv(scenario=scenario, change_step=change_step, seed=seed)
    returns: List[float] = []
    successes = 0
    cargos: List[int] = []
    reasons: Dict[str, int] = {}
    for episode in range(episodes):
        observation, _ = environment.reset(seed=seed + episode, options={"hard_reset": episode == 0})
        terminated = truncated = False
        while not terminated and not truncated:
            tensor = torch.as_tensor(observation[None], dtype=torch.float32, device=device)
            with torch.no_grad():
                action, _, _ = model.act(tensor, deterministic=True)
            observation, _, terminated, truncated, _ = environment.step(int(action.item()))
        result = environment.completed_episodes[-1]
        returns.append(float(result["return"]))
        successes += int(result["success"])
        cargos.append(int(result["cargo"]))
        reasons[result["reason"]] = reasons.get(result["reason"], 0) + 1
    model.train()
    return {
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "success_rate": successes / max(1, episodes),
        "mean_cargo": float(np.mean(cargos)),
        "shield_failures": float(reasons.get("shield_depleted", 0)),
        "timeouts": float(reasons.get("horizon_reached", 0)),
    }


def save_checkpoint(
    path: Path,
    model: ActorCritic,
    optimizer: torch.optim.Optimizer,
    config: PPOConfig,
    scenario: str,
    observation_dim: int,
    action_dim: int,
    update: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": asdict(config),
            "scenario": scenario,
            "observation_dim": observation_dim,
            "action_dim": action_dim,
            "update": update,
        },
        path,
    )


def train(config: PPOConfig, scenario: str, checkpoint: Path, log_path: Path, eval_episodes: int) -> None:
    set_global_seed(config.seed)
    device = choose_device(config.device)
    environments = make_envs(config, scenario)
    observation_dim = environments[0].observation_dim
    action_dim = environments[0].action_space_n
    model = ActorCritic(observation_dim, action_dim, config.hidden_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, eps=1e-5)

    observations = []
    for index, environment in enumerate(environments):
        observation, _ = environment.reset(seed=config.seed + index * 9973, options={"hard_reset": True})
        observations.append(observation)
    current_observations = np.stack(observations).astype(np.float32)
    current_dones = np.zeros(config.num_envs, dtype=np.float32)
    batch_size = config.num_envs * config.rollout_steps
    updates = max(1, config.total_timesteps // batch_size)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "update", "timesteps", "mean_episode_return", "success_rate", "mean_cargo",
        "policy_loss", "value_loss", "entropy", "approx_kl", "clip_fraction",
    ]
    with log_path.open("w", newline="", encoding="utf-8") as log_file:
        writer = csv.DictWriter(log_file, fieldnames=fieldnames)
        writer.writeheader()
        recent_returns: List[float] = []
        recent_successes: List[float] = []
        recent_cargos: List[float] = []

        print(f"device={device} scenario={scenario} obs_dim={observation_dim} actions={action_dim}")
        print(f"rollout_batch={batch_size} updates={updates} change_step={config.change_step}")
        for update in range(1, updates + 1):
            buffer = RolloutBuffer(config.rollout_steps, config.num_envs, observation_dim)
            for step in range(config.rollout_steps):
                tensor = torch.as_tensor(current_observations, dtype=torch.float32, device=device)
                with torch.no_grad():
                    actions, log_probabilities, values = model.act(tensor)
                action_array = actions.cpu().numpy()
                log_probability_array = log_probabilities.cpu().numpy()
                value_array = values.cpu().numpy()

                next_observations = []
                rewards = np.zeros(config.num_envs, dtype=np.float32)
                dones = np.zeros(config.num_envs, dtype=np.float32)
                for index, environment in enumerate(environments):
                    next_observation, reward, terminated, truncated, info = environment.step(int(action_array[index]))
                    done = bool(terminated or truncated)
                    rewards[index] = float(reward)
                    dones[index] = float(done)
                    if done:
                        recent_returns.append(float(info["episode_return"]))
                        recent_successes.append(float(info["success"]))
                        recent_cargos.append(float(info["cargo"]))
                        next_observation, _ = environment.reset()
                    next_observations.append(next_observation)

                buffer.add(
                    step,
                    current_observations,
                    action_array,
                    rewards,
                    dones,
                    log_probability_array,
                    value_array,
                )
                current_observations = np.stack(next_observations).astype(np.float32)
                current_dones = dones

            with torch.no_grad():
                final_tensor = torch.as_tensor(current_observations, dtype=torch.float32, device=device)
                _, final_values = model.distribution_and_value(final_tensor)
            buffer.finish(final_values.cpu().numpy(), current_dones, config.gamma, config.gae_lambda)
            metrics = ppo_update(model, optimizer, buffer.flattened(), config, device)

            timesteps = update * batch_size
            row = {
                "update": update,
                "timesteps": timesteps,
                "mean_episode_return": float(np.mean(recent_returns[-100:])) if recent_returns else 0.0,
                "success_rate": float(np.mean(recent_successes[-100:])) if recent_successes else 0.0,
                "mean_cargo": float(np.mean(recent_cargos[-100:])) if recent_cargos else 0.0,
                **metrics,
            }
            writer.writerow(row)
            log_file.flush()
            if update == 1 or update % max(1, updates // 10) == 0 or update == updates:
                evaluation = evaluate_policy(
                    model,
                    scenario=scenario,
                    episodes=eval_episodes,
                    seed=config.seed + 100_000,
                    change_step=config.change_step,
                    device=device,
                )
                print(
                    f"update={update:4d}/{updates} steps={timesteps:7d} "
                    f"train_return={row['mean_episode_return']:7.2f} "
                    f"train_success={row['success_rate']:.2f} "
                    f"eval_return={evaluation['mean_return']:7.2f} "
                    f"eval_success={evaluation['success_rate']:.2f} "
                    f"entropy={metrics['entropy']:.3f}"
                )

    save_checkpoint(checkpoint, model, optimizer, config, scenario, observation_dim, action_dim, updates)
    final_evaluation = evaluate_policy(
        model,
        scenario=scenario,
        episodes=eval_episodes,
        seed=config.seed + 200_000,
        change_step=config.change_step,
        device=device,
    )
    print(f"checkpoint={checkpoint}")
    print(f"log={log_path}")
    print("final_evaluation=" + ", ".join(f"{key}={value:.3f}" for key, value in final_evaluation.items()))


def evaluate_checkpoint(path: Path, scenario: str, episodes: int, seed: int, device_name: str) -> None:
    device = choose_device(device_name)
    checkpoint = torch.load(path, map_location=device)
    model = ActorCritic(
        checkpoint["observation_dim"],
        checkpoint["action_dim"],
        checkpoint["config"]["hidden_size"],
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    result = evaluate_policy(
        model,
        scenario=scenario,
        episodes=episodes,
        seed=seed,
        change_step=checkpoint["config"].get("change_step"),
        device=device,
    )
    print("checkpoint_evaluation=" + ", ".join(f"{key}={value:.3f}" for key, value in result.items()))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO on the difficult Star Ring Arena II environment.")
    parser.add_argument("--scenario", choices=SCENARIOS, default="mixed_shift")
    parser.add_argument("--total-timesteps", type=int, default=200_000)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--rollout-steps", type=int, default=256)
    parser.add_argument("--update-epochs", type=int, default=6)
    parser.add_argument("--minibatch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--change-step", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=20260916)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--checkpoint", type=Path, default=Path("results/ppo_v2.pt"))
    parser.add_argument("--log-path", type=Path, default=Path("results/ppo_v2_training.csv"))
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--eval-only", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.eval_only:
        evaluate_checkpoint(args.eval_only, args.scenario, args.eval_episodes, args.seed, args.device)
        return
    if args.total_timesteps <= 0:
        raise SystemExit("--total-timesteps must be positive")
    config = PPOConfig(
        total_timesteps=args.total_timesteps,
        num_envs=args.num_envs,
        rollout_steps=args.rollout_steps,
        update_epochs=args.update_epochs,
        minibatch_size=args.minibatch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        change_step=args.change_step,
        device=args.device,
    )
    train(config, args.scenario, args.checkpoint, args.log_path, args.eval_episodes)


if __name__ == "__main__":
    main()
