"""星环守卫的可复现非平稳有限状态环境。

环境协议与 arena_protocol.json、app.js 共用同一份配置：
- 8 个离散状态：S0...S7
- 2 个离散动作：A0 脉冲突击、A1 相位防守
- 6 类可控变化场景
- 变化前后可通过已知模型精确计算 Q* 和最优动作

接口采用 Gymnasium 风格，但不强制依赖第三方包，便于先接入自写
PPO、Stable-Baselines3 或其他训练框架。
"""

from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


ROOT = Path(__file__).resolve().parent
DEFAULT_PROTOCOL_PATH = ROOT / "arena_protocol.json"


Model = Tuple[List[List[float]], List[List[List[float]]]]


class NonStationaryArenaEnv:
    """可计算真值的时变 MDP。

    ``global_step`` 跨 episode 累计，因此环境变化不会因为 episode 结束而
    被重置。调用 ``reset(options={"hard_reset": True})`` 可以开始一轮新实验。
    """

    def __init__(
        self,
        scenario: str = "local_flip",
        change_step: Optional[int] = None,
        gradual_duration: Optional[int] = None,
        horizon: Optional[int] = None,
        seed: Optional[int] = None,
        protocol_path: Optional[str | Path] = None,
    ) -> None:
        path = Path(protocol_path) if protocol_path else DEFAULT_PROTOCOL_PATH
        with path.open("r", encoding="utf-8") as file:
            self.protocol: Dict[str, Any] = json.load(file)

        self.scenario = self._validate_scenario(scenario)
        self.change_step = int(change_step if change_step is not None else self.protocol["change_step"])
        self.gradual_duration = int(
            gradual_duration if gradual_duration is not None else self.protocol["gradual_duration"]
        )
        self.horizon = int(horizon if horizon is not None else self.protocol["horizon"])
        if self.change_step < 0:
            raise ValueError("change_step must be non-negative")
        if self.gradual_duration <= 0:
            raise ValueError("gradual_duration must be positive")
        if self.horizon <= 0:
            raise ValueError("horizon must be positive")

        self.gamma = float(self.protocol["gamma"])
        self.tie_tolerance = float(self.protocol["tie_tolerance"])
        self.rng = random.Random()
        self.seed = seed if seed is not None else int(self.protocol["random_seed"])
        self.rng.seed(self.seed)

        self.global_step = 0
        self.episode_step = 0
        self.episode_index = 0
        self.state = int(self.protocol["initial_state"])
        self.total_return = 0.0
        self.last_action: Optional[int] = None
        self.last_reward = 0.0
        self.history: List[Dict[str, Any]] = []

    @property
    def n_states(self) -> int:
        return len(self.protocol["states"])

    @property
    def n_actions(self) -> int:
        return len(self.protocol["actions"])

    @property
    def observation_space_n(self) -> int:
        """无 Gym 依赖时可直接读取的观测空间大小。"""
        return self.n_states

    @property
    def action_space_n(self) -> int:
        """无 Gym 依赖时可直接读取的动作空间大小。"""
        return self.n_actions

    def _validate_scenario(self, scenario: str) -> str:
        if scenario not in self.protocol["scenarios"]:
            choices = ", ".join(self.protocol["scenarios"])
            raise ValueError(f"unknown scenario {scenario!r}; choose one of: {choices}")
        return scenario

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[int, Dict[str, Any]]:
        """重置 episode，返回 ``(observation, info)``。

        默认只重置 episode 状态，保留环境版本时间线；训练智能体时可在
        episode 结束后直接调用。新实验请传入 ``options={"hard_reset": True}``。
        """
        options = options or {}
        if seed is not None:
            self.seed = int(seed)
            self.rng.seed(self.seed)
        if options.get("hard_reset", False):
            self.global_step = 0
            self.history.clear()
            self.episode_index = 0
        self.episode_step = 0
        self.state = int(options.get("initial_state", self.protocol["initial_state"]))
        self.total_return = 0.0
        self.last_action = None
        self.last_reward = 0.0
        return self.state, self._info(event="reset")

    def set_scenario(
        self,
        scenario: str,
        *,
        change_step: Optional[int] = None,
        hard_reset: bool = True,
    ) -> None:
        """切换场景；默认开启一轮全新的实验。"""
        self.scenario = self._validate_scenario(scenario)
        if change_step is not None:
            self.change_step = int(change_step)
        if hard_reset:
            self.reset(options={"hard_reset": True})

    def force_change(self) -> None:
        """立即把时间线推进到变化点，用于交互式演示。"""
        self.global_step = max(self.global_step, self.change_step)

    def phase_at(self, time_step: Optional[int] = None) -> str:
        time_step = self.global_step if time_step is None else int(time_step)
        if self.scenario in {"stable", "reward_shift", "transition_shift", "global_flip", "local_flip"}:
            return "base" if time_step < self.change_step else "updated"
        if time_step < self.change_step:
            return "base"
        if time_step < self.change_step + self.gradual_duration:
            return "transition"
        return "updated"

    def progress_at(self, time_step: Optional[int] = None) -> float:
        """返回渐变场景的 0...1 进度，其他场景返回 0 或 1。"""
        time_step = self.global_step if time_step is None else int(time_step)
        if self.scenario != "gradual_flip":
            return 0.0 if time_step < self.change_step else 1.0
        if time_step <= self.change_step:
            return 0.0
        return min(1.0, (time_step - self.change_step) / self.gradual_duration)

    def model_at(self, time_step: Optional[int] = None) -> Model:
        """返回给定环境时间点的 ``(R, P)``，不会修改环境内部状态。"""
        time_step = self.global_step if time_step is None else int(time_step)
        rewards: List[List[float]] = copy.deepcopy(self.protocol["base_rewards"])
        transitions: List[List[List[float]]] = copy.deepcopy(self.protocol["base_transitions"])

        if time_step < self.change_step or self.scenario == "stable":
            return rewards, transitions

        if self.scenario == "reward_shift":
            amount = float(self.protocol["reward_shift_amount"])
            rewards = [[value + amount for value in row] for row in rewards]
        elif self.scenario == "global_flip":
            self._swap_rewards(rewards, range(self.n_states))
        elif self.scenario == "local_flip":
            self._swap_rewards(rewards, self.protocol["local_flip_states"])
        elif self.scenario == "transition_shift":
            self._apply_transition_shift(transitions, self.protocol["transition_shift_states"])
        elif self.scenario == "gradual_flip":
            progress = self.progress_at(time_step)
            target = copy.deepcopy(rewards)
            self._swap_rewards(target, range(self.n_states))
            rewards = [
                [(1.0 - progress) * old + progress * new for old, new in zip(old_row, new_row)]
                for old_row, new_row in zip(rewards, target)
            ]
        return rewards, transitions

    def step(self, action: int) -> Tuple[int, float, bool, bool, Dict[str, Any]]:
        """执行一个动作并返回 Gymnasium 风格五元组。"""
        action = int(action)
        if action < 0 or action >= self.n_actions:
            raise ValueError(f"action must be in [0, {self.n_actions - 1}], got {action}")

        state_before = self.state
        rewards, transitions = self.model_at(self.global_step)
        reward = float(rewards[state_before][action])
        next_state = self._sample(transitions[state_before][action])

        self.history.append(
            {
                "time": self.global_step,
                "state": state_before,
                "action": action,
                "reward": reward,
                "next_state": next_state,
                "phase": self.phase_at(self.global_step),
            }
        )
        self.state = next_state
        self.last_action = action
        self.last_reward = reward
        self.total_return += reward
        self.global_step += 1
        self.episode_step += 1

        terminated = False
        truncated = self.episode_step >= self.horizon
        if truncated:
            self.episode_index += 1
            self.episode_step = 0
            self.state = int(self.protocol["initial_state"])
            self.total_return = 0.0

        info = self._info(event="step")
        info.update(
            {
                "state_before": state_before,
                "action": action,
                "reward": reward,
                "next_state": next_state,
                "terminated": terminated,
                "truncated": truncated,
            }
        )
        return self.state, reward, terminated, truncated, info

    def true_q_values(self, time_step: Optional[int] = None) -> List[List[float]]:
        rewards, transitions = self.model_at(time_step)
        return value_iteration(rewards, transitions, self.gamma)

    def optimal_actions(self, time_step: Optional[int] = None) -> List[int]:
        return greedy_actions(self.true_q_values(time_step), self.tie_tolerance)

    def decision_truth(self, time_step: Optional[int] = None) -> Dict[str, Any]:
        """返回当前环境下可由真实模型直接计算的决策变化真值。"""
        time_step = self.global_step if time_step is None else int(time_step)
        before = self.optimal_actions(max(0, self.change_step - 1))
        current = self.optimal_actions(time_step)
        flips = [before[index] != current[index] for index in range(self.n_states)]
        occupancy = self.state_occupancy()
        impact = sum(weight for weight, flipped in zip(occupancy, flips) if flipped)
        return {
            "reference_optimal_actions": before,
            "current_optimal_actions": current,
            "action_flip_mask": flips,
            "state_occupancy": occupancy,
            "decision_impact": impact,
            "phase": self.phase_at(time_step),
            "progress": self.progress_at(time_step),
        }

    def state_occupancy(self, window: Optional[int] = None) -> List[float]:
        """按最近访问轨迹估计当前策略的状态占用分布。"""
        size = int(window or self.protocol["monitor_window"])
        recent = self.history[-size:]
        counts = [0.0] * self.n_states
        if not recent:
            return [1.0 / self.n_states] * self.n_states
        for item in recent:
            counts[int(item["state"])] += 1.0
        total = sum(counts)
        return [value / total for value in counts]

    def _info(self, event: str) -> Dict[str, Any]:
        truth = self.decision_truth()
        rewards, transitions = self.model_at()
        return {
            "event": event,
            "scenario": self.scenario,
            "phase": self.phase_at(),
            "progress": self.progress_at(),
            "global_step": self.global_step,
            "episode_step": self.episode_step,
            "episode_index": self.episode_index,
            "state": self.state,
            "environment_version": "V0" if self.phase_at() == "base" else "V1",
            "rewards": rewards,
            "transitions": transitions,
            "q_values": self.true_q_values(),
            **truth,
        }

    def _sample(self, probabilities: Sequence[float]) -> int:
        threshold = self.rng.random()
        cumulative = 0.0
        for index, probability in enumerate(probabilities):
            cumulative += float(probability)
            if threshold <= cumulative:
                return index
        return len(probabilities) - 1

    @staticmethod
    def _swap_rewards(rewards: List[List[float]], states: Sequence[int]) -> None:
        for state in states:
            rewards[int(state)][0], rewards[int(state)][1] = rewards[int(state)][1], rewards[int(state)][0]

    @staticmethod
    def _apply_transition_shift(
        transitions: List[List[List[float]]],
        states: Sequence[int],
    ) -> None:
        size = len(transitions)
        for state in states:
            state = int(state)
            attack_target = (state - 2) % size
            defend_target = (state + 1) % size
            attack = [0.0] * size
            defend = [0.0] * size
            attack[state] = 0.01
            attack[attack_target] = 0.99
            defend[state] = 0.01
            defend[defend_target] = 0.99
            transitions[state][0] = attack
            transitions[state][1] = defend


def value_iteration(
    rewards: Sequence[Sequence[float]],
    transitions: Sequence[Sequence[Sequence[float]]],
    gamma: float,
    tolerance: float = 1e-10,
    max_iterations: int = 10_000,
) -> List[List[float]]:
    """使用已知 ``R``、``P`` 精确近似最优动作价值 ``Q*``。"""
    state_count = len(rewards)
    action_count = len(rewards[0])
    values = [0.0] * state_count
    q_values = [[0.0] * action_count for _ in range(state_count)]

    for _ in range(max_iterations):
        next_values = [0.0] * state_count
        for state in range(state_count):
            for action in range(action_count):
                expected = sum(
                    float(probability) * values[next_state]
                    for next_state, probability in enumerate(transitions[state][action])
                )
                q_values[state][action] = float(rewards[state][action]) + gamma * expected
            next_values[state] = max(q_values[state])
        difference = max(abs(left - right) for left, right in zip(next_values, values))
        values = next_values
        if difference < tolerance:
            break

    # Recompute once against the converged V so callers receive a self-consistent Q table.
    for state in range(state_count):
        for action in range(action_count):
            expected = sum(
                float(probability) * values[next_state]
                for next_state, probability in enumerate(transitions[state][action])
            )
            q_values[state][action] = float(rewards[state][action]) + gamma * expected
    return q_values


def greedy_actions(q_values: Sequence[Sequence[float]], tie_tolerance: float = 0.0) -> List[int]:
    """根据 Q 值选择动作；差距小于容忍阈值时使用较小动作编号作为稳定平局规则。"""
    actions: List[int] = []
    for row in q_values:
        best = max(float(value) for value in row)
        candidates = [index for index, value in enumerate(row) if best - float(value) <= tie_tolerance]
        actions.append(candidates[0])
    return actions


def summarize_truth(env: NonStationaryArenaEnv) -> None:
    """打印变化前后真实最优动作，便于锁定环境协议。"""
    before = env.optimal_actions(max(0, env.change_step - 1))
    after_time = env.change_step + env.gradual_duration if env.scenario == "gradual_flip" else env.change_step
    after = env.optimal_actions(after_time)
    flips = [index for index, (left, right) in enumerate(zip(before, after)) if left != right]
    print(f"scenario={env.scenario}")
    print(f"before={before}")
    print(f"after ={after}")
    print(f"flipped_states={flips}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect the Star Ring Arena environment truth.")
    parser.add_argument("--scenario", default="local_flip", choices=["stable", "reward_shift", "transition_shift", "global_flip", "local_flip", "gradual_flip"])
    parser.add_argument("--change-step", type=int, default=None)
    args = parser.parse_args()
    environment = NonStationaryArenaEnv(scenario=args.scenario, change_step=args.change_step)
    summarize_truth(environment)
