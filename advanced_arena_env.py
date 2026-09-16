"""星环守卫 II：深空远征 —— 面向 PPO 的高难度非平稳环境。

这份环境与 ``arena_env.py`` 有意分开：
- ``arena_env.py`` 保留 8 状态、2 动作、可计算 Q* 的论文机制验证环境；
- 本文件是更接近真实游戏的训练环境，奖励来自资源采集、风险事件、
  资源管理和最终撤离结果，不提供固定的状态-动作奖励表。

训练策略只应使用 ``reset`` 返回的 observation 和 ``step`` 返回的 reward。
隐藏资源库存、真实风险、版本参数和事件真值只能通过 ``oracle_snapshot``
提供给评估器，不能喂给 agent。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np


ROOT = Path(__file__).resolve().parent
DEFAULT_PROTOCOL_PATH = ROOT / "arena_v2_protocol.json"


class DeepSpaceExpeditionEnv:
    """部分可观测的随机事件驱动非平稳深空远征环境。"""

    ACTION_FORWARD = 0
    ACTION_REVERSE = 1
    ACTION_DASH = 2
    ACTION_HARVEST = 3
    ACTION_GUARD = 4
    ACTION_SCAN = 5
    ACTION_EXTRACT = 6

    def __init__(
        self,
        scenario: str = "mixed_shift",
        change_step: Optional[int] = None,
        horizon: Optional[int] = None,
        seed: Optional[int] = None,
        protocol_path: Optional[Union[str, Path]] = None,
    ) -> None:
        path = Path(protocol_path) if protocol_path else DEFAULT_PROTOCOL_PATH
        with path.open("r", encoding="utf-8") as file:
            self.protocol: Dict[str, Any] = json.load(file)

        self.scenario = self._validate_scenario(scenario)
        self.change_step = int(change_step if change_step is not None else self.protocol["change_step"])
        self.horizon = int(horizon if horizon is not None else self.protocol["episode_horizon"])
        self.gamma = float(self.protocol["gamma"])
        self.target_cargo = int(self.protocol["target_cargo"])
        self.seed = int(seed if seed is not None else self.protocol["seed"])
        self.rng = np.random.default_rng(self.seed)
        self.nodes = self.protocol["nodes"]
        self.actions = self.protocol["actions"]
        self.node_kinds = self.protocol["node_kinds"]
        self.route_rewrite_nodes = {int(node) for node in self.protocol["route_rewrite_nodes"]}

        if self.change_step < 0:
            raise ValueError("change_step must be non-negative")
        if self.horizon <= 0:
            raise ValueError("horizon must be positive")

        self.global_step = 0
        self.episode_step = 0
        self.episode_index = 0
        self.position = 0
        self.shield = 0.0
        self.energy = 0.0
        self.threat = 0.0
        self.cargo = 0
        self.episode_return = 0.0
        self.last_action = -1
        self.last_reward = 0.0
        self.last_damage = 0.0
        self.last_events: List[str] = []
        self.terminal_reason = ""
        self.episode_success = False
        self.resource_stock: List[int] = []
        self.hazard_real: List[float] = []
        self.scan_memory: Dict[int, Dict[str, float]] = {}
        self.episode_history: List[Dict[str, Any]] = []
        self.completed_episodes: List[Dict[str, Any]] = []
        self._reset_episode()

    @property
    def n_nodes(self) -> int:
        return len(self.nodes)

    @property
    def n_actions(self) -> int:
        return len(self.actions)

    @property
    def action_space_n(self) -> int:
        return self.n_actions

    @property
    def observation_space_shape(self) -> Tuple[int]:
        return (self.observation_dim,)

    @property
    def observation_dim(self) -> int:
        return self.n_nodes + len(self.node_kinds) * 3 + 10 + self.n_actions

    @property
    def action_names(self) -> List[str]:
        return [str(action["name"]) for action in self.actions]

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
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """返回 ``(observation, info)``，不把隐藏真值放进 info。"""
        options = options or {}
        if seed is not None:
            self.seed = int(seed)
            self.rng = np.random.default_rng(self.seed)
        if options.get("hard_reset", False):
            self.global_step = 0
            self.episode_index = 0
            self.completed_episodes.clear()
        self._reset_episode()
        return self._observation(), self._public_info(event="reset")

    def set_scenario(
        self,
        scenario: str,
        *,
        change_step: Optional[int] = None,
        hard_reset: bool = True,
    ) -> None:
        self.scenario = self._validate_scenario(scenario)
        if change_step is not None:
            self.change_step = int(change_step)
        if hard_reset:
            self.reset(options={"hard_reset": True})

    def phase_at(self, time_step: Optional[int] = None) -> str:
        time_step = self.global_step if time_step is None else int(time_step)
        if self.scenario in {"stable", "storm_surge", "route_rewrite", "resource_decay", "mixed_shift"}:
            return "base" if time_step < self.change_step or self.scenario == "stable" else "updated"
        if time_step < self.change_step:
            return "base"
        if time_step < self.change_step + int(self.protocol["gradual_duration"]):
            return "transition"
        return "updated"

    def progress_at(self, time_step: Optional[int] = None) -> float:
        time_step = self.global_step if time_step is None else int(time_step)
        if self.scenario != "gradual_update":
            return 0.0 if time_step < self.change_step else 1.0
        duration = max(1, int(self.protocol["gradual_duration"]))
        return float(np.clip((time_step - self.change_step) / duration, 0.0, 1.0))

    def dynamics_at(self, time_step: Optional[int] = None) -> Dict[str, Any]:
        """返回内部动力学参数；只供评估器使用，不进入 observation。"""
        time_step = self.global_step if time_step is None else int(time_step)
        progress = self.progress_at(time_step)
        target = {
            "hazard_multiplier": 1.0,
            "threat_gain": 1.0,
            "guard_multiplier": 1.0,
            "resource_multiplier": 1.0,
            "dash_risk": 1.0,
            "route_rewrite": False,
        }
        if self.scenario == "storm_surge":
            target.update(hazard_multiplier=1.65, threat_gain=1.35)
        elif self.scenario == "route_rewrite":
            target.update(route_rewrite=True, hazard_multiplier=1.08)
        elif self.scenario == "resource_decay":
            target.update(resource_multiplier=0.54, hazard_multiplier=1.06)
        elif self.scenario == "mixed_shift":
            target.update(
                hazard_multiplier=1.48,
                threat_gain=1.28,
                guard_multiplier=0.72,
                resource_multiplier=0.68,
                dash_risk=1.35,
                route_rewrite=True,
            )
        elif self.scenario == "gradual_update":
            mixed = {
                "hazard_multiplier": 1.48,
                "threat_gain": 1.28,
                "guard_multiplier": 0.72,
                "resource_multiplier": 0.68,
                "dash_risk": 1.35,
                "route_rewrite": progress >= 0.65,
            }
            for key in ("hazard_multiplier", "threat_gain", "guard_multiplier", "resource_multiplier", "dash_risk"):
                target[key] = 1.0 + progress * (mixed[key] - 1.0)
            target["route_rewrite"] = bool(mixed["route_rewrite"])
        if time_step < self.change_step and self.scenario != "stable":
            return {
                "hazard_multiplier": 1.0,
                "threat_gain": 1.0,
                "guard_multiplier": 1.0,
                "resource_multiplier": 1.0,
                "dash_risk": 1.0,
                "route_rewrite": False,
                "progress": 0.0,
            }
        target["progress"] = progress
        return target

    def oracle_snapshot(self) -> Dict[str, Any]:
        """返回隐藏环境真值；评估器可用，禁止作为 agent observation。"""
        return {
            "scenario": self.scenario,
            "phase": self.phase_at(),
            "progress": self.progress_at(),
            "dynamics": self.dynamics_at(),
            "position": self.position,
            "resource_stock": list(self.resource_stock),
            "hazard_real": list(self.hazard_real),
            "scan_memory": {str(key): dict(value) for key, value in self.scan_memory.items()},
            "global_step": self.global_step,
            "episode_step": self.episode_step,
            "shield": self.shield,
            "energy": self.energy,
            "threat": self.threat,
            "cargo": self.cargo,
            "target_cargo": self.target_cargo,
            "episode_success": self.episode_success,
        }

    def public_snapshot(self) -> Dict[str, Any]:
        """供可视化使用的公开状态；未扫描区域不暴露真实库存和风险。"""
        current_scan = self.scan_memory.get(self.position)
        return {
            "scenario": self.scenario,
            "phase": self.phase_at(),
            "progress": self.progress_at(),
            "version": "V0" if self.phase_at() == "base" else "V0→V1" if self.phase_at() == "transition" else "V1",
            "global_step": self.global_step,
            "episode_step": self.episode_step,
            "episode_index": self.episode_index,
            "position": self.position,
            "shield": round(self.shield, 2),
            "energy": round(self.energy, 2),
            "threat": round(self.threat, 2),
            "cargo": self.cargo,
            "target_cargo": self.target_cargo,
            "last_action": self.last_action,
            "last_reward": round(self.last_reward, 4),
            "last_damage": round(self.last_damage, 2),
            "events": list(self.last_events),
            "terminal_reason": self.terminal_reason,
            "success": self.episode_success,
            "scan": dict(current_scan) if current_scan else None,
            "nodes": [
                {
                    "id": int(node["id"]),
                    "name": node["name"],
                    "short": node["short"],
                    "kind": node["kind"],
                    "forward": node["forward"],
                    "reverse": node["reverse"],
                    "resource_capacity": node["resource_capacity"],
                    "scanned": int(node["id"]) in self.scan_memory,
                }
                for node in self.nodes
            ],
        }

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """执行一个动作，返回 Gymnasium 风格五元组。"""
        action = int(action)
        if action < 0 or action >= self.n_actions:
            raise ValueError(f"action must be in [0, {self.n_actions - 1}], got {action}")

        params = self.dynamics_at()
        start_position = self.position
        events: List[str] = []
        reward = -0.025
        terminated = False
        truncated = False
        action_valid = True
        self.last_damage = 0.0

        if action == self.ACTION_FORWARD:
            action_valid, action_reward = self._travel(direction=1, hops=1, params=params, events=events)
            reward += action_reward
        elif action == self.ACTION_REVERSE:
            action_valid, action_reward = self._travel(direction=-1, hops=1, params=params, events=events)
            reward += action_reward
        elif action == self.ACTION_DASH:
            action_valid, action_reward = self._travel(direction=1, hops=2, params=params, events=events)
            reward += action_reward
        elif action == self.ACTION_HARVEST:
            action_valid, action_reward = self._harvest(params=params, events=events)
            reward += action_reward
        elif action == self.ACTION_GUARD:
            action_valid, action_reward = self._guard(params=params, events=events)
            reward += action_reward
        elif action == self.ACTION_SCAN:
            action_valid, action_reward = self._scan(events=events)
            reward += action_reward
        elif action == self.ACTION_EXTRACT:
            action_valid, action_reward, terminated = self._extract(events=events)
            reward += action_reward

        if not terminated:
            reward += self._resolve_hazard(action=action, params=params, events=events)
            if self.shield <= 0:
                terminated = True
                self.terminal_reason = "shield_depleted"
                reward -= 12.0
                events.append("护盾归零，远征失败")

        self.threat = float(np.clip(self.threat, 0.0, 100.0))
        self.energy = float(np.clip(self.energy, 0.0, 100.0))
        self.shield = float(np.clip(self.shield, 0.0, 100.0))
        self.global_step += 1
        self.episode_step += 1
        self.last_action = action
        self.last_reward = float(reward)
        self.last_events = events
        self.episode_return += float(reward)
        self.episode_history.append(
            {
                "global_step": self.global_step - 1,
                "episode_step": self.episode_step,
                "position": start_position,
                "next_position": self.position,
                "action": action,
                "reward": float(reward),
                "events": list(events),
                "phase": self.phase_at(self.global_step - 1),
            }
        )

        if not terminated and self.episode_step >= self.horizon:
            truncated = True
            self.terminal_reason = "horizon_reached"
            if self.cargo < self.target_cargo:
                reward -= 2.0
                self.episode_return -= 2.0
                events.append("时间耗尽，资源未达标")
            else:
                reward -= 1.0
                self.episode_return -= 1.0
                events.append("时间耗尽，尚未完成撤离")
            self.last_reward = float(reward)
            self.episode_history[-1]["reward"] = float(reward)
        self.last_events = events

        if terminated or truncated:
            self.completed_episodes.append(
                {
                    "episode": self.episode_index,
                    "return": float(self.episode_return),
                    "success": bool(self.episode_success),
                    "cargo": int(self.cargo),
                    "reason": self.terminal_reason,
                    "global_step": self.global_step,
                }
            )

        old_phase = self.phase_at(self.global_step - 1)
        new_phase = self.phase_at(self.global_step)
        if old_phase != new_phase:
            events.append("环境版本发生变化")
            self.last_events = events

        info = self._public_info(event="step")
        info.update(
            {
                "action": action,
                "action_name": self.actions[action]["name"],
                "action_valid": action_valid,
                "state_before": start_position,
                "terminated": terminated,
                "truncated": truncated,
                "terminal_reason": self.terminal_reason,
                "episode_return": float(self.episode_return),
                "success": bool(self.episode_success),
            }
        )
        return self._observation(), float(reward), terminated, truncated, info

    def _reset_episode(self) -> None:
        self.episode_step = 0
        self.position = 0
        self.shield = float(self.protocol["initial_shield"])
        self.energy = float(self.protocol["initial_energy"])
        self.threat = float(self.protocol["initial_threat"])
        self.cargo = 0
        self.episode_return = 0.0
        self.last_action = -1
        self.last_reward = 0.0
        self.last_damage = 0.0
        self.last_events = ["新的远征开始：收集资源并返回星门"]
        self.terminal_reason = ""
        self.episode_success = False
        self.scan_memory = {}
        self.episode_history = []
        self.resource_stock = []
        self.hazard_real = []
        for node in self.nodes:
            capacity = int(node["resource_capacity"])
            if capacity <= 0:
                self.resource_stock.append(0)
            else:
                variation = int(self.rng.choice([-1, 0, 0, 0, 1]))
                self.resource_stock.append(max(0, capacity + variation))
            hazard = float(node["base_hazard"]) + float(self.rng.uniform(-0.025, 0.025))
            self.hazard_real.append(float(np.clip(hazard, 0.005, 0.85)))

    def _travel(
        self,
        *,
        direction: int,
        hops: int,
        params: Dict[str, Any],
        events: List[str],
    ) -> Tuple[bool, float]:
        cost = 4.0 if hops == 1 else 15.0
        threat_gain = 3.0 if hops == 1 else 8.0 * float(params["dash_risk"])
        if self.energy < cost:
            self.threat += 8.0
            events.append("能量不足，航行动作失败")
            return False, -0.42
        self.energy -= cost
        for _ in range(hops):
            self.position = self._next_position(self.position, direction, params)
        self.threat += threat_gain
        events.append(f"航行至 {self.nodes[self.position]['short']}")
        return True, -0.04 if hops == 1 else -0.14

    def _harvest(self, *, params: Dict[str, Any], events: List[str]) -> Tuple[bool, float]:
        cost = 7.0
        if self.energy < cost:
            self.threat += 7.0
            events.append("能量不足，无法稳定采集")
            return False, -0.38
        self.energy -= cost
        node = self.nodes[self.position]
        stock = self.resource_stock[self.position]
        if int(node["resource_capacity"]) <= 0 or stock <= 0:
            self.threat += 5.0
            events.append("这里没有可提取资源")
            return True, -0.22

        scan_bonus = 0.16 if self.position in self.scan_memory else 0.0
        success_probability = float(np.clip(0.58 * float(params["resource_multiplier"]) + scan_bonus, 0.18, 0.82))
        if float(self.rng.random()) > success_probability:
            self.threat += 7.0
            events.append("采集失败：资源被干扰或设备失稳")
            return True, -0.16

        amount = 1
        if stock >= 2 and float(self.rng.random()) < 0.28 * float(params["resource_multiplier"]):
            amount = 2
        amount = min(amount, stock, self.target_cargo - self.cargo)
        self.resource_stock[self.position] -= amount
        self.cargo += amount
        self.threat += 4.0 + self.hazard_real[self.position] * 12.0
        events.append(f"采集成功：+{amount} 资源，货舱 {self.cargo}/{self.target_cargo}")
        return True, 0.38 * amount

    def _guard(self, *, params: Dict[str, Any], events: List[str]) -> Tuple[bool, float]:
        safe_node = self.nodes[self.position]["kind"] in {"gate", "relay"}
        energy_cost = 0.0 if safe_node else 5.0
        if self.energy < energy_cost:
            self.threat += 4.0
            events.append("能量不足，戒备系统无法启动")
            return False, -0.3
        threat_before = self.threat
        self.energy -= energy_cost
        shield_gain = (18.0 if safe_node else 14.0) * float(params["guard_multiplier"])
        threat_drop = (30.0 if safe_node else 21.0) * float(params["guard_multiplier"])
        energy_gain = 20.0 if safe_node else 0.0
        self.energy = min(100.0, self.energy + energy_gain)
        self.shield = min(100.0, self.shield + shield_gain)
        self.threat = max(0.0, self.threat - threat_drop)
        if safe_node:
            events.append(f"安全节点补给：能量 +{energy_gain:.0f}，护盾 +{shield_gain:.1f}，威胁 -{threat_drop:.1f}")
        else:
            events.append(f"相位戒备：护盾 +{shield_gain:.1f}，威胁 -{threat_drop:.1f}")
        return True, 0.11 if safe_node or threat_before >= 35.0 else -0.015

    def _scan(self, *, events: List[str]) -> Tuple[bool, float]:
        if self.energy < 3.0:
            self.threat += 4.0
            events.append("能量不足，扫描中断")
            return False, -0.25
        self.energy -= 3.0
        self.scan_memory[self.position] = {
            "resource": float(self.resource_stock[self.position]),
            "hazard": float(self.hazard_real[self.position]),
            "time": float(self.global_step),
        }
        self.threat += 1.0
        events.append(
            f"扫描完成：库存信号 {self.resource_stock[self.position]}，风险 {self.hazard_real[self.position]:.2f}"
        )
        return True, -0.055

    def _extract(self, *, events: List[str]) -> Tuple[bool, float, bool]:
        if self.position == 0 and self.cargo >= self.target_cargo:
            self.episode_success = True
            self.terminal_reason = "successful_extraction"
            events.append(f"成功撤离：带回 {self.cargo} 份资源")
            return True, 22.0 + self.cargo * 1.2, True
        if self.position == 0:
            self.threat += 7.0
            events.append(f"撤离失败：需要 {self.target_cargo}，当前只有 {self.cargo}")
            return True, -0.55, False
        self.threat += 3.0
        events.append("撤离信标只能在星门基地使用")
        return True, -0.28, False

    def _resolve_hazard(self, *, action: int, params: Dict[str, Any], events: List[str]) -> float:
        node_hazard = self.hazard_real[self.position]
        action_risk = {
            self.ACTION_FORWARD: 0.012,
            self.ACTION_REVERSE: 0.006,
            self.ACTION_DASH: 0.07 * float(params["dash_risk"]),
            self.ACTION_HARVEST: 0.045,
            self.ACTION_GUARD: -0.025,
            self.ACTION_SCAN: 0.0,
            self.ACTION_EXTRACT: -0.01,
        }[action]
        probability = float(
            np.clip(
                node_hazard * float(params["hazard_multiplier"])
                + max(0.0, self.threat - 30.0) / 360.0
                + action_risk,
                0.005,
                0.88,
            )
        )
        if float(self.rng.random()) >= probability:
            self.threat = max(0.0, self.threat - 0.4)
            return 0.0
        damage = float(self.rng.integers(5, 18))
        if self.nodes[self.position]["kind"] == "storm":
            damage += float(self.rng.integers(3, 10))
        self.shield -= damage
        self.last_damage = damage
        self.threat += 6.0
        events.append(f"遭遇环境冲击：护盾 -{damage:.0f}")
        return -0.035 * damage

    def _next_position(self, position: int, direction: int, params: Dict[str, Any]) -> int:
        node = self.nodes[position]
        if direction > 0:
            if bool(params["route_rewrite"]) and position in self.route_rewrite_nodes:
                return int(node.get("alternate_forward", node["forward"]))
            return int(node["forward"])
        return int(node["reverse"])

    def _observation(self) -> np.ndarray:
        node = self.nodes[self.position]
        forward_node = self.nodes[self._next_position(self.position, 1, self.dynamics_at())]
        reverse_node = self.nodes[self._next_position(self.position, -1, self.dynamics_at())]
        features: List[float] = []
        features.extend(self._one_hot(self.position, self.n_nodes))
        features.extend(self._one_hot(self.node_kinds.index(node["kind"]), len(self.node_kinds)))
        features.extend(self._one_hot(self.node_kinds.index(forward_node["kind"]), len(self.node_kinds)))
        features.extend(self._one_hot(self.node_kinds.index(reverse_node["kind"]), len(self.node_kinds)))

        scan = self.scan_memory.get(self.position)
        features.extend(
            [
                self.shield / 100.0,
                self.energy / 100.0,
                self.threat / 100.0,
                min(1.0, self.cargo / max(1, self.target_cargo)),
                self.episode_step / max(1, self.horizon),
                (min(3.0, scan["resource"]) / 3.0) if scan else 0.0,
                float(scan["hazard"]) if scan else 0.0,
                1.0 if scan else 0.0,
                min(1.0, self.last_damage / 30.0),
                float(np.clip(self.last_reward / 5.0, -1.0, 1.0)),
            ]
        )
        features.extend(self._one_hot(self.last_action, self.n_actions, allow_unknown=True))
        observation = np.asarray(features, dtype=np.float32)
        if observation.shape[0] != self.observation_dim:
            raise RuntimeError(f"observation dimension mismatch: {observation.shape[0]} != {self.observation_dim}")
        return observation

    @staticmethod
    def _one_hot(index: int, size: int, allow_unknown: bool = False) -> List[float]:
        result = [0.0] * size
        if 0 <= int(index) < size:
            result[int(index)] = 1.0
        elif not allow_unknown:
            raise ValueError(f"index {index} outside one-hot size {size}")
        return result

    def _public_info(self, *, event: str) -> Dict[str, Any]:
        return {
            "event": event,
            "global_step": self.global_step,
            "episode_step": self.episode_step,
            "episode_index": self.episode_index,
            "position": self.position,
            "shield": float(self.shield),
            "energy": float(self.energy),
            "threat": float(self.threat),
            "cargo": int(self.cargo),
            "target_cargo": self.target_cargo,
            "events": list(self.last_events),
        }


def run_random_episodes(scenario: str, episodes: int, seed: int) -> None:
    env = DeepSpaceExpeditionEnv(scenario=scenario, seed=seed)
    successes = 0
    returns: List[float] = []
    for episode in range(episodes):
        obs, _ = env.reset(seed=seed + episode, options={"hard_reset": episode == 0})
        terminated = truncated = False
        while not terminated and not truncated:
            action = int(env.rng.integers(env.n_actions))
            obs, reward, terminated, truncated, info = env.step(action)
        result = env.completed_episodes[-1]
        successes += int(result["success"])
        returns.append(float(result["return"]))
    print(f"scenario={scenario} episodes={episodes} random_success_rate={successes / max(1, episodes):.3f}")
    print(f"mean_return={np.mean(returns):.3f} std_return={np.std(returns):.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect the difficult PPO-ready Star Ring Arena II environment.")
    parser.add_argument("--scenario", default="mixed_shift", choices=["stable", "storm_surge", "route_rewrite", "resource_decay", "mixed_shift", "gradual_update"])
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260916)
    args = parser.parse_args()
    run_random_episodes(args.scenario, args.episodes, args.seed)
