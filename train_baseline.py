"""用最小 Q-learning 基线跑通星环守卫环境。

这不是论文最终的 PPO 实现，而是一个可复现的接入样例：

    python train_baseline.py --scenario local_flip --steps 2400

后续接 PPO 时，保留 ``env.reset()``、``env.step(action)`` 和 ``info`` 中的
真值字段即可，不需要修改游戏规则。
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path
from typing import List, Optional

from arena_env import NonStationaryArenaEnv, greedy_actions


SCENARIOS = ["stable", "reward_shift", "transition_shift", "global_flip", "local_flip", "gradual_flip"]


def choose_action(
    q_table: List[List[float]],
    state: int,
    *,
    epsilon: float,
    rng: random.Random,
) -> int:
    if rng.random() < epsilon:
        return rng.randrange(len(q_table[state]))
    return greedy_actions([q_table[state]])[0]


def run_training(
    scenario: str,
    steps: int,
    seed: int,
    learning_rate: float,
    epsilon: float,
    output: Optional[Path] = None,
) -> None:
    rng = random.Random(seed)
    env = NonStationaryArenaEnv(scenario=scenario, seed=seed)
    state, _ = env.reset(seed=seed, options={"hard_reset": True})
    q_table = [[0.0, 0.0] for _ in range(env.n_states)]
    rows = []

    for time_step in range(steps):
        action = choose_action(q_table, state, epsilon=epsilon, rng=rng)
        next_state, reward, _, truncated, info = env.step(action)
        next_value = max(q_table[next_state])
        target = reward + env.gamma * next_value
        q_table[state][action] += learning_rate * (target - q_table[state][action])

        rows.append(
            {
                "step": time_step,
                "environment_step": info["global_step"],
                "phase": info["phase"],
                "state": state,
                "action": action,
                "reward": round(reward, 6),
                "decision_impact": round(info["decision_impact"], 6),
                "optimal_action": info["current_optimal_actions"][state],
                "episode": info["episode_index"],
            }
        )
        state = next_state
        if truncated:
            state, _ = env.reset()

        if (time_step + 1) % max(1, steps // 10) == 0 or time_step == steps - 1:
            optimal = info["current_optimal_actions"][state]
            print(
                f"step={time_step + 1:5d} | phase={info['phase']:<10} | "
                f"impact={info['decision_impact']:.2f} | next_state=S{state} | "
                f"q={q_table[state]} | optimal=A{optimal}"
            )

    print("\n真实环境协议")
    before = env.optimal_actions(max(0, env.change_step - 1))
    after_time = env.change_step + env.gradual_duration if scenario == "gradual_flip" else env.change_step
    after = env.optimal_actions(after_time)
    flips = [index for index, (left, right) in enumerate(zip(before, after)) if left != right]
    print(f"scenario={scenario}")
    print(f"before_optimal={before}")
    print(f"after_optimal ={after}")
    print(f"flipped_states={flips}")

    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"\n已写入逐步日志：{output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a small Q-learning baseline on Star Ring Arena.")
    parser.add_argument("--scenario", choices=SCENARIOS, default="local_flip")
    parser.add_argument("--steps", type=int, default=2400)
    parser.add_argument("--seed", type=int, default=20260916)
    parser.add_argument("--learning-rate", type=float, default=0.12)
    parser.add_argument("--epsilon", type=float, default=0.12)
    parser.add_argument("--output", type=Path, default=None, help="optional CSV path for step-level logs")
    args = parser.parse_args()
    if args.steps <= 0:
        parser.error("--steps must be positive")
    run_training(
        scenario=args.scenario,
        steps=args.steps,
        seed=args.seed,
        learning_rate=args.learning_rate,
        epsilon=args.epsilon,
        output=args.output,
    )


if __name__ == "__main__":
    main()
