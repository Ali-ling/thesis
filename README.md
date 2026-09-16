# 星环守卫 · Non-Stationary Arena

这是一个为毕业设计搭建的第一版游戏化非平稳强化学习仿真环境。智能体在 8 个区域组成的星环航线上，每一步选择两个动作之一：

- `A0`：脉冲突击
- `A1`：相位防守

环境使用可计算真值的有限状态 MDP。游戏运行到指定步数后，奖励函数或转移概率可以发生预先定义的变化，以模拟游戏版本更新。浏览器 Demo 和 Python 训练接口共享 `arena_protocol.json`，避免“演示规则”和“训练规则”不一致。

## 1. 启动浏览器 Demo

在当前目录运行：

```bash
python3 -m http.server 8000
```

然后打开 <http://localhost:8000>。

浏览器 Demo 支持：

- 手动选择 `脉冲突击` 或 `相位防守`；也可以按键盘 `1` / `2`；
- 自动策略：贪心、带探索、随机；空格键可以开始或暂停；
- 六类变化场景：无变化、奖励平移、转移扰动、全局翻转、局部翻转、渐变翻转；
- 调整变化发生步数，手动触发环境更新；
- 查看当前状态、奖励、版本、真实最优动作、状态占用和决策影响比例；
- 运行到足够轨迹后，查看参考窗口 / 近期窗口的简化估计和低/中/高证据等级。

如果浏览器提示无法加载 `arena_protocol.json`，说明页面是通过 `file://` 打开的，请改用上面的静态服务器地址。

## 2. Python 环境接口

环境不依赖第三方包，采用 Gymnasium 风格的 `reset` 和 `step` 接口：

```python
from arena_env import NonStationaryArenaEnv

env = NonStationaryArenaEnv(scenario="local_flip", seed=20260916)
observation, info = env.reset(options={"hard_reset": True})

for _ in range(1000):
    action = 0
    observation, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        observation, info = env.reset()
```

`info` 中包含研究需要的真值字段：

- `q_values`：当前环境模型的真实 `Q*`；
- `reference_optimal_actions`：变化前最优动作；
- `current_optimal_actions`：当前版本最优动作；
- `action_flip_mask`：每个状态是否发生动作偏好翻转；
- `state_occupancy`：按最近访问轨迹计算的状态占用分布；
- `decision_impact`：按状态占用加权的真实决策影响比例；
- `phase`、`progress`、`environment_version`：当前环境版本阶段。

### 查看每个场景的真实翻转状态

```bash
for scenario in stable reward_shift transition_shift global_flip local_flip gradual_flip; do
  python3 arena_env.py --scenario "$scenario"
done
```

### 运行最小 Q-learning 基线

这不是最终论文 PPO，只用于验证训练接口和环境变化链路：

```bash
python3 train_baseline.py \
  --scenario local_flip \
  --steps 2400 \
  --seed 20260916 \
  --output results/local_flip_baseline.csv
```

可将 `scenario` 替换为六类场景之一。后续接入 PPO 时，保留 `env.reset()`、`env.step(action)` 和日志中的 `info` 字段即可。

## 3. 环境协议

`arena_protocol.json` 是唯一规则源，当前主协议为：

- 状态空间：`Discrete(8)`；
- 动作空间：`Discrete(2)`；
- 折扣因子：`γ = 0.92`；
- 默认回合长度：`80 steps`；
- 默认变化点：`T+48`；
- 默认监测窗口：`24 + 24` 步；
- 决策影响阈值：`δ0 = 0.20`、`δ1 = 0.55`；
- 固定随机种子：`20260916`。

这套环境第一版有意保持“可解释、可复现、可计算真值”，而不是追求复杂画面。下一步可以在不改变 MDP 接口的前提下接入标准 PPO、滚动窗口估计器、区块自助法和四组更新基线。

## 4. 文件说明

```text
index.html          浏览器游戏页面
styles.css          游戏与监测面板样式
app.js              浏览器端环境、交互和可视化
arena_protocol.json 浏览器与 Python 共用的环境协议
arena_env.py        Python 非平稳 MDP 与真实 Q* 计算
train_baseline.py   最小 Q-learning 接入示例
毕业设计.md         研究方案和验收标准
```
