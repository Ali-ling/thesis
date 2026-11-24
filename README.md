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

### 运行决策变化检测验证

检测器使用经验奖励、经验转移和 Bellman 最优解估计动作翻转，不读取真实 `Q*`：

```bash
python3 validate_detector.py \
  --window-sizes 500 \
  --seeds 3 \
  --bootstrap-samples 60 \
  --output-dir results/detector_h500_pilot
```

结果会写入逐次运行明细和场景汇总 CSV。首轮预实验显示 `H=100` 覆盖不足，`H=500` 在六类场景中平均覆盖率接近或达到 `1.00`。

运行单元测试：

```bash
python3 -m unittest -v test_decision_detector.py test_update_scheduler.py test_ppo_arena.py test_run_arena_experiments.py test_calibrate_ppo_budget.py test_analyze_formal_results.py
python3 -m unittest -v test_open_bandit_topk.py
```

### 真实数据下载与审计

Open Bandit 小样例：

```bash
python3 download_open_bandit_sample.py
python3 audit_open_bandit.py --root data/open_bandit/sample
```

完整 BTS/All：

```bash
python3 download_open_bandit_full.py
python3 audit_open_bandit_full.py
```

完整数据位于 `data/open_bandit/full`，约 6 GB，已被 `.gitignore` 排除，不应提交到 Git。

### 运行 8 状态 PPO 两阶段训练

先在稳定环境训练共享 checkpoint：

```bash
python3 train_ppo_arena.py \
  --mode pretrain \
  --timesteps 20000 \
  --checkpoint results/arena_ppo_pretrained.pt \
  --log-path results/arena_ppo_pretrained.csv
```

再从同一 checkpoint 启动变化后的在线适应：

```bash
python3 train_ppo_arena.py \
  --mode adapt \
  --scenario local_flip \
  --schedule decision \
  --timesteps 20000 \
  --change-step 4000 \
  --load-checkpoint results/arena_ppo_pretrained.pt \
  --checkpoint results/arena_ppo_local_flip_decision.pt \
  --log-path results/arena_ppo_local_flip_decision.csv
```

将 `--schedule decision` 改为 `--schedule fixed`，即可运行关闭检测器的固定 PPO 基线。当前 PPO 主实验窗口为 `H=1500`，预训练使用 12000 步和熵系数 `0.02`，正式统计使用 200 次区块自助重抽样。

### 运行四组匹配实验

统一编排器会为每个随机种子训练一次稳定 checkpoint，再让固定、周期、漂移和证据驱动方法从同一模型开始：

```bash
python3 run_arena_experiments.py \
  --scenarios stable local_flip global_flip \
  --methods fixed periodic drift decision \
  --pretrain-timesteps 12000 \
  --online-timesteps 16000 \
  --change-step 2000 \
  --monitor-window 1500 \
  --entropy-coef 0.02 \
  --monitor-min-visits 2 \
  --resume \
  --save-trajectories \
  --bootstrap-samples 30 \
  --output-dir results/stage3_pilot_v3
```

当前检测器支持 `H=1500` 的 PPO 主实验和 `H=500` 的离线均匀探索验证。正式多种子实验应使用 200 次区块自助重抽样，并报告逐次运行明细。

隔离运行的种子结果可用以下命令合并：

```bash
python3 merge_pilot_outputs.py \
  results/stage3_v3/seed_20270401 \
  results/stage3_v3/seed_20270402 \
  results/stage3_v3/seed_20270403 \
  --output-dir results/stage3_v3
```

检查策略动作覆盖：

```bash
python3 inspect_policy.py results/budget_calibration_entropy02/seed_20270131/pretrained.pt
```

## 3. 环境协议

`arena_protocol.json` 是唯一规则源，当前主协议为：

- 状态空间：`Discrete(8)`；
- 动作空间：`Discrete(2)`；
- 折扣因子：`γ = 0.92`；
- 默认回合长度：`80 steps`；
- 默认变化点：`T+48`；
- 浏览器演示监测窗口：`24 + 24` 步；
- 论文检测预实验窗口：`500 + 500` 步（见 `实验协议_v1.md`）；
- 决策影响阈值：`δ0 = 0.20`、`δ1 = 0.55`；
- 固定随机种子：`20260916`。

这套环境第一版有意保持“可解释、可复现、可计算真值”，而不是追求复杂画面。当前已接入滚动窗口估计器、决策影响指标、区块自助法、8 状态 PPO、四组更新机制和匹配实验编排器；尚未完成的是正式多种子实验和论文结果分析。

## 4. 文件说明

```text
index.html          浏览器游戏页面
styles.css          游戏与监测面板样式
app.js              浏览器端环境、交互和可视化
arena_protocol.json 浏览器与 Python 共用的环境协议
arena_env.py        Python 非平稳 MDP 与真实 Q* 计算
train_baseline.py   最小 Q-learning 接入示例
decision_detector.py 窗口估计、I_t、区块自助法和证据分级
update_scheduler.py 证据等级到 PPO 优化轮数的映射
validate_detector.py 六场景离线检测验证与 CSV 输出
train_ppo_arena.py 8 状态 PPO、在线监测和两阶段训练入口
run_arena_experiments.py 四组方法共享 checkpoint 的实验编排器
calibrate_ppo_budget.py 稳定 PPO 预算校准
inspect_policy.py    动作概率与监测覆盖检查
analyze_training_log.py 单次运行恢复与触发指标
merge_pilot_outputs.py 多种子隔离结果合并
analyze_formal_results.py 正式分组统计与配对 bootstrap
plot_formal_results.py 正式实验论文图
run_update_ablation.py 更新强度、决策和 oracle 消融
analyze_update_ablation.py 消融配对统计与后验 oracle
download_open_bandit_sample.py Open Bandit 小样例下载
download_open_bandit_full.py Open Bandit BTS/All 下载
download_kuairand_pure.py KuaiRand-Pure 官方下载与校验（等待数据源恢复）
audit_open_bandit.py Open Bandit 小样例审计
audit_open_bandit_full.py Open Bandit 完整数据审计
build_open_bandit_application.py 真实数据上下文和动作聚合
open_bandit_topk.py 真实日志 Top-K 决策变化模型
analyze_open_bandit_topk.py Open Bandit 窗口应用分析
calibrate_open_bandit_topk_null.py Top-K 安慰剂阈值校准
run_open_bandit_semisynthetic.py 真实曝光背景下的已知变化注入
inspect_open_bandit_features.py 用户特征基数审计
inspect_open_bandit_application.py 真实数据动作支持审计
实验协议_v1.md      首轮均匀探索检测与调度记录
实验协议_v2.md      PPO 主实验窗口和四组基线设置
实验协议_v3.md      当前正式实验预算、探索和窗口协议
第三阶段试点报告.md 当前三场景试点结果与限制
预算与窗口校准报告.md 方向复核及 H=1500 修订依据
v3试点报告.md       v3 参数、覆盖修订和正式实验门槛
正式实验报告_v1.md  10 种子六场景正式结果与边界
消融与Oracle实验报告.md 更新强度与真实价值反馈实验
真实数据审计报告_样例.md 小样例字段和可行性审计
真实数据审计报告_完整.md BTS/All 完整数据审计
真实数据应用报告_v1.md 真实日志 Top-K 决策变化应用结果
真实数据半合成实验报告_v1.md 已知变化注入与确定性 Top-K 失败诊断
真实数据应用验证总结.md 三次真实数据粒度与已知变化验证总结
毕业设计.md         研究方案和验收标准
```
