# 跨设备项目交接说明

> 最后同步：2026-11-24  
> 分支：`main`  
> 远程：`git@github.com:Ali-ling/thesis.git`

## 1. 项目目标

论文暂定题目：

> 非平稳强化学习中的决策相关变化检验与证据驱动的分级策略更新研究

核心问题不是“环境统计量是否变化”，而是：

> 当前策略实际访问的状态或上下文中，最优动作是否发生变化；证据强弱是否足以改变 PPO 更新强度。

当前贡献边界：

- 不宣称首次提出非平稳检测或变化触发更新；
- 不宣称证据驱动 PPO 在所有场景提高最终回报；
- 主要验证决策变化检测的选择性、更新成本和适用边界。

## 2. 当前完成度

已完成：

- 8 状态、2 动作的已知真值环境；
- 经验 `R/P`、价值迭代、动作翻转、状态占用加权和区块自助法；
- 8 状态 PPO；
- 固定、周期、奖励漂移 CUSUM、决策证据四组更新调度；
- 10 个种子 × 6 场景 × 4 方法的正式实验；
- 更新强度消融与 oracle 调度分析；
- Open Bandit 完整数据下载、审计、真实窗口分析；
- Open Bandit 半合成已知变化注入原型；
- 20 个主测试和 2 个 Open Bandit 测试。

未完成：

- 没有取得可访问的 KuaiRand-Pure 文件；
- 真实数据决策变化统计量尚未通过已知变化召回验证；
- 真实数据章节尚未形成“多数正向、少量负向”的证据；
- 论文正文尚未系统撰写。

## 3. 冻结的有限状态实验参数

参见 `实验协议_v3.md`。

```text
pretrain_timesteps = 12000
entropy_coef = 0.02
behavior_epsilon = 0
online_timesteps = 16000
change_step = 2000
monitor_window = 1500
monitor_interval = 256
monitor_min_visits = 2
bootstrap_samples = 200
base_epochs = 3
standard_epochs = 6
strong_epochs = 9
```

正式实验种子：

```text
20270501–20270510
```

这些种子已经用于主结果。后续确认性实验必须使用新种子。

## 4. 正式实验结论

证据驱动方法：

- 在稳定、奖励平移、转移扰动、局部翻转中没有高影响误报；
- 在全局和渐变翻转中高影响触发率为 90%；
- 未触发种子为 `20270505`；
- 相对固定 PPO 的最终价值差在所有场景的 95% CI 均跨过 0；
- 更新轮数明显少于周期方法，但没有恢复速度优势。

最稳妥的论文结论：

> 决策证据调度降低了无效加强更新，但在当前设置下没有观察到最终策略回报的显著提升。

详细结果：

- `正式实验报告_v1.md`
- `results/formal_v3/analysis/group_summary.csv`
- `results/formal_v3/analysis/decision_vs_fixed.csv`
- `results/formal_v3/figures/`

## 5. 更新强度消融

关键结论：

- 全局翻转中，fixed3 和 fixed9 都显著差于 fixed6；
- 稳定场景中 fixed9 显著差于 fixed6；
- decision 与 fixed6 的最终价值差大多不显著；
- 动态价值反馈 oracle 会过度更新，不是严格上界；
- 后验固定强度 oracle 与 decision 的均值差仅约 `0.07–0.11`。

详见 `消融与Oracle实验报告.md`。

## 6. Open Bandit 真实数据状态

完整数据规模：

```text
12,357,200 exposures
61,208 clicks
80 actions
2019-11-24 to 2019-11-30
```

真实数据应用尝试：

- 商品级上下文动作模型；
- 商品类别聚类；
- Top-3 集合；
- 区块 bootstrap；
- 安慰剂阈值校准；
- 真实曝光背景下的已知变化注入。

结果：

- 真实窗口基本全部为“不确定”；
- 商品级和类别级半合成实验没有通过已知高影响召回；
- 当前确定性 Top-K 方法不适配 Open Bandit 的低点击率和相近动作价值。

不得把 Open Bandit 结果写成真实应用成功。详细原因：

- `真实数据应用验证总结.md`
- `真实数据半合成实验报告_v1.md`

## 7. 原始数据不随仓库分发

`.gitignore` 排除了：

```text
data/
results/ 中未显式跟踪的原始日志
*.pt
*.trajectory.csv.gz
```

只把最终汇总 CSV 和论文图强制加入 Git。

### Open Bandit 小样例

```bash
python download_open_bandit_sample.py
python audit_open_bandit.py --root data/open_bandit/sample
```

### Open Bandit 完整 BTS/All

```bash
python download_open_bandit_full.py
python audit_open_bandit_full.py
```

完整文件约 6 GB。下载来源是官方 Hugging Face 镜像，目标大小：

```text
6321017454 bytes
```

### Open Bandit 聚合数据

商品级第二版：

```bash
python build_open_bandit_application.py \
  --output-dir data/open_bandit/application_v2 \
  --context-clusters 8 --top-actions 10 --positions all
```

商品类别版：

```bash
python build_open_bandit_application.py \
  --item-context data/open_bandit/full/item_context.csv \
  --output-dir data/open_bandit/application_categories \
  --context-clusters 8 --action-clusters 5 --positions all
```

## 8. KuaiRand-Pure

官方数据：

- <https://kuairand.com/>
- <https://github.com/chongminggao/KuaiRand>
- <https://zenodo.org/records/10439422>

目标文件：

```text
KuaiRand-Pure.tar.gz
MD5: 0820331067a3784d9691136f772b35a7
```

放置位置：

```text
data/kuairand/pure/KuaiRand-Pure.tar.gz
```

下载与校验：

```bash
python download_kuairand_pure.py
```

2026-11-24 尝试时 Zenodo 返回 500/502，Hugging Face 第三方镜像返回 429，
因此尚无 KuaiRand 数据。

## 9. 新电脑启动步骤

```bash
git clone git@github.com:Ali-ling/thesis.git
cd thesis
python -m pip install -r requirements-ppo.txt
python -m pip install -r requirements-analysis.txt
```

先阅读：

1. `AGENTS.md`
2. `AGENT_HANDOFF.md`
3. `实验协议_v3.md`
4. `正式实验报告_v1.md`
5. `真实数据应用验证总结.md`

验证测试：

```bash
python -m unittest -v test_decision_detector.py test_update_scheduler.py test_ppo_arena.py test_run_arena_experiments.py test_calibrate_ppo_budget.py test_analyze_formal_results.py
python -m unittest -v test_open_bandit_topk.py
```

两个命令应分开运行。

## 10. 推荐下一步

优先级 1：

- 找到可信、可访问的 KuaiRand-Pure；
- 先审计时间跨度、随机曝光比例、点击/长播放率、用户与视频类别覆盖；
- 不直接开始最终实验。

优先级 2：

- 使用视频内容类别或标签预先划分类别动作；
- 固定时间窗口、上下文聚类和阈值；
- 在真实窗口中寻找可重复的动作类别偏好变化；
- 用已知变化注入验证召回和误报。

优先级 3：

- 达到“多数正向、少量负向”的真实数据证据后，再写真实数据应用章节；
- 若仍无正向结果，论文应限定为负结果和边界分析，不能继续调阈值。

## 11. 当前禁止事项

- 不要删除 Open Bandit 负结果；
- 不要根据正式测试结果重新选择窗口或阈值；
- 不要用 `20270501–20270510` 作为新方法确认性实验种子；
- 不要把半合成结果称为真实在线实验；
- 不要提交原始数据或大模型文件。
