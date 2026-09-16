(() => {
  "use strict";

  const $ = (selector) => document.querySelector(selector);
  const clone = (value) => JSON.parse(JSON.stringify(value));
  const clamp = (value, min, max) => Math.min(max, Math.max(min, value));
  const formatNumber = (value, digits = 1) => Number(value || 0).toFixed(digits);
  const actionName = (protocol, action) => protocol.actions[action]?.short || `A${action}`;

  const ui = {
    protocolStatus: $("#protocolStatus"),
    protocolChip: $(".protocol-chip"),
    heroStep: $("#heroStep"),
    heroHorizon: $("#heroHorizon"),
    heroReturn: $("#heroReturn"),
    heroImpact: $("#heroImpact"),
    phaseBadge: $("#phaseBadge"),
    phaseLabel: $("#phaseLabel"),
    currentLocation: $("#currentLocation"),
    currentLocationDescription: $("#currentLocationDescription"),
    lastReward: $("#lastReward"),
    episodeStep: $("#episodeStep"),
    versionLabel: $("#versionLabel"),
    arenaMap: $("#arenaMap"),
    action0Button: $("#action0Button"),
    action1Button: $("#action1Button"),
    decisionHint: $("#decisionHint"),
    evidenceBadge: $("#evidenceBadge"),
    impactReadout: $("#impactReadout"),
    impactReadoutNote: $("#impactReadoutNote"),
    impactTrackFill: $("#impactTrackFill"),
    delta0Label: $("#delta0Label"),
    delta1Label: $("#delta1Label"),
    truthFlipStates: $("#truthFlipStates"),
    windowStatus: $("#windowStatus"),
    updateRecommendation: $("#updateRecommendation"),
    stateSignalGrid: $("#stateSignalGrid"),
    monitorNote: $("#monitorNote"),
    eventLog: $("#eventLog"),
    scenarioSelect: $("#scenarioSelect"),
    scenarioDescription: $("#scenarioDescription"),
    scenarioTruth: $("#scenarioTruth"),
    changeStepInput: $("#changeStepInput"),
    changeStepValue: $("#changeStepValue"),
    policySelect: $("#policySelect"),
    resetButton: $("#resetButton"),
    autoButton: $("#autoButton"),
    autoButtonLabel: $("#autoButtonLabel"),
    injectButton: $("#injectButton"),
    stateCode: $("#stateCode"),
    stateIndex: $("#stateIndex"),
    lastAction: $("#lastAction"),
    lastActionId: $("#lastActionId"),
    seedLabel: $("#seedLabel"),
    chartLatest: $("#chartLatest"),
    returnBars: $("#returnBars"),
    gammaLabel: $("#gammaLabel"),
    horizonLabel: $("#horizonLabel"),
    footerProtocol: $("#footerProtocol"),
    clearLogButton: $("#clearLogButton"),
  };

  const app = {
    protocol: null,
    env: null,
    scenario: "local_flip",
    seed: 20260916,
    policy: "greedy",
    running: false,
    timer: null,
    logs: [],
  };

  class ArenaModel {
    constructor(protocol, scenario, seed) {
      this.protocol = protocol;
      this.scenario = scenario;
      this.changeStep = Number(protocol.change_step);
      this.gradualDuration = Number(protocol.gradual_duration);
      this.horizon = Number(protocol.horizon);
      this.gamma = Number(protocol.gamma);
      this.tieTolerance = Number(protocol.tie_tolerance);
      this.seed = seed;
      this.reset(seed);
    }

    get stateCount() {
      return this.protocol.states.length;
    }

    reset(seed = this.seed) {
      this.seed = seed;
      this.random = this.makeRandom(seed);
      this.globalStep = 0;
      this.episodeStep = 0;
      this.episodeIndex = 0;
      this.state = Number(this.protocol.initial_state);
      this.episodeReturn = 0;
      this.lastAction = null;
      this.lastReward = 0;
      this.lastState = this.state;
      this.history = [];
      this.episodeReturns = [];
    }

    setScenario(scenario) {
      this.scenario = scenario;
      this.reset(this.seed);
    }

    makeRandom(seed) {
      let value = Number(seed) >>> 0;
      return () => {
        value += 0x6D2B79F5;
        let result = value;
        result = Math.imul(result ^ (result >>> 15), result | 1);
        result ^= result + Math.imul(result ^ (result >>> 7), result | 61);
        return ((result ^ (result >>> 14)) >>> 0) / 4294967296;
      };
    }

    phaseAt(time = this.globalStep) {
      const fixedScenarios = ["stable", "reward_shift", "transition_shift", "global_flip", "local_flip"];
      if (fixedScenarios.includes(this.scenario)) {
        return time < this.changeStep || this.scenario === "stable" ? "base" : "updated";
      }
      if (time < this.changeStep) return "base";
      if (time < this.changeStep + this.gradualDuration) return "transition";
      return "updated";
    }

    progressAt(time = this.globalStep) {
      if (this.scenario !== "gradual_flip") return time < this.changeStep ? 0 : 1;
      if (time <= this.changeStep) return 0;
      return clamp((time - this.changeStep) / this.gradualDuration, 0, 1);
    }

    modelAt(time = this.globalStep) {
      const rewards = clone(this.protocol.base_rewards);
      const transitions = clone(this.protocol.base_transitions);
      if (time < this.changeStep || this.scenario === "stable") {
        return { rewards, transitions };
      }

      if (this.scenario === "reward_shift") {
        const amount = Number(this.protocol.reward_shift_amount);
        for (let state = 0; state < rewards.length; state += 1) {
          rewards[state] = rewards[state].map((value) => value + amount);
        }
      } else if (this.scenario === "global_flip") {
        this.swapRewards(rewards, [...Array(this.stateCount).keys()]);
      } else if (this.scenario === "local_flip") {
        this.swapRewards(rewards, this.protocol.local_flip_states);
      } else if (this.scenario === "transition_shift") {
        this.applyTransitionShift(transitions, this.protocol.transition_shift_states);
      } else if (this.scenario === "gradual_flip") {
        const progress = this.progressAt(time);
        const target = clone(rewards);
        this.swapRewards(target, [...Array(this.stateCount).keys()]);
        for (let state = 0; state < rewards.length; state += 1) {
          rewards[state] = rewards[state].map((value, action) => (
            (1 - progress) * value + progress * target[state][action]
          ));
        }
      }
      return { rewards, transitions };
    }

    swapRewards(rewards, states) {
      states.forEach((state) => {
        [rewards[state][0], rewards[state][1]] = [rewards[state][1], rewards[state][0]];
      });
    }

    applyTransitionShift(transitions, states) {
      states.forEach((state) => {
        const attackTarget = (state - 2 + this.stateCount) % this.stateCount;
        const defendTarget = (state + 1) % this.stateCount;
        const attack = Array(this.stateCount).fill(0);
        const defend = Array(this.stateCount).fill(0);
        attack[state] = 0.01;
        attack[attackTarget] = 0.99;
        defend[state] = 0.01;
        defend[defendTarget] = 0.99;
        transitions[state][0] = attack;
        transitions[state][1] = defend;
      });
    }

    step(action) {
      const chosenAction = Number(action);
      if (![0, 1].includes(chosenAction)) throw new Error("动作必须是 0 或 1");
      const model = this.modelAt(this.globalStep);
      const stateBefore = this.state;
      const reward = Number(model.rewards[stateBefore][chosenAction]);
      const nextState = this.sample(model.transitions[stateBefore][chosenAction]);
      const phaseBefore = this.phaseAt(this.globalStep);
      this.history.push({
        time: this.globalStep,
        state: stateBefore,
        action: chosenAction,
        reward,
        nextState,
        phase: phaseBefore,
      });
      this.lastState = stateBefore;
      this.lastAction = chosenAction;
      this.lastReward = reward;
      this.state = nextState;
      this.episodeReturn += reward;
      this.globalStep += 1;
      this.episodeStep += 1;
      let episodeFinished = false;
      if (this.episodeStep >= this.horizon) {
        this.episodeReturns.push(this.episodeReturn);
        this.episodeIndex += 1;
        this.episodeStep = 0;
        this.state = Number(this.protocol.initial_state);
        this.episodeReturn = 0;
        episodeFinished = true;
      }
      return { reward, stateBefore, nextState, phaseBefore, episodeFinished };
    }

    forceChange() {
      this.globalStep = Math.max(this.globalStep, this.changeStep);
    }

    sample(probabilities) {
      const threshold = this.random();
      let cumulative = 0;
      for (let index = 0; index < probabilities.length; index += 1) {
        cumulative += Number(probabilities[index]);
        if (threshold <= cumulative) return index;
      }
      return probabilities.length - 1;
    }

    qValuesAt(time = this.globalStep) {
      const model = this.modelAt(time);
      return valueIteration(model.rewards, model.transitions, this.gamma);
    }

    optimalActionsAt(time = this.globalStep) {
      return greedyActions(this.qValuesAt(time), this.tieTolerance);
    }

    stateOccupancy(samples = this.history.slice(-Number(this.protocol.monitor_window))) {
      const counts = Array(this.stateCount).fill(0);
      if (!samples.length) return counts.map(() => 1 / this.stateCount);
      samples.forEach((item) => { counts[item.state] += 1; });
      return counts.map((value) => value / samples.length);
    }

    truthAt(time = this.globalStep) {
      const referenceActions = this.optimalActionsAt(Math.max(0, this.changeStep - 1));
      const currentActions = this.optimalActionsAt(time);
      const flipMask = currentActions.map((action, state) => action !== referenceActions[state]);
      const occupancy = this.stateOccupancy();
      const impact = flipMask.reduce((sum, flipped, state) => sum + (flipped ? occupancy[state] : 0), 0);
      return {
        referenceActions,
        currentActions,
        flipMask,
        occupancy,
        impact,
      };
    }

    estimateMonitor() {
      const window = Number(this.protocol.monitor_window);
      const reference = this.history.slice(-window * 2, -window);
      const recent = this.history.slice(-window);
      if (reference.length < window || recent.length < window) {
        return { ready: false, reference, recent, coverage: 0 };
      }
      const referenceModel = estimateModel(reference, this.protocol);
      const recentModel = estimateModel(recent, this.protocol);
      const referenceQ = valueIteration(referenceModel.rewards, referenceModel.transitions, this.gamma);
      const recentQ = valueIteration(recentModel.rewards, recentModel.transitions, this.gamma);
      const referenceActions = greedyActions(referenceQ, this.tieTolerance);
      const recentActions = greedyActions(recentQ, this.tieTolerance);
      const flipMask = recentActions.map((action, state) => action !== referenceActions[state]);
      const occupancy = this.stateOccupancy(recent);
      const impact = flipMask.reduce((sum, flipped, state) => sum + (flipped ? occupancy[state] : 0), 0);
      const coverage = coverageRate([...reference, ...recent], this.stateCount);
      let level = "mid";
      if (impact <= Number(this.protocol.delta0)) level = "low";
      if (impact >= Number(this.protocol.delta1) && coverage >= 0.38) level = "high";
      return {
        ready: true,
        reference,
        recent,
        referenceQ,
        recentQ,
        referenceActions,
        recentActions,
        flipMask,
        occupancy,
        impact,
        coverage,
        level,
      };
    }

    snapshot() {
      const phase = this.phaseAt();
      const truth = this.truthAt();
      return {
        scenario: this.scenario,
        phase,
        progress: this.progressAt(),
        version: phase === "base" ? "V0" : phase === "transition" ? "V0→V1" : "V1",
        globalStep: this.globalStep,
        episodeStep: this.episodeStep,
        episodeIndex: this.episodeIndex,
        episodeReturn: this.episodeReturn,
        state: this.state,
        lastState: this.lastState,
        lastAction: this.lastAction,
        lastReward: this.lastReward,
        qValues: this.qValuesAt(),
        ...truth,
        monitor: this.estimateMonitor(),
      };
    }
  }

  function valueIteration(rewards, transitions, gamma, tolerance = 1e-9, maxIterations = 5000) {
    const values = Array(rewards.length).fill(0);
    const qValues = rewards.map((row) => Array(row.length).fill(0));
    for (let iteration = 0; iteration < maxIterations; iteration += 1) {
      const nextValues = values.map((_, state) => {
        for (let action = 0; action < rewards[state].length; action += 1) {
          const expected = transitions[state][action].reduce((sum, probability, nextState) => (
            sum + Number(probability) * values[nextState]
          ), 0);
          qValues[state][action] = Number(rewards[state][action]) + gamma * expected;
        }
        return Math.max(...qValues[state]);
      });
      const difference = Math.max(...nextValues.map((value, state) => Math.abs(value - values[state])));
      for (let state = 0; state < values.length; state += 1) values[state] = nextValues[state];
      if (difference < tolerance) break;
    }
    for (let state = 0; state < rewards.length; state += 1) {
      for (let action = 0; action < rewards[state].length; action += 1) {
        const expected = transitions[state][action].reduce((sum, probability, nextState) => (
          sum + Number(probability) * values[nextState]
        ), 0);
        qValues[state][action] = Number(rewards[state][action]) + gamma * expected;
      }
    }
    return qValues;
  }

  function greedyActions(qValues, tolerance = 0) {
    return qValues.map((row) => {
      const best = Math.max(...row);
      const candidates = row
        .map((value, action) => ({ value, action }))
        .filter(({ value }) => best - value <= tolerance);
      return candidates[0].action;
    });
  }

  function estimateModel(samples, protocol) {
    const rewards = clone(protocol.base_rewards);
    const transitions = clone(protocol.base_transitions);
    const rewardSums = rewards.map((row) => row.map(() => 0));
    const rewardCounts = rewards.map((row) => row.map(() => 0));
    const transitionCounts = transitions.map((state) => state.map((row) => row.map(() => 0)));
    samples.forEach((item) => {
      rewardSums[item.state][item.action] += Number(item.reward);
      rewardCounts[item.state][item.action] += 1;
      transitionCounts[item.state][item.action][item.nextState] += 1;
    });
    for (let state = 0; state < rewards.length; state += 1) {
      for (let action = 0; action < rewards[state].length; action += 1) {
        if (rewardCounts[state][action] > 0) rewards[state][action] = rewardSums[state][action] / rewardCounts[state][action];
        const total = transitionCounts[state][action].reduce((sum, value) => sum + value, 0);
        if (total > 0) transitions[state][action] = transitionCounts[state][action].map((value) => value / total);
      }
    }
    return { rewards, transitions };
  }

  function coverageRate(samples, stateCount) {
    const pairs = new Set(samples.map((item) => `${item.state}:${item.action}`));
    return pairs.size / (stateCount * 2);
  }

  function addLog(message, type = "system", time = app.env?.globalStep ?? 0) {
    app.logs.unshift({ message, type, time });
    app.logs = app.logs.slice(0, 16);
    renderLog();
  }

  function renderLog() {
    if (!app.logs.length) {
      ui.eventLog.innerHTML = '<div class="empty-log">尚未产生事件。选择一个场景并开始探索。</div>';
      return;
    }
    ui.eventLog.innerHTML = app.logs.map((entry) => `
      <div class="log-entry log-${entry.type}">
        <span class="log-time">T+${String(entry.time).padStart(3, "0")}</span>
        <i class="log-dot"></i>
        <span class="log-message">${entry.message}</span>
      </div>
    `).join("");
  }

  function populateScenarios() {
    ui.scenarioSelect.innerHTML = Object.entries(app.protocol.scenarios).map(([key, scenario]) => (
      `<option value="${key}">${scenario.label} · ${scenario.tag}</option>`
    )).join("");
    ui.scenarioSelect.value = app.scenario;
  }

  function setScenario(scenario, reason = "切换场景") {
    stopAuto();
    app.scenario = scenario;
    app.env.setScenario(scenario);
    app.logs = [];
    addLog(`${reason}：${app.protocol.scenarios[scenario].label}，实验时间线已重置。`, "system");
    render();
  }

  function resetExperiment(reason = "实验已重置") {
    stopAuto();
    app.env.reset(app.seed);
    app.logs = [];
    addLog(`${reason}，随机种子 ${app.seed}。`, "system");
    render();
  }

  function runAction(action, source = "manual") {
    if (!app.env) return;
    const beforePhase = app.env.phaseAt();
    const result = app.env.step(action);
    const afterPhase = app.env.phaseAt();
    if (source === "manual") {
      addLog(`执行 A${action}「${actionName(app.protocol, action)}」：获得 ${formatNumber(result.reward)} 奖励，进入 S${result.nextState}。`, "action");
    }
    if (beforePhase !== afterPhase && afterPhase !== "base") {
      const scenario = app.protocol.scenarios[app.scenario];
      addLog(`环境更新生效：${scenario.label}进入 ${app.env.phaseAt() === "transition" ? "渐变阶段" : "V1 新版本"}。`, "update");
    }
    if (result.episodeFinished) {
      const completedReturn = app.env.episodeReturns[app.env.episodeReturns.length - 1];
      addLog(`回合结束：累计回报 ${formatNumber(completedReturn)}，下一回合从 S${app.protocol.initial_state} 开始。`, "system");
    }
    render();
  }

  function selectAutoAction() {
    const state = app.env.state;
    if (app.policy === "random") return Math.floor(app.env.random() * 2);
    const optimal = app.env.optimalActionsAt();
    if (app.policy === "epsilon" && app.env.random() < 0.18) return 1 - optimal[state];
    return optimal[state];
  }

  function autoTick() {
    if (!app.running) return;
    runAction(selectAutoAction(), "auto");
    app.timer = window.setTimeout(autoTick, 620);
  }

  function startAuto() {
    if (app.running) return;
    app.running = true;
    ui.autoButton.classList.add("running");
    ui.autoButtonLabel.textContent = "暂停运行";
    ui.autoButton.querySelector(".play-icon").textContent = "Ⅱ";
    autoTick();
  }

  function stopAuto() {
    app.running = false;
    if (app.timer) window.clearTimeout(app.timer);
    app.timer = null;
    ui.autoButton?.classList.remove("running");
    if (ui.autoButtonLabel) ui.autoButtonLabel.textContent = "自动运行";
    const icon = ui.autoButton?.querySelector(".play-icon");
    if (icon) icon.textContent = "▶";
  }

  function toggleAuto() {
    if (app.running) stopAuto();
    else startAuto();
  }

  function phaseText(snapshot) {
    if (snapshot.phase === "base") return "V0 旧版本";
    if (snapshot.phase === "transition") return `V0 → V1 · ${Math.round(snapshot.progress * 100)}%`;
    return "V1 新版本";
  }

  function renderMap(snapshot) {
    ui.arenaMap.innerHTML = app.protocol.states.map((stateInfo, state) => {
      const q = snapshot.qValues[state];
      const optimal = snapshot.currentActions[state];
      const occupancy = snapshot.occupancy[state] * 100;
      const current = state === snapshot.state;
      const visited = app.env.history.some((item) => item.state === state);
      const flipped = snapshot.flipMask[state];
      return `
        <div class="state-card${current ? " current" : ""}${visited ? " visited" : ""}${flipped ? " flipped" : ""}" title="S${state} · ${stateInfo.name}">
          <span class="state-index">S${state}</span>
          <span class="state-name">${stateInfo.name}</span>
          <span class="state-action">最优：<b>A${optimal} ${actionName(app.protocol, optimal)}</b>${flipped ? " · 翻转" : ""}</span>
          <span class="state-meta">Q* <b>${formatNumber(Math.max(...q), 2)}</b> · <b>${Math.round(occupancy)}%</b></span>
        </div>
      `;
    }).join("");
  }

  function renderMonitor(snapshot) {
    const monitor = snapshot.monitor;
    const impact = snapshot.impact;
    ui.heroImpact.textContent = `${Math.round(impact * 100)}%`;
    ui.impactReadout.textContent = `${Math.round(impact * 100)}%`;
    ui.impactTrackFill.style.width = `${Math.round(impact * 100)}%`;
    ui.truthFlipStates.textContent = snapshot.flipMask.map((flipped, state) => flipped ? `S${state}` : null).filter(Boolean).join(", ") || "无";
    ui.delta0Label.textContent = Number(app.protocol.delta0).toFixed(2);
    ui.delta1Label.textContent = Number(app.protocol.delta1).toFixed(2);
    const window = Number(app.protocol.monitor_window);
    ui.windowStatus.textContent = monitor.ready ? `${monitor.recent.length} + ${monitor.reference.length} 步` : `${app.env.history.length} / ${window * 2} 步`;

    ui.evidenceBadge.className = "evidence-badge";
    if (!monitor.ready) {
      ui.evidenceBadge.classList.add("evidence-idle");
      ui.evidenceBadge.textContent = "等待窗口";
      ui.updateRecommendation.textContent = "基础更新";
      ui.impactReadoutNote.textContent = "等待连续窗口";
      ui.monitorNote.textContent = `真实真值为 ${Math.round(impact * 100)}%；窗口估计需要至少 ${window * 2} 步轨迹，当前还差 ${Math.max(0, window * 2 - app.env.history.length)} 步。`;
    } else {
      const labels = { low: ["低影响证据", "基础更新"], mid: ["不确定证据", "标准更新"], high: ["高影响证据", "加强更新"] };
      const [badge, recommendation] = labels[monitor.level];
      ui.evidenceBadge.classList.add(`evidence-${monitor.level}`);
      ui.evidenceBadge.textContent = badge;
      ui.updateRecommendation.textContent = recommendation;
      ui.impactReadoutNote.textContent = `窗口估计 ${Math.round(monitor.impact * 100)}% · 覆盖 ${Math.round(monitor.coverage * 100)}%`;
      ui.monitorNote.textContent = `参考窗口与近期窗口已就绪；估计动作翻转状态为 ${monitor.flipMask.map((flipped, state) => flipped ? `S${state}` : null).filter(Boolean).join(", ") || "无"}。后续 PPO 可将该证据映射为更新轮数。`;
    }

    ui.stateSignalGrid.innerHTML = app.protocol.states.map((stateInfo, state) => {
      const currentAction = snapshot.currentActions[state];
      const flipped = snapshot.flipMask[state];
      const occupancy = snapshot.occupancy[state] * 100;
      return `
        <div class="signal-card${flipped ? " signal-flip" : ""}">
          <div class="signal-top"><span>S${state}</span><b>${Math.round(occupancy)}%</b></div>
          <div class="signal-action"><b>A${currentAction}</b> ${actionName(app.protocol, currentAction)}</div>
          <div class="signal-bar"><span style="width:${Math.max(4, Math.round(occupancy))}%"></span></div>
        </div>
      `;
    }).join("");
  }

  function renderChart() {
    const samples = app.env.history.slice(-28);
    const maxAbs = Math.max(1, ...samples.map((item) => Math.abs(item.reward)));
    ui.returnBars.innerHTML = samples.length
      ? samples.map((item) => `<span class="chart-bar${item.reward < 0 ? " negative" : ""}" style="height:${Math.max(8, Math.round(Math.abs(item.reward) / maxAbs * 100))}%" title="T+${item.time}: ${formatNumber(item.reward)}"></span>`).join("")
      : '<span class="chart-bar" style="height:8%"></span>'.repeat(8);
    const latest = samples[samples.length - 1];
    ui.chartLatest.textContent = latest ? `${latest.reward >= 0 ? "+" : ""}${formatNumber(latest.reward)}` : "—";
  }

  function render(snapshot = app.env.snapshot()) {
    const scenario = app.protocol.scenarios[app.scenario];
    const currentState = app.protocol.states[snapshot.state];
    const currentQ = snapshot.qValues[snapshot.state];
    const bestAction = snapshot.currentActions[snapshot.state];
    const phaseClass = snapshot.phase === "base" ? "phase-base" : snapshot.phase === "transition" ? "phase-transition" : "phase-updated";
    ui.protocolStatus.textContent = "环境协议已加载";
    ui.protocolChip.classList.add("ready");
    ui.heroStep.textContent = snapshot.globalStep;
    ui.heroHorizon.textContent = app.env.horizon;
    ui.heroReturn.textContent = formatNumber(snapshot.episodeReturn);
    ui.phaseBadge.className = `phase-badge ${phaseClass}`;
    ui.phaseLabel.textContent = phaseText(snapshot);
    ui.currentLocation.textContent = `S${snapshot.state} · ${currentState.name}`;
    ui.currentLocationDescription.textContent = `推荐动作：A${bestAction} ${actionName(app.protocol, bestAction)} · Q* = ${formatNumber(Math.max(...currentQ), 2)}`;
    ui.lastReward.textContent = formatNumber(snapshot.lastReward);
    ui.episodeStep.textContent = snapshot.episodeStep;
    ui.versionLabel.textContent = snapshot.version;
    ui.action0Button.setAttribute("aria-label", `执行脉冲突击，当前 Q 值 ${formatNumber(currentQ[0], 2)}`);
    ui.action1Button.setAttribute("aria-label", `执行相位防守，当前 Q 值 ${formatNumber(currentQ[1], 2)}`);
    ui.decisionHint.textContent = `当前真实 Q*：A0 ${formatNumber(currentQ[0], 2)} / A1 ${formatNumber(currentQ[1], 2)}；建议 A${bestAction}。`;
    ui.scenarioDescription.textContent = scenario.description;
    ui.scenarioTruth.textContent = `真值：${scenario.truth}`;
    ui.changeStepValue.textContent = app.env.changeStep;
    ui.changeStepInput.value = app.env.changeStep;
    ui.injectButton.disabled = app.scenario === "stable";
    ui.injectButton.style.opacity = app.scenario === "stable" ? "0.45" : "1";
    ui.stateCode.textContent = `S${snapshot.state}`;
    ui.stateIndex.textContent = snapshot.state;
    ui.lastAction.textContent = snapshot.lastAction === null ? "—" : actionName(app.protocol, snapshot.lastAction);
    ui.lastActionId.textContent = snapshot.lastAction === null ? "—" : `A${snapshot.lastAction}`;
    ui.seedLabel.textContent = app.seed;
    ui.gammaLabel.textContent = Number(app.protocol.gamma).toFixed(2);
    ui.horizonLabel.textContent = `${app.env.horizon} steps`;
    ui.footerProtocol.textContent = app.protocol.protocol_version;
    renderMap(snapshot);
    renderMonitor(snapshot);
    renderChart();
  }

  function bindEvents() {
    ui.action0Button.addEventListener("click", () => runAction(0));
    ui.action1Button.addEventListener("click", () => runAction(1));
    ui.scenarioSelect.addEventListener("change", (event) => setScenario(event.target.value));
    ui.policySelect.addEventListener("change", (event) => {
      app.policy = event.target.value;
      addLog(`自动策略切换为「${event.target.options[event.target.selectedIndex].textContent.split(" · ")[0]}」。`, "system");
    });
    ui.changeStepInput.addEventListener("input", (event) => {
      ui.changeStepValue.textContent = event.target.value;
    });
    ui.changeStepInput.addEventListener("change", (event) => {
      app.env.changeStep = Number(event.target.value);
      resetExperiment(`变化点已设为 T+${app.env.changeStep}`);
    });
    ui.resetButton.addEventListener("click", () => resetExperiment());
    ui.autoButton.addEventListener("click", toggleAuto);
    ui.injectButton.addEventListener("click", () => {
      if (app.scenario === "stable") return;
      const before = app.env.globalStep;
      app.env.forceChange();
      addLog(`手动触发环境更新：时间线由 T+${before} 推进至 T+${app.env.globalStep}。`, "update");
      render();
    });
    ui.clearLogButton.addEventListener("click", () => {
      app.logs = [];
      renderLog();
    });
    window.addEventListener("keydown", (event) => {
      if (event.target.matches("input, select, textarea")) return;
      if (event.key === "1") runAction(0);
      if (event.key === "2") runAction(1);
      if (event.key === " ") {
        event.preventDefault();
        toggleAuto();
      }
    });
  }

  async function init() {
    try {
      const response = await fetch("arena_protocol.json", { cache: "no-store" });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      app.protocol = await response.json();
      app.seed = Number(app.protocol.random_seed);
      app.scenario = "local_flip";
      app.env = new ArenaModel(app.protocol, app.scenario, app.seed);
      populateScenarios();
      bindEvents();
      addLog(`协议加载完成：${app.protocol.description}`, "system");
      addLog(`当前场景为「${app.protocol.scenarios[app.scenario].label}」，变化将在 T+${app.env.changeStep} 发生。`, "system");
      render();
    } catch (error) {
      ui.protocolStatus.textContent = "协议加载失败";
      ui.protocolChip.classList.remove("ready");
      ui.eventLog.innerHTML = `<div class="empty-log">无法加载 arena_protocol.json。请在项目目录运行 <code>python -m http.server 8000</code> 后访问 http://localhost:8000。<br /><small>${error.message}</small></div>`;
    }
  }

  window.addEventListener("DOMContentLoaded", init);
})();
