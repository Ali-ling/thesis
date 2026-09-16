(() => {
  "use strict";

  const $ = (selector) => document.querySelector(selector);
  const clone = (value) => JSON.parse(JSON.stringify(value));
  const clamp = (value, min, max) => Math.min(max, Math.max(min, value));
  const format = (value, digits = 2) => Number(value || 0).toFixed(digits);

  const ui = {
    protocolStatus: $("#protocolStatus"),
    versionPill: $(".version-pill"),
    heroCargo: $("#heroCargo"),
    heroStep: $("#heroStep"),
    heroVersion: $("#heroVersion"),
    heroEpisodeStep: $("#heroEpisodeStep"),
    phaseIndicator: $("#phaseIndicator"),
    phaseText: $("#phaseText"),
    currentNode: $("#currentNode"),
    currentBrief: $("#currentBrief"),
    objectiveHint: $("#objectiveHint"),
    lastAction: $("#lastAction"),
    lastReward: $("#lastReward"),
    routeMap: $("#routeMap"),
    routeHint: $("#routeHint"),
    shieldValue: $("#shieldValue"),
    energyValue: $("#energyValue"),
    threatValue: $("#threatValue"),
    cargoValue: $("#cargoValue"),
    shieldBar: $("#shieldBar"),
    energyBar: $("#energyBar"),
    threatBar: $("#threatBar"),
    cargoBar: $("#cargoBar"),
    actionGrid: $("#actionGrid"),
    decisionNote: $("#decisionNote"),
    eventStream: $("#eventStream"),
    scenarioSelect: $("#scenarioSelect"),
    scenarioDescription: $("#scenarioDescription"),
    changeStepValue: $("#changeStepValue"),
    scenarioTag: $("#scenarioTag"),
    changeStepInput: $("#changeStepInput"),
    resetButton: $("#resetButton"),
    injectButton: $("#injectButton"),
    intelStatus: $("#intelStatus"),
    intelMain: $("#intelMain"),
    intelTarget: $("#intelTarget"),
    intelHorizon: $("#intelHorizon"),
    clearLogButton: $("#clearLogButton"),
  };

  const app = { protocol: null, env: null, scenario: "mixed_shift", seed: 20260916, logs: [] };

  class ExpeditionModel {
    constructor(protocol, scenario, seed) {
      this.protocol = protocol;
      this.scenario = scenario;
      this.changeStep = Number(protocol.change_step);
      this.horizon = Number(protocol.episode_horizon);
      this.targetCargo = Number(protocol.target_cargo);
      this.nodeKinds = protocol.node_kinds;
      this.routeRewriteNodes = new Set(protocol.route_rewrite_nodes);
      this.setSeed(seed);
      this.reset();
    }

    setSeed(seed) {
      this.seed = Number(seed);
      this.random = this.makeRandom(this.seed);
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

    reset() {
      this.globalStep = 0;
      this.episodeStep = 0;
      this.episodeIndex = 0;
      this.position = 0;
      this.shield = Number(this.protocol.initial_shield);
      this.energy = Number(this.protocol.initial_energy);
      this.threat = Number(this.protocol.initial_threat);
      this.cargo = 0;
      this.episodeReturn = 0;
      this.lastAction = -1;
      this.lastReward = 0;
      this.lastDamage = 0;
      this.lastEvents = ["新的远征开始：收集资源并返回星门"];
      this.terminalReason = "";
      this.success = false;
      this.scanMemory = {};
      this.history = [];
      this.completed = [];
      this.resourceStock = [];
      this.hazardReal = [];
      this.protocol.nodes.forEach((node) => {
        const capacity = Number(node.resource_capacity);
        const variation = Math.floor(this.random() * 5) === 0 ? -1 : (Math.floor(this.random() * 6) === 0 ? 1 : 0);
        this.resourceStock.push(capacity > 0 ? Math.max(0, capacity + variation) : 0);
        this.hazardReal.push(clamp(Number(node.base_hazard) + (this.random() * .05 - .025), .005, .85));
      });
    }

    setScenario(scenario) {
      this.scenario = scenario;
      this.reset();
    }

    phaseAt(time = this.globalStep) {
      if (["stable", "storm_surge", "route_rewrite", "resource_decay", "mixed_shift"].includes(this.scenario)) {
        return this.scenario === "stable" || time < this.changeStep ? "base" : "updated";
      }
      if (time < this.changeStep) return "base";
      if (time < this.changeStep + Number(this.protocol.gradual_duration)) return "transition";
      return "updated";
    }

    progressAt(time = this.globalStep) {
      if (this.scenario !== "gradual_update") return time < this.changeStep ? 0 : 1;
      return clamp((time - this.changeStep) / Number(this.protocol.gradual_duration), 0, 1);
    }

    dynamicsAt(time = this.globalStep) {
      const progress = this.progressAt(time);
      const result = { hazard: 1, threat: 1, guard: 1, resource: 1, dash: 1, rewrite: false };
      if (this.scenario === "storm_surge") Object.assign(result, { hazard: 1.65, threat: 1.35 });
      if (this.scenario === "route_rewrite") Object.assign(result, { rewrite: true, hazard: 1.08 });
      if (this.scenario === "resource_decay") Object.assign(result, { resource: .54, hazard: 1.06 });
      if (this.scenario === "mixed_shift") Object.assign(result, { hazard: 1.48, threat: 1.28, guard: .72, resource: .68, dash: 1.35, rewrite: true });
      if (this.scenario === "gradual_update") {
        Object.assign(result, {
          hazard: 1 + progress * .48,
          threat: 1 + progress * .28,
          guard: 1 - progress * .28,
          resource: 1 - progress * .32,
          dash: 1 + progress * .35,
          rewrite: progress >= .65,
        });
      }
      if (time < this.changeStep && this.scenario !== "stable") return { hazard: 1, threat: 1, guard: 1, resource: 1, dash: 1, rewrite: false };
      return result;
    }

    nextPosition(position, direction, dynamics = this.dynamicsAt()) {
      const node = this.protocol.nodes[position];
      if (direction > 0 && dynamics.rewrite && this.routeRewriteNodes.has(position)) return Number(node.alternate_forward ?? node.forward);
      return direction > 0 ? Number(node.forward) : Number(node.reverse);
    }

    step(action) {
      const params = this.dynamicsAt();
      const actionId = Number(action);
      const events = [];
      let reward = -.025;
      let terminated = false;
      let truncated = false;
      let valid = true;
      const startPosition = this.position;
      this.lastDamage = 0;
      if (actionId === 0) reward += this.travel(1, 1, params, events);
      else if (actionId === 1) reward += this.travel(-1, 1, params, events);
      else if (actionId === 2) reward += this.travel(1, 2, params, events);
      else if (actionId === 3) { const result = this.harvest(params, events); valid = result.valid; reward += result.reward; }
      else if (actionId === 4) { const result = this.guard(params, events); valid = result.valid; reward += result.reward; }
      else if (actionId === 5) { const result = this.scan(events); valid = result.valid; reward += result.reward; }
      else if (actionId === 6) { const result = this.extract(events); valid = result.valid; reward += result.reward; terminated = result.terminated; }
      else throw new Error("动作超出范围");

      if (!terminated) {
        reward += this.resolveHazard(actionId, params, events);
        if (this.shield <= 0) {
          terminated = true;
          this.terminalReason = "shield_depleted";
          reward -= 12;
          events.push("护盾归零，远征失败");
        }
      }
      this.threat = clamp(this.threat, 0, 100);
      this.energy = clamp(this.energy, 0, 100);
      this.shield = clamp(this.shield, 0, 100);
      this.globalStep += 1;
      this.episodeStep += 1;
      this.lastAction = actionId;
      this.lastReward = reward;
      this.lastEvents = events;
      this.episodeReturn += reward;
      this.history.push({ time: this.globalStep - 1, action: actionId, reward, position: startPosition, nextPosition: this.position, events });
      if (!terminated && this.episodeStep >= this.horizon) {
        truncated = true;
        this.terminalReason = "horizon_reached";
        if (this.cargo < this.targetCargo) { reward -= 2; this.episodeReturn -= 2; events.push("时间耗尽，资源未达标"); }
        else { reward -= 1; this.episodeReturn -= 1; events.push("时间耗尽，尚未完成撤离"); }
      }
      const oldPhase = this.phaseAt(this.globalStep - 1);
      const newPhase = this.phaseAt(this.globalStep);
      if (oldPhase !== newPhase) events.push("环境版本发生变化");
      this.lastEvents = events;
      if (terminated || truncated) this.completed.push({ return: this.episodeReturn, success: this.success, cargo: this.cargo, reason: this.terminalReason });
      return { reward, terminated, truncated, valid };
    }

    travel(direction, hops, params, events) {
      const cost = hops === 1 ? 4 : 15;
      if (this.energy < cost) { this.threat += 8; events.push("能量不足，航行动作失败"); return -.42; }
      this.energy -= cost;
      for (let index = 0; index < hops; index += 1) this.position = this.nextPosition(this.position, direction, params);
      this.threat += hops === 1 ? 3 : 8 * params.dash;
      events.push(`航行至 ${this.protocol.nodes[this.position].short}`);
      return hops === 1 ? -.04 : -.14;
    }

    harvest(params, events) {
      if (this.energy < 7) { this.threat += 7; events.push("能量不足，无法稳定采集"); return { valid: false, reward: -.38 }; }
      this.energy -= 7;
      const node = this.protocol.nodes[this.position];
      const stock = this.resourceStock[this.position];
      if (Number(node.resource_capacity) <= 0 || stock <= 0) { this.threat += 5; events.push("这里没有可提取资源"); return { valid: true, reward: -.22 }; }
      const scanBonus = this.scanMemory[this.position] ? .16 : 0;
      const successProbability = clamp(.58 * params.resource + scanBonus, .18, .82);
      if (this.random() > successProbability) { this.threat += 7; events.push("采集失败：资源被干扰或设备失稳"); return { valid: true, reward: -.16 }; }
      let amount = 1;
      if (stock >= 2 && this.random() < .28 * params.resource) amount = 2;
      amount = Math.min(amount, stock, this.targetCargo - this.cargo);
      this.resourceStock[this.position] -= amount;
      this.cargo += amount;
      this.threat += 4 + this.hazardReal[this.position] * 12;
      events.push(`采集成功：+${amount} 资源，货舱 ${this.cargo}/${this.targetCargo}`);
      return { valid: true, reward: .38 * amount };
    }

    guard(params, events) {
      const safe = ["gate", "relay"].includes(this.protocol.nodes[this.position].kind);
      const cost = safe ? 0 : 5;
      if (this.energy < cost) { this.threat += 4; events.push("能量不足，戒备系统无法启动"); return { valid: false, reward: -.3 }; }
      const threatBefore = this.threat;
      this.energy -= cost;
      const shieldGain = (safe ? 18 : 14) * params.guard;
      const threatDrop = (safe ? 30 : 21) * params.guard;
      const energyGain = safe ? 20 : 0;
      this.energy = Math.min(100, this.energy + energyGain);
      this.shield = Math.min(100, this.shield + shieldGain);
      this.threat = Math.max(0, this.threat - threatDrop);
      events.push(safe ? `安全节点补给：能量 +${energyGain}，护盾 +${format(shieldGain, 1)}` : `相位戒备：护盾 +${format(shieldGain, 1)}，威胁 -${format(threatDrop, 1)}`);
      return { valid: true, reward: safe || threatBefore >= 35 ? .11 : -.015 };
    }

    scan(events) {
      if (this.energy < 3) { this.threat += 4; events.push("能量不足，扫描中断"); return { valid: false, reward: -.25 }; }
      this.energy -= 3;
      this.scanMemory[this.position] = { resource: this.resourceStock[this.position], hazard: this.hazardReal[this.position], time: this.globalStep };
      this.threat += 1;
      events.push(`扫描完成：库存信号 ${this.resourceStock[this.position]}，风险 ${format(this.hazardReal[this.position])}`);
      return { valid: true, reward: -.055 };
    }

    extract(events) {
      if (this.position === 0 && this.cargo >= this.targetCargo) {
        this.success = true;
        this.terminalReason = "successful_extraction";
        events.push(`成功撤离：带回 ${this.cargo} 份资源`);
        return { valid: true, reward: 22 + this.cargo * 1.2, terminated: true };
      }
      if (this.position === 0) { this.threat += 7; events.push(`撤离失败：需要 ${this.targetCargo}，当前只有 ${this.cargo}`); return { valid: true, reward: -.55, terminated: false }; }
      this.threat += 3;
      events.push("撤离信标只能在星门基地使用");
      return { valid: true, reward: -.28, terminated: false };
    }

    resolveHazard(action, params, events) {
      const node = this.protocol.nodes[this.position];
      const risks = [.012, .006, .07 * params.dash, .045, -.025, 0, -.01];
      const probability = clamp(this.hazardReal[this.position] * params.hazard + Math.max(0, this.threat - 30) / 360 + risks[action], .005, .88);
      if (this.random() >= probability) { this.threat = Math.max(0, this.threat - .4); return 0; }
      let damage = 5 + Math.floor(this.random() * 13);
      if (node.kind === "storm") damage += 3 + Math.floor(this.random() * 7);
      this.shield -= damage;
      this.lastDamage = damage;
      this.threat += 6;
      events.push(`遭遇环境冲击：护盾 -${damage}`);
      return -.035 * damage;
    }

    snapshot() {
      const node = this.protocol.nodes[this.position];
      const scan = this.scanMemory[this.position] || null;
      return {
        scenario: this.scenario,
        phase: this.phaseAt(),
        progress: this.progressAt(),
        version: this.phaseAt() === "base" ? "V0" : this.phaseAt() === "transition" ? "V0→V1" : "V1",
        globalStep: this.globalStep,
        episodeStep: this.episodeStep,
        position: this.position,
        shield: this.shield,
        energy: this.energy,
        threat: this.threat,
        cargo: this.cargo,
        targetCargo: this.targetCargo,
        lastAction: this.lastAction,
        lastReward: this.lastReward,
        lastDamage: this.lastDamage,
        lastEvents: this.lastEvents,
        terminalReason: this.terminalReason,
        success: this.success,
        scan,
        nodes: this.protocol.nodes.map((item, index) => ({ ...item, scanned: Boolean(this.scanMemory[index]) })),
      };
    }
  }

  function addLog(message, type = "normal", time = app.env?.globalStep || 0) {
    app.logs.unshift({ message, type, time });
    app.logs = app.logs.slice(0, 18);
    renderLog();
  }

  function eventType(message) {
    if (message.includes("成功撤离")) return "success";
    if (message.includes("护盾") || message.includes("失败") || message.includes("不足") || message.includes("冲击")) return "warning";
    if (message.includes("版本")) return "update";
    return "normal";
  }

  function renderLog() {
    if (!app.logs.length) { ui.eventStream.innerHTML = '<div class="empty-stream">远征尚未开始。</div>'; return; }
    ui.eventStream.innerHTML = app.logs.map((entry) => `<div class="event-entry ${entry.type}"><span class="event-time">T+${String(entry.time).padStart(4, "0")}</span><i class="event-dot"></i><span class="event-message">${entry.message}</span></div>`).join("");
  }

  function renderScenarioOptions() {
    ui.scenarioSelect.innerHTML = Object.entries(app.protocol.scenarios).map(([key, scenario]) => `<option value="${key}">${scenario.label} · ${scenario.tag}</option>`).join("");
    ui.scenarioSelect.value = app.scenario;
  }

  function renderRoute(snapshot) {
    ui.routeMap.innerHTML = snapshot.nodes.map((node, index) => {
      const danger = ["storm", "ruins"].includes(node.kind);
      const current = index === snapshot.position;
      const signal = node.scanned ? "已扫描" : "未知";
      const mapOrder = index < 6 ? index : index;
      return `<div class="route-node${current ? " current" : ""}${node.scanned ? " scanned" : ""}${danger ? " danger" : ""}" data-node="${index}" style="order:${mapOrder}"><div class="node-code"><span>S${String(index).padStart(2, "0")}</span><b>${current ? "HERE" : node.kind.toUpperCase()}</b></div><div class="node-name">${node.short}</div><div class="node-meta"><i class="node-signal"></i><span>${signal}</span></div><i class="node-connection"></i></div>`;
    }).join("");
    ui.routeHint.textContent = snapshot.scan ? `S${snapshot.position} 已获得局部情报；其他节点库存仍未知` : "未知节点：扫描后获得局部情报";
  }

  function renderActions() {
    ui.actionGrid.innerHTML = app.protocol.actions.map((action, index) => {
      const danger = index === 2 || index === 6;
      return `<button class="action-button${danger ? " action-danger" : ""}" type="button" data-action="${index}"><span class="action-key">${index + 1}</span><strong>${action.name}</strong><small>${action.description}</small></button>`;
    }).join("");
    ui.actionGrid.querySelectorAll("[data-action]").forEach((button) => button.addEventListener("click", () => takeAction(Number(button.dataset.action))));
  }

  function renderIntel(snapshot) {
    if (!snapshot.scan) {
      ui.intelStatus.textContent = "未扫描";
      ui.intelStatus.className = "intel-status";
      ui.intelMain.innerHTML = '<div class="intel-empty"><span>?</span><b>情报未知</b><small>执行「扫描节点」查看当前位置的风险与库存信号。</small></div>';
      return;
    }
    ui.intelStatus.textContent = "已扫描";
    ui.intelStatus.className = "intel-status ready";
    ui.intelMain.innerHTML = `<div class="intel-readout"><div class="intel-value"><span>可见库存</span><b>${snapshot.scan.resource}</b><small>资源单位</small></div><div class="intel-value"><span>局部风险</span><b>${format(snapshot.scan.hazard)}</b><small>基础概率信号</small></div></div>`;
  }

  function render(snapshot) {
    const node = snapshot.nodes[snapshot.position];
    const phaseClass = snapshot.phase === "base" ? "phase-base" : snapshot.phase === "transition" ? "phase-transition" : "phase-updated";
    ui.protocolStatus.textContent = "V2 协议已加载";
    ui.versionPill.classList.add("ready");
    ui.heroCargo.textContent = `${snapshot.cargo}/${snapshot.targetCargo}`;
    ui.heroStep.textContent = String(snapshot.globalStep).padStart(4, "0");
    ui.heroVersion.textContent = snapshot.version;
    ui.heroEpisodeStep.textContent = `${String(snapshot.episodeStep).padStart(3, "0")}/${app.env.horizon}`;
    ui.phaseIndicator.className = `phase-indicator ${phaseClass}`;
    ui.phaseText.textContent = snapshot.phase === "base" ? "V0 稳定版本" : snapshot.phase === "transition" ? `V0 → V1 · ${Math.round(snapshot.progress * 100)}%` : "V1 新版本";
    ui.currentNode.textContent = `S${snapshot.position} · ${node.name}`;
    ui.currentBrief.textContent = node.brief;
    ui.lastAction.textContent = snapshot.lastAction < 0 ? "—" : app.protocol.actions[snapshot.lastAction].name;
    ui.lastReward.textContent = `奖励 ${snapshot.lastReward >= 0 ? "+" : ""}${format(snapshot.lastReward)}`;
    ui.objectiveHint.textContent = snapshot.cargo >= snapshot.targetCargo ? "资源已达标，找到星门并执行撤离。" : "未知库存不会自动显示，先扫描再决定。";
    ui.shieldValue.textContent = Math.round(snapshot.shield);
    ui.energyValue.textContent = Math.round(snapshot.energy);
    ui.threatValue.textContent = Math.round(snapshot.threat);
    ui.cargoValue.textContent = `${snapshot.cargo} / ${snapshot.targetCargo}`;
    ui.shieldBar.style.width = `${snapshot.shield}%`;
    ui.energyBar.style.width = `${snapshot.energy}%`;
    ui.threatBar.style.width = `${snapshot.threat}%`;
    ui.cargoBar.style.width = `${Math.min(100, snapshot.cargo / snapshot.targetCargo * 100)}%`;
    ui.scenarioDescription.textContent = app.protocol.scenarios[app.scenario].description;
    ui.changeStepValue.textContent = app.env.changeStep;
    ui.changeStepInput.value = app.env.changeStep;
    ui.scenarioTag.textContent = app.protocol.scenarios[app.scenario].tag;
    ui.intelTarget.textContent = `${snapshot.targetCargo} 份`;
    ui.intelHorizon.textContent = `${app.env.horizon} 步`;
    ui.decisionNote.textContent = snapshot.lastEvents.length ? snapshot.lastEvents.join("；") : "行动结果由当前局势、隐藏库存和随机事件共同决定。没有预先给定的最优动作。";
    renderRoute(snapshot);
    renderIntel(snapshot);
  }

  function takeAction(action) {
    if (!app.env || app.env.terminalReason) return;
    const result = app.env.step(action);
    const actionName = app.protocol.actions[action].name;
    const eventMessages = app.env.lastEvents.length ? app.env.lastEvents : [`执行${actionName}`];
    eventMessages.forEach((message) => addLog(message, eventType(message), app.env.globalStep));
    addLog(`动作「${actionName}」完成：奖励 ${result.reward >= 0 ? "+" : ""}${format(result.reward)}。`, "normal", app.env.globalStep);
    if (result.terminated || result.truncated) {
      if (app.env.success) addLog("任务成功：你把资源带回了星门。", "success", app.env.globalStep);
      else addLog(`远征结束：${app.env.terminalReason === "shield_depleted" ? "护盾耗尽" : "时间耗尽"}。点击「重新出发」再试一次。`, "warning", app.env.globalStep);
    }
    render(app.env.snapshot());
  }

  function resetGame(reason = "新的远征已开始") {
    app.env.reset();
    app.logs = [];
    addLog(reason, "normal", app.env.globalStep);
    render(app.env.snapshot());
  }

  function bindEvents() {
    ui.scenarioSelect.addEventListener("change", (event) => {
      app.scenario = event.target.value;
      app.env.setScenario(app.scenario);
      resetGame(`版本剧本切换为「${app.protocol.scenarios[app.scenario].label}」`);
    });
    ui.changeStepInput.addEventListener("input", (event) => { ui.changeStepValue.textContent = event.target.value; });
    ui.changeStepInput.addEventListener("change", (event) => {
      app.env.changeStep = Number(event.target.value);
      resetGame(`版本更新时点设为 T+${app.env.changeStep}`);
    });
    ui.resetButton.addEventListener("click", () => resetGame());
    ui.injectButton.addEventListener("click", () => {
      const before = app.env.globalStep;
      app.env.globalStep = Math.max(app.env.globalStep, app.env.changeStep);
      addLog(`调试更新：时间线由 T+${before} 推进至 T+${app.env.globalStep}。`, "update", app.env.globalStep);
      render(app.env.snapshot());
    });
    ui.clearLogButton.addEventListener("click", () => { app.logs = []; renderLog(); });
    window.addEventListener("keydown", (event) => {
      if (event.target.matches("input, select, textarea")) return;
      const number = Number(event.key);
      if (number >= 1 && number <= 7) takeAction(number - 1);
    });
  }

  async function init() {
    try {
      const response = await fetch("arena_v2_protocol.json", { cache: "no-store" });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      app.protocol = await response.json();
      app.scenario = app.protocol.default_scenario;
      app.seed = Number(app.protocol.seed);
      app.env = new ExpeditionModel(app.protocol, app.scenario, app.seed);
      renderScenarioOptions();
      renderActions();
      bindEvents();
      addLog(`V2 协议加载完成：${app.protocol.description}`, "normal");
      addLog("提示：随机试玩成功率很低，真正的 PPO 需要从完整轨迹中学习资源与风险的长期关系。", "update");
      render(app.env.snapshot());
    } catch (error) {
      ui.protocolStatus.textContent = "协议加载失败";
      ui.eventStream.innerHTML = `<div class="empty-stream">无法加载 arena_v2_protocol.json。请通过本地 HTTP 服务访问页面。<br /><small>${error.message}</small></div>`;
    }
  }

  window.addEventListener("DOMContentLoaded", init);
})();
