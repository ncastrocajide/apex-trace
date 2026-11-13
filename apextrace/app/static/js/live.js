// Live view: the lap in progress against the reference lap.
//
// The server pushes one message per telemetry frame over a WebSocket.
// Messages only update state; the screen is redrawn at most once per
// animation frame, and the trace at most ten times a second, so a 60 Hz
// stream costs the same to display as a 20 Hz one.

import { ChartStack } from "./charts.js";
import { api, clear, cssVar, deltaClass, el, emit, lapTime, on, signed } from "./util.js";

const TRACE_INTERVAL = 100; // ms between redraws of the delta trace
const BAR_RANGES = [1, 2, 5, 10, 30, 60]; // seconds, half-width of the bar

const state = {
  connected: false,
  status: null,
  reference: null,
  pinned: false,
  lap: null,
  sample: null,
  trace: { d: [], delta: [] },
  laps: [],
  allLaps: [],
  notes: [],
  dirty: true,
  traceDrawn: 0,
  chartFor: undefined,
};

const $ = (id) => document.getElementById(id);
let chart;
let retry = 1000;

export function initLive() {
  chart = new ChartStack($("live-chart"));
  $("reference").addEventListener("change", async (event) => {
    try {
      await api.setReference(event.target.value || null);
    } catch (error) {
      note(`could not change the reference: ${error.message}`);
    }
  });
  on("laps-changed", refreshAllLaps);
  matchMedia("(prefers-color-scheme: light)").addEventListener("change", () => {
    state.chartFor = undefined;
    state.dirty = true;
  });
  refreshAllLaps();
  connect();
  requestAnimationFrame(frame);
}

// --- the socket --------------------------------------------------------

function connect() {
  const scheme = location.protocol === "https:" ? "wss" : "ws";
  const socket = new WebSocket(`${scheme}://${location.host}/ws/live`);
  socket.onopen = () => {
    state.connected = true;
    retry = 1000;
    state.dirty = true;
  };
  socket.onmessage = (event) => handle(JSON.parse(event.data));
  socket.onclose = () => {
    state.connected = false;
    state.dirty = true;
    setTimeout(connect, retry);
    retry = Math.min(retry * 2, 8000);
  };
}

function handle(message) {
  switch (message.type) {
    case "hello":
      state.status = message.status;
      state.reference = message.reference;
      state.pinned = message.pinned;
      state.lap = message.lap;
      state.laps = message.laps;
      state.trace = { d: message.trace.map((p) => p[0]), delta: message.trace.map((p) => p[1]) };
      state.sample = null;
      break;
    case "sample":
      if (message.lap !== state.lap) {
        state.lap = message.lap;
        state.trace = { d: [], delta: [] };
      }
      state.sample = message;
      if (message.delta != null) {
        const { d, delta } = state.trace;
        // A flashback moves the car back along the lap; so does the trace.
        while (d.length && d[d.length - 1] >= message.d) {
          d.pop();
          delta.pop();
        }
        d.push(message.d);
        delta.push(message.delta);
      }
      break;
    case "lap":
      state.laps.push(message.lap);
      emit("laps-changed");
      break;
    case "reference":
      state.reference = message.reference;
      state.pinned = message.pinned;
      state.trace = { d: [], delta: [] };
      break;
    case "skipped":
      note(`lap ${message.lap_number} not kept: ${message.reason}`);
      break;
    case "status":
      state.status = message;
      break;
  }
  state.dirty = true;
}

function note(text) {
  state.notes.unshift(text);
  state.notes.length = Math.min(state.notes.length, 6);
  state.dirty = true;
}

async function refreshAllLaps() {
  try {
    state.allLaps = await api.laps();
    state.dirty = true;
  } catch { /* the socket status already says the app is unreachable */ }
}

// --- drawing -----------------------------------------------------------

function frame(now) {
  if (state.dirty) {
    state.dirty = false;
    drawStatus();
    drawHero();
    drawTables();
    if (now - state.traceDrawn > TRACE_INTERVAL) {
      state.traceDrawn = now;
      drawTrace();
    } else {
      state.dirty = true; // come back for the trace
    }
  }
  requestAnimationFrame(frame);
}

function receiving() {
  return state.connected && Boolean(state.status?.receiving);
}

function drawStatus() {
  const pill = $("stream");
  const status = state.status;
  let text;
  pill.classList.toggle("on", receiving());
  pill.classList.toggle("off", !state.connected);
  if (!state.connected) text = "app not reachable";
  else if (receiving()) text = `live · ${status.track ?? status.source}`;
  else if (status?.source === "idle") text = "no live source";
  else text = "waiting for telemetry";
  $("stream-text").textContent = text;

  let line = "Connecting…";
  if (state.connected && status) {
    if (receiving()) line = `Receiving from ${status.source}${status.track ? ` · ${status.track}` : ""}`;
    else if (status.source === "idle") line = "No live source. Start the app without --no-live to listen for the game.";
    else if (status.source.startsWith("udp:")) {
      line = `Waiting for telemetry on UDP port ${status.source.slice(4)}. In F1 25: Settings › Telemetry › UDP Telemetry on, format 2025.`;
    } else line = `Waiting for ${status.source}`;
  } else if (!state.connected) {
    line = "Lost the connection to the app. Retrying…";
  }
  $("live-status").textContent = line;

  clear($("stream-stats")).append(
    ...[
      ["Source", status?.source ?? "–"],
      ["Track", status?.track ?? "–"],
      ["Packets", status ? String(status.packets) : "–"],
      ["Unreadable", status ? String(status.bad_packets) : "–"],
      ["Recording", status?.recording ? "yes" : "no"],
    ].flatMap(([k, v]) => [el("dt", { text: k }), el("dd", { text: v })]),
  );
  clear($("live-notes")).append(...state.notes.map((text) => el("li", { text })));
}

function drawHero() {
  const sample = state.sample;
  const delta = sample?.delta ?? null;
  const value = $("delta-value");
  value.textContent = signed(delta, 2);
  value.className = `delta-value ${deltaClass(delta, 0.005)}${receiving() ? "" : " stale"}`;

  const trend = $("delta-trend");
  if (!state.reference) {
    trend.textContent = "Finish a lap to set a reference, or pick one above.";
    trend.className = "delta-trend";
  } else if (delta == null) {
    trend.innerHTML = "&nbsp;";
    trend.className = "delta-trend";
  } else if (sample.valid === false) {
    trend.textContent = "lap invalidated";
    trend.className = "delta-trend loss";
  } else if (Math.abs(sample.trend) < 0.01) {
    trend.textContent = "■ holding";
    trend.className = "delta-trend";
  } else {
    const gaining = sample.trend < 0;
    trend.textContent = `${gaining ? "▼ gaining" : "▲ losing"} ${Math.abs(sample.trend).toFixed(2)} s over the last 150 m`;
    trend.className = `delta-trend ${gaining ? "gain" : "loss"}`;
  }

  // Bar: zero in the middle, time lost to the right. The range steps
  // up as needed so the bar never pins at the end.
  const size = Math.abs(delta ?? 0);
  const range = BAR_RANGES.find((r) => r >= size) ?? BAR_RANGES[BAR_RANGES.length - 1];
  const half = Math.min(size / range, 1) * 50;
  const fill = $("delta-bar-fill");
  fill.style.width = `${half}%`;
  fill.style.left = (delta ?? 0) < 0 ? `${50 - half}%` : "50%";
  fill.className = `delta-bar-fill ${deltaClass(delta, 0.005)}`;
  $("delta-scale-min").textContent = `−${range.toFixed(1)}`;
  $("delta-scale-max").textContent = `+${range.toFixed(1)}`;

  $("stat-predicted").textContent = lapTime(sample?.predicted ?? null);
  $("stat-current").textContent = sample ? lapTime(sample.t) : "–";
  $("stat-reference").textContent = state.reference ? lapTime(state.reference.lap_time) : "–";
  $("stat-last").textContent = lapTime(state.laps[state.laps.length - 1]?.lap_time ?? null);
  $("stat-speed").textContent = sample ? sample.speed.toFixed(0) : "–";
  $("stat-gear").textContent = sample ? gearName(sample.gear) : "–";
  $("pedal-throttle").style.height = `${sample?.throttle ?? 0}%`;
  $("pedal-brake").style.height = `${sample?.brake ?? 0}%`;
}

function gearName(gear) {
  if (gear === 0) return "N";
  return gear < 0 ? "R" : String(gear);
}

function drawTables() {
  // Reference picker: automatic, or any lap the app holds.
  const select = $("reference");
  const wanted = state.pinned && state.reference ? state.reference.id : "";
  const signature = JSON.stringify([state.allLaps.map((l) => l.id), wanted]);
  if (select.dataset.signature !== signature) {
    select.dataset.signature = signature;
    clear(select).append(
      el("option", { value: "", text: "Session best (automatic)" }),
      ...state.allLaps.map((lap) =>
        el("option", { value: lap.id, text: `${lap.label} · ${lapTime(lap.lap_time)}` }),
      ),
    );
    select.value = wanted;
  }

  const best = Math.min(...state.laps.map((lap) => lap.lap_time));
  const rows = [...state.laps].reverse().map((lap) =>
    el(
      "tr",
      { class: lap.id === state.reference?.id ? "is-reference" : null },
      el("td", { text: lap.label }),
      el("td", { text: lapTime(lap.lap_time) }),
      el("td", { class: deltaClass(lap.lap_time - best), text: lap.lap_time === best ? "best" : signed(lap.lap_time - best) }),
    ),
  );
  const wrap = clear($("live-laps"));
  if (!rows.length) {
    wrap.append(el("p", { class: "hint", text: "No complete laps yet in this session." }));
  } else {
    wrap.append(
      el("table", {},
        el("thead", {}, el("tr", {}, el("th", { text: "Lap" }), el("th", { text: "Time" }), el("th", { text: "To best" }))),
        el("tbody", {}, ...rows)),
    );
  }

  $("live-lap-label").textContent =
    state.lap == null ? "" : `lap ${state.lap}${state.reference ? ` against ${state.reference.label}` : ""}`;
}

function drawTrace() {
  // The chart is rebuilt only when its x axis changes (a new
  // reference, so a new lap length); otherwise just its data.
  const key = state.reference ? `${state.reference.id}:${state.reference.length}` : "none";
  if (state.chartFor !== key) {
    state.chartFor = key;
    chart.render({
      x: state.trace.d,
      xRange: state.reference ? [0, state.reference.length] : undefined,
      segments: [],
      panels: [
        {
          title: "Delta to reference [s]", height: 220, zero: true, digits: 3, minSpan: 0.2,
          series: [{ label: "Δ", color: cssVar("--text"), values: state.trace.delta }],
        },
      ],
    });
  } else {
    chart.update(state.trace.d, state.trace.delta);
  }
}
