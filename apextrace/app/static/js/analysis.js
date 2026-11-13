// Analysis view: the lap list, and one lap or a pair of laps on screen.
//
// Lap A is the reference; lap B is compared against it. With only one
// of them picked the view shows that lap on its own.

import { ChartStack } from "./charts.js";
import { TrackMap } from "./trackmap.js";
import { api, clear, cssVar, deltaClass, el, fixed, lapTime, on, signed } from "./util.js";

const state = { laps: [], a: null, b: null, payload: null, request: 0 };

const $ = (id) => document.getElementById(id);
let charts;
let map;

const CHANNEL_PANELS = {
  Speed: { title: "Speed [km/h]", height: 190 },
  Throttle: { title: "Throttle [%]", height: 110, range: [-4, 104] },
  Brake: { title: "Brake [%]", height: 110, range: [-4, 104] },
  Gear: { title: "Gear", height: 100, stepped: true, digits: 0 },
  RPM: { title: "Engine speed [rpm]", height: 110, digits: 0 },
  Steering: { title: "Steering [% of lock]", height: 120, zero: true },
};

const METRICS = [
  ["full_throttle", "Full throttle"],
  ["braking", "Braking"],
  ["coasting", "Coasting"],
];

export function initAnalysis() {
  charts = new ChartStack($("charts"), { onCursor: (index) => map.setCursor(index) });
  map = new TrackMap($("map"));

  $("file").addEventListener("change", (event) => {
    uploadFiles([...event.target.files]);
    event.target.value = "";
  });
  const drop = $("drop");
  for (const type of ["dragenter", "dragover"]) {
    drop.addEventListener(type, (event) => {
      event.preventDefault();
      drop.classList.add("over");
    });
  }
  for (const type of ["dragleave", "drop"]) {
    drop.addEventListener(type, () => drop.classList.remove("over"));
  }
  drop.addEventListener("drop", (event) => {
    event.preventDefault();
    uploadFiles([...event.dataTransfer.files]);
  });
  $("fastf1-form").addEventListener("submit", loadFastF1);

  on("laps-changed", refreshLaps);
  // Chart colours are read from CSS when a chart is built, so a theme
  // switch needs the charts rebuilt.
  matchMedia("(prefers-color-scheme: light)").addEventListener("change", () => render());

  refreshLaps();
}

// --- lap list ----------------------------------------------------------

async function refreshLaps() {
  try {
    state.laps = await api.laps();
  } catch (error) {
    say(error.message, true);
    return;
  }
  const ids = new Set(state.laps.map((lap) => lap.id));
  const before = [state.a, state.b];
  if (!ids.has(state.a)) state.a = null;
  if (!ids.has(state.b)) state.b = null;
  renderLaps();
  if (before[0] !== state.a || before[1] !== state.b) load();
}

function renderLaps() {
  $("laps-hint").hidden = state.laps.length > 0;
  clear($("laps")).append(
    ...state.laps.map((lap) => {
      const isA = lap.id === state.a;
      const isB = lap.id === state.b;
      const where = [originName(lap.origin), lap.track].filter(Boolean).join(" · ");
      return el(
        "li",
        { class: `lap${isA ? " is-a" : ""}${isB ? " is-b" : ""}` },
        el(
          "div",
          { class: "lap-pick" },
          el("button", {
            class: "pick-a", text: "A", "aria-pressed": String(isA),
            title: "Use as reference lap (A)", onclick: () => pick("a", lap.id),
          }),
          el("button", {
            class: "pick-b", text: "B", "aria-pressed": String(isB),
            title: "Compare against A (B)", onclick: () => pick("b", lap.id),
          }),
        ),
        el("div", { class: "lap-label", text: lap.label, title: lap.label }),
        el("div", { class: "lap-time", text: lapTime(lap.lap_time) }),
        el(
          "div",
          { class: "lap-meta" },
          el("span", { text: where }),
          el("button", {
            class: "lap-remove", text: "remove", "aria-label": `Remove ${lap.label}`,
            onclick: () => removeLap(lap.id),
          }),
        ),
      );
    }),
  );
}

// Where a lap came from, in the user's terms: two laps can share a
// label (lap 3 of a capture, lap 3 driven just now) and differ in this.
function originName(origin) {
  if (origin === "live") return "live";
  if (origin === "fastf1") return "F1 timing";
  return origin.startsWith("upload:") ? origin.slice(7) : origin;
}

function pick(slot, id) {
  const other = slot === "a" ? "b" : "a";
  state[slot] = state[slot] === id ? null : id;
  if (state[other] === id) state[other] = null;
  renderLaps();
  load();
}

async function removeLap(id) {
  try {
    await api.remove(id);
  } catch (error) {
    say(error.message, true);
  }
  refreshLaps();
}

// --- loading laps ------------------------------------------------------

function say(text, isError = false) {
  const node = $("load-message");
  node.textContent = text;
  node.classList.toggle("error", isError);
}

async function uploadFiles(files) {
  if (!files.length) return;
  const lines = [];
  const added = [];
  let failed = false;
  for (const file of files) {
    say(`reading ${file.name}…`);
    try {
      const result = await api.upload(file, $("steering-lock").value);
      added.push(...result.added);
      const notes = result.notes.length ? ` (${result.notes.join("; ")})` : "";
      lines.push(`${file.name}: ${result.added.length} laps${notes}`);
    } catch (error) {
      failed = true;
      lines.push(`${file.name}: ${error.message}`);
    }
  }
  say(lines.join(" · "), failed);
  await refreshLaps();
  autoPick(added);
}

async function loadFastF1(event) {
  event.preventDefault();
  const form = event.target;
  const button = form.querySelector("button");
  const params = Object.fromEntries(new FormData(form));
  params.year = Number(params.year);
  button.disabled = true;
  say("downloading from F1 timing; the first lap of a session takes a minute…");
  try {
    const lap = await api.fastf1(params);
    say(`${lap.label}: ${lapTime(lap.lap_time)}`);
    await refreshLaps();
    autoPick([lap]);
  } catch (error) {
    say(error.message, true);
  } finally {
    button.disabled = false;
  }
}

// With nothing on screen yet, show what was just loaded: the quickest
// new lap as A and the next quickest as B.
function autoPick(added) {
  if (!added.length || state.a || state.b) return;
  const ranked = [...added].sort((x, y) => x.lap_time - y.lap_time);
  state.a = ranked[0].id;
  state.b = ranked[1]?.id ?? null;
  renderLaps();
  load();
}

// --- the result --------------------------------------------------------

async function load() {
  const request = ++state.request;
  const single = state.a ?? state.b;
  try {
    let payload = null;
    if (state.a && state.b) payload = { kind: "compare", ...(await api.compare(state.a, state.b)) };
    else if (single) payload = { kind: "single", ...(await api.lap(single)) };
    if (request !== state.request) return; // a newer selection overtook this one
    state.payload = payload;
    state.error = null;
  } catch (error) {
    if (request !== state.request) return;
    state.payload = null;
    state.error = error.message;
  }
  render();
}

function render() {
  const { payload } = state;
  const empty = $("analysis-empty");
  $("analysis-result").hidden = !payload;
  empty.hidden = Boolean(payload);
  if (!payload) {
    charts.destroy();
    clear(empty).append(
      el("h1", { text: state.error ? "These laps cannot be compared" : "Pick a lap" }),
      state.error
        ? el("p", { text: state.error })
        : el("p", {},
            "Mark one lap as ", el("b", { class: "chip a", text: "A" }),
            " to look at it on its own, and a second as ", el("b", { class: "chip b", text: "B" }),
            " to compare it against A. Positive delta means B is behind."),
    );
    return;
  }
  if (payload.kind === "compare") renderCompare(payload);
  else renderSingle(payload);
}

function lapHead(chip, label, time) {
  return el(
    "div",
    { class: "result-lap" },
    el("div", { class: "who" }, el("span", { class: `chip ${chip}`, text: chip.toUpperCase() }), el("span", { text: label, title: label })),
    el("div", { class: "time", text: lapTime(time) }),
  );
}

function renderCompare(p) {
  clear($("result-head")).append(
    lapHead("a", p.a.label, p.a.lap_time),
    lapHead("b", p.b.label, p.b.lap_time),
    el(
      "div",
      { class: "result-delta" },
      el("div", { class: "label", text: "B against A" }),
      el("div", { class: `value ${deltaClass(p.final_delta)}`, text: `${signed(p.final_delta)} s` }),
    ),
  );

  const notices = [];
  if (p.cross_source) {
    notices.push(
      "These laps come from different sources. Car, tyres and track model differ, so read this comparison for its shapes, not for its number.",
    );
  }
  if (p.corners_error) notices.push(`No corner table: ${p.corners_error}.`);
  const notice = $("result-notice");
  notice.hidden = notices.length === 0;
  notice.textContent = notices.join(" ");

  const colours = { a: cssVar("--lap-a"), b: cssVar("--lap-b") };
  charts.render({
    x: p.distance,
    segments: p.segments,
    panels: [
      {
        title: "Delta, B against A [s]", height: 170, zero: true, digits: 3, minSpan: 0.1,
        series: [{ label: "Δ", color: cssVar("--text"), values: p.delta }],
      },
      ...Object.entries(p.channels).map(([name, values]) => ({
        ...CHANNEL_PANELS[name],
        series: [
          { label: "A", color: colours.a, values: values.a },
          { label: "B", color: colours.b, values: values.b },
        ],
      })),
    ],
  });

  renderMap(
    p.map && { x: p.map.x, y: p.map.y, values: p.map.loss_per_100m, mode: "diverging" },
    "Where the time goes",
    "Coloured by time lost per 100 m: teal where B gains on A, orange where it loses.",
  );
  renderMetrics([["A", "a", p.metrics.a], ["B", "b", p.metrics.b]]);
  renderCornerComparison(p.corners);
  renderBehaviour(p.behaviour);
}

function renderSingle(p) {
  const chip = state.a ? "a" : "b";
  clear($("result-head")).append(lapHead(chip, p.label, p.lap_time));
  $("result-notice").hidden = true;

  const colour = cssVar(chip === "a" ? "--lap-a" : "--lap-b");
  charts.render({
    x: p.distance,
    segments: p.corners,
    panels: Object.entries(p.channels).map(([name, values]) => ({
      ...CHANNEL_PANELS[name],
      series: [{ label: chip.toUpperCase(), color: colour, values }],
    })),
  });

  renderMap(
    p.map && { x: p.map.x, y: p.map.y, values: p.channels.Speed, mode: "sequential" },
    "Track map",
    "Coloured by speed: orange slow, teal fast.",
  );
  renderMetrics([[chip.toUpperCase(), chip, p.metrics]]);
  renderCornerList(p.corners);
  renderBehaviour(p.behaviour);
}

function renderMap(data, title, hint) {
  $("map-title").textContent = title;
  $("map-hint").textContent = data ? hint : "This lap has no position channels, so there is no map to draw.";
  map.render(data || null);
}

function renderMetrics(groups) {
  clear($("metrics")).append(
    ...groups.map(([name, chip, metrics]) =>
      el(
        "div",
        { class: "metric-group" },
        el("h3", {}, el("span", { class: `chip ${chip}`, text: name }), " share of the lap distance"),
        ...METRICS.map(([key, label]) => {
          const share = metrics[key] ?? 0;
          const fill = el("i");
          fill.style.width = `${(share * 100).toFixed(1)}%`;
          fill.style.background = cssVar(chip === "a" ? "--lap-a" : "--lap-b");
          return el(
            "div",
            { class: "metric-row" },
            el("span", { text: label }),
            el("div", { class: "bar" }, fill),
            el("span", { class: "num", text: `${(share * 100).toFixed(1)}%` }),
          );
        }),
      ),
    ),
  );
}

function table(columns, rows, footer) {
  const head = el("tr", {}, ...columns.map(([title, unit]) => el("th", {}, title, unit ? el("small", { text: unit }) : null)));
  const body = rows.map((cells) =>
    el("tr", {}, ...cells.map((cell) => (cell instanceof Node ? cell : el("td", { text: cell })))),
  );
  return el(
    "table",
    {},
    el("thead", {}, head),
    el("tbody", {}, ...body),
    footer ? el("tfoot", {}, el("tr", {}, ...footer.map((cell) => (cell instanceof Node ? cell : el("td", { text: cell }))))) : null,
  );
}

const timeCell = (value) => el("td", { class: deltaClass(value), text: signed(value) });
const metres = (value) => (value == null ? "–" : signed(value, 1));

function renderCornerComparison(corners) {
  const panel = $("corners-panel");
  panel.hidden = !corners;
  if (!corners) return;
  const sum = (key) => corners.reduce((total, row) => total + (row[key] ?? 0), 0);
  clear($("corners")).append(
    table(
      [
        ["Corner"], ["Apex A", "km/h"], ["Apex B", "km/h"],
        ["Brake", "m, + later"], ["Apex", "m, + deeper"], ["Power", "m, + later"],
        ["Coast A", "m"], ["Coast B", "m"], ["Entry", "s"], ["Exit", "s"], ["Total", "s"],
      ],
      corners.map((row) => [
        row.corner, fixed(row.apex_a_kmh), fixed(row.apex_b_kmh),
        metres(row.brake_m), metres(row.apex_m), metres(row.power_m),
        fixed(row.coast_a_m, 0), fixed(row.coast_b_m, 0),
        timeCell(row.entry_s), timeCell(row.exit_s), timeCell(row.total_s),
      ]),
      ["Lap", "", "", "", "", "", "", "", timeCell(sum("entry_s")), timeCell(sum("exit_s")), timeCell(sum("total_s"))],
    ),
  );
}

function renderCornerList(corners) {
  const panel = $("corners-panel");
  panel.hidden = corners.length === 0;
  if (!corners.length) return;
  clear($("corners")).append(
    table(
      [["Corner"], ["Apex", "m"], ["Apex speed", "km/h"], ["Brake point", "m"], ["Full throttle", "m"]],
      corners.map((c) => [c.label, fixed(c.apex, 0), fixed(c.apex_speed), fixed(c.brake_point, 0), fixed(c.throttle_point, 0)]),
    ),
  );
}

// Filled in by the behaviour heuristics when the payload carries them.
function renderBehaviour(behaviour) {
  const panel = $("behaviour-panel");
  panel.hidden = !behaviour?.rows?.length;
  if (panel.hidden) return;
  clear($("behaviour")).append(
    table(
      behaviour.columns.map((c) => [c.title, c.unit]),
      behaviour.rows.map((row) => behaviour.columns.map((c) => formatBehaviour(row[c.key], c))),
    ),
  );
}

function formatBehaviour(value, column) {
  if (typeof value === "string") return value;
  if (value == null) return "–";
  return column.signed ? signed(value, column.digits ?? 1) : fixed(value, column.digits ?? 1);
}
