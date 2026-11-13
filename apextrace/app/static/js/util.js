// Small shared helpers: DOM building, number formatting, the API.

/** Build an element. Text is always set as text, never parsed as HTML:
 *  lap labels come from telemetry files and must not be trusted. */
export function el(tag, props = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(props)) {
    if (value == null || value === false) continue;
    if (key === "class") node.className = value;
    else if (key === "text") node.textContent = value;
    else if (key.startsWith("on")) node.addEventListener(key.slice(2), value);
    else node.setAttribute(key, value === true ? "" : value);
  }
  for (const child of children.flat()) {
    if (child == null) continue;
    node.append(child instanceof Node ? child : document.createTextNode(child));
  }
  return node;
}

export function clear(node) {
  node.replaceChildren();
  return node;
}

export function cssVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

/** 83.412 -> "1:23.412"; under a minute -> "58.204". */
export function lapTime(seconds) {
  if (seconds == null || !Number.isFinite(seconds)) return "–";
  const minutes = Math.floor(seconds / 60);
  const rest = (seconds - minutes * 60).toFixed(3);
  return minutes > 0 ? `${minutes}:${rest.padStart(6, "0")}` : rest;
}

/** A delta with an explicit sign; a true minus, and "0" without one. */
export function signed(value, digits = 3) {
  if (value == null || !Number.isFinite(value)) return "–";
  const rounded = Number(value.toFixed(digits));
  if (rounded === 0) return (0).toFixed(digits);
  return (rounded > 0 ? "+" : "−") + Math.abs(rounded).toFixed(digits);
}

/** "gain" for time gained (negative delta), "loss" for time lost. */
export function deltaClass(value, tolerance = 0.0005) {
  if (value == null || !Number.isFinite(value) || Math.abs(value) < tolerance) return "";
  return value < 0 ? "gain" : "loss";
}

export function fixed(value, digits = 1) {
  return value == null || !Number.isFinite(value) ? "–" : value.toFixed(digits);
}

async function request(path, options = {}) {
  let response;
  try {
    response = await fetch(path, options);
  } catch {
    throw new Error("the app is not reachable; is it still running?");
  }
  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try {
      const body = await response.json();
      if (typeof body.detail === "string") detail = body.detail;
      else if (Array.isArray(body.detail)) {
        detail = body.detail.map((d) => `${d.loc?.slice(-1)[0]}: ${d.msg}`).join("; ");
      }
    } catch { /* not JSON; keep the status line */ }
    throw new Error(detail);
  }
  return response.json();
}

export const api = {
  laps: () => request("/api/laps"),
  lap: (id) => request(`/api/laps/${encodeURIComponent(id)}`),
  remove: (id) => request(`/api/laps/${encodeURIComponent(id)}`, { method: "DELETE" }),
  compare: (a, b) =>
    request(`/api/compare?a=${encodeURIComponent(a)}&b=${encodeURIComponent(b)}`),
  upload: (file) => {
    const body = new FormData();
    body.append("file", file);
    return request("/api/upload", { method: "POST", body });
  },
  fastf1: (params) =>
    request("/api/fastf1", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(params),
    }),
  setReference: (lapId) =>
    request("/api/live/reference", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ lap_id: lapId }),
    }),
};

/** A tiny event bus so the two views can tell each other about laps. */
export const bus = new EventTarget();
export const emit = (name, detail) => bus.dispatchEvent(new CustomEvent(name, { detail }));
export const on = (name, handler) => bus.addEventListener(name, (e) => handler(e.detail));
