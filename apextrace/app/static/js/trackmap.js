// Track map: the driven line, coloured by a value along it.
//
// Two colourings. "diverging" is for a comparison: time lost per 100 m,
// teal where lap B gains and orange-red where it loses, fading to
// neutral where nothing happens. "sequential" is for a single lap:
// speed, slow to fast.

import { cssVar } from "./util.js";

function mix(a, b, t) {
  return a.map((v, i) => Math.round(v + (b[i] - v) * t));
}

function parse(colour) {
  const probe = document.createElement("canvas").getContext("2d");
  probe.fillStyle = colour;
  const hex = probe.fillStyle; // normalised to #rrggbb
  return [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16));
}

export class TrackMap {
  constructor(canvas) {
    this.canvas = canvas;
    this.data = null;
    this.cursor = null;
    new ResizeObserver(() => this.draw()).observe(canvas);
  }

  /** data: { x, y, values, mode: "diverging" | "sequential" } or null. */
  render(data) {
    this.data = data;
    this.cursor = null;
    this.draw();
  }

  setCursor(index) {
    if (index === this.cursor) return;
    this.cursor = index;
    this.draw();
  }

  draw() {
    const { canvas, data } = this;
    const ratio = devicePixelRatio || 1;
    const width = canvas.clientWidth;
    const height = canvas.clientHeight;
    if (!width || !height) return;
    canvas.width = Math.round(width * ratio);
    canvas.height = Math.round(height * ratio);
    const ctx = canvas.getContext("2d");
    ctx.scale(ratio, ratio);
    ctx.clearRect(0, 0, width, height);
    if (!data) return;

    const points = [];
    for (let i = 0; i < data.x.length; i++) {
      if (data.x[i] != null && data.y[i] != null) points.push(i);
    }
    if (points.length < 2) return;

    const xs = points.map((i) => data.x[i]);
    const ys = points.map((i) => data.y[i]);
    const [minX, maxX] = [Math.min(...xs), Math.max(...xs)];
    const [minY, maxY] = [Math.min(...ys), Math.max(...ys)];
    const pad = 18;
    // One scale for both axes: a track map with the wrong aspect
    // ratio is a different track.
    const scale = Math.min(
      (width - 2 * pad) / Math.max(maxX - minX, 1),
      (height - 2 * pad) / Math.max(maxY - minY, 1),
    );
    const offsetX = (width - (maxX - minX) * scale) / 2;
    const offsetY = (height - (maxY - minY) * scale) / 2;
    const px = (i) => offsetX + (data.x[i] - minX) * scale;
    const py = (i) => height - offsetY - (data.y[i] - minY) * scale; // y up

    const colour = this.colourer();
    ctx.lineWidth = 4;
    ctx.lineCap = "round";
    for (let k = 1; k < points.length; k++) {
      const [i, j] = [points[k - 1], points[k]];
      ctx.strokeStyle = colour(data.values[j]);
      ctx.beginPath();
      ctx.moveTo(px(i), py(i));
      ctx.lineTo(px(j), py(j));
      ctx.stroke();
    }

    // Start/finish line marker.
    ctx.fillStyle = cssVar("--text");
    ctx.beginPath();
    ctx.arc(px(points[0]), py(points[0]), 3.5, 0, 2 * Math.PI);
    ctx.fill();

    const at = this.cursor;
    if (at != null && data.x[at] != null && data.y[at] != null) {
      ctx.fillStyle = cssVar("--surface");
      ctx.strokeStyle = cssVar("--text");
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(px(at), py(at), 6, 0, 2 * Math.PI);
      ctx.fill();
      ctx.stroke();
    }
  }

  colourer() {
    const values = this.data.values.filter((v) => v != null);
    const neutral = parse(cssVar("--neutral"));
    const rgb = (c) => `rgb(${c[0]}, ${c[1]}, ${c[2]})`;

    if (this.data.mode === "diverging") {
      const gain = parse(cssVar("--gain"));
      const loss = parse(cssVar("--loss"));
      // Scale to a high percentile, not the maximum, so one spike does
      // not wash the rest of the lap out to neutral.
      const sorted = values.map(Math.abs).sort((a, b) => a - b);
      const limit = Math.max(sorted[Math.floor(sorted.length * 0.97)] ?? 0, 1e-4);
      return (v) => {
        if (v == null) return rgb(neutral);
        const t = Math.min(Math.abs(v) / limit, 1);
        return rgb(mix(neutral, v < 0 ? gain : loss, Math.sqrt(t)));
      };
    }

    const slow = parse(cssVar("--loss"));
    const fast = parse(cssVar("--gain"));
    const min = Math.min(...values);
    const span = Math.max(Math.max(...values) - min, 1e-9);
    return (v) => {
      if (v == null) return rgb(neutral);
      const t = (v - min) / span;
      return rgb(t < 0.5 ? mix(slow, neutral, t * 2) : mix(neutral, fast, t * 2 - 1));
    };
  }
}
