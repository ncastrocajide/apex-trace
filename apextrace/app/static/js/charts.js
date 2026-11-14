// Stacked telemetry traces on a shared distance axis, built on uPlot.
//
// Every panel shares one x axis: the cursor is synced across panels,
// and zooming one zooms them all, so a point on the delta trace can be
// read straight down through speed, throttle and brake at the same
// piece of track. Corner segments are drawn behind the traces as
// alternating bands with the apex marked.

import { cssVar, el } from "./util.js";

const Y_AXIS_WIDTH = 52;
let stackCount = 0;

export class ChartStack {
  /** onCursor(index | null) is called as the cursor moves over a trace. */
  constructor(container, { onCursor } = {}) {
    this.container = container;
    this.onCursor = onCursor;
    this.plots = [];
    this.syncKey = `apex-${++stackCount}`;
    this.zooming = false;
    this.spec = null;
    this.resizer = new ResizeObserver(() => this.resize());
    this.resizer.observe(container);
  }

  /**
   * spec: { x: number[], segments: [{start, apex, end, label}],
   *         panels: [{ title, height, zero, stepped, range,
   *                    series: [{ label, color, values }] }] }
   */
  render(spec) {
    this.destroy();
    this.spec = spec;
    const width = this.width();
    const last = spec.panels.length - 1;
    spec.panels.forEach((panel, i) => {
      const host = el("div", { class: "chart" }, el("div", { class: "chart-title", text: panel.title }));
      this.container.append(host);
      const data = [spec.x, ...panel.series.map((s) => s.values)];
      this.plots.push(new uPlot(this.options(panel, width, i === last, i === 0), data, host));
    });
  }

  /** Replace the data of the first panel in place (the live trace). */
  update(x, ...values) {
    this.plots[0]?.setData([x, ...values]);
  }

  width() {
    return Math.max(320, Math.floor(this.container.clientWidth));
  }

  options(panel, width, isLast, isFirst) {
    const text = cssVar("--muted");
    const grid = cssVar("--grid");
    const axis = { stroke: text, grid: { stroke: grid, width: 1 }, ticks: { stroke: grid, width: 1 }, font: "11px system-ui, sans-serif" };
    const segments = this.spec.segments ?? [];

    return {
      width,
      height: panel.height ?? 150,
      padding: [14, 30, 0, 0],
      cursor: {
        sync: { key: this.syncKey, setSeries: false },
        drag: { x: true, y: false },
        points: { size: 6 },
        // Only the vertical line: it marks a position on track, which
        // is shared by every panel. A horizontal one would not be.
        y: false,
      },
      legend: { live: true },
      scales: {
        // A fixed x range (the live trace: the whole lap from the
        // start) or one fitted to the data (analysis, zoomable).
        x: this.spec.xRange ? { time: false, range: this.spec.xRange } : { time: false },
        y: { range: yRange(panel) },
      },
      axes: [
        // Distance labels only under the last panel; the others keep
        // their grid lines so the eye can follow a position down.
        { ...axis, values: isLast ? (_, ticks) => ticks.map((t) => `${t} m`) : () => [], size: isLast ? 30 : 6 },
        { ...axis, size: Y_AXIS_WIDTH, ...(panel.stepped ? { incrs: [1, 2, 5, 10], space: 18 } : {}) },
      ],
      series: [
        { label: "Distance", value: (_, v) => (v == null ? "–" : `${v.toFixed(0)} m`) },
        ...panel.series.map((s) => ({
          label: s.label,
          stroke: s.color,
          width: 1.5,
          points: { show: false },
          paths: panel.stepped ? uPlot.paths.stepped({ align: 1 }) : undefined,
          value: (_, v) => (v == null ? "–" : v.toFixed(panel.digits ?? 1)),
        })),
      ],
      hooks: {
        drawClear: [(u) => drawSegments(u, segments, isFirst, panel.zero)],
        setCursor: [(u) => this.onCursor?.(u.cursor.idx ?? null)],
        setScale: [(u, key) => this.syncZoom(u, key)],
      },
    };
  }

  // Propagate an x zoom from the panel the user dragged on to the rest.
  syncZoom(source, key) {
    if (key !== "x" || this.zooming) return;
    this.zooming = true;
    const { min, max } = source.scales.x;
    for (const plot of this.plots) {
      if (plot !== source) plot.setScale("x", { min, max });
    }
    this.zooming = false;
  }

  resize() {
    const width = this.width();
    for (const plot of this.plots) {
      if (plot.width !== width) plot.setSize({ width, height: plot.height });
    }
  }

  destroy() {
    for (const plot of this.plots) plot.destroy();
    this.plots = [];
    this.container.replaceChildren();
  }
}

// The y range of a panel: fixed if the panel says so; otherwise fitted
// to the data, but never tighter than the panel's minimum span. A delta
// of a few milliseconds is flat, and should look flat, not be blown up
// to fill the panel.
function yRange(panel) {
  if (panel.range) return panel.range;
  return (u, min, max) => {
    if (min == null || max == null) [min, max] = [0, 0];
    const span = Math.max(max - min, panel.minSpan ?? 0);
    const mid = (min + max) / 2;
    const pad = (span || 1) * 0.08;
    return [mid - span / 2 - pad, mid + span / 2 + pad];
  };
}

function drawSegments(u, segments, labelled, zeroLine) {
  const { ctx, bbox } = u;
  const ratio = devicePixelRatio || 1;
  ctx.save();
  ctx.beginPath();
  ctx.rect(bbox.left, bbox.top, bbox.width, bbox.height);
  ctx.clip();

  segments.forEach((segment, i) => {
    const x0 = u.valToPos(segment.start, "x", true);
    const x1 = u.valToPos(segment.end, "x", true);
    if (i % 2 === 0) {
      ctx.fillStyle = cssVar("--band");
      ctx.fillRect(x0, bbox.top, x1 - x0, bbox.height);
    }
    const apex = u.valToPos(segment.apex, "x", true);
    ctx.strokeStyle = cssVar("--grid");
    ctx.setLineDash([3 * ratio, 3 * ratio]);
    ctx.lineWidth = ratio;
    ctx.beginPath();
    ctx.moveTo(apex, bbox.top);
    ctx.lineTo(apex, bbox.top + bbox.height);
    ctx.stroke();
    if (labelled) {
      ctx.setLineDash([]);
      ctx.fillStyle = cssVar("--muted");
      ctx.font = `${10 * ratio}px system-ui, sans-serif`;
      ctx.textAlign = "center";
      ctx.fillText(segment.label, apex, bbox.top + 11 * ratio);
    }
  });

  if (zeroLine) {
    const y = u.valToPos(0, "y", true);
    ctx.setLineDash([]);
    ctx.strokeStyle = cssVar("--muted");
    ctx.lineWidth = ratio;
    ctx.beginPath();
    ctx.moveTo(bbox.left, y);
    ctx.lineTo(bbox.left + bbox.width, y);
    ctx.stroke();
  }
  ctx.restore();
}
