# apex-trace

Source-agnostic race telemetry analysis.

The goal: ingest lap telemetry from any source (real F1 timing data via
[FastF1](https://github.com/theOehrly/Fast-F1), or racing sims like Assetto
Corsa and F1 25), normalise it onto a canonical distance-based representation,
and run the same comparison engine on top: cumulative time delta, channel
overlays, corner-by-corner segmentation and driving metrics.

**Status: M0 (project skeleton and end-to-end data-chain proof).**

## Why source-agnostic

Comparing two laps rigorously means resampling every channel onto a common
distance axis. Once that normalisation exists, the analysis engine stops
caring where the data came from: adding a new telemetry source is just
writing one more loader that produces the same canonical `Lap`. That
source-independence is the point of the project.

## Stack

Python 3.12 · pandas · NumPy · SciPy · matplotlib · FastF1 · managed with
[uv](https://github.com/astral-sh/uv)

## Quick start

```bash
uv sync
uv run python examples/hello_fastf1.py
```

The first run downloads one F1 session into a local cache (`.fastf1_cache/`,
takes a minute or two); subsequent runs are near-instant.

## Roadmap

- [x] **M0**: skeleton, environment, FastF1 data chain
- [ ] **M1**: canonical `Lap` (uniform distance grid) + FastF1 loader
- [ ] **M2**: cumulative time-delta engine + channel overlays
- [ ] **M3**: corner segmentation + driving metrics
- [ ] **M4**: second loader (sim telemetry) through the same, untouched engine
- [ ] **M5**: setup-symptom heuristics (exploratory)

## Honest limits

Cross-source comparisons (real F1 vs sim) are qualitative: car, tyres,
downforce and track geometry all differ, so a numeric delta between sources
does not mean "you are X seconds slower". Within one source the comparison
is quantitative. FastF1 telemetry has no steering channel and a near on/off
brake signal; the richer channels live in sim data.
