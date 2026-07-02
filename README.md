# apex-trace

Source-agnostic race telemetry analysis.

The goal: ingest lap telemetry from any source (real F1 timing data via
[FastF1](https://github.com/theOehrly/Fast-F1), or racing sims like Assetto
Corsa and F1 25), normalise it onto a canonical distance-based representation,
and run the same comparison engine on top: cumulative time delta, channel
overlays, corner-by-corner segmentation and driving metrics.

**Status: M2 (delta engine validated against official F1 timing).**

## Headline result

![Cumulative time delta and channel overlays, Leclerc vs Norris, Monza 2024 qualifying](docs/delta_overlay.png)

Leclerc vs Norris, Monza 2024 qualifying. The top panel is the cumulative
time delta (positive means Leclerc behind); the channels below explain it.
Leclerc gains steadily on the straights with a lower-drag Ferrari and brakes
visibly later into the Roggia chicane (the sharp spike at 2100 m), but pays
on every corner exit; the Lesmos and Ascari account for the 0.134 s he
concedes over the lap.

The engine is validated against official timing gaps in five scenarios
(fast, slow and street tracks; qualifying and race laps): zero measured
error for the default method. Reproduce with `examples/validate_delta.py`.

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
uv run python examples/hello_fastf1.py       # smoke test: lap times
uv run python examples/plot_speed_trace.py   # one lap through the pipeline
uv run python examples/compare_laps.py       # delta + overlays, two laps
uv run python examples/validate_delta.py     # engine vs official timing
```

The first run downloads one F1 session into a local cache (`.fastf1_cache/`,
takes a minute or two); subsequent runs are near-instant.

Loading a lap in your own code:

```python
from apextrace.loaders.fastf1_loader import load_fastf1_lap

lap = load_fastf1_lap(2024, "Italian Grand Prix", "Q", "NOR")
lap.data.head()   # canonical channels on a uniform 5 m distance grid
lap.lap_time      # seconds
```

## Roadmap

- [x] **M0**: skeleton, environment, FastF1 data chain
- [x] **M1**: canonical `Lap` (uniform distance grid) + FastF1 loader
- [x] **M2**: cumulative time-delta engine + channel overlays
- [ ] **M3**: corner segmentation + driving metrics
- [ ] **M4**: second loader (sim telemetry) through the same, untouched engine
- [ ] **M5**: setup-symptom heuristics (exploratory)

## Honest limits

Cross-source comparisons (real F1 vs sim) are qualitative: car, tyres,
downforce and track geometry all differ, so a numeric delta between sources
does not mean "you are X seconds slower". Within one source the comparison
is quantitative. FastF1 telemetry has no steering channel and a near on/off
brake signal; the richer channels live in sim data.
