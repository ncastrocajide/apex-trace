"""M4 demo: a sim lap against the real pole lap, through one engine.

Your fastest lap from an F1 25 capture against the pole lap of the same
circuit from official F1 timing. Two loaders that share nothing, two
canonical Laps, one comparison.

    uv run python examples/cross_source.py monza.f1cap
    uv run python examples/cross_source.py spa.f1cap --year 2024 \
        --gp "Belgian Grand Prix" --driver LEC

Read the result for its shapes, not for its number. The car, the tyres,
the downforce and the track model all differ between a game and the
real thing, so the total says little; where along the lap the two
traces part company (a braking point, a minimum speed, how early the
throttle comes back) is the part that carries over.

The laps must be of the same circuit: the engine reconciles two
distance axes that differ by metres, not two different tracks, so the
script refuses a pair whose lengths disagree by more than 3 %.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # render straight to file, no display needed

from apextrace.delta import delta_time
from apextrace.loaders.f1_25 import load_f1_25_lap
from apextrace.loaders.fastf1_loader import load_fastf1_lap
from apextrace.plots import plot_delta_overlay
from apextrace.report import lap_metrics

OUT_FILE = Path(__file__).resolve().parent / "cross_source.png"
MAX_LENGTH_MISMATCH = 0.03


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path, help="F1 25 capture file")
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--gp", default="Italian Grand Prix")
    parser.add_argument("--session", default="Q")
    parser.add_argument("--driver", default="NOR", help="pole sitter's code")
    args = parser.parse_args()

    real = load_fastf1_lap(args.year, args.gp, args.session, args.driver)
    sim = load_f1_25_lap(args.capture)

    length_real = float(real.data["Distance"].iloc[-1])
    length_sim = float(sim.data["Distance"].iloc[-1])
    if abs(length_real - length_sim) > MAX_LENGTH_MISMATCH * length_real:
        raise SystemExit(
            f"not the same circuit: {real.label} is {length_real:.0f} m long "
            f"and {sim.label} is {length_sim:.0f} m"
        )

    print(f"\nreal: {real.label:<32} {real.lap_time:8.3f} s  {length_real:6.0f} m")
    print(f"sim:  {sim.label:<32} {sim.lap_time:8.3f} s  {length_sim:6.0f} m")

    delta = delta_time(real, sim)
    print(f"\nsim against real: {delta.iloc[-1]:+.3f} s (qualitative, see above)")
    for lap in (real, sim):
        m = lap_metrics(lap)
        print(
            f"{lap.label}: {m['full_throttle']:.1%} full throttle, "
            f"{m['braking']:.1%} braking, {m['coasting']:.1%} coasting"
        )

    fig, _ = plot_delta_overlay(real, sim)
    fig.savefig(OUT_FILE, dpi=150)
    print(f"\nsaved {OUT_FILE}")


if __name__ == "__main__":
    main()
