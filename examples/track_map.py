"""M3 demo: track map from position channels with official corner marks.

The driven line of the pole lap at Monza, coloured by speed, with the
official corner designations located on the map. The count sanity-check
is visual: every dark (slow) patch on the line must sit next to a mark.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from apextrace.loaders.fastf1_loader import (
    load_fastf1_corner_marks,
    load_fastf1_lap,
)
from apextrace.plots import plot_track_map

OUT_FILE = Path(__file__).resolve().parent / "track_map.png"


def main() -> None:
    lap = load_fastf1_lap(2024, "Italian Grand Prix", "Q", "NOR")
    marks = load_fastf1_corner_marks(2024, "Italian Grand Prix", "Q")
    fig, _ = plot_track_map(lap, marks)
    fig.savefig(OUT_FILE, dpi=150)
    print(f"saved {OUT_FILE}")


if __name__ == "__main__":
    main()
