"""Longitudinal and lateral acceleration, derived from speed and position.

No source hands over accelerations in the canonical schema, and none
has to: a car's planar motion is fully described by how fast it goes
and where. Along the path, acceleration is the rate of change of speed.
Across it, acceleration is speed squared times the curvature of the
line being driven. Both are computed here from the Speed and X/Y
channels a Lap already has, so they exist for every source with a
position channel, F1 timing data included, on one definition.

    a_long = dv/dt = v * dv/ds
    a_lat  = v^2 * kappa,   kappa = (x' y'' - y' x'') / (x'^2 + y'^2)^(3/2)

with every derivative taken along distance, the lap's own axis.

These are estimates, and the price is differentiation. Speed arrives
quantised (whole km/h in both F1 sources) and position is noisy, and a
derivative amplifies exactly that, a second derivative more so. So the
derivatives are taken with a Savitzky-Golay filter: a low-order
polynomial fitted over a sliding window, differentiated analytically.
That is smoothing, and the house rule is that smoothing is declared,
never hidden: the two windows are parameters, in metres, and they are
the knob that trades noise against detail. A window wider than a corner
flattens its peak. The defaults were chosen from the noise the sources
have, not tuned on real laps, and a figure quoted from here should say
which windows it was computed with.

Three more limits. The motion is taken as planar, so elevation and
banking are ignored. The sign of the lateral acceleration follows the
handedness of the source's map, which differs between sources: compare
magnitudes across sources, signs only within one. And nothing here is
a measurement of grip: it is what the car did, which is at most what
the tyres could do.
"""

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from apextrace.corners import Corner, segment_lap
from apextrace.lap import Lap

G = 9.80665  # standard gravity [m/s^2]
NAN = float("nan")


def _window(metres: float, step: float, polyorder: int, available: int) -> int:
    """A filter window in samples: odd, wide enough to fit, no wider than the lap."""
    points = max(int(round(metres / step)) | 1, polyorder + 2 | 1)
    return min(points, available if available % 2 else available - 1)


def derived_g(
    lap: Lap,
    speed_window: float = 50.0,
    position_window: float = 100.0,
    polyorder: int = 3,
) -> pd.DataFrame:
    """Longitudinal and lateral acceleration along a lap, in g.

    Returns a DataFrame on the lap's grid with Distance, GLong (positive
    accelerating, negative braking) and GLat (signed; NaN throughout if
    the lap has no X/Y channels). Nothing is stored back on the Lap.

    speed_window, position_window:
        Length of track [m] the smoothing polynomial is fitted over,
        for the speed derivative and for the path curvature. Position
        needs the wider one: curvature is a second derivative.
    polyorder:
        Order of that polynomial. Three keeps a constant-curvature arc
        and a constant acceleration exact.
    """
    dist = lap.data["Distance"].to_numpy(dtype=float)
    step = float(dist[1] - dist[0])
    # The grid is uniform except for its last point, which sits exactly
    # on the line. The filter assumes uniform spacing, so it runs on the
    # uniform part and the last point takes its neighbour's value.
    uniform = dist.size if np.isclose(dist[-1] - dist[-2], step) else dist.size - 1

    speed = lap.data["Speed"].to_numpy(dtype=float)[:uniform] / 3.6  # m/s
    window = _window(speed_window, step, polyorder, uniform)
    smooth = savgol_filter(speed, window, polyorder)
    slope = savgol_filter(speed, window, polyorder, deriv=1, delta=step)
    g_long = smooth * slope / G

    g_lat = np.full(uniform, NAN)
    if "X" in lap.channels and "Y" in lap.channels:
        window = _window(position_window, step, polyorder, uniform)
        x = lap.data["X"].to_numpy(dtype=float)[:uniform]
        y = lap.data["Y"].to_numpy(dtype=float)[:uniform]
        dx = savgol_filter(x, window, polyorder, deriv=1, delta=step)
        dy = savgol_filter(y, window, polyorder, deriv=1, delta=step)
        ddx = savgol_filter(x, window, polyorder, deriv=2, delta=step)
        ddy = savgol_filter(y, window, polyorder, deriv=2, delta=step)
        curvature = (dx * ddy - dy * ddx) / np.power(dx * dx + dy * dy, 1.5)
        g_lat = smooth * smooth * curvature / G

    def on_grid(values: np.ndarray) -> np.ndarray:
        return values if uniform == dist.size else np.append(values, values[-1])

    return pd.DataFrame(
        {"Distance": dist, "GLong": on_grid(g_long), "GLat": on_grid(g_lat)}
    )


@dataclass
class CornerG:
    """Acceleration metrics for one corner, in g and metres.

    Lateral and combined figures are NaN for a lap without position
    channels; the longitudinal ones need only speed.
    """

    corner: str
    brake_g: float  # peak deceleration
    accel_g: float  # peak acceleration
    lat_g: float  # peak lateral acceleration, magnitude
    combined_g: float  # peak of the two together
    trail_g_m: float  # decelerating and cornering at once
    power_g_m: float  # accelerating and cornering at once


def g_report(
    lap: Lap,
    corners: list[Corner] | None = None,
    combined_fraction: float = 0.3,
    **windows: float,
) -> pd.DataFrame:
    """Acceleration metrics for every corner of a lap, one row per corner.

    The two overlap lengths are the vehicle-dynamics reading of a
    corner's phases: how far the car is slowing and turning together
    (trail braking, seen from the car rather than from the pedal) and
    how far it is turning and accelerating together (power applied
    before the steering is unwound). Each needs both accelerations
    above combined_fraction of that lap's own envelope, taken as the
    98th percentile so a single spike does not set the scale.

    corners defaults to the lap's own segmentation. windows are passed
    to derived_g.
    """
    if corners is None:
        corners = segment_lap(lap)
    table = derived_g(lap, **windows)
    dist = table["Distance"].to_numpy()
    step = float(dist[1] - dist[0])
    g_long = table["GLong"].to_numpy()
    g_lat = table["GLat"].to_numpy()
    has_lat = not np.isnan(g_lat).all()

    if has_lat:
        long_scale = float(np.percentile(np.abs(g_long), 98))
        lat_scale = float(np.percentile(np.abs(g_lat), 98))
        cornering = np.abs(g_lat) >= combined_fraction * lat_scale
        slowing = g_long <= -combined_fraction * long_scale
        driving = g_long >= combined_fraction * long_scale

    rows = []
    for corner in corners:
        seg = (dist >= corner.start) & (dist <= corner.end)
        row = CornerG(
            corner=corner.name or f"#{corner.number}",
            brake_g=float(max(-np.min(g_long[seg]), 0.0)),
            accel_g=float(max(np.max(g_long[seg]), 0.0)),
            lat_g=NAN,
            combined_g=NAN,
            trail_g_m=NAN,
            power_g_m=NAN,
        )
        if has_lat:
            row.lat_g = float(np.max(np.abs(g_lat[seg])))
            row.combined_g = float(np.max(np.hypot(g_long[seg], g_lat[seg])))
            row.trail_g_m = float(np.sum(seg & cornering & slowing) * step)
            row.power_g_m = float(np.sum(seg & cornering & driving) * step)
        rows.append(asdict(row))

    report = pd.DataFrame(rows)
    report.attrs["lap"] = lap.label
    return report
