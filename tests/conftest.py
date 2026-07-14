"""Shared builders for synthetic laps with answers known in closed form.

No real telemetry in the test suite: every assertion compares against a
value computed by hand, the same patch-test philosophy used to validate
the pipeline manually since M1. Checks against real data (official
timing, sector gaps) live in examples/validate_*.py and need the
network, so they stay out of CI on purpose.
"""

import numpy as np
import pandas as pd
import pytest
from scipy.integrate import cumulative_trapezoid

from apextrace.lap import Lap


@pytest.fixture
def make_lap():
    """Factory: a Lap on an explicit distance grid, self-consistent.

    Time is the integral of slowness (3.6 / speed) over distance, the
    same relation the delta engine exploits, so the synthetic kinematics
    hold exactly by construction.
    """

    def _make(
        dist,
        speed_kmh,
        throttle=None,
        brake=None,
        label="synthetic",
        source="synthetic",
    ) -> Lap:
        dist = np.asarray(dist, dtype=float)
        speed = np.asarray(speed_kmh, dtype=float)
        n = dist.size
        data = pd.DataFrame(
            {
                "Distance": dist,
                "Time": cumulative_trapezoid(3.6 / speed, dist, initial=0.0),
                "Speed": speed,
                "Throttle": (
                    np.full(n, 100.0)
                    if throttle is None
                    else np.asarray(throttle, dtype=float)
                ),
                "Brake": (
                    np.zeros(n) if brake is None else np.asarray(brake, dtype=float)
                ),
                "Gear": np.full(n, 8),
            }
        )
        return Lap(data=data, label=label, source=source)

    return _make


@pytest.fixture
def dipped_speed():
    """Factory: a speed trace with Gaussian dips (synthetic corners).

    Each dip is (centre [m], floor [km/h], width [m]); the trace sits at
    `base` elsewhere, so apex positions and speeds are known exactly.
    """

    def _dipped(dist, base=300.0, dips=((1000.0, 100.0, 120.0),)):
        dist = np.asarray(dist, dtype=float)
        speed = np.full(dist.size, base)
        for centre, floor, width in dips:
            speed = speed - (base - floor) * np.exp(-(((dist - centre) / width) ** 2))
        return speed

    return _dipped
