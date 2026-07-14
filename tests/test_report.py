"""The report's structural invariant on synthetic laps."""

import numpy as np
import pytest

from apextrace.delta import delta_time
from apextrace.report import corner_report, lap_metrics


def test_per_corner_deltas_sum_to_the_lap_delta(make_lap, dipped_speed):
    # Segments tile the lap, so the per-corner deltas must telescope to
    # the full-lap delta exactly. This is the M3 internal invariant.
    dist = np.arange(0.0, 4001.0, 5.0)
    lap_a = make_lap(
        dist, dipped_speed(dist, dips=((1000.0, 100.0, 120.0), (2600.0, 150.0, 150.0)))
    )
    lap_b = make_lap(
        dist, dipped_speed(dist, dips=((1000.0, 95.0, 120.0), (2600.0, 160.0, 150.0)))
    )
    table = corner_report(lap_a, lap_b)
    final = delta_time(lap_a, lap_b).iloc[-1]
    assert table["total_s"].sum() == pytest.approx(final, abs=1e-9)
    assert list(table["corner"]) == ["#1", "#2"]


def test_lap_metrics_fractions(make_lap):
    dist = np.arange(0.0, 1001.0, 5.0)
    n = dist.size
    throttle = np.where(dist < 500.0, 100.0, 0.0)
    brake = np.where(dist >= 750.0, 100.0, 0.0)
    metrics = lap_metrics(
        make_lap(dist, np.full(n, 200.0), throttle=throttle, brake=brake)
    )
    # 0-495 full throttle, 500-745 coasting, 750-1000 braking.
    assert metrics["full_throttle"] == pytest.approx(0.5, abs=0.02)
    assert metrics["coasting"] == pytest.approx(0.25, abs=0.02)
    assert metrics["braking"] == pytest.approx(0.25, abs=0.02)
