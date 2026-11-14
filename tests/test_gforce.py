"""Derived accelerations against motions whose accelerations are known.

A circle driven at constant speed has v^2 / R of lateral acceleration
and none along the path. A straight driven at constant acceleration has
that acceleration and no lateral. Everything else is built from those.
"""

import numpy as np
import pytest

from apextrace.gforce import G, derived_g, g_report

DIST = np.arange(0.0, 3001.0, 5.0)
RADIUS = 3000.0 / (2.0 * np.pi)
ANGLE = DIST / RADIUS


@pytest.fixture
def on_circle(make_lap):
    """Factory: a lap round a 3 km circle at the given speed trace."""

    def _lap(speed, mirrored=False, **kwargs):
        lap = make_lap(DIST, speed, **kwargs)
        lap.data["X"] = RADIUS * np.cos(ANGLE)
        lap.data["Y"] = RADIUS * np.sin(ANGLE) * (-1.0 if mirrored else 1.0)
        return lap

    return _lap


def test_constant_speed_on_a_circle(on_circle):
    g = derived_g(on_circle(np.full(DIST.size, 200.0)))
    expected = (200.0 / 3.6) ** 2 / RADIUS / G
    assert g["GLat"].to_numpy() == pytest.approx(expected, rel=0.01)
    assert np.abs(g["GLong"]).max() < 1e-9


def test_lateral_sign_follows_the_handedness_of_the_map(on_circle):
    """Mirror the map and the same corner turns the other way."""
    speed = np.full(DIST.size, 200.0)
    left = derived_g(on_circle(speed))["GLat"]
    right = derived_g(on_circle(speed, mirrored=True))["GLat"]
    assert (left > 0).all() and (right < 0).all()
    assert np.allclose(left, -right)


def test_constant_acceleration_on_a_straight(make_lap):
    accel = 5.0  # m/s^2
    speed = np.sqrt((100.0 / 3.6) ** 2 + 2.0 * accel * DIST) * 3.6
    lap = make_lap(DIST, speed)
    lap.data["X"] = DIST
    lap.data["Y"] = np.zeros(DIST.size)

    g = derived_g(lap)
    assert g["GLong"].to_numpy() == pytest.approx(accel / G, rel=0.01)
    assert np.abs(g["GLat"]).max() < 1e-9


def test_without_position_only_the_lateral_is_missing(make_lap):
    """F1 timing laps without a map still have a speed to differentiate."""
    g = derived_g(make_lap(DIST, np.full(DIST.size, 200.0)))
    assert g["GLat"].isna().all()
    assert not g["GLong"].isna().any()


def test_smoothing_is_what_makes_quantised_speed_usable(make_lap):
    """Both F1 sources report whole km/h; a raw derivative amplifies that."""
    accel = 5.0
    exact = np.sqrt((100.0 / 3.6) ** 2 + 2.0 * accel * DIST) * 3.6
    quantised = np.round(exact)

    raw = (quantised / 3.6) * np.gradient(quantised / 3.6, DIST) / G
    filtered = derived_g(make_lap(DIST, quantised))["GLong"].to_numpy()
    inner = slice(10, -10)
    raw_error = np.abs(raw[inner] - accel / G).max()
    filtered_error = np.abs(filtered[inner] - accel / G).max()
    assert filtered_error < 0.15
    assert filtered_error < raw_error / 3.0


def test_the_window_trades_noise_against_detail(on_circle):
    """Declared, not hidden: a wider window is quieter."""
    rng = np.random.default_rng(1)
    lap = on_circle(np.full(DIST.size, 200.0))
    lap.data["X"] += rng.normal(0.0, 0.1, DIST.size)
    lap.data["Y"] += rng.normal(0.0, 0.1, DIST.size)

    inner = slice(20, -20)
    wide = derived_g(lap, position_window=100.0)["GLat"].to_numpy()[inner]
    narrow = derived_g(lap, position_window=50.0)["GLat"].to_numpy()[inner]
    assert wide.std() < narrow.std() / 3.0
    assert wide.mean() == pytest.approx((200.0 / 3.6) ** 2 / RADIUS / G, rel=0.01)


def test_the_exact_end_point_of_the_grid_is_handled(on_circle, make_lap):
    """The last grid point sits on the line, off the uniform spacing."""
    dist = np.append(np.arange(0.0, 3000.0, 5.0), 2998.3)
    lap = make_lap(dist, np.full(dist.size, 200.0))
    lap.data["X"] = RADIUS * np.cos(dist / RADIUS)
    lap.data["Y"] = RADIUS * np.sin(dist / RADIUS)

    g = derived_g(lap)
    assert len(g) == dist.size
    assert g["GLat"].iloc[-1] == pytest.approx(g["GLat"].iloc[-2])


def test_corner_report_reads_braking_and_cornering_together(on_circle, dipped_speed):
    """One corner on the circle: slowing in, cornering throughout, driving out."""
    speed = dipped_speed(DIST, base=280.0, dips=((1500.0, 100.0, 120.0),))
    report = g_report(on_circle(speed))
    assert len(report) == 1
    row = report.iloc[0]

    metres_per_second = speed / 3.6
    exact_long = metres_per_second * np.gradient(metres_per_second, DIST) / G
    assert row["brake_g"] == pytest.approx(-exact_long.min(), rel=0.03)
    assert row["accel_g"] == pytest.approx(exact_long.max(), rel=0.03)
    # Lateral peaks where the car is fastest on this constant-radius track.
    assert row["lat_g"] == pytest.approx((280.0 / 3.6) ** 2 / RADIUS / G, rel=0.02)
    assert row["combined_g"] >= max(row["brake_g"], row["lat_g"])
    # Always turning here, so the overlaps are the braking and driving zones.
    assert row["trail_g_m"] > 100.0 and row["power_g_m"] > 100.0
    assert row["trail_g_m"] == pytest.approx(row["power_g_m"], rel=0.1)


def test_corner_report_without_a_map_keeps_the_longitudinal_figures(
    make_lap, dipped_speed
):
    report = g_report(
        make_lap(DIST, dipped_speed(DIST, dips=((1500.0, 100.0, 120.0),)))
    )
    row = report.iloc[0]
    assert row["brake_g"] > 1.0 and row["accel_g"] > 1.0
    assert np.isnan(row["lat_g"]) and np.isnan(row["trail_g_m"])
