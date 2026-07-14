"""Delta engine against closed-form answers on synthetic laps."""

import numpy as np
import pytest

from apextrace.delta import delta_time


def test_constant_speed_delta_matches_closed_form(make_lap):
    dist = np.arange(0.0, 1001.0, 5.0)
    lap_a = make_lap(dist, np.full(dist.size, 180.0))  # 50.0 m/s
    lap_b = make_lap(dist, np.full(dist.size, 120.0))  # 33.3 m/s
    # Slowness difference: 3.6/120 - 3.6/180 = 0.01 s/m, so the delta
    # grows linearly: delta(s) = 0.01 * s.
    delta = delta_time(lap_a, lap_b)
    np.testing.assert_allclose(
        delta.to_numpy(), 0.01 * delta.index.to_numpy(), atol=1e-9
    )


def test_time_and_speed_methods_agree_on_ideal_data(make_lap):
    dist = np.arange(0.0, 2001.0, 5.0)
    lap_a = make_lap(dist, 200.0 + 50.0 * np.sin(dist / 300.0))
    lap_b = make_lap(dist, 200.0 + 40.0 * np.cos(dist / 250.0))
    d_time = delta_time(lap_a, lap_b, method="time")
    d_speed = delta_time(lap_a, lap_b, method="speed")
    # On synthetic data both methods see the exact same kinematics, so
    # any disagreement would be an implementation bug, not data noise.
    np.testing.assert_allclose(d_speed.to_numpy(), d_time.to_numpy(), atol=1e-9)


def test_landmarks_change_the_interior_but_pin_the_final_delta(make_lap):
    dist = np.arange(0.0, 2001.0, 5.0)
    lap_a = make_lap(dist, 200.0 + 50.0 * np.sin(dist / 300.0))
    lap_b = make_lap(dist, 200.0 + 40.0 * np.cos(dist / 250.0))
    plain = delta_time(lap_a, lap_b)
    warped = delta_time(
        lap_a, lap_b, landmarks=(np.array([700.0, 1300.0]), np.array([690.0, 1310.0]))
    )
    assert warped.iloc[-1] == pytest.approx(plain.iloc[-1], abs=1e-9)
    assert np.max(np.abs(warped.to_numpy() - plain.to_numpy())) > 0.0
    assert warped.attrs["landmarks"] == 2


def test_non_monotonic_landmarks_raise(make_lap):
    dist = np.arange(0.0, 1001.0, 5.0)
    lap = make_lap(dist, np.full(dist.size, 180.0))
    with pytest.raises(ValueError, match="increasing"):
        delta_time(
            lap, lap, landmarks=(np.array([600.0, 300.0]), np.array([300.0, 600.0]))
        )


def test_unknown_method_raises(make_lap):
    dist = np.arange(0.0, 1001.0, 5.0)
    lap = make_lap(dist, np.full(dist.size, 180.0))
    with pytest.raises(ValueError, match="method"):
        delta_time(lap, lap, method="banana")
