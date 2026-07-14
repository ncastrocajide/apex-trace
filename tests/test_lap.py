"""Patch tests of the distance resampler: the M1 checks, now permanent."""

import numpy as np
import pandas as pd
import pytest

from apextrace.lap import resample_to_distance


def poisoned_table() -> pd.DataFrame:
    """The M1 patch case: linear channels plus a poisoned duplicate row.

    Time and Throttle are linear in Distance (0.1x and 4x), so linear
    interpolation must reproduce them exactly. The duplicate row at 7 m
    carries absurd values and must die in the monotonicity cleanup.
    """
    return pd.DataFrame(
        {
            "Distance": [0.0, 4.0, 7.0, 7.0, 12.0, 18.0, 25.0],
            "Time": [0.0, 0.4, 0.7, 99.0, 1.2, 1.8, 2.5],
            "Throttle": [0.0, 16.0, 28.0, 99.0, 48.0, 72.0, 100.0],
            "Gear": [1, 1, 2, 9, 2, 3, 3],
        }
    )


def test_linear_channels_are_reproduced_exactly():
    out = resample_to_distance(poisoned_table(), step=5.0)
    np.testing.assert_allclose(out["Time"], out["Distance"] * 0.1, atol=1e-12)
    np.testing.assert_allclose(out["Throttle"], out["Distance"] * 4.0, atol=1e-12)


def test_poisoned_duplicate_row_is_dropped():
    out = resample_to_distance(poisoned_table(), step=5.0)
    assert not (out["Time"] > 90).any()
    assert not (out["Gear"] == 9).any()


def test_gear_uses_zero_order_hold():
    out = resample_to_distance(poisoned_table(), step=5.0)
    # Gear switches from 1 to 2 at 7 m: the grid point at 5 m must keep
    # the last known gear (1), not interpolate a 1.4.
    assert out.loc[out["Distance"] == 5.0, "Gear"].item() == 1
    assert out["Gear"].dtype.kind == "i"


def test_grid_ends_exactly_on_the_last_sample():
    out = resample_to_distance(poisoned_table(), step=4.0)
    assert out["Distance"].iloc[-1] == 25.0


def test_missing_distance_column_raises():
    with pytest.raises(ValueError, match="Distance"):
        resample_to_distance(pd.DataFrame({"Speed": [1.0, 2.0]}))
