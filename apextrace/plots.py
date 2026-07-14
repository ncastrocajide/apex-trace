"""Comparison figures built on canonical Laps.

Every function returns (fig, axes) so callers can tweak, show or save.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection

from apextrace.delta import aligned_axes, delta_time
from apextrace.lap import Lap


def plot_delta_overlay(
    lap_a: Lap,
    lap_b: Lap,
    channels: tuple[str, ...] = ("Speed", "Throttle", "Brake"),
    method: str = "time",
):
    """Cumulative time delta on top, channel overlays below, shared x.

    Where the delta rises, lap_b is losing time to lap_a right there.
    Channels missing from either lap are simply not drawn for that lap.
    """
    delta = delta_time(lap_a, lap_b, method=method)
    dist_a, dist_b = aligned_axes(lap_a, lap_b)

    n_panels = 1 + len(channels)
    fig, axes = plt.subplots(
        n_panels, 1, sharex=True, figsize=(12, 2.2 * n_panels)
    )

    top = axes[0]
    top.plot(delta.index, delta.to_numpy(), color="tab:red", linewidth=1.4)
    top.axhline(0.0, color="black", linewidth=0.8)
    top.set_ylabel("Delta t [s]")
    top.set_title(
        f"{lap_b.label} vs {lap_a.label}  "
        f"(final {delta.iloc[-1]:+.3f} s, method: {delta.attrs['method']})"
    )

    for ax, channel in zip(axes[1:], channels):
        for lap, dist in ((lap_a, dist_a), (lap_b, dist_b)):
            if channel in lap.channels:
                ax.plot(dist, lap.data[channel], label=lap.label, linewidth=1.0)
        ax.set_ylabel(channel)

    axes[1].legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("Distance [m]")
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig, axes


def plot_track_map(lap: Lap, marks: pd.DataFrame | None = None,
                   other: Lap | None = None):
    """Driven line coloured by speed, with official corner marks.

    Needs the optional X/Y channels (raises if the source lacks them).
    marks: table with X, Y and Name columns, e.g. from
    load_fastf1_corner_marks. A second lap draws underneath as a thin
    grey line, which makes different driving lines visible on track.
    """
    if "X" not in lap.channels or "Y" not in lap.channels:
        raise ValueError(f"{lap.label!r} has no X/Y channels to draw")

    x = lap.data["X"].to_numpy(dtype=float)
    y = lap.data["Y"].to_numpy(dtype=float)
    speed = lap.data["Speed"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(9, 7))

    if other is not None:
        ax.plot(other.data["X"], other.data["Y"], color="grey",
                linewidth=0.9, label=other.label)

    # One coloured segment per grid step: LineCollection maps a value
    # (the mean speed of the step) through the colormap for each one.
    points = np.column_stack([x, y])
    segments = np.stack([points[:-1], points[1:]], axis=1)
    line = LineCollection(segments, cmap="viridis", linewidths=2.2)
    line.set_array((speed[:-1] + speed[1:]) / 2.0)
    ax.add_collection(line)
    fig.colorbar(line, ax=ax, label=f"Speed [km/h], {lap.label}")

    if marks is not None:
        ax.plot(marks["X"], marks["Y"], "o", color="black", markersize=3)
        for _, mark in marks.iterrows():
            ax.annotate(str(mark["Name"]), (mark["X"], mark["Y"]),
                        textcoords="offset points", xytext=(6, 6),
                        fontsize=8, fontweight="bold")

    ax.set_aspect("equal")
    ax.autoscale()
    ax.margins(0.05)
    ax.set_axis_off()
    ax.set_title(lap.label if other is None
                 else f"{lap.label} (coloured) vs {other.label} (grey)")
    fig.tight_layout()
    return fig, ax
