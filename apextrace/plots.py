"""Comparison figures built on canonical Laps.

Every function returns (fig, axes) so callers can tweak, show or save.
"""

import matplotlib.pyplot as plt

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
