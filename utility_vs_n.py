"""Experiment 4 (Figure 4): the utility of the coarse and the randomized releases as the number of
records n grows (Section 7, Table 2: loss in utility as n -> infinity).

For each n, controls.controls_table draws controls.RELEASES releases h of n records from
Mult_n(theta) and publishes each raw, coarse (the whole numbers of a QID column in bins of width w,
Section 7.1) and randomized (GRR_k(p') on the sensitive value, at level eps, Section 7.2). The
analyst, with the benchmark prior uni, estimates theta from it, and her utility is the utility gain
c(w, theta) = (2 - ||w - theta||^2) / kd of her estimate (Definition 4.4), at most 2/kd. Its mean
over the releases, with its 1 - controls.ALPHA confidence interval, is plotted against n:
    (a) for the coarse releases, whose gain stops short of 2/kd: however many records, the analyst
        does not learn how a coarse group splits into fine ones (Proposition 7.2);
    (b) for the randomized releases, whose gain tends to 2/kd, but needs more records the smaller
        eps is: the noisy release behaves like a raw release of n rho_k^2 records (Proposition 7.4);
in both, the raw release, as the reference.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import LogLocator, MaxNLocator

from common import STYLE, save
from controls import BEFORE, COARSE_SCALE, NOISY_SCALE, RAW, controls_table, utility

LINE_WIDTH = 1  # of the curves of the mean utility gains
BAND_ALPHA = 0.25  # opacity of their confidence intervals, in the color of the curve
# Markers at each n: a diamond for the raw release (as in controls.plot), and one shape per level,
# from the weakest protection to the strongest, so that curves of close colors stay apart
RAW_MARKER, MARKERS, MARKER_SIZE = "D", ["o", "s", "^", "v", "P", "X"], 6
# A marker every MARK_EVERY values of n, shifted by one from a level to the next, so that the markers
# of curves on top of each other do not overlap
MARK_EVERY = 4


def utility_table(theta: pd.DataFrame, sizes: list[int], column: str, widths: list[int],
                  epsilons: list[float]) -> pd.DataFrame:
    """The utility gain c of the releases of controls.controls_table for each number of records n of
    sizes: its mean over the controls.RELEASES releases of n records and the half-width of its
    confidence interval ("utility gain CI"), a row per (n, release), with the releases in the order of
    controls_table: raw, coarse with the QID column `column` in bins of each width of widths, and
    randomized at each level eps of epsilons."""
    tables = {n: controls_table(theta, n, column, widths, epsilons) for n in sizes}
    return pd.concat(tables, names=["n", "release"])[["utility gain", "utility gain CI"]]


def curve(ax, rows: pd.DataFrame, label: str, **style) -> None:
    """The mean utility gain of rows (a row per n) against n, with its confidence interval as a band
    in the same color."""
    n, mean, half = rows.index, rows["utility gain"], rows["utility gain CI"]
    (line,) = ax.plot(n, mean, label=label, linewidth=LINE_WIDTH, markersize=MARKER_SIZE, **style)
    ax.fill_between(n, mean - half, mean + half, color=line.get_color(), alpha=BAND_ALPHA,
                    linewidth=0)


def gain_axis(ax, low: float, kd: int) -> None:
    """y from just below low to just above the maximum gain 2/kd, in units of 10^p, the order of
    magnitude of 2/kd: round ticks, with none next to 2/kd, which is the last tick, written "2/kd".
    The room above 2/kd keeps the curves that tend to it in full."""
    top = 2 / kd
    span = top - low
    ax.set_ylim(low - 0.03 * span, top + 0.02 * span)
    p = int(np.floor(np.log10(top)))
    ticks = MaxNLocator(6, steps=[1, 2, 5, 10]).tick_values(low, top)
    digits = max(0, -int(np.floor(np.log10((ticks[1] - ticks[0]) / 10.0**p) + 1e-9)))
    inner = [t for t in ticks if low - 0.03 * span <= t < top - span / 10]
    ax.set_yticks([*inner, top], labels=[*(f"{t / 10.0**p:.{digits}f}" for t in inner), "$2/kd$"])
    ax.set_ylabel(rf"Utility gain ($\times 10^{{{p}}}$)")


def plot(theta: pd.DataFrame, table: pd.DataFrame, widths: list[int], epsilons: list[float],
         paths: tuple[str, str] | None = None,
         figsize: tuple[float, float] = (5.5, 3.4)) -> None:
    """Figure 4: the mean utility gain c in the table of utility_table, made with the given widths
    and epsilons, against the number of records n, on a log scale, with its confidence interval as
    a band, in two figures, for subfigures (a) and (b): (a) for the coarse releases, with the QID
    column in bins of each width w of widths; (b) for the randomized ones, at each level eps of
    epsilons; in both, the raw release,
    dashed. The levels are colored as in controls.plot, evenly by rank, from their weakest protection
    (the narrowest bins, the largest eps), dark, to their strongest, light, and listed in that order,
    the order of the curves from top to bottom. Dash-dotted: the utility before the release, the gain
    c(theta_0, theta) of the analyst's best guess before it, theta_0, the mean of her prior
    (Section 6). Both figures have the same y axis, which zooms in on the gains from that one, or the
    lowest one of the raw and coarse releases, to their maximum 2/kd (gain_axis): the randomized
    releases enter from below once they pass it. A legend above the axes. If paths is given, (a)
    and (b) are saved to paths[0] and paths[1]. figsize is in inches, for each figure: by default
    drawn 5.5 in wide, as the other figures, for one column of the paper (3.5 in), so that the text
    prints at 0.64 of its size, 7.6 pt for the ticks and the legends and 10 pt for the axis
    labels."""
    releases = table.index.unique("release")  # in the order of controls_table
    raw, coarse, noisy = releases[0], releases[1:1 + len(widths)], releases[1 + len(widths):]
    sizes = table.index.unique("n")
    kd = theta.size
    before = utility(np.full(theta.shape, 1 / kd), theta)  # c(theta_0, theta), theta_0 uniform
    shown = table.drop(noisy, level="release")["utility gain"]  # the raw and coarse releases
    low = min(before, shown.min())  # the bottom of the y axis of both figures
    # Set globally, not just while plotting: Jupyter draws the figures after the cell
    plt.rcParams.update(STYLE)
    # Each figure: its releases, their levels, color scale and label of a level (gamma_w for bins of
    # w, eps = level), and the direction of the levels in which they protect more (+1 for wider
    # bins, -1 for smaller eps)
    panels = [(coarse, widths, COARSE_SCALE.reversed(), r"$\LARGE\gamma_{{{}}}$", 1),
              (noisy, epsilons, NOISY_SCALE, r"$\varepsilon = {}$", -1)]
    for (names, values, scale, label, sign), path in zip(panels, paths or (None, None)):
        fig, ax = plt.subplots(figsize=figsize, layout="constrained")
        values = np.asarray(values, dtype=float)
        colors = scale(np.argsort(np.argsort(values)) / max(len(values) - 1, 1))
        curve(ax, table.xs(raw, level="release"), "Raw release", color=RAW, linestyle="--", zorder=4)
        # From the weakest protection to the strongest
        for j, i in enumerate(np.argsort(sign * values)):
            curve(ax, table.xs(names[i], level="release"), label.format(f"{values[i]:.4g}"),
                  color=colors[i], marker=MARKERS[j % len(MARKERS)],
                  markevery=(j % MARK_EVERY, MARK_EVERY))
        ax.axhline(before, color=BEFORE, linewidth=0.7, linestyle="-.", zorder=1)
        ax.annotate(r"Before the release, $c(\theta_0,\theta)$", (1, before),
                    xycoords=("axes fraction", "data"), xytext=(-2, 3), textcoords="offset points",
                    ha="right", va="bottom", fontsize=11, color="0.2")
        ax.set_xscale("log")
        ax.set_xlim(min(sizes), max(sizes))
        ax.xaxis.set_major_locator(LogLocator(numticks=10))  # every power of 10
        ax.grid(color="0.3", linewidth=0.4, alpha=0.13)
        ax.set_axisbelow(True)
        ax.set_xlabel("Number of records $n$")
        gain_axis(ax, low, kd)
        # Above the axes, clear of the curves, in columns filled from top to bottom
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1), ncol=3, frameon=False,
                  borderaxespad=0.3, columnspacing=1.2, handlelength=2.5)
        if path is not None:
            save(fig, path, dpi=300)
