"""Experiment 2 (Figure 2): the risk in particular releases.

The population theta (common.py) is the distribution of the records over QID columns and a
sensitive column with k values. For each release size n, RELEASES releases h of n records are
drawn from Mult_n(theta) (common.sample_releases). In a release h, the risk in a QID group q with
h(q, S) >= 1 is
    r_q(h) = max_s h(q, s) / h(q, S),
the fraction of its members whose sensitive value the membership-aware adversary infers, whatever
she knows about the population (Proposition 5.1). Over all the groups, the fraction of inferred
individuals is rbar(h) = (1/n) sum_q max_s h(q, s), and by Proposition 5.2(a) its expected value
is the risk for a target drawn from theta,
    sum_q theta(q, S) G_g[pi^{delta_theta,1} |> R]|_{q_t = q},
computed exactly by risk_bounds.risk. For each n, the risks r_q(h) are plotted as a box per group,
over the releases in which the group is not empty, with the base-rate risk
Phi(theta) = max_s theta(s | q) of the group, around which they concentrate as n grows
(Proposition 5.2(c)); next to them, rbar(h) is plotted as a box, with its expected value. Above
the panels, the share theta(q, S) of each group in the population is plotted as a bar.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgba
from matplotlib.ticker import MaxNLocator

from common import (COLORS, STYLE, base_rate_risk, group_label, group_share, sample_releases,
                    save)
from risk_bounds import group_ticks, risk

RELEASES = 1000  # releases h drawn from Mult_n(theta) for each release size n
# Okabe-Ito blue for the risks r_q(h) in the groups, bluish green for rbar(h) in all, vermillion
# for Phi(theta); gray for the shares theta(q, S) of the groups, which are context, not risks
BOX, ALL, BASE_RATE, SHARE = COLORS[0], COLORS[2], COLORS[1], "0.72"


def box_style(color: str) -> dict:
    """Style of the boxes of the risks, in the given color, for Axes.boxplot."""
    return dict(
        widths=0.5, patch_artist=True,
        boxprops=dict(facecolor=to_rgba(color, 0.18), edgecolor=color, linewidth=0.8),
        medianprops=dict(color=color, linewidth=1.4), whiskerprops=dict(color=color, linewidth=0.8),
        capprops=dict(color=color, linewidth=0.8),
        flierprops=dict(marker="o", markersize=2.5, markerfacecolor="none", markeredgecolor=color,
                        markeredgewidth=0.5, alpha=0.6))


def sorted_groups(theta: pd.DataFrame) -> list[tuple]:
    """The QID groups of theta, from the smallest to the largest."""
    return sorted(theta.index, key=lambda q: group_share(theta, q))


def release_risks(theta: pd.DataFrame, n: int) -> pd.DataFrame:
    """r_q(h) = max_s h(q, s) / h(q, S) in the RELEASES releases h of n records of
    common.sample_releases: a row per release and a column per QID group q, NaN where the group is
    empty, with the groups in the order of sorted_groups."""
    h = sample_releases(theta, n, RELEASES)
    size = h.sum(axis=2)
    risks = np.divide(h.max(axis=2), size, out=np.full(size.shape, np.nan), where=size > 0)
    return pd.DataFrame(risks, columns=theta.index)[sorted_groups(theta)]


def risk_table(theta: pd.DataFrame, n: int) -> pd.DataFrame:
    """The risk in each QID group q, in the releases of n records of release_risks, with the
    groups in the same order: the expected group size mu = n theta(q, S), the base-rate risk
    max_s theta(s | q), the share of the releases in which the group is not empty, and the
    minimum, quartiles and maximum of r_q(h) over these releases."""
    risks = release_risks(theta, n)
    groups = list(risks.columns)
    share = np.array([group_share(theta, q) for q in groups])
    low, q1, median, q3, high = risks.quantile([0, 0.25, 0.5, 0.75, 1]).to_numpy()
    return pd.DataFrame({
        "group": [group_label(theta, q) for q in groups],
        "share": share,
        "mu": n * share,
        "base-rate risk": [base_rate_risk(theta, q) for q in groups],
        "present": risks.notna().mean().to_numpy(),
        "min": low,
        "q1": q1,
        "median": median,
        "q3": q3,
        "max": high,
    })


def fractions(theta: pd.DataFrame, n: int) -> np.ndarray:
    """The fraction of inferred individuals rbar(h) = (1/n) sum_q max_s h(q, s) in each of the
    RELEASES releases h of n records of common.sample_releases."""
    return sample_releases(theta, n, RELEASES).max(axis=2).sum(axis=1) / n


def expected_fraction(theta: pd.DataFrame, n: int) -> float:
    """The expected value of rbar(h) by Proposition 5.2(a),
    sum_q theta(q, S) G_g[pi^{delta_theta,1} |> R]|_{q_t = q}, with G_g computed exactly by
    risk_bounds.risk."""
    return sum(group_share(theta, q) * risk(theta, q, n) for q in theta.index)


def fraction_table(theta: pd.DataFrame, sizes: list[int]) -> pd.DataFrame:
    """Proposition 5.2(a) for each release size n of sizes: the mean of the fraction of inferred
    individuals rbar(h) over the releases of fractions, with its standard error, and its expected
    value sum_q theta(q, S) G_g[pi^{delta_theta,1} |> R]|_{q_t = q}, computed exactly; and the
    base-rate risk averaged over the groups in the same way, sum_q theta(q, S) max_s theta(s | q),
    which is the limit of the expected value as n grows (Corollary 5.2)."""
    share = np.array([group_share(theta, q) for q in theta.index])
    rows = []
    for n in sizes:
        fraction = fractions(theta, n)
        rows.append({
            "base-rate risk": share @ [base_rate_risk(theta, q) for q in theta.index],
            "expected": expected_fraction(theta, n),
            "sampled": fraction.mean(),
            "standard error": fraction.std(ddof=1) / np.sqrt(len(fraction)),
        })
    return pd.DataFrame(rows, index=pd.Index(sizes, name="n"))


def plot_shares(ax, theta: pd.DataFrame) -> None:
    """The share theta(q, S) of each QID group q in the population: a bar per group, in the order
    of sorted_groups and as wide as the boxes of plot_risks, to go above its panels, with its
    percentage on top. The y-axis is linear, so that the shares can be compared directly.
    theta(q, S) is written in the upper left corner, which the bars leave empty, on a white box as
    the n of plot_risks."""
    shares = np.array([group_share(theta, q) for q in sorted_groups(theta)])
    bars = ax.bar(np.arange(len(shares)), shares, width=0.5, color=SHARE, linewidth=0)
    # Two significant digits: 40%, 1.3%, 0.031%
    ax.bar_label(bars, [rf"{100 * s:.2g}\%" for s in shares], padding=2, fontsize=10, color="0.35")
    ax.set_ylim(0, 1.3 * shares.max())  # room for the percentages above the bars
    ax.yaxis.set_major_locator(MaxNLocator(3))
    ax.yaxis.set_major_formatter(lambda y, _: rf"{100 * y:g}\%")
    ax.set_ylabel("Share")
    ax.grid(axis="y", color="0.3", linewidth=0.4, alpha=0.13)
    ax.set_axisbelow(True)
    ax.text(0.005, 0.95, r"$\theta(q, S)$", transform=ax.transAxes, ha="left", va="top",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=1.5))


def plot_risks(ax, theta: pd.DataFrame, n: int) -> None:
    """The risks r_q(h) in the releases of n records of release_risks: for each QID group q, a box
    of r_q(h) over the releases in which the group is not empty, with the percentage of these
    releases below it when it is not 100, and the base-rate risk Phi(theta) of the group as a
    horizontal line; on the right, the risk in all the groups together (plot_fraction). The y-axis
    spans [1/k, 1], where every risk lies, with room below for the percentages; the groups are
    named on the x-axis by risk_bounds.group_ticks, and n is written in the upper right corner."""
    table = risk_table(theta, n)
    risks = release_risks(theta, n).to_numpy()
    x = np.arange(len(table))
    boxes = ax.boxplot([r[~np.isnan(r)] for r in risks.T], positions=x, **box_style(BOX))
    boxes["boxes"][0].set_label(r"Risk $r_q(h)$ in group $q$")
    ax.hlines(table["base-rate risk"], x - 0.4, x + 0.4, color=BASE_RATE, linewidth=1.6, zorder=3,
              label=r"Base-rate risk $\Phi(\theta)$")
    k = theta.shape[1]
    span = 1 - 1 / k
    ax.set_ylim(1 / k - 0.14 * span, 1 + 0.02 * span)  # no risk lies below 1/k: room for the text
    ax.yaxis.set_major_locator(MaxNLocator(6))  # e.g. 0.5, 0.6, ..., 1 for k = 2
    present = table["present"].to_numpy()
    absent = np.flatnonzero(present < 1)
    for i in absent:
        ax.text(i, 1 / k - 0.075 * span, rf"{100 * present[i]:.1f}\%", ha="center", va="center",
                fontsize=10, color="0.35")
    if len(absent):  # a legend entry with the glyph % as its marker
        ax.plot([], [], linestyle="none", marker=r"$\%$", markersize=9, color="0.35",
                label=r"Releases in which group $q$ appears, if not all")
    ax.grid(axis="y", color="0.3", linewidth=0.4, alpha=0.13)
    ax.set_axisbelow(True)
    group_ticks(ax, table["group"])  # no x-axis label: the tick labels name the groups
    plot_fraction(ax, theta, n, len(table))  # after group_ticks, which would remove its tick
    ax.set_ylabel(r"Risk $r_q(h)$")
    power = round(np.log10(n))
    # Inside the axes, in the upper right corner, which the large groups leave empty
    ax.text(0.99, 0.96, rf"$n = 10^{{{power}}}$" if 10 ** power == n else rf"$n = {n}$",
            transform=ax.transAxes, ha="right", va="top",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=1.5))


def plot_fraction(ax, theta: pd.DataFrame, n: int, x: float) -> None:
    """The risk in all the groups together, at x on the x-axis of plot_risks: a box of the fraction
    of inferred individuals rbar(h) in the releases of n records of fractions, and its expected
    value of Proposition 5.2(a), expected_fraction, as a diamond. Its tick is added to those
    already on the axis, and its label slanted."""
    fraction = fractions(theta, n)
    boxes = ax.boxplot([fraction], positions=[x], manage_ticks=False, **box_style(ALL))
    boxes["boxes"][0].set_label(r"Risk $\bar r(h)$ (average of all groups)")
    ax.plot(x, expected_fraction(theta, n), "D", color="0.1", markerfacecolor="white",
            markersize=6, zorder=3, label=r"Expected value $\mathbb{E}[\bar r(h)]$, Prop. 5.2(a)")
    ticks = list(ax.get_xticks())
    labels = [ax.xaxis.get_major_formatter()(t, i) for i, t in enumerate(ticks)]
    ax.set_xticks(ticks + [x], labels + [r"All groups, $\bar r(h)$"])  # the others keep their style
    ax.xaxis.get_major_ticks()[-1].label1.set(rotation=45, ha="right", rotation_mode="anchor")
    ax.set_xlim(-0.5, x + 0.5)


def plot(theta: pd.DataFrame, sizes: list[int], path: str | None = None,
         figsize: tuple[float, float] | None = None) -> None:
    """Figure 2: the shares of the QID groups of theta in the population (plot_shares), and below
    them a panel of plot_risks for each release size n of sizes, one above the other, saved to path
    if given. figsize is in inches, by default 11.5 wide and 2.5 + 1.9 per panel of plot_risks
    tall; the shares take 0.6 of the height of such a panel."""
    # Set globally, not just while plotting: Jupyter draws the figure after the cell
    plt.rcParams.update(STYLE)
    fig, axes = plt.subplots(len(sizes) + 1, 1, figsize=figsize or (11.5, 2.5 + 1.9 * len(sizes)),
                             sharex=True, height_ratios=[0.6] + [1] * len(sizes),
                             layout="constrained", squeeze=False)
    plot_shares(axes[0, 0], theta)
    for ax, n in zip(axes[1:, 0], sizes):
        plot_risks(ax, theta, n)
    for ax in axes[:, 0]:
        ax.label_outer()  # the groups are named below the last panel only
    fig.align_ylabels(axes[:, 0])  # Share and Risk, whatever the width of their tick labels
    # The entries of all the panels, each once (a panel may have no percentages), above them in
    # two columns: the groups in the first, all of them (rbar(h)) in the second
    legend = {label: h for ax in axes[1:, 0] for h, label in zip(*ax.get_legend_handles_labels())}
    legend = dict(sorted(legend.items(), key=lambda entry: r"\bar r" in entry[0]))
    fig.legend(list(legend.values()), list(legend.keys()), loc="outside upper center", ncol=2,
               borderaxespad=0)  # at the top edge: the room the layout leaves is below it
    if path is not None:
        save(fig, path, dpi=300)
