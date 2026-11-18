"""Experiment 5 (Figure 5): the statistical and the differential leakage by group size.

The population theta (common.py) is the distribution of the records of data/pums2010_acs1.csv over
QID columns and a sensitive column with k values. The adversary has the benchmark prior uni and wants
to infer the sensitive value of a target with QID q. Before any data she guesses at random. After a
release, her optimal guess is a most frequent value of group q in it, at random among ties, whether
the target is in the release (Proposition 5.1) or not (theta(. | q) given h is then
Dir(1 + h(q, .)), Section 3.2). Her successes are evaluated in the true population theta, for each
group q:
    prior risk  P(q) = 1/k;
    out-risk    O(q): the release h ~ Mult_n(theta) does not contain the target, and the adversary
                succeeds with probability theta(guess | q), that the target, drawn from her group,
                has the value guessed;
    in-risk     I(q): the release is h' + e_(q, s), with h' ~ Mult_{n-1}(theta) and the target's
                value s ~ theta(. | q), and I(q) = sum_s theta(s | q) Pr[s wins]. Whatever the
                adversary's prior, it is the risk G_g[pi^{delta_theta,1} |> R] of Experiment 1
                (Proposition B.3(b)), which risk_bounds.risk computes exactly; here it is
                simulated like the out-risk.
Their ratios are multiplicative leakages (Definition 3.3): the statistical leakage O/P, what the
release teaches about the population; the differential leakage I/O, what the target's own record
adds; and the inferential leakage I/P, their product. On a log scale the two parts add up, and the
share of the inferential leakage that is differential, log(I/O) / log(I/P), lies in [0, 1], as
P <= O <= Phi_q(theta) <= I, with Phi_q(theta) = max_s theta(s | q) the base-rate risk (up to the
noise of the simulation).

The size of group q in a release is nearly Poisson(m), m = n theta(q, S): two groups with the same
theta(. | q) have the same risks at the same expected group size m, whatever the number of records
n. plot (Figure 5(a)) shows this on pairs of groups with the same base-rate risk and sizes far
apart (pairs): the share of each group along a grid of m (pair_table), against n (top), where the
two groups of a pair lie far apart, and against m (bottom), where they fall on one curve, from
about 1 in small groups (the leakage is differential) to about 0 in large ones (statistical).
plot_groups (Figure 5(b)) shows the share of all the groups for a few n (leakage_table).
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator

from common import COLORS, STYLE, base_rate_risk, group_label, group_share, save

RELEASES = 1000  # releases h and h' of all the groups for each number of records n (leakage_table)
CURVE_RELEASES = 100_000  # releases of a single group at each point of its curve (pair_table)
SEED = 0  # of the releases: the tables and the plots of the same population show the same releases
M_GRID = 10 ** (np.arange(-12, 19) / 6)  # expected group sizes m of the curves: 10^-2 to 10^3
# A pair of groups: the larger has a base-rate risk within LEVEL_TOLERANCE of the level asked for,
# and the smaller one within PAIR_TOLERANCE of the larger one's
LEVEL_TOLERANCE, PAIR_TOLERANCE = 0.01, 0.005
MARKERS = ["o", "s", "^", "D", "v"]  # one per number of records n in plot_groups, in their order
MARKER_SIZE, POINT_AREA = 6, 14  # of the markers of n in the legend (points) and of the groups (points^2)
# The curves of plot: the larger group of a pair as a wide translucent band, and the smaller one as a
# thin dashed line, so that both show where they fall on each other
CURVE_STYLES = [dict(linewidth=4, alpha=0.5, solid_capstyle="butt"), dict(linewidth=1.3, linestyle="--")]
# The base-rate risk, from 1/k to 1: one hue, vermillion, light to dark, with its lightest step still
# visible on white
BASE_RATE_SCALE = LinearSegmentedColormap.from_list(
    "base_rate", ["#EDA27A", "#DD7C45", "#D55E00", "#A34500", "#6A2C00"])


def tie_average(scores: np.ndarray, values: np.ndarray) -> np.ndarray:
    """The mean of values[..., q, s] over the values s with the largest scores[..., q, s]: the
    expected gain of a guess at random among the most frequent values of group q. In an empty group
    all the k values tie."""
    top = scores == scores.max(axis=-1, keepdims=True)
    return (top * values).sum(axis=-1) / top.sum(axis=-1)


def successes(theta: pd.DataFrame, n: int, groups: list[tuple] | None = None,
              releases: int = RELEASES, seed: int | tuple[int, ...] = SEED
              ) -> tuple[np.ndarray, np.ndarray]:
    """The adversary's success in each of `releases` releases of n records, as arrays [release, q],
    for the QID groups q of groups (by default all those of theta), in their order:
        out     with h ~ Mult_n(theta): theta(guess | q), for a guess at random among the most
                frequent values of group q in h;
        inside  with h' ~ Mult_{n-1}(theta), drawn separately: sum_s theta(s | q) Pr[s wins], where s
                wins if it is a most frequent value of group q once the target's record (q, s) is
                added to h', with probability 1 over the number of such values.
    The cells of the other groups are drawn as a single one, which leaves the counts of these
    groups as in Mult_n(theta)."""
    cells = theta.to_numpy() if groups is None else theta.loc[groups].to_numpy()
    k = cells.shape[1]
    cond = cells / cells.sum(axis=1, keepdims=True)  # theta(s | q)
    pvals = np.append(cells.ravel(), max(0.0, 1 - cells.sum()))  # the other groups last
    rng = np.random.default_rng(seed)
    shape = (releases, *cells.shape)
    h = rng.multinomial(n, pvals, releases)[:, :-1].reshape(shape)
    others = rng.multinomial(n - 1, pvals, releases)[:, :-1].reshape(shape)
    unit = np.eye(k, dtype=int)  # unit[s]: the target's record, in cell (q, s) of every group q
    inside = sum(cond[:, s] * tie_average(others + unit[s], unit[s]) for s in range(k))
    return tie_average(h, cond), inside


def leakages(prior: float, out: np.ndarray, inside: np.ndarray) -> dict:
    """From the prior risk P, the out-risk O and the in-risk I: the statistical leakage O/P, the
    differential leakage I/O, the inferential leakage I/P, their product, and the share of the
    inferential leakage that is differential, log(I/O) / log(I/P)."""
    return {
        "statistical leakage": out / prior,
        "differential leakage": inside / out,
        "inferential leakage": inside / prior,
        "differential share": np.log(inside / out) / np.log(inside / prior),
    }


def leakage_table(theta: pd.DataFrame, sizes: list[int]) -> pd.DataFrame:
    """A row per number of records n of sizes and QID group q, indexed by n and the QID columns:
    the expected group size m = n theta(q, S); the prior risk P = 1/k, the out-risk O and the
    in-risk I, the means of successes over RELEASES releases, and the base-rate risk
    Phi_q(theta) = max_s theta(s | q); and the leakages and the share of leakages."""
    cells = theta.to_numpy()
    prior = 1 / cells.shape[1]
    base_rate = [base_rate_risk(theta, q) for q in theta.index]
    tables = {}
    for n in sizes:
        out, inside = (x.mean(axis=0) for x in successes(theta, n))
        tables[n] = pd.DataFrame({
            "m": n * cells.sum(axis=1),
            "prior risk": prior,
            "out-risk": out,
            "in-risk": inside,
            "base-rate risk": base_rate,
            **leakages(prior, out, inside),
        }, index=theta.index)
    return pd.concat(tables, names=["n"])


def pairs(theta: pd.DataFrame, choices: list[tuple[float, dict[str, str]]]) -> list[tuple]:
    """For each (level, values) of choices, a pair (larger, smaller) of QID groups with about that
    base-rate risk: the larger is the largest group whose Phi_q(theta) is within LEVEL_TOLERANCE of
    the level; the smaller is the group with the QID values `values`, e.g. {"RAC1P": "6"} for an
    Asian group, whose Phi_q(theta) is the closest to the larger one's, which must be within
    PAIR_TOLERANCE of it."""
    share = theta.sum(axis=1)
    base_rate = theta.max(axis=1) / share
    result = []
    for level, values in choices:
        near = share[(base_rate - level).abs() <= LEVEL_TOLERANCE]
        if near.empty:
            raise ValueError(f"no group has a base-rate risk within {LEVEL_TOLERANCE} of {level}")
        larger = near.idxmax()
        chosen = np.all([theta.index.get_level_values(col) == v for col, v in values.items()], axis=0)
        distance = (base_rate[chosen] - base_rate.loc[larger]).abs().drop(index=[larger], errors="ignore")
        if distance.empty or distance.min() > PAIR_TOLERANCE:
            raise ValueError(f"no group with {values} has a base-rate risk within {PAIR_TOLERANCE} "
                             f"of that of {group_label(theta, larger)}")
        result.append((larger, distance.idxmin()))
    return result


def pair_table(theta: pd.DataFrame, pairs: list[tuple], grid: np.ndarray = M_GRID) -> pd.DataFrame:
    """The columns of leakage_table for the QID groups of pairs, along the expected group sizes m of
    grid: for each group q, a row per number of records n = max(1, round(m / theta(q, S))), each n
    once, indexed by the QID columns and n, with the label of the group and its number of records in
    theta, and m = n theta(q, S). The risks are means over CURVE_RELEASES releases of the group at
    each n, with a seed of its own: its curve is smooth across n, and the groups of a pair are drawn
    independently."""
    prior = 1 / theta.shape[1]
    tables = {}
    for i, q in enumerate(q for pair in pairs for q in pair):
        share = group_share(theta, q)
        sizes = np.unique(np.maximum(1, np.round(np.asarray(grid) / share))).astype(int)
        out, inside = np.array([[x.mean() for x in successes(theta, n, [q], CURVE_RELEASES, (SEED, i))]
                                for n in sizes]).T
        tables[q] = pd.DataFrame({
            "group": group_label(theta, q),
            "records": round(share * theta.attrs["records"]),
            "m": sizes * share,
            "prior risk": prior,
            "out-risk": out,
            "in-risk": inside,
            "base-rate risk": base_rate_risk(theta, q),
            **leakages(prior, out, inside),
        }, index=pd.Index(sizes, name="n"))
    return pd.concat(tables, names=theta.index.names)


def size_label(n: int) -> str:
    """n = 10^p for a power of 10, else n in full, in LaTeX."""
    power = round(np.log10(n))
    return rf"$n = 10^{{{power}}}$" if 10 ** power == n else rf"$n = {n}$"


def marker_handle(marker: str, color: str, label: str = "") -> Line2D:
    """A legend entry with a marker alone, as the groups of plot_groups."""
    return Line2D([], [], linestyle="none", marker=marker, markersize=MARKER_SIZE, color=color,
                  markeredgecolor="white", markeredgewidth=0.5, label=label)


def share_axis(ax, xlabel: str) -> None:
    """x on a log scale, labelled at every power of 10, and the share from 0 to 1 on y, with faint
    grid lines."""
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(LogLocator(numticks=12))
    ax.set_xlabel(xlabel)
    ax.set_ylim(-0.04, 1.04)
    ax.grid(color="0.3", linewidth=0.4, alpha=0.13)
    ax.set_axisbelow(True)


def plot(theta: pd.DataFrame, pairs: list[tuple], path: str | None = None,
         figsize: tuple[float, float] = (5.5, 5.4)) -> None:
    """Figure 5(a): the share of the inferential leakage that is differential, log(I/O) / log(I/P),
    of the QID groups of pairs (pair_table), against the number of records n (top) and against the
    expected group size m = n theta(q, S) (bottom), both on a log scale. A color per pair, in the
    order of common.COLORS (Okabe-Ito blue, vermillion, green, ...); the larger group a wide band,
    the smaller one dashed (CURVE_STYLES). Above the panels, a legend as a table, a row per pair:
    its base-rate risk, then its larger and its smaller group, named by common.group_label, with
    their numbers of records. Saved to path if given; figsize is in inches: by default drawn 5.5 in
    wide, as the other figures, for one column of the paper."""
    table = pair_table(theta, pairs)
    curves = dict(iter(table.groupby("group", sort=False)))  # the rows of each group, by its label
    # Set globally, not just while plotting: Jupyter draws the figure after the cell
    plt.rcParams.update(STYLE)
    fig, (by_n, by_m) = plt.subplots(2, 1, figsize=figsize, layout="constrained")
    columns = [[], [], []]  # the legend: base-rate risks, larger groups, smaller groups
    for pair, color in zip(pairs, COLORS):
        base_rate = curves[group_label(theta, pair[0])]["base-rate risk"].iloc[0]
        columns[0].append((Line2D([], [], linestyle="none"), rf"$\Phi_q(\theta) \approx {base_rate:.2f}$"))
        for q, style, column in zip(pair, CURVE_STYLES, columns[1:]):
            rows = curves[group_label(theta, q)]
            by_n.plot(rows.index.get_level_values("n"), rows["differential share"], color=color,
                      **style)
            (line,) = by_m.plot(rows["m"], rows["differential share"], color=color, **style)
            column.append((line, f"{group_label(theta, q)} ({rows['records'].iloc[0]:,})"))
    share_axis(by_n, "Number of records $n$")
    share_axis(by_m, r"Expected group size $m = n\,\theta(q,S)$")
    # In words: the paper names no I, O or P
    fig.supylabel("Share of differential leakage", fontsize=plt.rcParams["axes.labelsize"])
    # Filled column by column: the base-rate risks, then the larger groups, then the smaller ones
    handles, labels = zip(*(entry for column in columns for entry in column))
    fig.legend(handles, labels, loc="outside upper center", ncol=3, frameon=False, borderaxespad=0,
               handlelength=1.8, handletextpad=0.5, columnspacing=0.8)
    if path is not None:
        save(fig, path, dpi=300)


def plot_groups(theta: pd.DataFrame, sizes: list[int], path: str | None = None,
                figsize: tuple[float, float] = (5.5, 3.4)) -> None:
    """Figure 5(b): the share of the inferential leakage that is differential, log(I/O) / log(I/P),
    of every QID group for each number of records n of sizes (leakage_table), against the expected
    group size m = n theta(q, S) on a log scale, with a marker per n, colored by the base-rate risk
    of the group (BASE_RATE_SCALE, from 1/k to 1). Its color bar is in the lower left corner, which
    no group reaches (small groups have a share close to 1), and the markers of n are in a legend
    above. Saved to path if given; figsize in inches."""
    table = leakage_table(theta, sizes)
    k = theta.shape[1]
    norm = Normalize(1 / k, 1)
    # Set globally, not just while plotting: Jupyter draws the figure after the cell
    plt.rcParams.update(STYLE)
    fig, ax = plt.subplots(figsize=figsize, layout="constrained")
    for n, marker in zip(sizes, MARKERS):
        groups = table.xs(n, level="n")
        points = ax.scatter(groups["m"], groups["differential share"], s=POINT_AREA,
                            c=groups["base-rate risk"], cmap=BASE_RATE_SCALE, norm=norm,
                            marker=marker, edgecolors="white", linewidths=0.3)
    share_axis(ax, r"Expected group size $m = n\,\theta(q,S)$")
    ax.set_ylabel("Share of\ndifferential leakage")
    ticks = [1 / k, (1 + 1 / k) / 2, 1]
    bar = fig.colorbar(points, cax=ax.inset_axes([0.05, 0.2, 0.3, 0.05]), orientation="horizontal")
    bar.set_ticks(ticks, labels=[f"{t:g}" for t in ticks])
    bar.ax.set_title(r"Base-rate risk $\Phi_q(\theta)$", fontsize=plt.rcParams["legend.fontsize"],
                     pad=4)
    bar.outline.set_linewidth(0.5)
    fig.legend(handles=[marker_handle(marker, "0.35", size_label(n))
                        for n, marker in zip(sizes, MARKERS)],
               loc="outside upper center", ncol=len(sizes), frameon=False, borderaxespad=0)
    if path is not None:
        save(fig, path, dpi=300)
