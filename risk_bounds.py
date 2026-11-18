"""Experiment 1 (Figure 1): the differential leakage across groups.

The adversary knows the population theta (common.py), the target's QID q_t, and that the target is
in a release of n records drawn from theta. Her prior risk is then the base-rate risk
Phi(theta) = max_s theta(s | q_t) (Theorem 5.1(a)), and the gap between her risk
G_g[pi^{delta_theta,1} |> R] and Phi(theta) is entirely differential leakage. For a target in each
QID group, risk computes her risk exactly, and bounds_table puts it next to its bounds of
Theorem 5.1(b),
    lower  Phi(theta) + (1 - Phi(theta)) (1 - theta(q_t, S))^(n - 1),
    upper  Phi(theta) + sqrt((1 - sum_s theta(s | q_t)^2) / (n theta(q_t, S))),
which plot draws with Phi(theta) (Figure 1(a)). plot_distribution draws the distribution of the
ages and the prevalence of disability by age (Figure 1(b)).
"""

from functools import reduce

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MultipleLocator
from scipy.signal import convolve
from scipy.stats import binom

from common import COLORS, STYLE, base_rate_risk, conditional, group_label, group_share, save

NEGLIGIBLE = 1e-20  # values of a binomial count with a smaller probability are left out
MAX_TICK_LABELS = 60  # at most this number of groups are named on the x-axis
# Figure 1 has slightly thicker axes and ticks than the other figures
FIGURE_1_STYLE = {**STYLE, "axes.linewidth": 0.8, "xtick.major.width": 0.8,
                  "ytick.major.width": 0.8, "xtick.major.size": 4, "ytick.major.size": 4}


def likely_counts(trials: int, p: float) -> tuple[int, int]:
    """First and last value c of a binomial count whose probability, C(trials, c) p^c
    (1 - p)^(trials - c), exceeds NEGLIGIBLE. The other values, at most trials + 1 of them, have
    total probability below (trials + 1) NEGLIGIBLE."""
    c = np.flatnonzero(binom.pmf(np.arange(trials + 1), trials, p) > NEGLIGIBLE)
    return int(c[0]), int(c[-1])


def scaled_terms(lo: int, hi: int, mu: float) -> np.ndarray:
    """mu^c / c! for c = lo, ..., hi, scaled to sum to 1, so that products of many such
    polynomials stay within floating-point range. They are built from the ratios mu / c, as mu^c
    and c! overflow."""
    log_terms = np.concatenate(([0.0], np.cumsum(np.log(mu / np.arange(lo + 1, hi + 1)))))
    terms = np.exp(log_terms - log_terms.max())
    return terms / terms.sum()


def poly_product(polys: list[np.ndarray], start: np.ndarray, length: int) -> np.ndarray:
    """Coefficients of x^0, ..., x^(length - 1) in the polynomial start * prod(polys)."""
    product = reduce(lambda u, v: convolve(u, v)[:length], polys, start[:length])
    return np.pad(product, (0, length - len(product)))


def expected_max_shares(m_lo: int, m_hi: int, cond: np.ndarray) -> np.ndarray:
    """f(m) = E[max_s C_s] / m for C drawn from Mult_m(cond) and m = m_lo, ..., m_hi: the
    probability that the adversary guesses the value of a target whose group has m members in the
    release, when she guesses its most frequent value (Proposition 5.1).

    - For a count X, E[X] = sum_b b Pr[X = b] = sum_{b >= 0} Pr[X > b].
    - By the formula of Mult_m (Section 4), Pr[max_s C_s <= b] is the sum of
      m! prod_s cond_s^{c_s} / c_s! over the counts c with sum m and every c_s <= b, i.e. m! times
      the coefficient of x^m in prod_s sum_{c <= b} (cond_s x)^c / c!. Dividing it by the same
      coefficient without the bound b (where the probabilities sum to 1) cancels m!, and lets the
      terms be scaled: writing (mu cond_s)^c, for any mu > 0, only adds a factor mu^m.
    - C_s is binomial with parameters m and cond_s. Its values that likely_counts leaves out both
      for m_lo and for m_hi are left out: as m grows, the probability of a value below the mean
      decreases and above it increases, so it stays below NEGLIGIBLE for every m in between.
    """
    m = np.arange(m_lo, m_hi + 1)
    lo = np.array([likely_counts(m_lo, p)[0] for p in cond])
    hi = np.array([likely_counts(m_hi, p)[1] for p in cond])
    terms = [scaled_terms(a, z, (m_lo + m_hi) / 2 * p) for a, z, p in zip(lo, hi, cond)]
    length = m_hi - lo.sum() + 1  # the polynomials are stored from x^lo.sum() up to x^m_hi
    i = m - lo.sum()  # index of the coefficient of x^m
    done = hi <= lo.max()  # counts that the bound b below never cuts: multiplied once
    base = poly_product([a for a, d in zip(terms, done) if d], np.ones(1), length)
    cut = [(a, l) for a, l, d in zip(terms, lo, done) if not d]
    full = poly_product([a for a, _ in cut], base, length)[i]
    expected = np.full(len(m), float(lo.max()))  # Pr[max_s C_s <= b] is 0 for b < max_s lo_s
    for b in range(lo.max(), hi.max()):
        bounded = poly_product([a[: b - l + 1] for a, l in cut], base, length)[i]
        ratio = np.divide(bounded, full, out=np.ones(len(m)), where=full > 0)
        expected += 1 - np.clip(ratio, 0, 1)  # the clip removes rounding outside [0, 1]
    return expected / m


def risk(theta: pd.DataFrame, q: tuple, n: int) -> float:
    """G_g[pi^{delta_theta,1} |> R], computed exactly: the risk for a target in QID group q in a
    release of n records. The target's group has m members, with m - 1 binomial with parameters
    n - 1 and theta(q, S), and the adversary then succeeds with probability f(m) of
    expected_max_shares."""
    cond = conditional(theta, q)
    cond = cond[cond > 0]  # values absent from the group have count 0 in every release
    if len(cond) == 1:  # a single sensitive value in the group: the adversary always guesses it
        return 1.0
    t = group_share(theta, q)
    first, last = likely_counts(n - 1, t)  # likely numbers m - 1 of other members
    m = np.arange(first + 1, last + 2)
    weights = binom.pmf(m - 1, n - 1, t)
    # The weights sum to 1 up to n NEGLIGIBLE; dividing by their sum cancels their rounding errors.
    return float(weights @ expected_max_shares(m[0], m[-1], cond) / weights.sum())


def lower_bound(theta: pd.DataFrame, q: tuple, n: int) -> float:
    """Phi(theta) + (1 - Phi(theta)) (1 - theta(q, S))^(n - 1): the lower bound of Theorem 5.1(b)
    on the risk for a target in QID group q."""
    prior = base_rate_risk(theta, q)
    return prior + (1 - prior) * (1 - group_share(theta, q)) ** (n - 1)


def upper_bound(theta: pd.DataFrame, q: tuple, n: int) -> float:
    """Phi(theta) + sqrt((1 - sum_s theta(s | q)^2) / (n theta(q, S))): the upper bound of
    Theorem 5.1(b) on the risk for a target in QID group q, which can exceed 1."""
    return base_rate_risk(theta, q) + float(
        np.sqrt((1 - np.sum(conditional(theta, q) ** 2)) / (n * group_share(theta, q))))


def qid_order(q: tuple) -> tuple:
    """Sort key of QID group q: column by column, numeric codes as numbers (so age 9 comes before
    age 10)."""
    return tuple((0, int(v), "") if v.isdigit() else (1, 0, v) for v in q)


def bounds_table(theta: pd.DataFrame, n: int) -> pd.DataFrame:
    """For a target in each QID group q, in the order of the QID codes (qid_order), and a release of
    n records: the group's share theta(q, S), its base-rate risk Phi(theta), and the risk with its
    lower and upper bounds of Theorem 5.1(b)."""
    groups = sorted(theta.index, key=qid_order)
    return pd.DataFrame({
        "group": [group_label(theta, q) for q in groups],
        "share": [group_share(theta, q) for q in groups],
        "base-rate risk": [base_rate_risk(theta, q) for q in groups],
        "lower bound": [lower_bound(theta, q, n) for q in groups],
        "risk": [risk(theta, q, n) for q in groups],
        "upper bound": [upper_bound(theta, q, n) for q in groups],
    })


def group_ticks(ax, names: list[str], max_tick_labels: int = MAX_TICK_LABELS) -> None:
    """Name the groups at x = 0, 1, ... on the x-axis by names: at most max_tick_labels of them,
    one every 1, 2, 5, 10, ... groups, slanted if one is longer than 3 characters."""
    names = list(names)
    step = next((s for s in (1, 2, 5, 10, 20, 50, 100) if len(names) / s <= max_tick_labels), 200)
    ticks = list(range(0, len(names), step))
    rotate = max((len(names[i]) for i in ticks), default=0) > 3
    ax.set_xticks(ticks, [names[i] for i in ticks], rotation=45 if rotate else 0,
                  ha="right" if rotate else "center", rotation_mode="anchor")


def plot(table: pd.DataFrame, path: str, xlabel: str, figsize: tuple[float, float] = (12.4, 5.0),
         max_tick_labels: int = MAX_TICK_LABELS, markersize: float = 6,
         legend_cols: int = 2) -> None:
    """Figure 1(a): for each QID group of bounds_table, the base-rate risk Phi(theta) as a short
    horizontal line, and the risk between its lower and upper bounds, capped at 1, which no risk
    exceeds. The groups are named on the x-axis by group_ticks; figsize is in inches; a narrow
    figure needs fewer tick labels (max_tick_labels) and smaller markers (in points), and the
    legend (legend_cols columns) may fit only in one column."""
    x = np.arange(len(table))
    base_rate = table["base-rate risk"]
    lower, upper = table["lower bound"], np.minimum(table["upper bound"], 1)
    # Okabe-Ito orange for the upper bound, blue for the risk, bluish green for the lower bound and
    # vermillion for the base-rate risk
    upper_color, risk_color, lower_color, base_rate_color = (COLORS[3], COLORS[0], COLORS[2],
                                                             COLORS[1])
    # White-centred markers, edged in the color of their series
    markers = dict(linestyle="none", markersize=markersize, markerfacecolor="white",
                   markeredgewidth=0.9)
    # Set globally, not just while plotting: Jupyter draws the figure again after the cell, and
    # LaTeX then still needs the preamble of the style
    plt.rcParams.update(FIGURE_1_STYLE)
    fig, ax = plt.subplots(figsize=figsize)
    ax.vlines(x, lower, upper, color="0.3", linewidth=0.8, alpha=0.3, zorder=0)
    baseline = ax.hlines(base_rate, x - 0.4, x + 0.4, color=base_rate_color,
                         linewidth=1.0, alpha=0.85, zorder=1)
    upper_line, = ax.plot(x, upper, marker="v", color=upper_color, **markers)
    lower_line, = ax.plot(x, lower, marker="^", color=lower_color, **markers)
    # Drawn last, so that it stays visible where it meets its bounds
    risk_line, = ax.plot(x, table["risk"], marker="o", color=risk_color, **markers)
    group_ticks(ax, table["group"], max_tick_labels)
    ax.set_xlabel(xlabel, labelpad=7)
    ax.set_ylabel("Risk", labelpad=7)
    ax.grid(axis="y", color="0.3", linewidth=0.4, alpha=0.13)  # faint horizontal lines only
    ax.set_axisbelow(True)
    ax.set_xlim(-1.5, len(table) + 0.5)  # a tiny space before the first group and after the last
    # Inside the axes, in the lower left corner. Its columns are filled one after the other: with
    # two, the bounds, then the risks.
    legend = ax.legend(
        [upper_line, lower_line, risk_line, baseline],
        ["Upper bound (at most 1)", "Lower bound",
         r"$G_g[\pi^{\delta_\theta,1} \vartriangleright \mathsf{R}]$", r"Base-rate risk $\Phi(\theta)$"],
        loc="lower left", ncol=legend_cols, frameon=True, fancybox=False,
        borderaxespad=0.6, borderpad=0.6, handlelength=1.8, handletextpad=0.5,
        columnspacing=1.0)
    fig.tight_layout()
    legend.get_frame().set_linewidth(0.6)
    legend.get_frame().set_alpha(0.7)
    save(fig, path)


def plot_distribution(data_path: str, path: str) -> None:
    """Figure 1(b), side by side: the distribution of the ages, theta(age), and the prevalence of
    disability (DIS = 1) at each age, theta(disability | age), from the records of data_path, in
    the style of plot."""
    data = pd.read_csv(data_path, usecols=["AGEP", "DIS"]).dropna()
    ages = data["AGEP"].value_counts().sort_index()  # number of people of each age
    disabled = data.loc[data["DIS"] == 1, "AGEP"].value_counts().reindex(ages.index, fill_value=0)
    age_color, disability_color = COLORS[0], COLORS[3]  # Okabe-Ito blue and orange
    plt.rcParams.update(FIGURE_1_STYLE)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5.5, 2), sharex=True, layout="constrained")
    ax1.bar(ages.index, 100 * ages / len(data), width=0.8, color=age_color, alpha=0.85,
            linewidth=0)
    ax1.set_ylabel("Population (\\%)", labelpad=7)
    ax2.bar(disabled.index, 100 * disabled / ages, width=0.8, color=disability_color, alpha=0.85,
            linewidth=0)
    ax2.set_ylabel("Disabled (\\%)", labelpad=7)
    # The titles go above the panels: inside, the narrow panels have no empty corner for them
    for ax, title in [(ax1, r"$\theta(\mathit{age})$"),
                      (ax2, r"$\theta(\mathit{disability} \mid \mathit{age})$")]:
        ax.set_title(title, fontsize=14, pad=4)
        ax.set_xlim(ages.index[0] - 1.5, ages.index[-1] + 1.5)  # a tiny space at the ends
        ax.xaxis.set_major_locator(MultipleLocator(20))  # a label every 20 years: narrow panels
        ax.margins(y=0.02)
        ax.grid(axis="y", color="0.3", linewidth=0.4, alpha=0.13)  # faint horizontal lines only
        ax.set_axisbelow(True)
        ax.set_xlabel("Age", labelpad=4)  # below both panels
    fig.get_layout_engine().set(w_pad=0.08)  # some room between the panels
    save(fig, path)
