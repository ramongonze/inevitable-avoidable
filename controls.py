"""Experiment 3 (Figure 3): coarsening the QIDs and randomizing the sensitive attribute against
attribute inference, and what they cost in utility (Section 7).

The population theta (common.py) is the distribution of the records of data/pums2010_acs1.csv over
QID columns, with d groups, and a sensitive column with k values. RELEASES releases h of n records
are drawn from Mult_n(theta) (common.sample_releases), and each is published
    raw         as it is;
    coarse      as h^gamma, with the whole numbers of a QID column grouped into bins of `width`
                values, e.g. ages 0-4, 5-9, ... (Section 7.1); the raw release is the coarse one
                with gamma the identity;
    randomized  after the sensitive value of each record is randomized by GRR_k(p'),
                p' = e^eps / (e^eps + k - 1), which is eps-locally differentially private, with the
                QID kept: M = I_d (x) GRR_k(p') (Section 7.2).
An attacker guesses, for all the members of a QID group q of the release, the most frequent value
of the group of q in the published histogram (the coarse group gamma(q) in h^gamma), at random among
ties, and the risk is the fraction of the n individuals whose value she guesses. On the raw release
it is rbar(h) of Proposition 5.2, and the guess is optimal whatever she knows (Proposition 5.1).
The analyst, with the benchmark prior uni (kappa = kd and theta_0 uniform), estimates theta by
    raw, coarse   her optimal estimate of Proposition 7.2 (6.2(a) for the raw release);
    randomized    the projection onto DA of (h/n) M^-1, as in the proof of Proposition 7.4.
Her utility is the utility gain of her estimate w in the population theta,
c(w, theta) = (2 - ||w - theta||^2) / kd (Definition 4.4), between 0 and its maximum 2/kd, reached
for w = theta. Even her estimate without any release, theta_0, has a gain close to 2/kd, so the plot
zooms in on the gains from about hers to 2/kd, and the tables write them in units of a power of 10.
Each risk and utility gain is reported as its mean over the releases with its 1 - ALPHA confidence
interval (Student t), in a table, and plotted as a point of the trade-off between the two.
"""

import itertools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.text import Text
from matplotlib.ticker import MaxNLocator
from matplotlib.transforms import Bbox
from scipy import stats

from common import COLORS, STYLE, sample_releases, save

RELEASES = 1000  # releases h of n records drawn from Mult_n(theta), the same for every setting
ALPHA = 0.05  # the confidence intervals have level 1 - ALPHA
SEED = 1  # of the randomization of the sensitive values (the releases have common.SEED)
# The color of each point of a control, by its parameter, from its strongest protection to its
# weakest, light to dark, in hues of its own: Okabe-Ito orange to blue (common.COLORS) from the
# smallest eps to the largest, and pink to dark purple (ColorBrewer RdPu) from the widest bins to the
# narrowest
NOISY_SCALE = LinearSegmentedColormap.from_list("noisy", [COLORS[3], COLORS[0]])
COARSE_SCALE = LinearSegmentedColormap.from_list(
    "coarse", ["#F768A1", "#DD3497", "#AE017E", "#7A0177", "#49006A"])
RAW = "0.15"  # near black for the raw release
# The reference lines, in Okabe-Ito hues (common.COLORS) that no point uses: bluish green for
# the base-rate risk, vermillion for the utility before the release
BASE_RATE, BEFORE = COLORS[2], COLORS[1]
MARKER_SIZE = 8  # of every point, in the plot and in the legend, in points
RAW_SIZE = 1.3 * MARKER_SIZE  # of the raw release, larger, so the points it equals show inside
# The points within CLUSTER of the raw release, as a share of the axes, semi-transparent, so that
# the ones on top of each other show
CLUSTER, CLUSTER_ALPHA = 0.08, 0.6
LABEL_STEPS = (12, 20, 28, 36)  # distances in points from a point to its label, tried in turn


def coarse_groups(theta: pd.DataFrame, column: str, width: int) -> np.ndarray:
    """gamma: the coarse group of each QID group of theta, as an index 0, 1, ... in the order of
    theta.index, with the whole numbers v of the QID column `column` grouped into the bins
    [w floor(v / w), w floor(v / w) + w) of w = width values (e.g. ages 0-4, 5-9, ... for width
    5), and the other QID columns kept."""
    i = theta.index.names.index(column)
    coarse = [q[:i] + (int(q[i]) // width,) + q[i + 1:] for q in theta.index]
    index = {g: c for c, g in enumerate(dict.fromkeys(coarse))}
    return np.array([index[g] for g in coarse])


def coarsen(x: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """x[..., q, s] summed over the QID groups q with the same coarse group, for gamma = groups:
    the coarse release h^gamma of the releases h, or the coarse population theta^gamma of theta."""
    return np.stack([x[..., groups == c, :].sum(axis=-2) for c in range(groups.max() + 1)],
                    axis=-2)


def keep_probability(eps: float | np.ndarray, r: int) -> float | np.ndarray:
    """e^eps / (e^eps + r - 1): the probability with which GRR_r keeps a value at level eps."""
    return np.exp(eps) / (np.exp(eps) + r - 1)


def rho(eps: float | np.ndarray, r: int) -> float | np.ndarray:
    """rho_r = (p r - 1) / (r - 1) for p = keep_probability(eps, r), i.e.
    (e^eps - 1) / (e^eps + r - 1): the eigenvalue of GRR_r(p) on the vectors that sum to 0
    (Proposition 7.4)."""
    return (np.exp(eps) - 1) / (np.exp(eps) + r - 1)


def grr(eps: float, k: int) -> np.ndarray:
    """GRR_k(p'), p' = e^eps / (e^eps + k - 1): row s keeps the value s with probability p' and
    replaces it by each of the k - 1 other values with probability (1 - p') / (k - 1)."""
    p = keep_probability(eps, k)
    return np.full((k, k), (1 - p) / (k - 1)) + (p - (1 - p) / (k - 1)) * np.eye(k)


def randomize(h: np.ndarray, eps: float, rng: np.random.Generator) -> np.ndarray:
    """The releases h[..., q, s] after the sensitive value of each record is randomized by
    GRR_k(p'): the h[..., q, s] records with value s are spread over the k values as
    Mult(h[..., q, s], GRR_k(p')[s])."""
    return rng.multinomial(h, grr(eps, h.shape[-1])).sum(axis=-2)


def inferred(scores: np.ndarray, h: np.ndarray) -> np.ndarray:
    """The fraction of the n individuals of each release h[..., q, s] whose sensitive value an
    attacker guesses, when she guesses for all the members of QID group q a value s with the
    largest scores[..., q, s], at random among ties: (1/n) sum_q of the mean of h[..., q, s] over
    these values. With scores = h, it is rbar(h) of Proposition 5.2."""
    top = scores == scores.max(axis=-1, keepdims=True)
    return ((h * top).sum(axis=-1) / top.sum(axis=-1)).sum(axis=-1) / h.sum(axis=(-2, -1))


def coarse_estimate(published: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """The analyst's optimal estimate after the coarse release published[..., q, s] = h^gamma(q', s),
    q' = gamma(q), for the benchmark prior uni: theta_hat(q, s) = (1 + h^gamma(q', s) / j) / (kd + n),
    with j the number of QID groups in q' (Proposition 7.2 with kappa = kd and theta_0 uniform; for
    the raw release, (1 + h(q, s)) / (kd + n) of Proposition 6.2(a))."""
    split = published / np.bincount(groups)[groups][:, None]  # h^gamma(q', s) / j
    kd = split.shape[-2] * split.shape[-1]
    return (1 + split) / (kd + split.sum(axis=(-2, -1), keepdims=True))


def project(v: np.ndarray) -> np.ndarray:
    """The Euclidean projection of each v[..., :] onto the simplex: max(v - tau, 0), with tau such
    that it sums to 1, found from the coordinates of v in decreasing order (Held, Wolfe and
    Crowder, 1974)."""
    u = -np.sort(-v, axis=-1)
    excess = np.cumsum(u, axis=-1) - 1
    positive = (u > excess / np.arange(1, v.shape[-1] + 1)).sum(axis=-1, keepdims=True)
    tau = np.take_along_axis(excess, positive - 1, axis=-1) / positive
    return np.maximum(v - tau, 0)


def randomized_estimate(noisy: np.ndarray, eps: float) -> np.ndarray:
    """The analyst's estimate after the randomized release noisy[..., q, s] = h~(q, s), as in the
    proof of Proposition 7.4: the projection onto DA of theta_hat = (h~/n) M^-1. As
    M^-1 = I_d (x) GRR_k(p')^-1 and GRR_k(p')^-1 = (I - (1 - rho_k) J / k) / rho_k,
    theta_hat(q, s) = (h~(q, s) - (1 - rho_k) h~(q, S) / k) / (n rho_k)."""
    k = noisy.shape[-1]
    r = rho(eps, k)
    n = noisy.sum(axis=(-2, -1), keepdims=True)
    estimate = (noisy - (1 - r) / k * noisy.sum(axis=-1, keepdims=True)) / (n * r)
    return project(estimate.reshape(*estimate.shape[:-2], -1)).reshape(estimate.shape)


def utility(w: np.ndarray, theta: pd.DataFrame | np.ndarray) -> np.ndarray:
    """The utility gain c(w, theta) = (2 - ||w - theta||^2) / kd (Definition 4.4) of each estimate
    w[..., q, s] in the population theta: at most 2/kd, for w = theta."""
    cells = np.asarray(theta)
    return (2 - ((w - cells) ** 2).sum(axis=(-2, -1))) / (cells.shape[-2] * cells.shape[-1])


def interval(name: str, values: np.ndarray) -> dict:
    """{name: the mean of values over the releases, name + " CI": the half-width of its 1 - ALPHA
    confidence interval, from Student's t distribution with one degree of freedom less than the
    number of releases}."""
    half = stats.t.ppf(1 - ALPHA / 2, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values))
    return {name: values.mean(), f"{name} CI": half}


def summary(risk: np.ndarray, gain: np.ndarray) -> dict:
    """A row of controls_table: the risk and the utility gain in each release, with their
    intervals."""
    return {**interval("risk", risk), **interval("utility gain", gain)}


def controls_table(theta: pd.DataFrame, n: int, column: str, widths: list[int],
                   epsilons: list[float]) -> pd.DataFrame:
    """The risk and the utility gain c (utility), each as its mean over RELEASES releases of n
    records with the half-width of its confidence interval (column "... CI"): for the raw release,
    the coarse releases with the QID column `column` in bins of each width of widths, and the
    randomized releases at each level eps of epsilons, all made from the same releases."""
    h = sample_releases(theta, n, RELEASES)
    rng = np.random.default_rng(SEED)
    coarsenings = {"Raw": np.arange(len(theta))}  # gamma the identity
    coarsenings |= {f"{column} in bins of {w}": coarse_groups(theta, column, w) for w in widths}
    rows = {}
    for name, groups in coarsenings.items():
        published = coarsen(h, groups)[..., groups, :]  # h^gamma(gamma(q), s) for each group q
        rows[name] = summary(inferred(published, h),
                             utility(coarse_estimate(published, groups), theta))
    for eps in epsilons:
        noisy = randomize(h, eps, rng)
        rows[f"GRR, ε = {eps:g}"] = summary(inferred(noisy, h),
                                            utility(randomized_estimate(noisy, eps), theta))
    return pd.DataFrame.from_dict(rows, orient="index")


def decimals(ci: float) -> int:
    """The number of decimals that rounds at the second significant digit of ci (4 if ci = 0)."""
    return max(0, 1 - int(np.floor(np.log10(ci)))) if ci > 0 else 4


def formatted_table(table: pd.DataFrame) -> pd.DataFrame:
    """The table of controls_table with each mean and the half-width of its confidence interval in
    the same cell, m ± c, both rounded at the second significant digit of the half-width; the
    utility gains in units of 10^p, the order of magnitude of the largest one (e.g. 10^-3 for
    kd = 480), given in their header."""
    p = int(np.floor(np.log10(table["utility gain"].max())))
    unit = f" (×10^{p})" if p else ""
    out = pd.DataFrame(index=table.index)
    for column, name, scale in [("risk", "Risk", 1),
                                ("utility gain", "Utility gain" + unit, 10.0 ** -p)]:
        out[name] = [f"{m:.{decimals(c)}f} ± {c:.{decimals(c)}f}"
                     for m, c in zip(table[column] * scale, table[f"{column} CI"] * scale)]
    return out


def label_points(ax, points: np.ndarray, labels: list[tuple[float, float, str, tuple[int, int]]]
                 ) -> None:
    """Write each label (x, y, text, (sx, sy)) next to its point (x, y), in neutral ink on a light
    white backing, at the first place that lies inside ax and overlaps neither a marker at points
    (rows (x, y)) nor a label written before: diagonally from the point, LABEL_STEPS points away
    from the nearest, on its side (sx, sy) first (e.g. (-1, 1) for the upper left), then on the
    others, or the nearest on its side if no place is free; joined to the point by a line. The
    layout of the figure must be final."""
    fig = ax.figure
    fig.canvas.draw()  # the final positions of the axes and the points
    renderer, pixels = fig.canvas.get_renderer(), fig.dpi / 72  # pixels per point
    half = 4 * pixels  # half the side of a marker with its ring
    taken = [Bbox.from_extents(px - half, py - half, px + half, py + half)
             for px, py in ax.transData.transform(points)]
    for x, y, text, (sx, sy) in labels:
        note = ax.annotate(text, (x, y), (0, 0), textcoords="offset points", fontsize=11,
                           color="0.2", zorder=5,
                           bbox=dict(boxstyle="square,pad=0.1", facecolor="white",
                                     edgecolor="none", alpha=0.8),
                           arrowprops=dict(arrowstyle="-", color="0.4", linewidth=0.6, shrinkB=4))
        note.set_in_layout(False)
        sides = [(sx, sy), (-sx, sy), (sx, -sy), (-sx, -sy)]
        # The last place, next to the point on its side, only if no other place is free
        for step, (u, v) in [*itertools.product(LABEL_STEPS, sides), (LABEL_STEPS[0], (sx, sy))]:
            note.xyann = (step * u, step * v)
            note.set(ha="left" if u > 0 else "right", va="bottom" if v > 0 else "top")
            note.update_positions(renderer)
            box = Text.get_window_extent(note, renderer).padded(pixels)  # without the line
            inside = (ax.bbox.x0 <= box.x0 and box.x1 <= ax.bbox.x1 and ax.bbox.y0 <= box.y0
                      and box.y1 <= ax.bbox.y1)
            if inside and not any(box.overlaps(other) for other in taken):
                break
        taken.append(box)


def plot(theta: pd.DataFrame, table: pd.DataFrame, widths: list[int], epsilons: list[float],
         path: str | None = None, figsize: tuple[float, float] = (6.4, 4.6)) -> None:
    """Figure 3: the trade-off between risk and utility of the controls in the table of
    controls_table, made with the given widths and epsilons: a point per
    setting at its mean utility gain c (x) and risk (y), with their confidence intervals as error
    bars, for the raw release (diamond), the coarse releases (circles, bins of each width of
    widths) and the randomized ones (squares, at each level eps of epsilons). The points of each
    control are colored by its parameter, in hues of its own (NOISY_SCALE, COARSE_SCALE), read on a
    color bar of its own, its strongest protection (the smallest eps, the widest bins) at the
    bottom; the smallest, middle and largest parameters are written next to their points
    (label_points); those within CLUSTER of the raw release are semi-transparent.
    Dashed: the base-rate (population) risk sum_q theta(q, S) max_s theta(s | q), below which no
    control brings the risk for an attacker who knows theta (Propositions 7.1(b) and 7.3(a));
    dash-dotted: the utility before the release, the gain c(theta_0, theta) in theta of the
    analyst's best guess before it, theta_0, the mean of her prior (Section 6). The x axis zooms in
    on the gains from that one, its first tick, written in full, to their maximum 2/kd, its last
    tick, at its right end, with a grid line; the best releases lie towards the lower right
    corner. One legend above the axes; saved to path if given; figsize is in inches."""
    raw, coarse, noisy = table.iloc[:1], table.iloc[1:1 + len(widths)], table.iloc[1 + len(widths):]
    population = theta.to_numpy().max(axis=1).sum()
    # Set globally, not just while plotting: Jupyter draws the figure after the cell
    plt.rcParams.update(STYLE)
    fig, ax = plt.subplots(figsize=figsize, layout="constrained")
    # x from the gain of the prior estimate theta_0, or the lowest gain, to the maximum 2/kd of the
    # population learned exactly; y around the risks and the population risk
    kd = theta.size
    before = utility(np.full(theta.shape, 1 / kd), theta)  # c(theta_0, theta), theta_0 uniform
    low = min(before, table["utility gain"].min())
    ax.set_xlim(low - 0.03 * (2 / kd - low), 2 / kd)
    bottom, top = min(table["risk"].min(), population), max(table["risk"].max(), population)
    ax.set_ylim(bottom - 0.08 * (top - bottom), top + 0.15 * (top - bottom))  # labels on top
    (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
    # x in units of 10^p, the order of magnitude of 2/kd: the gain before the release at the left,
    # in full (one more decimal than the round ticks), and 2/kd at the right end, with no round
    # tick next to either
    p = int(np.floor(np.log10(2 / kd)))
    ticks = MaxNLocator(6, steps=[1, 2, 5, 10]).tick_values(x0, x1)
    digits = max(0, -int(np.floor(np.log10((ticks[1] - ticks[0]) / 10.0**p) + 1e-9)))
    gap = (x1 - x0) / 10
    inner = [t for t in ticks if x0 <= t < x1 - gap and abs(t - before) > gap]
    ax.set_xticks([before, *inner, x1],
                  labels=[f"{before / 10.0**p:.{digits + 1}f}",
                          *(f"{t / 10.0**p:.{digits}f}" for t in inner), "$2/kd$"])

    def points(rows: pd.DataFrame, color, marker: str, zorder: int) -> None:
        """The rows of table at (utility gain, risk) with their error bars, filled with color
        (one or one per row) and ringed in white where they overlap, or open in RAW for "none"."""
        open_ = isinstance(color, str) and color == "none"
        near = (np.hypot((rows["utility gain"] - raw["utility gain"].iloc[0]) / (x1 - x0),
                         (rows["risk"] - raw["risk"].iloc[0]) / (y1 - y0)) < CLUSTER).to_numpy()
        alpha = np.where(near, CLUSTER_ALPHA, 1)
        ax.errorbar(rows["utility gain"], rows["risk"], xerr=rows["utility gain CI"],
                    yerr=rows["risk CI"], fmt="none",
                    ecolor=RAW if open_ else np.column_stack([color[:, :3], alpha]),
                    elinewidth=0.8, zorder=2)
        ax.scatter(rows["utility gain"], rows["risk"], marker=marker,
                   s=(RAW_SIZE if open_ else MARKER_SIZE)**2,
                   zorder=zorder,
                   **(dict(facecolors="none", edgecolors=RAW, linewidths=1.2) if open_ else
                      dict(c=color, edgecolors="white", linewidths=0.6, alpha=alpha)))

    # The randomized releases first: each color bar goes between the axes and the previous one.
    # Their labels go above their curve, to the upper left, and those of the coarse ones below it
    labels = []
    # The symbol above each bar and the label of a point at level v of its control
    for rows, values, scale, marker, symbol, label, side in [
            (noisy, epsilons, NOISY_SCALE, "s", r"\varepsilon", r"\varepsilon = {:.4g}", (-1, 1)),
            (coarse, widths, COARSE_SCALE.reversed(), "o", r"\mbox{\LARGE$\gamma$}_{b}", r"\mbox{{\LARGE$\gamma$}}_{{{:.4g}}}",
             (1, -1))]:
        values = np.asarray(values, dtype=float)
        norm = Normalize(values.min(), values.max())
        points(rows, scale(norm(values)), marker, 3)
        # Shorter than the axes, with its symbol above it, out of the layout, which would add the
        # height of the symbol above the axes once per bar
        bar = fig.colorbar(ScalarMappable(norm, scale), ax=ax, aspect=25, pad=0.02, shrink=0.85,
                           anchor=(0, 0))
        bar.ax.set_title(f"${symbol}$", fontsize=plt.rcParams["axes.labelsize"]).set_in_layout(False)
        # The ends of the bar among its ticks, e.g. the smallest eps, with no tick next to them
        gap = (norm.vmax - norm.vmin) / 10
        inner = MaxNLocator(5).tick_values(norm.vmin, norm.vmax)
        ticks = [norm.vmin, *(t for t in inner if norm.vmin + gap < t < norm.vmax - gap), norm.vmax]
        bar.set_ticks(ticks, labels=[f"{t:.4g}" for t in ticks])
        if rows is coarse:  # the widest bins, the strongest protection, at the bottom too
            bar.ax.invert_yaxis()
        # Labelled: for eps, the first and third levels of epsilons and the largest (the last of the
        # range, before any extra level appended to it); for gamma, the smallest width, the one closest
        # to the middle of the bar, and the largest
        if rows is noisy:
            picks = [0, min(2, len(values) - 1), values.argmax()]
        else:
            middle = np.abs(values - (values.min() + values.max()) / 2).argmin()
            picks = [values.argmin(), middle, values.argmax()]
        labels += [(rows["utility gain"].iloc[i], rows["risk"].iloc[i],
                    f"${label.format(values[i])}$", side)
                   for i in dict.fromkeys(picks)]
    # On top, open, the raw release, which the randomized releases approach as eps grows
    points(raw, "none", "D", 4)
    handles = [Line2D([], [], linestyle="none", marker=marker, label=label,
                      markersize=RAW_SIZE if marker == "D" else MARKER_SIZE,
                      markerfacecolor=face, markeredgecolor=edge, markeredgewidth=width)
               for marker, face, edge, width, label in [
                   ("o", "0.6", "white", 0.6, r"Coarsening $\mbox{\LARGE$\gamma$}_{b}$: age in bins of $b$ years"),
                   ("s", "0.6", "white", 0.6, r"GRR at level $\varepsilon$"),
                   ("D", "none", RAW, 1.2, "Raw release")]]
    handles.append(ax.axhline(population, color=BASE_RATE, linewidth=1, linestyle="--",
                              zorder=1, label="Base-rate risk"))
    handles.append(ax.axvline(before, color=BEFORE, linewidth=1, linestyle="-.", zorder=1,
                              label=r"Utility before release, $c(\theta_0,\theta)$"))
    grid = dict(color="0.3", linewidth=0.4, alpha=0.13)
    ax.grid(**grid)
    ax.set_axisbelow(True)
    # The grid line of 2/kd, on the right edge of the axes, in full: the grid clips it to half
    ax.axvline(x1, **grid, clip_on=False, zorder=0.5)
    ax.set_xlabel(rf"Utility gain ($\times 10^{{{p}}}$)")
    ax.set_ylabel("Risk")
    # The releases in the first column of the legend, the reference lines in the second
    fig.legend(handles=handles, loc="outside upper center", ncol=2, borderaxespad=0)
    label_points(ax, table[["utility gain", "risk"]].to_numpy(), labels)
    if path is not None:
        save(fig, path, dpi=300)
