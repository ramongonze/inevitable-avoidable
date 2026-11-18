"""Shared code of the experiments: the population theta, its QID groups, releases drawn from it,
and the style of the figures.

theta is the distribution of the records of data/pums2010_acs1.csv (build_pums2010.py) over QID
columns and a sensitive column: theta(q, s) is the fraction of the records with QID q and
sensitive value s. The file itself is the population; the survey weights are not used (Section 8).
"""

import os

import numpy as np
import pandas as pd

LABELS = {  # names of the race codes; any other code is shown as a number
    "RAC1P": {"1": "White", "2": "Black", "3": "American Indian", "4": "Alaska Native",
              "5": "AIAN other", "6": "Asian", "7": "NHPI", "8": "Other race",
              "9": "Two or more"},
}
SEED = 0  # of the releases of sample_releases: the tables and figures of a setting show the same
STYLE = {  # style of the figures; their text goes through LaTeX
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{amssymb}",
    "font.family": "serif",
    "font.serif": ["Source Serif 4"],
    "font.size": 14,
    "axes.labelsize": 16,
    "axes.titlesize": 17,
    "legend.fontsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "pdf.fonttype": 42,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "lines.linewidth": 1.1,
    "lines.markersize": 3.6,
    "legend.fancybox": False,
    "legend.framealpha": 0.7,
    "patch.linewidth": 0.6,  # legend frame
}
# Okabe-Ito colors: blue, vermillion, bluish green, orange, reddish purple, sky blue, black
COLORS = ["#0072B2", "#D55E00", "#009E73", "#E69F00", "#CC79A7", "#56B4E9", "#000000"]


def load_population(path: str, qid: list[str], sensitive: str) -> pd.DataFrame:
    """Population theta: theta.loc[q, s] is the share of the records with QID q (a tuple with one
    code per QID column) and sensitive value s. Records with an empty cell (N/A) in these columns
    are left out; theta.attrs["records"] is the number of the others."""
    columns = [*qid, sensitive]
    data = pd.read_csv(path, dtype=str, usecols=columns).dropna()
    print(f"Population: {len(data):,} records with no N/A in {', '.join(columns)}")
    counts = data.groupby(columns).size().unstack(fill_value=0)
    if not isinstance(counts.index, pd.MultiIndex):  # a single QID column: make the groups 1-tuples
        counts.index = pd.MultiIndex.from_arrays([counts.index])
    theta = counts / counts.to_numpy().sum()
    theta.attrs["records"] = len(data)
    return theta


def group_share(theta: pd.DataFrame, q: tuple) -> float:
    """theta(q, S): the share of the population in QID group q."""
    return float(theta.loc[q].sum())


def conditional(theta: pd.DataFrame, q: tuple) -> np.ndarray:
    """theta(. | q): the distribution of the sensitive value within QID group q."""
    return theta.loc[q].to_numpy() / group_share(theta, q)


def base_rate_risk(theta: pd.DataFrame, q: tuple) -> float:
    """Phi_q(theta) = max_s theta(s | q): the base-rate risk of QID group q, the share of its most
    frequent sensitive value (Section 4)."""
    return float(conditional(theta, q).max())


def group_label(theta: pd.DataFrame, q: tuple) -> str:
    """The name of QID group q: its codes with a name in LABELS first, then the others as numbers
    (e.g. "06" as 6), e.g. "White, 78" for race White and age 78."""
    named, other = [], []
    for col, v in zip(theta.index.names, q):
        if v in LABELS.get(col, {}):
            named.append(LABELS[col][v])
        else:
            other.append(str(int(v)) if v.isdigit() else v)
    return ", ".join(named + other)


def sample_releases(theta: pd.DataFrame, n: int, releases: int) -> np.ndarray:
    """Counts h[i, q, s] of the releases i = 0, ..., releases - 1 of n records each, drawn from
    Mult_n(theta) with the seed SEED, with the groups q in the order of theta.index and the
    sensitive values s in that of theta.columns."""
    cells = theta.to_numpy()
    h = np.random.default_rng(SEED).multinomial(n, cells.ravel(), size=releases)
    return h.reshape(releases, *cells.shape)


def save(fig, path: str, dpi: int = 200) -> None:
    """Save the figure to path, in the format of its extension, creating its folder if needed."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.05)
