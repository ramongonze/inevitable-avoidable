# The Inevitable and the Avoidable: Decomposing Attribute-Inference Leakage in Data Releases — artifact

This folder reproduces the experiments of Section 8 of the paper: Figures 1–5 and the numbers
that the text quotes from them. Each experiment is a Jupyter notebook, which writes its figures to
`figures/`.

| Notebook | Experiment (Section 8) | Figure | Files |
|---|---|---|---|
| `Experiment_1.ipynb` | 1. Differential leakage across groups | Figure 1 | `figures/figure1a.pdf`, `figures/figure1b.pdf` |
| `Experiment_2.ipynb` | 2. Risk in particular releases | Figure 2 | `figures/figure2.pdf` |
| `Experiment_3.ipynb` | 3. Coarsening and randomization | Figure 3 | `figures/figure3.pdf` |
| `Experiment_4.ipynb` | 4. Utility as the dataset grows | Figure 4 | `figures/figure4a.pdf`, `figures/figure4b.pdf` |
| `Experiment_5.ipynb` | 5. The two leakages by group size | Figure 5 | `figures/figure5a.pdf`, `figures/figure5b.pdf` |

The notebooks use these modules:

| File | Contents |
|---|---|
| `build_pums2010.py` | Download of the 2010 ACS PUMS and extraction of the columns used |
| `common.py` | The population θ, its groups, the releases drawn from it, and the style of the figures |
| `risk_bounds.py` | Experiment 1: the exact risk and its bounds (Theorem 5.1); the exact risk is also used by Experiment 2 |
| `release_risks.py` | Experiment 2 |
| `controls.py` | Experiment 3: coarsening and GRR, their risk and utility; also used by Experiment 4 |
| `utility_vs_n.py` | Experiment 4 |
| `leakages.py` | Experiment 5 |

## Requirements

- Python 3.14, the version used for the paper (the pinned packages need Python 3.11 or later).
- LaTeX with `dvipng`, since the text of the figures is typeset with LaTeX:
  - Debian/Ubuntu: `sudo apt install texlive-latex-extra texlive-fonts-recommended cm-super dvipng`
  - macOS: `brew install texlive`, or MacTeX.
- About 1 GB of disk space, and an internet connection for the data (first run only).

## Step by step

1. Check that LaTeX works: `latex --version` and `dvipng --version`.
2. Open a terminal in this folder (the one with this README).
3. Create and activate a virtual environment:
   ```
   python3.14 -m venv .venv
   source .venv/bin/activate
   ```
4. Install the packages: `pip install -r requirements.txt`
5. Download and prepare the data: `python build_pums2010.py`

   This downloads the 2010 ACS 1-year PUMS person file of the U.S. Census Bureau
   (`csv_pus.zip`, about 600 MB) into `data/`, and writes `data/pums2010_acs1.csv` with the
   columns used (age `AGEP`, race `RAC1P`, disability `DIS`, marital status `MAR`) of all its
   3,061,692 records. Each notebook also does this if the file is missing.
6. Run the notebooks, either
   - interactively: `jupyter lab`, then open each `Experiment_X.ipynb` and run all its cells
     (Run → Run All Cells), or
   - all at once: `jupyter nbconvert --to notebook --execute --inplace Experiment_*.ipynb`
7. The figures are in `figures/` (see the table above); the numbers are in the notebooks.

## Notes

- The population θ is the PUMS file itself: θ(q, s) is the fraction of its records with QID q and
  sensitive value s. The survey weights are not used.
- The releases are drawn with fixed seeds, so every run gives the same figures and numbers.
- On a laptop, the five notebooks run in about a minute in total (Experiment 4, the longest, in
  about 20 s); the download of the data depends on the connection.
- The notebooks are distributed with their outputs, as obtained with the steps above.
