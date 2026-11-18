"""Build data/pums2010_acs1.csv, the population of the experiments, from the 2010 American
Community Survey (ACS) 1-year Public Use Microdata Sample (PUMS) of the U.S. Census Bureau.

The national person file, csv_pus.zip (about 600 MB), is downloaded into data/, and the columns
that the experiments use are extracted from all its records: AGEP (age), RAC1P (race), DIS
(disability) and MAR (marital status). Run it once with `python build_pums2010.py`; the notebooks
also run it, and it does nothing if data/pums2010_acs1.csv already exists.
"""

import fnmatch
import os
import urllib.request
import zipfile

import pandas as pd

URL = "https://www2.census.gov/programs-surveys/acs/data/pums/2010/1-Year/csv_pus.zip"
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
ZIP_PATH = os.path.join(DATA_DIR, "csv_pus.zip")
DATA_PATH = os.path.join(DATA_DIR, "pums2010_acs1.csv")
MEMBER_PATTERN = "ss10pus*.csv"  # the person files in the zip
COLUMNS = ["AGEP", "RAC1P", "DIS", "MAR"]
CHUNK_SIZE = 250_000  # records read at a time


def download(url: str, path: str) -> None:
    """Download url to path, printing the progress every 10%. Skip if path exists."""
    if os.path.exists(path):
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    last_pct = -10

    def progress(blocks: int, block_size: int, total: int) -> None:
        nonlocal last_pct
        if total <= 0:
            return
        done = min(blocks * block_size, total)
        pct = 100 * done // total
        if pct >= last_pct + 10:
            last_pct = pct - pct % 10
            print(f"  {pct:3d}%  {done / 1e6:.0f} / {total / 1e6:.0f} MB", flush=True)

    print(f"Downloading {url}")
    tmp_path = path + ".part"  # renamed when complete: an interrupted download leaves no zip
    urllib.request.urlretrieve(url, tmp_path, reporthook=progress)
    os.replace(tmp_path, path)


def build_dataset() -> str:
    """Write the columns COLUMNS of all the records of the person files of the zip to DATA_PATH,
    downloading the zip first if needed, and return DATA_PATH. The codes and the empty cells are
    copied as they are. Nothing is done if DATA_PATH already exists."""
    if os.path.exists(DATA_PATH):
        return DATA_PATH
    download(URL, ZIP_PATH)
    print(f"Extracting {', '.join(COLUMNS)} into data/{os.path.basename(DATA_PATH)}")
    tmp_path = DATA_PATH + ".part"  # renamed when complete: an interrupted run leaves no CSV
    records = 0
    with zipfile.ZipFile(ZIP_PATH) as zf, open(tmp_path, "w", newline="") as out:
        members = sorted(name for name in zf.namelist()
                         if fnmatch.fnmatch(os.path.basename(name), MEMBER_PATTERN))
        if not members:
            raise RuntimeError(f"no {MEMBER_PATTERN} file in {os.path.basename(ZIP_PATH)}")
        out.write(",".join(COLUMNS) + "\n")
        for name in members:
            with zf.open(name) as f:
                # dtype=str and na_filter=False keep the codes and the empty cells as they are
                for chunk in pd.read_csv(f, usecols=COLUMNS, dtype=str, na_filter=False,
                                         chunksize=CHUNK_SIZE):
                    chunk[COLUMNS].to_csv(out, header=False, index=False)
                    records += len(chunk)
    os.replace(tmp_path, DATA_PATH)
    print(f"{records:,} records written")
    return DATA_PATH


if __name__ == "__main__":
    build_dataset()
