# jetcorrml

Code for the paper *Learning to Calibrate Jet Energies* by B. Malaescu, B. Nachman, M. Ram,
S. Schramm and D. Whiteson. The code is by D. Whiteson.

## Requirements

- macOS or Linux.
- `python3`, version 3.8 or later, to start `run_paper.py`. It uses only the standard
  library, so the `python3` already on a Mac (or a conda base) is enough; nothing needs
  updating.
- [uv](https://docs.astral.sh/uv/) on the `PATH`. Every program of the run is started with
  `uv run --locked`, which builds the environment from `uv.lock` the first time: Python
  3.14 (`.python-version`; uv fetches it if it is missing, even when a newer Python is
  installed) and the eight packages the code uses (awkward, matplotlib, numpy, pandas,
  pyarrow, scikit-learn, scipy, torch, with what they need), each at its locked version.
  This needs internet once. `--locked` makes uv stop rather than resolve again if
  `uv.lock` and `pyproject.toml` disagree.
- The two simulated samples (below).
- Time: about 1.5 hours for the full run with 6 jobs side by side on a 16-core MacBook Pro (M-series).

`python3 run_paper.py --dry-run` checks all of this in a few seconds and lists the steps.
Before the first step, a run also checks the environment uv built: the Python version,
and every installed package against its `uv.lock` version. It stops if anything differs.

## Install

```
git clone https://github.com/danielwhiteson/jetcorrml && cd jetcorrml
uv sync --locked
```

(`run_paper.py` builds the environment itself if you skip it.)

## Inputs

The simulated samples (parquet) are at [doi:10.5281/zenodo.24043234](https://doi.org/10.5281/zenodo.24043234),
with a description of how they were made and what each file holds. Download
`jetcorrml_dijet.zip` and `jetcorrml_large_r.zip` into this folder and unzip them:

```
unzip jetcorrml_dijet.zip      # inputs/raw/jj*.pq          dijets, pp -> jj (36 files)
unzip jetcorrml_large_r.zip    # inputs/large_r/wgamma*.pq  photon + W(-> q qbar), large-R jets (54 files)
```

or give folders holding them with `--raw` and `--large-r-raw`. Keep the file names: the
files are read in sorted order of their names, which sets the order of the events.

## Reproduce the paper

```
python3 run_paper.py --dry-run     # list the steps
python3 run_paper.py               # run them; about 1.5 hours
```

A rerun skips the steps that are done, unless what they read has changed since. To compare
with an earlier run, add `--baseline <dir>/paper_check`.

Output:

```
paper_check/REPORT.txt     every figure and table, in paper order, and how it compares
output/paper/figures/      the paper's figures
output/paper/data/         the numbers behind each figure
output/paper/tables/       the tables: their numbers (.csv) and their LaTeX (.tex)
paper_check_logs/          one log per step
```

## Map

| Paper | Code |
|---|---|
| Injections (III C, IV A, V A, VII A) | `src/steps/inject.py`, `inject_large_r.py`, `divide.py` |
| Pile-up (IV) | `src/steps/pileup_nn.py`, `pileup_parametric.py` |
| Absolute scale (V) | `src/steps/absolute_nn.py`, `absolute_parametric.py` |
| Sub-jets (VI) | `src/steps/pileup_nn_large_r.py`, `absolute_nn_large_r.py` |
| In situ (VII) | `src/steps/insitu_nn.py`, `insitu_parametric.py` |
| Nuisance parameters (VIII) | `src/steps/band.py` |
| Figures | `scripts/figures.py`; the list of the paper's items is `src/jetcorrml/paper/registry.py` |
| Table II, III, IV, V | `scripts/table_pileup.py`, `table_absolute.py`, `table_insitu.py`, `table_density.py` |
| Statements checked outside the run | `scripts/study_absolute_norm.py` (median vs core peak, IX), `scripts/study_band_regions.py` (where the band's variance lies, VIII) |
| The run | `run_paper.py`; its steps and their files are `src/jetcorrml/pipeline.py` |

Settings the results depend on: `src/jetcorrml/settings.py` (the datasets and their injections, the
distortions, the trainings, the in-situ passes, the band); `get(mini=True)` is the reduced run of `--mini`.
Every file the run reads or writes: `src/jetcorrml/paths.py`.
