"""Every file the pipeline reads or writes.

Paths are relative to the top of the tree, where the runner starts every step.
Jets are stored as parquet files ending in .pq.  ``ds`` is a dataset name
(e.g. ``lo_rho05``), ``d`` a data/simulation distortion
(e.g. ``expo``), ``s`` a division of the sample into simulation and data, and
``kind`` one of its two halves, ``"MC"`` or ``"DATA"``.
"""

from glob import glob

# ── inputs (linked into the tree by the runner) ──────────────────────────────
RAW = "data/raw"                          # the dijet sample: <prefix>*.pq
LARGE_R_RAW = "data/large_r/raw"          # the large-R sample: wgamma*.pq
LARGE_R_PREFIX = "wgamma"


def raw_dijet_files(prefix):
    """The dijet sample's files, in the order they are read (the injection draws
    random numbers jet by jet, so the order matters)."""
    return sorted(glob(f"{RAW}/{prefix}*.pq"))


def raw_large_r_files():
    """The large-R sample's files, in the order they are read."""
    return sorted(glob(f"{LARGE_R_RAW}/{LARGE_R_PREFIX}*.pq"))


# ── small-R chain ────────────────────────────────────────────────────────────
def injected(ds):
    """Jets after the pile-up and absolute-scale injection."""
    return f"data/injected/{ds}.pq"


def nn_pileup(ds):
    """Jets after the NN pile-up correction."""
    return f"data/pileup/nn/{ds}.pq"


def parametric_pileup(ds):
    """Jets after the parametric pile-up correction."""
    return f"data/pileup/parametric/{ds}.pq"


def nn_absolute(ds):
    """Jets after the NN pile-up correction and the NN absolute scale."""
    return f"data/absolute/nn/{ds}.pq"


def nn_absolute_factor(ds):
    """The overall factor the NN absolute scale ends with (json), and the median response before it."""
    return f"data/absolute/nn/{ds}_factor.json"


def parametric_absolute(ds, pileup):
    """Jets after the parametric absolute scale, applied after the ``pileup`` ("nn" or
    "parametric") pile-up correction: the "cross" and the "full" chain of Fig. 11."""
    return f"data/absolute/parametric-{'cross' if pileup == 'nn' else 'full'}/{ds}.pq"


# ── large-R chain (one dataset) ──────────────────────────────────────────────
def large_r_injected():
    return "data/large_r/injected.pq"


def large_r_pileup():
    """Large-R jets after the NN pile-up correction."""
    return "data/large_r/pileup_nn.pq"


def _large_r_absolute_stem(subjets):
    return f"data/large_r/absolute_nn{'_subjets' if subjets else ''}"


def large_r_absolute(subjets):
    """Large-R jets after the NN absolute scale, with or without the sub-jet inputs."""
    return _large_r_absolute_stem(subjets) + ".pq"


def large_r_model(subjets):
    """The trained large-R response network (the probe figure reads it)."""
    return _large_r_absolute_stem(subjets) + ".model.pt"


# ── divisions and in situ ────────────────────────────────────────────────────
def division(ds, d, s, kind):
    """One half of division s of dataset ds, the data half carrying distortion d."""
    return f"data/division/{ds}/{d}/{s}_{kind}.pq"


def insitu(ds, d, s, method, kind, k=None):
    """The in-situ output of ``method`` ("nn" or "parametric") on division s: the
    result (``k`` None), or the output of an earlier pass k."""
    top = f"data/insitu/{method}" if k is None else f"data/insitu/{method}/pass{k}"
    return f"{top}/{ds}/{d}/{s}_{kind}.pq"


# ── the band's retrainings (on division 0 of the in-situ dataset) ────────────
def retraining_dir(d):
    return f"data/band/{d}"


def retraining(ds, d, b, kind):
    """Both halves of division 0 after retraining b."""
    return f"{retraining_dir(d)}/retraining{b}_{kind}.pq"


def retraining_grid(ds, d, b):
    """The scale of retraining b on a fixed (pT, |eta|) grid."""
    return f"{retraining_dir(d)}/retraining{b}_grid.npz"


def retraining_log(ds, d, b):
    return f"{retraining_dir(d)}/retraining{b}.log"


def retraining_grids(ds, d):
    """Every retraining's grid, in the order the band is built from: sorted by file
    name, so for 30 retrainings 0, 10-19, 1, 20-29, 2, 3, ..., 9 ("_" sorts after the
    digits).  The covariance sums over the retrainings in this order, so it is part of
    the result."""
    return sorted(glob(f"{retraining_dir(d)}/retraining*_grid.npz"))


def retraining_of_grid(grid):
    """The retraining number b of a grid file from retraining_grids."""
    return int(grid.rsplit("/retraining", 1)[-1].split("_grid.npz")[0])


# ── the paper's figures and the numbers they draw ────────────────────────────
PAPER = "output/paper"


def paper_figure(subdir, name):
    return f"{PAPER}/figures/{subdir}/{name}.pdf"


def paper_data(name):
    return f"{PAPER}/data/{name}.npz"


def paper_table(name, ext):
    """A table of the paper: its numbers (ext "csv") or its text ("tex", or "txt" for
    numbers the paper quotes in its text)."""
    return f"{PAPER}/tables/{name}.{ext}"
