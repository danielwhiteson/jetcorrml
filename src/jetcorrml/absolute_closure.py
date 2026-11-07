"""Closure metrics for the absolute energy-scale correction.

The absolute stage exists to put the median response at unity everywhere, so
what it should be judged on is how well it does that, bin by bin.  The
resolution sigma_68 is a poor figure of merit here: for a correction that is a
deterministic function c(E_reco, |eta|), applying it is a change of variable, so
at fixed E_gen the width transforms by the local derivative of the map,

    sigma_corr / sigma_reco  =  c * |1 + dln c / dln E| .

sigma_68 therefore responds to the SLOPE of the correction, not to its accuracy,
and a correction which deviates from the truth can be narrower than the exact
one.  On a closure test with the injected response of Sec. "Absolute energy
scale", the exact numerical inversion gives median 1.0003 and sigma_68 0.1258,
while the same inversion perturbed by a 4% wiggle gives median 1.0074 and
sigma_68 0.1203 -- a worse correction with a better resolution.  Resolution is
still worth reporting, but it cannot be the primary metric.

The metric defined here is the analogue of the pile-up figure of merit: within
each bin the deviation of the median response from unity is measured, and the
figure of merit is the root mean square of those deviations across bins.
Squaring before averaging weights the largest per-bin deviations most and
prevents bins of opposite sign from cancelling, which a signed mean would allow.
A perfect correction gives zero in every bin.

Three binnings are reported: in E_gen, in |eta_det|, and over the
two-dimensional grid of both.  The two-dimensional one is the most complete,
since a correction can be right when averaged over eta at each energy and still
wrong in individual cells.
"""
from pathlib import Path

import awkward as ak
import numpy as np
from scipy.stats import norm

from jetcorrml import paths, settings
from jetcorrml.jets import flat

# Default bin edges.
#
# E_gen: log-spaced from the closure floor of 50 GeV.  The floor is where the
# reco-pT threshold stops biasing the response distribution, and the upper edge
# covers the bulk of the dijet sample.
DEFAULT_E_EDGES = np.geomspace(50.0, 600.0, 11)

# |eta_det|: the calorimeter-region boundaries used elsewhere in the pipeline,
# truncated at the acceptance of the sample.
DEFAULT_ETA_EDGES = np.array([0.0, 0.3, 0.8, 1.1, 1.37, 1.52, 2.1, 2.5])


def _median_dev(resp, mask, min_count):
    """median(resp) - 1 on a mask, or nan if the bin is too sparse."""
    if int(mask.sum()) < min_count:
        return np.nan
    return float(np.median(resp[mask]) - 1.0)


def _median_sem(resp, mask, min_count):
    """Standard error of the median in a bin, 1.253 * sigma / sqrt(N).

    The spread is taken from the 68% inter-quantile range rather than the
    standard deviation, so that the long low-response tail does not inflate it.
    This sets the level below which a per-bin closure deviation is consistent
    with statistical fluctuation, and hence the floor of the RMS figure of
    merit: a correction cannot be shown to close better than the sample can
    resolve.
    """
    n = int(mask.sum())
    if n < min_count:
        return np.nan
    r = resp[mask]
    sigma = 0.5 * (np.quantile(r, 0.8413) - np.quantile(r, 0.1587))
    return float(1.2533 * sigma / np.sqrt(n))


def _rms(a):
    a = np.asarray(a, dtype=float)
    return float(np.sqrt(np.nanmean(a**2))) if np.isfinite(a).any() else np.nan


def _maxabs(a):
    a = np.asarray(a, dtype=float)
    return float(np.nanmax(np.abs(a))) if np.isfinite(a).any() else np.nan


def closure_metrics(
    resp,
    e_gen,
    eta_det,
    sel=None,
    e_edges=None,
    eta_edges=None,
    min_count=50,
):
    """Per-bin closure of the response, and its RMS across bins.

    Parameters
    ----------
    resp : array
        E_corr / E_gen per jet (or E_reco / E_gen for the uncorrected reference).
    e_gen, eta_det : array
        Particle-level energy and detector pseudorapidity per jet.
    sel : bool array, optional
        Closure selection.  Jets outside it are ignored.
    e_edges, eta_edges : array, optional
        Bin edges.  Defaults above.
    min_count : int
        Bins holding fewer jets than this are dropped rather than reported, a
        median from a handful of jets being uninformative.

    Returns
    -------
    dict with

      closure_rms_e, closure_rms_eta, closure_rms_2d
          RMS across bins of median(resp) - 1.  Zero is ideal.
      closure_max_e, closure_max_eta, closure_max_2d
          The largest single-bin deviation, which the RMS can hide.
      dev_e, dev_eta, dev_2d
          The per-bin deviations themselves, for plotting.  dev_2d is indexed
          [i_e, i_eta].
      n_e, n_eta, n_2d
          Jet counts per bin.
      e_centres, eta_centres, e_edges, eta_edges
    """
    resp = np.asarray(resp, dtype=float)
    e_gen = np.asarray(e_gen, dtype=float)
    abs_eta = np.abs(np.asarray(eta_det, dtype=float))
    sel = (np.ones(len(resp), dtype=bool) if sel is None
           else np.asarray(sel, dtype=bool))

    e_edges = DEFAULT_E_EDGES if e_edges is None else np.asarray(e_edges, float)
    eta_edges = (DEFAULT_ETA_EDGES if eta_edges is None
                 else np.asarray(eta_edges, float))
    n_e, n_eta = len(e_edges) - 1, len(eta_edges) - 1

    i_e = np.digitize(e_gen, e_edges) - 1
    i_eta = np.digitize(abs_eta, eta_edges) - 1
    # jets outside the binning are excluded rather than piled into the edge bins
    in_range = (i_e >= 0) & (i_e < n_e) & (i_eta >= 0) & (i_eta < n_eta)
    ok = sel & in_range & np.isfinite(resp)

    dev_e = np.full(n_e, np.nan)
    sem_e = np.full(n_e, np.nan)
    cnt_e = np.zeros(n_e, dtype=int)
    for a in range(n_e):
        m = ok & (i_e == a)
        cnt_e[a] = int(m.sum())
        dev_e[a] = _median_dev(resp, m, min_count)
        sem_e[a] = _median_sem(resp, m, min_count)

    dev_eta = np.full(n_eta, np.nan)
    sem_eta = np.full(n_eta, np.nan)
    cnt_eta = np.zeros(n_eta, dtype=int)
    for b in range(n_eta):
        m = ok & (i_eta == b)
        cnt_eta[b] = int(m.sum())
        dev_eta[b] = _median_dev(resp, m, min_count)
        sem_eta[b] = _median_sem(resp, m, min_count)

    dev_2d = np.full((n_e, n_eta), np.nan)
    sem_2d = np.full((n_e, n_eta), np.nan)
    cnt_2d = np.zeros((n_e, n_eta), dtype=int)
    for a in range(n_e):
        sel_a = ok & (i_e == a)
        if not sel_a.any():
            continue
        for b in range(n_eta):
            m = sel_a & (i_eta == b)
            cnt_2d[a, b] = int(m.sum())
            dev_2d[a, b] = _median_dev(resp, m, min_count)
            sem_2d[a, b] = _median_sem(resp, m, min_count)

    return {
        "closure_rms_e": _rms(dev_e),
        "closure_rms_eta": _rms(dev_eta),
        "closure_rms_2d": _rms(dev_2d),
        "closure_max_e": _maxabs(dev_e),
        "closure_max_eta": _maxabs(dev_eta),
        "closure_max_2d": _maxabs(dev_2d),
        # statistical floor: the RMS a perfectly closing correction would show
        # from the finite number of jets per bin.  A figure of merit at or below
        # its floor is measuring the sample, not the correction.
        "closure_floor_e": _rms(sem_e),
        "closure_floor_eta": _rms(sem_eta),
        "closure_floor_2d": _rms(sem_2d),
        "dev_e": dev_e,
        "dev_eta": dev_eta,
        "dev_2d": dev_2d,
        "sem_e": sem_e,
        "sem_eta": sem_eta,
        "sem_2d": sem_2d,
        "n_e": cnt_e,
        "n_eta": cnt_eta,
        "n_2d": cnt_2d,
        "e_centres": np.sqrt(e_edges[:-1] * e_edges[1:]),
        "eta_centres": 0.5 * (eta_edges[:-1] + eta_edges[1:]),
        "e_edges": e_edges,
        "eta_edges": eta_edges,
    }


# ─────────────────────────── the jets the closure is judged on ───────────────────────────
def closure_floor(eta, e_min=settings.CLOSURE_E_MIN):
    """E_gen below which a jet is left out: where the reconstructed-pT threshold biases R."""
    return np.maximum(e_min, settings.ABSOLUTE_E_BUFFER * settings.RECO_PT_MIN * np.cosh(eta))


def closure_jets(path):
    """(R = E/E_gen, E_gen, eta_det, the selected jets) of an absolute-scale output: the jets
    matched within the matching distance, with E_gen above the closure floor."""
    ev = ak.from_parquet(str(path))
    eta = flat(ev.jet.eta)
    e_corr = flat(ev.jet.E)
    e_gen = flat(ev.gen.pt) * np.cosh(flat(ev.gen.eta))
    dr = flat(ev.deltaR) if "deltaR" in ak.fields(ev) else np.zeros(len(eta))
    sel = (e_gen >= closure_floor(eta)) & (dr < settings.MATCH_DR) & (e_gen > 0)
    return e_corr / np.maximum(e_gen, 1.0), e_gen, eta, sel


def sigma68(x):
    """Half the central +-1 sigma inter-quantile range."""
    return 0.5 * (np.quantile(x, norm.cdf(1)) - np.quantile(x, norm.cdf(-1)))


# ─────────────────────────── Table III and Fig. 10: the scan ───────────────────────────
# label -> the file of a dataset (both on the NN pile-up correction)
SERIES = [
    ("nn", paths.nn_absolute),
    ("parametric", lambda ds: paths.parametric_absolute(ds, "nn")),
]


def closure_of_scan(e_edges=DEFAULT_E_EDGES, eta_edges=DEFAULT_ETA_EDGES):
    """The closure of each dataset of the scan and each series: ({(regime, rho_muN, label):
    metrics}, the files not found)."""
    out, missing = {}, []
    for dataset in settings.PILEUP_SCAN:            # lo-pT, then hi-pT, in rho_muN order
        for label, path_of in SERIES:
            p = Path(path_of(dataset.name))
            if not p.exists():
                missing.append(str(p))
                continue
            resp, e_gen, eta, sel = closure_jets(p)
            m = closure_metrics(resp, e_gen, eta, sel=sel, e_edges=e_edges, eta_edges=eta_edges)
            m["median"] = float(np.median(resp[sel]))
            m["sigma68"] = float(sigma68(resp[sel]))
            out[(dataset.regime, dataset.rho, label)] = m
    return out, missing


def regime_mean(res, regime, label, key):
    """The mean over the scan's rho_muN of one metric, NaN left out."""
    v = [res[(regime, r, label)][key] for r in settings.RHO_MUN
         if (regime, r, label) in res
         and res[(regime, r, label)][key] == res[(regime, r, label)][key]]
    return sum(v) / len(v) if v else float("nan")


# ─────────────────────────── Fig. 11: the resolution ───────────────────────────
def resolution_series(rho):
    """The (key, path) series of the resolution figure at rho_muN = ``rho``: the NN chain, the
    parametric absolute scale on the NN pile-up correction ("cross") and on the parametric one
    ("full"), each in the lo- and hi-pT regime."""
    lo, hi = (settings.pileup_dataset(rho, regime).name for regime in settings.REGIMES)
    return [
        ("nn_lo", Path(paths.nn_absolute(lo))),
        ("nn_hi", Path(paths.nn_absolute(hi))),
        ("cross_lo", Path(paths.parametric_absolute(lo, "nn"))),
        ("cross_hi", Path(paths.parametric_absolute(hi, "nn"))),
        ("full_lo", Path(paths.parametric_absolute(lo, "parametric"))),
        ("full_hi", Path(paths.parametric_absolute(hi, "parametric"))),
    ]


def resolution_jets(path):
    """(R, |eta|, E_gen) of the jets the closure is judged on, in the file's own float type
    (unlike closure_jets, which works in float64)."""
    ev = ak.from_parquet(str(path))
    jet_E = np.asarray(ak.flatten(ev.jet.E))
    jet_eta = np.asarray(ak.flatten(ev.jet.eta))
    gen_pt = np.asarray(ak.flatten(ev.gen.pt))
    gen_eta = np.asarray(ak.flatten(ev.gen.eta))
    dr = np.asarray(ak.flatten(ev.deltaR))
    e_gen = gen_pt * np.cosh(gen_eta)
    sel = (e_gen >= closure_floor(jet_eta)) & (dr < settings.MATCH_DR) & (e_gen > 0)
    R = jet_E[sel] / e_gen[sel]
    return R, np.abs(jet_eta[sel]), e_gen[sel]


def profile_sigma68(x, r, edges, log_x=False, min_count=30):
    """sigma_68 of r in bins of x (the bins with at least min_count jets)."""
    cx, cy = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (x >= lo) & (x < hi)
        if int(m.sum()) >= min_count:
            cx.append(np.sqrt(lo * hi) if log_x else 0.5 * (lo + hi))
            cy.append(sigma68(r[m]))
    return np.array(cx), np.array(cy)
