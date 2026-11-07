"""
Where the pipeline's outputs are, and how the paper figures read them.

Nothing here trains or applies a network: these are the files the pipeline
has already written.  Each correction step writes its jets in the order it
read them, so the file before a step and the file after it line up jet by
jet; `_same_jets` checks that before anything is compared.
"""

from functools import lru_cache
from pathlib import Path

import awkward as ak
import numpy as np

from jetcorrml import paths, settings
from jetcorrml.jets import flat, per_jet


# The run whose outputs are read, and the raw dijet files it read (<PREFIX>*.pq).
# scripts/figures.py sets these from --mini and --prefix before anything is computed.
RUN = settings.PAPER
PREFIX = settings.SAMPLE_PREFIX


def injected_file(ds):
    return Path(paths.injected(ds))


def nn_pileup_file(ds):
    return Path(paths.nn_pileup(ds))


def nn_absolute_file(ds):
    return Path(paths.nn_absolute(ds))


def parametric_pileup_file(ds):
    return Path(paths.parametric_pileup(ds))


def raw_files():
    """The raw dijet files, in the order the injection reads them."""
    out = paths.raw_dijet_files(PREFIX)
    if not out:
        raise FileNotFoundError(f"no raw files {paths.RAW}/{PREFIX}*.pq")
    return out


def _read(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"{path} is not there (run from the top of the pipeline folder, "
                                f"after the step that writes it)")
    return ak.from_parquet(path)


def _same_jets(a, b, pa, pb):
    ga, gb = flat(a.gen.pt), flat(b.gen.pt)
    if len(ga) != len(gb) or not np.array_equal(ga, gb):
        raise ValueError(f"{pa} and {pb} do not hold the same jets in the same order")


def pileup_eta_bins():
    """The |eta| bins of the pile-up step."""
    return np.array(settings.PILEUP_ETA_BINS, dtype=float)


@lru_cache(maxsize=2)
def injected(ds):
    """Pile-up variables of the injected sample, one entry per jet."""
    p1 = injected_file(ds)
    a = _read(p1)
    return {"npv": flat(a.event.npv), "mu": flat(a.event.mu)}, [p1]


@lru_cache(maxsize=2)
def pileup(ds):
    """Jets before and after the NN pile-up correction, flat.

    pt_reco is the injected pT (before any correction); pt_nn the pT the
    step wrote, pt_reco - rho*A - delta_NN.
    """
    p1, p2 = injected_file(ds), nn_pileup_file(ds)
    a, b = _read(p1), _read(p2)
    _same_jets(a, b, p1, p2)
    return {
        "pt_reco": flat(a.jet.pt),
        "pt_gen": flat(a.gen.pt),
        "eta_det": flat(a.jet.eta),
        "rho": flat(a.event.rho),
        "npv": flat(a.event.npv),
        "mu": flat(a.event.mu),
        "area_pt": flat(a.jet.areaE / np.cosh(a.jet.eta)),
        "pt_nn": flat(b.jet.pt),
    }, [p1, p2]


@lru_cache(maxsize=2)
def pileup_parametric(ds):
    """pT after the parametric pile-up correction (rho*A and the residual fit), for the
    jets of pileup(ds) in the same order; and the file it was read from."""
    p1, p2 = injected_file(ds), parametric_pileup_file(ds)
    a, b = _read(p1), _read(p2)
    ga, gb = flat(a.gen.pt), flat(b.gen.pt)
    if len(ga) != len(gb) or not np.allclose(ga, gb, rtol=0, atol=1e-6):
        raise ValueError(f"{p1} and {p2} do not hold the same jets in the same order")
    return flat(b.jet.pt), p2


@lru_cache(maxsize=2)
def absolute(ds):
    """Jets before and after the NN absolute correction, flat."""
    p2, p3 = nn_pileup_file(ds), nn_absolute_file(ds)
    a, b = _read(p2), _read(p3)
    _same_jets(a, b, p2, p3)
    pg, eg = flat(a.gen.pt), flat(a.gen.eta)
    return {
        "e_gen": pg * np.cosh(eg),
        "e_before": flat(a.jet.E),
        "e_after": flat(b.jet.E),
        "eta_det": flat(a.jet.eta),
        "dr": flat(a.deltaR),
    }, [p2, p3]


# ───────── the jets of the injection and appendix figures ─────────
def injection_residuals(files):
    """The residual pT - pT_gen of the jets of `files` taken together, with and without the
    rho*A subtraction, and their N_PV and mu (Fig. 5)."""
    arrs = [ak.from_parquet(f) for f in files]
    data = ak.concatenate(arrs) if len(arrs) > 1 else arrs[0]
    jet, gen = data["jet"], data["gen"]
    pt_reco = flat(jet.pt)
    pt_gen = flat(gen.pt)
    area_pt = flat(jet.areaE) / np.cosh(flat(jet.eta))
    rho = per_jet(data, "rho")
    npv = per_jet(data, "npv")
    mu = per_jet(data, "mu")
    residual = pt_reco - pt_gen - rho * area_pt
    residual_nosub = pt_reco - pt_gen  # no rho*A subtraction (raw pile-up still in)
    return {"residual": residual, "residual_nosub": residual_nosub,
            "npv": npv, "mu": mu, "n": len(residual)}


def density_residuals(files, dr_max):
    """Per-jet residuals and pile-up variables of the files, before and after the forward
    pile-up density is injected (Fig. 20): the jets matched within dr_max."""
    keys = ("resid", "resid_sub", "npv", "mu", "abs_eta", "rho_A")
    cols = {k: [] for k in keys}
    for p in files:
        ev = ak.from_parquet(p)
        eta = flat(ev.jet.eta)
        pt_reco = flat(ev.jet.pt)
        pt_gen = flat(ev.gen.pt)
        area_pt = flat(ev.jet.areaE) / np.cosh(eta)
        rho = per_jet(ev, "rho")
        rho_A = rho * area_pt
        resid = pt_reco - pt_gen

        sel = np.isfinite(resid) & np.isfinite(rho_A)
        if "deltaR" in ak.fields(ev):
            sel &= flat(ev.deltaR) < dr_max

        cols["resid"].append(resid[sel])
        cols["resid_sub"].append((resid - rho_A)[sel])
        cols["npv"].append(per_jet(ev, "npv")[sel])
        cols["mu"].append(per_jet(ev, "mu")[sel] if "mu" in ak.fields(ev.event)
                          else np.full(int(sel.sum()), np.nan))
        cols["abs_eta"].append(np.abs(eta)[sel])
        cols["rho_A"].append(rho_A[sel])
    if not cols["resid"]:
        raise SystemExit("no input files loaded")
    return {k: np.concatenate(v) for k, v in cols.items()}


def pileup_jets(path):
    """(pt, pt_gen, rho*A, npv, mu) of an injected or pile-up-corrected file (Figs. 21-23)."""
    ev = ak.from_parquet(str(path))
    eta = flat(ev.jet.eta)
    rho_a = per_jet(ev, "rho") * (flat(ev.jet.areaE) / np.cosh(eta))
    mu = per_jet(ev, "mu") if "mu" in ak.fields(ev.event) else np.full(len(eta), np.nan)
    return flat(ev.jet.pt), flat(ev.gen.pt), rho_a, per_jet(ev, "npv"), mu
