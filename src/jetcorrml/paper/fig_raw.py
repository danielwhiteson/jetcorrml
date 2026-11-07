"""
Paper figures of the raw sample (section "Samples"): jet kinematics, pile-up
observables, detector response.

The numbers are histograms and profiles of the first MAX_EVENTS events of the
raw files, in the order the injection reads them.
"""

from functools import lru_cache

import awkward as ak
import numpy as np

from jetcorrml import settings
from jetcorrml.jets import flat
from jetcorrml.profiles import binned_mean

from . import inputs, store
from . import style as S

MAX_EVENTS = 3_000_000        # events of the raw sample the figures read


def _finite(a):
    a = np.asarray(a, dtype=float)
    return a[np.isfinite(a)]


@lru_cache(maxsize=1)
def _raw():
    """The raw sample: the jets of the first files, up to MAX_EVENTS events."""
    files = inputs.raw_files()
    arrs, used, total = [], [], 0
    for f in files:
        a = ak.from_parquet(f)
        arrs.append(a)
        used.append(f)
        total += len(a)
        if total >= MAX_EVENTS:
            break
    data = ak.concatenate(arrs) if len(arrs) > 1 else arrs[0]
    return data["jet"], data["gen"], data["event"], used


def _hist_numbers(reco, gen, logx=False, nbins=60, rng=None, whole=False):
    """Bin edges and normalised contents of a histogram.
    whole=True: for a quantity that takes whole values only, one bin per value; equal bins
    wider than one value and narrower than two held one value in some and two in others."""
    reco = _finite(reco)
    data = [reco] + ([_finite(gen)] if gen is not None else [])
    if rng is None:
        rng = (min(d.min() for d in data), max(np.quantile(d, 0.999) for d in data))
    if whole and all(np.all(d == np.round(d)) for d in data):
        bins = np.arange(int(np.floor(rng[0])) - 0.5, int(np.ceil(rng[1])) + 1.5)
    elif logx and rng[0] > 0:
        bins = np.logspace(np.log10(max(rng[0], 1e-3)), np.log10(rng[1]), nbins)
    else:
        bins = np.linspace(rng[0], rng[1], nbins)
    out = {"edges": bins, "reco": np.histogram(reco, bins=bins, density=True)[0]}
    if gen is not None:
        out["gen"] = np.histogram(data[1], bins=bins, density=True)[0]
    return out


# what each histogram shows: (reco, gen, options)
def _hist_spec(jets, gen, event, which):
    f = flat
    if which == "jet_pt":
        return f(jets.pt), f(gen.pt), dict(logx=True)
    if which == "jet_y":
        return f(jets.y), (f(gen.y) if "y" in ak.fields(gen) else None), dict(rng=(-5, 5))
    if which == "jet_mass":
        return f(jets.mass), f(gen.mass), {}
    if which == "event_rho":
        return f(event.rho), None, {}
    if which == "event_npv":
        return f(event.npv), None, dict(nbins=40, whole=True)
    if which == "event_mu":
        return f(event.mu), None, dict(nbins=40)
    raise KeyError(which)


def compute_hist(name, which):
    jets, gen, event, files = _raw()
    reco, g, opt = _hist_spec(jets, gen, event, which)
    store.save(name, _hist_numbers(reco, g, **opt), files, {"quantity": which, "events": int(len(jets))})


_HIST_LOOK = {   # x label, log x, log y
    "jet_pt": (r"Jet $p_T$ [GeV]", True, True),
    "jet_y": (r"Jet rapidity $y$", False, False),
    "jet_mass": ("Jet mass [GeV]", False, True),
    "event_rho": (S.LAB["rho"], False, True),
    "event_npv": (S.LAB["npv"], False, False),
    "event_mu": (S.LAB["mu"], False, False),
}


def _steps(ax, edges, h, **kw):
    ax.stairs(h, edges, **kw)


def draw_hist(name, subdir, which):
    d = store.load(name)
    xlabel, logx, logy = _HIST_LOOK[which]
    fig, ax = S.figure("third", height=1.85)
    two = "gen" in d
    _steps(ax, d["edges"], d["reco"], color=S.COL["reco"], lw=1.1, label=S.LAB["reco"] if two else None)
    if two:
        _steps(ax, d["edges"], d["gen"], color=S.COL["ref"], lw=1.0, ls="--", label=S.LAB["gen"])
    if logx:
        S.plain_log_x(ax)
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(S.LAB["norm"])
    if not logy:
        ax.set_ylim(bottom=0)
    if two:
        S.legend_top(ax, "upper right")
    return S.save(fig, store.fig_path(subdir, name))


# ───────────── detector response, in energy (the variable of the absolute correction) ─────────────
def _response_e(jets, gen):
    """E_reco / E_gen with E_gen = pT_gen cosh(eta_gen), as the absolute-scale step defines it."""
    pt_g, eta_g = flat(gen.pt), flat(gen.eta)
    e_g = pt_g * np.cosh(eta_g)
    return flat(jets.E) / np.where(e_g > 0, e_g, np.nan), e_g, np.abs(eta_g)


def compute_response_e(name):
    jets, gen, event, files = _raw()
    resp, _e, _eta = _response_e(jets, gen)
    r = _finite(resp)
    bins = np.linspace(0.5, 6.0, 70)                     # as the pT version
    store.save(name, {"edges": bins, "reco": np.histogram(r, bins=bins, density=True)[0],
                      "median": np.median(r)}, files, {"response": "E_reco / (pT_gen cosh eta_gen)"})


def draw_response_e(name, subdir):
    d = store.load(name)
    fig, ax = S.figure("third", height=1.85)
    _steps(ax, d["edges"], d["reco"], color=S.COL["reco"], lw=1.1)
    ax.set_yscale("log")
    ax.set_xlabel(S.LAB["eresp"])
    ax.set_ylabel(S.LAB["norm"])
    return S.save(fig, store.fig_path(subdir, name))


def compute_response_e_vs_eta(name):
    jets, gen, event, files = _raw()
    resp, e_g, eta_g = _response_e(jets, gen)
    m = np.isfinite(resp) & (e_g > 0) & np.isfinite(eta_g)
    edges = np.linspace(0.0, max(2.5, float(np.quantile(eta_g[m], 0.999))), 16)
    xc, mu, se = binned_mean(eta_g[m], resp[m], edges, 5)
    store.save(name, {"x": xc, "y": mu, "e": se, "edges": edges}, files,
               {"response": "E_reco / (pT_gen cosh eta_gen)", "against": "eta"})


def draw_response_e_vs_eta(name, subdir):
    d = store.load(name)
    fig, ax = S.figure("third", height=1.85)
    ax.errorbar(d["x"], d["y"], yerr=d["e"], fmt=S.MARK["reco"], ms=S.ms(2.8), lw=0.8, color=S.COL["reco"])
    ax.axhline(1.0, color=S.COL["aux"], lw=0.6)
    ax.set_xlabel(r"Particle-level jet $|\eta|$")
    ax.set_ylabel(S.LAB["eresp_mean"])
    return S.save(fig, store.fig_path(subdir, name))


CENTRAL_ETA = settings.CENTRAL_ETA   # particle-level |eta| below this is "central" (the boundary of the in-situ section)
REGIONS = (("central", r"$|\eta|<%g$", "o", True), ("forward", r"$|\eta|>%g$", "s", False))


def compute_response_e_regions(name):
    """As fig_response_E_vs_E (same jets, same bins), the jets divided at |eta| = CENTRAL_ETA."""
    jets, gen, event, files = _raw()
    resp, e_g, eta_g = _response_e(jets, gen)
    m = np.isfinite(resp) & (e_g > 0) & np.isfinite(eta_g)
    edges = np.logspace(np.log10(20), np.log10(max(200, np.quantile(e_g[m], 0.99))), 22)
    out = {"edges": edges, "boundary": CENTRAL_ETA}
    for key, sel in (("central", eta_g < CENTRAL_ETA), ("forward", eta_g >= CENTRAL_ETA)):
        out[f"{key}_x"], out[f"{key}_y"], out[f"{key}_e"] = binned_mean(e_g[m & sel], resp[m & sel], edges, 5)
        out[f"{key}_jets"] = int((m & sel).sum())
    store.save(name, out, files, {"response": "E_reco / (pT_gen cosh eta_gen)", "against": "E",
                                  "regions": f"particle-level |eta| below / above {CENTRAL_ETA:g}"})


def draw_response_e_regions(name, subdir):
    d = store.load(name)
    fig, ax = S.figure("third", height=1.85)
    for key, lab, mk, filled in REGIONS:
        ax.errorbar(d[f"{key}_x"], d[f"{key}_y"], yerr=d[f"{key}_e"], fmt=mk, ms=S.ms(2.8), lw=0.8,
                    color=S.COL["reco"], mfc=S.COL["reco"] if filled else "white", mew=0.6,
                    label=lab % float(d["boundary"]))
    ax.axhline(1.0, color=S.COL["aux"], lw=0.6)
    S.plain_log_x(ax)
    ax.set_xlabel(r"Particle-level jet $E$ [GeV]")
    ax.set_ylabel(S.LAB["eresp_mean"])
    S.legend_top(ax, "upper right", ncols=2, columnspacing=0.6, handletextpad=0.1)
    return S.save(fig, store.fig_path(subdir, name))
