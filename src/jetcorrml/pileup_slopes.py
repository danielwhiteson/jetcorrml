"""
The pile-up step's figure of merit (Tables II and V, Fig. 9): the RMS over |eta|
bins of the slope of the jet pT against N_PV and against mu.

In each |eta_det| bin the pT is regressed linearly on N_PV and on mu; the
figure of merit is the RMS of those slopes over the bins, in GeV per unit N_PV
(or mu).  Zero is ideal; the RMS keeps slopes of opposite sign in different
regions from cancelling.  No selection is applied, as in the pipeline.
"""

from pathlib import Path

import awkward as ak
import numpy as np
from scipy.stats import linregress

from jetcorrml import paths, settings
from jetcorrml.jets import flat, per_jet


def load_jets(path, subtract_rho_a=False):
    """The jets of an injected or pile-up-corrected file: pt, pt_gen, npv, mu, eta (eta_det).

    subtract_rho_a: pt - rho*A in place of pt, for the injected file, since the
    corrections act on pt - rho*A (the 'no correction' baseline)."""
    ev = ak.from_parquet(str(path))
    eta = flat(ev.jet.eta)
    pt = flat(ev.jet.pt)
    if subtract_rho_a:
        pt = pt - per_jet(ev, "rho") * (flat(ev.jet.areaE) / np.cosh(eta))
    mu = per_jet(ev, "mu") if "mu" in ak.fields(ev.event) else np.full(len(pt), np.nan)
    return {"pt": pt, "pt_gen": flat(ev.gen.pt), "npv": per_jet(ev, "npv"), "mu": mu, "eta": eta}


def _rms(a):
    return float(np.sqrt(np.nanmean(a ** 2))) if np.isfinite(a).any() else float("nan")


def rms_slopes(pt, npv, mu, eta, eta_bins, min_jets=10, outside="last", forward_from=None):
    """RMS over |eta| bins of d(pt)/d(N_PV) and d(pt)/d(mu), and the slopes of each bin.

    outside: "last" counts a jet beyond the last edge in the last bin (Table II);
    "drop" leaves it out (Table V, with the bins of its figure).
    forward_from: also the RMS over the bins whose centre is at or above it (the
    forward pile-up density acts there; the RMS over all bins dilutes it)."""
    abs_eta = np.abs(eta)
    n = len(eta_bins) - 1
    idx = np.clip(np.digitize(abs_eta, eta_bins) - 1, 0, n - 1)
    inside = abs_eta < eta_bins[-1] if outside == "drop" else np.ones(len(abs_eta), dtype=bool)
    s_npv = np.full(n, np.nan)
    s_mu = np.full(n, np.nan)
    counts = np.zeros(n, dtype=int)
    have_mu = np.isfinite(mu).all() and np.ptp(mu) > 0
    for i in range(n):
        m = (idx == i) & inside
        counts[i] = int(m.sum())
        if counts[i] < min_jets:
            continue
        s_npv[i] = linregress(npv[m], pt[m]).slope
        if have_mu and np.ptp(mu[m]) > 0:
            s_mu[i] = linregress(mu[m], pt[m]).slope
    centres = 0.5 * (np.asarray(eta_bins[:-1]) + np.asarray(eta_bins[1:]))
    out = dict(rms_npv=_rms(s_npv), rms_mu=_rms(s_mu), slopes_npv=s_npv, slopes_mu=s_mu,
               counts=counts, centres=centres)
    if forward_from is not None:
        fwd = centres >= forward_from
        out["rms_npv_fwd"] = _rms(s_npv[fwd]) if fwd.any() else float("nan")
        out["rms_mu_fwd"] = _rms(s_mu[fwd]) if fwd.any() else float("nan")
    return out


def residual_slopes_of_scan(eta_bins):
    """Table II and Fig. 9: for each dataset of the pile-up scan and each correction, the
    RMS over |eta| bins of the slopes of the residual pt - pt_gen.  Returns
    ({dataset: {rho, regime, nn_npv, nn_mu, parametric_npv, parametric_mu}}, the files not found)."""
    out, missing = {}, []
    for dataset in settings.PILEUP_SCAN:            # lo-pT, then hi-pT, in rho_muN order
        ds = dataset.name
        row = {"rho": dataset.rho, "regime": dataset.regime,
               "nn_npv": None, "nn_mu": None, "parametric_npv": None, "parametric_mu": None}
        for tag, path in (("nn", Path(paths.nn_pileup(ds))),
                          ("parametric", Path(paths.parametric_pileup(ds)))):
            if not path.exists():
                missing.append(str(path))
                continue
            d = load_jets(path)
            m = rms_slopes(d["pt"] - d["pt_gen"], d["npv"], d["mu"], d["eta"], eta_bins)
            row[f"{tag}_npv"] = m["rms_npv"]
            row[f"{tag}_mu"] = m["rms_mu"]
        out[ds] = row
    return out, missing
