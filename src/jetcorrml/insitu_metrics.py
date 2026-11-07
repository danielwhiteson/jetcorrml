"""
The agreement of corrected data with simulation in the in-situ section (Table IV, Figs. 16, 17).

The in-situ files are read directly (jet/gen/event/deltaR), flattened to one entry
per jet.  Every binned number is population-gated: bins below ``min_count`` jets
are left empty.  The same functions score both methods.
"""
from __future__ import annotations

import awkward as ak
import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq
from scipy.stats import ks_2samp

from jetcorrml import settings

# ---- binning (shared so Page 1 and Page 4 stack on identical axes) ----------
# CANONICAL SCORING GRID -- must match what the NN loss/bridge optimise and the
# in-run terminal verdict dump (steps.insitu_nn).  gen-pT edges mirror
# the deployable --qm-pt-edges (30,60,100,200) with a sample floor (20, parton
# cut) and a finite cap (300); |eta| edges mirror --qm-eta-edges (0.5) + the
# --zjet-eta-max central boundary (0.8) + --dijet-eta-fwd-edges (1.5,2.0).
# If you change the run config, change these to match or the NN-vs-parametric
# comparison is scored on a different grid than the model was trained on.
PT_EDGES = np.array([20, 30, 60, 100, 200, 300], dtype=float)
ABSETA_EDGES = np.array([0.0, 0.5, 0.8, 1.5, 2.0, 2.5])
RHO_EDGES = np.linspace(0.0, 40.0, 11)  # 4 GeV/area bins covering the
CENTRAL = settings.CENTRAL_ETA   # |eta| < CENTRAL defines the central reference region


# ----------------------------------------------------------------------------
#  IO: flatten the nested per-event parquet to flat per-jet numpy arrays
# ----------------------------------------------------------------------------
def flatten_insitu(path: str) -> dict[str, np.ndarray]:
    """Return flat per-jet arrays from a nested in-situ output (or None)."""
    if not path:
        return None
    table = pq.read_table(path)
    flat = pc.list_flatten(table.column(0).combine_chunks())

    def leaf(dotted: str) -> np.ndarray:
        a = flat
        for key in dotted.split("."):
            a = pc.struct_field(a, key)
        return np.asarray(a)

    out = {
        "jet_pt": leaf("jet.pt").astype(float),
        "jet_eta": leaf("jet.eta").astype(float),
        "gen_pt": leaf("gen.pt").astype(float),
        "gen_eta": leaf("gen.eta").astype(float),
        "rho": leaf("event.rho").astype(float),
        "deltaR": leaf("deltaR").astype(float),
    }
    out["abs_eta"] = np.abs(out["jet_eta"])
    out["response"] = out["jet_pt"] / out["gen_pt"]
    return out


def apply_selection(d: dict, gen_pt_min: float, dr_max: float) -> dict:
    if d is None:
        return None
    m = (d["gen_pt"] >= gen_pt_min) & (d["deltaR"] < dr_max) & np.isfinite(d["response"])
    return {k: v[m] for k, v in d.items()}


# ----------------------------------------------------------------------------
#  small statistics helpers
# ----------------------------------------------------------------------------
def profile_median(x, y, edges, min_count):
    """Median of y in bins of x. Returns centres, median, n, approx SE (masked)."""
    idx = np.digitize(x, edges) - 1
    nb = len(edges) - 1
    centres = 0.5 * (edges[:-1] + edges[1:])
    med = np.full(nb, np.nan)
    n = np.zeros(nb, dtype=int)
    se = np.full(nb, np.nan)
    for b in range(nb):
        yy = y[idx == b]
        n[b] = yy.size
        if yy.size >= min_count:
            med[b] = np.median(yy)
            # SE of the median ~ 1.2533 * std / sqrt(N)
            se[b] = 1.2533 * np.std(yy) / np.sqrt(yy.size)
    return centres, med, n, se


def r_ratio_1d(mc, data, key, edges, min_count, central=False):
    """R_data / R_mc as a 1-D profile over `key` (marginalising other vars).

    R = response = reco/gen = jet/ref (ref is the gen jet in this study, so this
    is the ATLAS-style data/MC response double ratio).  Returns (centres,
    ratio, ok_mask) with ratio = NaN where either side is sparse.

    If `central` is True, both samples are first restricted to |eta| < CENTRAL.
    Use this for the pT projection so it matches the diagnose triptych's
    central "vs pT" pane (the standard JES response-vs-pT reference region);
    leave it False for the |eta| / rho projections, whose purpose is to expose
    the forward / rho dependence.
    """
    if central:
        cm = mc["abs_eta"] < CENTRAL
        cd = data["abs_eta"] < CENTRAL
        mc   = {k: v[cm] for k, v in mc.items()}
        data = {k: v[cd] for k, v in data.items()}
    xc, mm, mn, _ = profile_median(mc[key],   mc["response"],   edges, min_count)
    _,  dm, dn, _ = profile_median(data[key], data["response"], edges, min_count)
    ok = (mn >= min_count) & (dn >= min_count) & np.isfinite(mm) & np.isfinite(dm)
    ratio = np.full(len(xc), np.nan)
    ratio[ok] = dm[ok] / mm[ok]
    return xc, ratio, ok


def max_r_dev_1d(mc, data, key, edges, min_count, central=False):
    """max |R_data/R_mc - 1| over 1-D bins of `key`, or None if all sparse."""
    _, ratio, ok = r_ratio_1d(mc, data, key, edges, min_count, central=central)
    if not ok.any():
        return None
    return float(np.abs(ratio[ok] - 1.0).max())


# ─────────────────────────── the table's metrics ───────────────────────────
MIN_COUNT = 50
GEN_PT_MIN = 0.0         # all jets
DR_MAX = settings.MATCH_DR
# The pair selection of the asymmetry test: its own (the in-situ training uses 2.5 and 0.25).
DIJET_PT_MIN     = 25.0
DIJET_DPHI_MIN   = 2.7
DIJET_PT3_FRAC   = 0.3
DIJET_ETA_CTRL   = settings.CENTRAL_ETA

# Forward grids for the new metrics (forward-closure split + asymmetry-vs-eta).
# gen_pt is fine at low pT (the corner); upper edges match the canonical metric.
FWD_ETA_EDGES = [0.8, 1.5, 2.0, 2.5]


def dijet_asymmetry(division_path, insitu_path=None):
    """Dijet asymmetry with the pairs selected AFTER correction and the pT cut on
    the central jet only.

    The two leading jets are found in the pT that the sample is judged in: the
    in-situ-corrected pT when `insitu_path` is given (corrected data), otherwise the
    pT in `division_path` itself (MC, or uncorrected data).  Exactly one of the two
    must be central (|eta| < DIJET_ETA_CTRL); the central jet must have
    pT > DIJET_PT_MIN; there is NO pT cut on the forward jet; the pair must be
    back-to-back and the third jet below DIJET_PT3_FRAC of the CENTRAL jet's pT.
        A = (pT_central - pT_forward) / (0.5 * (pT_central + pT_forward))
    with every pT taken from the same (corrected) source.  Nothing in the
    selection depends on the forward jet's pT, so a mis-scaled forward jet cannot
    move itself in or out of the sample; the test is minimal for the true
    correction.  Returns (A, |eta| of the forward jet)."""
    raw = ak.from_parquet(str(division_path))
    use_pt = ak.from_parquet(str(insitu_path)).jet.pt if insitu_path is not None else raw.jet.pt
    n_jets = ak.num(use_pt)
    order = ak.argsort(-use_pt, axis=1)

    def col(a, i):
        return ak.to_numpy(ak.fill_none(ak.pad_none(a[order], 3, axis=1), 0.0)[:, i]).astype(float)

    pt1, pt2, pt3 = col(use_pt, 0), col(use_pt, 1), col(use_pt, 2)
    eta1, eta2 = col(raw.jet.eta, 0), col(raw.jet.eta, 1)
    phi1, phi2 = col(raw.jet.phi, 0), col(raw.jet.phi, 1)
    dphi = np.abs(phi1 - phi2) % (2 * np.pi)
    dphi = np.where(dphi > np.pi, 2 * np.pi - dphi, dphi)
    c1 = np.abs(eta1) < DIJET_ETA_CTRL
    c2 = np.abs(eta2) < DIJET_ETA_CTRL
    pt_c = np.where(c1, pt1, pt2)
    pt_f = np.where(c1, pt2, pt1)
    sel = (ak.to_numpy(n_jets >= 2) & (pt2 > 0)
           & ((c1 & ~c2) | (~c1 & c2))
           & (pt_c > DIJET_PT_MIN)
           & (dphi > DIJET_DPHI_MIN)
           & (pt3 / np.where(pt_c > 0, pt_c, 1.0) < DIJET_PT3_FRAC))
    if not sel.any():
        return np.array([], dtype=float), np.array([], dtype=float)
    abseta_fwd = np.where(c1, np.abs(eta2), np.abs(eta1))[sel]
    return (pt_c[sel] - pt_f[sel]) / (0.5 * (pt_c[sel] + pt_f[sel])), abseta_fwd


def dijet_selected_forward(division_path, insitu_path):
    """Per-forward-jet (|eta|, gen_pt, corrected response) for jets entering the
    FIXED dijet selection.  BLACK BOX / method-agnostic: the selection, the jet
    ordering, and gen_pt all come from the SHARED inputs; the ONLY method-
    dependent input is `insitu_path` (the corrected jet pT).  Uses the OR cut
    (>=1 forward jet) -- the topology both the NN and the parametric chain actually fit
    on -- and returns every jet of a selected pair that is forward (|eta|>=0.8):

        response = (corrected pT from the in-situ step) / (gen pT)

    For MC pass the MC in-situ file (uncorrected == reference); identical code
    path.  Returns dict {abs_eta, gen_pt, response} (possibly empty)."""
    empty = {k: np.array([], dtype=float) for k in ("abs_eta", "gen_pt", "response")}
    raw = ak.from_parquet(str(division_path))
    n_jets = ak.num(raw.jet.pt)
    order = ak.argsort(-raw.jet.pt, axis=1)
    srt_pt, srt_eta, srt_phi = raw.jet.pt[order], raw.jet.eta[order], raw.jet.phi[order]

    def col(a, i):
        return ak.to_numpy(ak.fill_none(ak.pad_none(a, 3, axis=1), 0.0)[:, i])

    pt1, pt2, pt3 = col(srt_pt, 0), col(srt_pt, 1), col(srt_pt, 2)
    eta1, eta2 = col(srt_eta, 0), col(srt_eta, 1)
    phi1, phi2 = col(srt_phi, 0), col(srt_phi, 1)
    dphi = np.abs(phi1 - phi2) % (2 * np.pi)
    dphi = np.where(dphi > np.pi, 2 * np.pi - dphi, dphi)
    pt_avg = 0.5 * (pt1 + pt2)
    pt3_frac = pt3 / np.where(pt_avg > 0, pt_avg, 1.0)   # avoid 0/0 warning
    c1 = np.abs(eta1) < DIJET_ETA_CTRL
    c2 = np.abs(eta2) < DIJET_ETA_CTRL
    sel = (ak.to_numpy(n_jets >= 2)
           & (pt1 > DIJET_PT_MIN) & (pt2 > DIJET_PT_MIN)
           & (dphi > DIJET_DPHI_MIN)
           & (pt_avg > 0) & (pt3_frac < DIJET_PT3_FRAC)
           & ~(c1 & c2))   # OR cut: at least one forward jet
    if not sel.any():
        return empty
    s5 = ak.from_parquet(str(insitu_path))
    ptc1, ptc2 = col(s5.jet.pt[order], 0), col(s5.jet.pt[order], 1)
    g1, g2 = col(s5.gen.pt[order], 0), col(s5.gen.pt[order], 1)
    f1 = sel & (np.abs(eta1) >= DIJET_ETA_CTRL)   # leading jet forward
    f2 = sel & (np.abs(eta2) >= DIJET_ETA_CTRL)   # sub-leading jet forward
    abseta = np.concatenate([np.abs(eta1[f1]), np.abs(eta2[f2])])
    genpt = np.concatenate([g1[f1], g2[f2]])
    ptcorr = np.concatenate([ptc1[f1], ptc2[f2]])
    good = genpt > 5.0   # require a valid gen match
    return {
        "abs_eta": abseta[good],
        "gen_pt": genpt[good],
        "response": ptcorr[good] / np.maximum(genpt[good], 1e-6),
    }


def cell_data(division_mc, division_data, insitu_mc, insitu_data, gen_pt_min, dr_max):
    """Bundle all per-cell data needed for the four metrics and the PDF plots.
    Returns None if either in-situ file is missing.

    `division_mc` / `division_data` are the *input* files of the in-situ step (post-
    distortion, pre-correction).  The dijet-asymmetry metric needs them for the
    jet directions and, for MC and uncorrected data, the pT; see
    dijet_asymmetry().  Without them the asymmetry is left empty.
    """
    if not insitu_mc.exists() or not insitu_data.exists():
        return None
    _empty = np.array([], dtype=float)
    a_mc, a_eta_mc = (dijet_asymmetry(division_mc, insitu_path=None)
                      if division_mc.exists() else (_empty, _empty))
    a_data, a_eta_data = (dijet_asymmetry(division_data, insitu_path=insitu_data)
                          if division_data.exists() else (_empty, _empty))
    # Uncorrected ("raw") data: the shared division, before any in-situ correction
    # correction.  Method-independent -- one curve giving the pre-correction
    # baseline so the plots show how far each correction travels.
    a_data_raw, a_eta_data_raw = (dijet_asymmetry(division_data, insitu_path=None)
                                  if division_data.exists() else (_empty, _empty))
    return {
        "mc":     apply_selection(flatten_insitu(str(insitu_mc)),   gen_pt_min, dr_max),
        "data":   apply_selection(flatten_insitu(str(insitu_data)), gen_pt_min, dr_max),
        "data_raw": (apply_selection(flatten_insitu(str(division_data)), gen_pt_min, dr_max)
                     if division_data.exists() else None),
        "a_mc":   a_mc,
        "a_data": a_data,
        "a_data_raw": a_data_raw,           # uncorrected data asymmetry (raw-raw)
        "a_eta_mc":   a_eta_mc,    # forward |eta| per asymmetry pair (shared raw
        "a_eta_data": a_eta_data,  # selection; identical NN vs parametric)
        "a_eta_data_raw": a_eta_data_raw,
        # dijet-SELECTED forward jets (black box: selection from the shared division,
        # corrected pT from this method's in-situ output).  None if the division is missing.
        "djf_mc":   (dijet_selected_forward(division_mc, insitu_mc)
                     if division_mc.exists() else None),
        "djf_data": (dijet_selected_forward(division_data, insitu_data)
                     if division_data.exists() else None),
    }


def metrics_from(cd):
    """Extract the 5 scalar metrics from a cell_data() dict (or all-None).

    Returns (KS_r, KS_a, maxdev_pt, maxdev_eta, maxdev_rho) where:
      KS_r        = KS(corr, MC) on global response distribution
      KS_a        = KS(corr, MC) on dijet asymmetry distribution
      maxdev_pt   = max |R_data/R_mc - 1| over 1-D gen_pt bins, restricted to
                    |eta| < CENTRAL (matches the diagnose triptych central
                    "vs pT" pane).  R = reco/gen = jet/ref; this is the
                    ATLAS-style data/MC response double ratio.
      maxdev_eta  = max |R_data/R_mc - 1| over 1-D |eta| bins (all pT; exposes
                    the forward residual)
      maxdev_rho  = max |R_data/R_mc - 1| over 1-D rho  bins (all pT, all |eta|)
    """
    if cd is None:
        return (None, None, None, None, None)
    ks_a = (float(ks_2samp(cd["a_data"], cd["a_mc"]).statistic)
            if cd["a_mc"].size and cd["a_data"].size else None)
    return (
        float(ks_2samp(cd["data"]["response"], cd["mc"]["response"]).statistic),
        ks_a,
        max_r_dev_1d(cd["mc"], cd["data"], "gen_pt",  PT_EDGES,     MIN_COUNT, central=True),
        max_r_dev_1d(cd["mc"], cd["data"], "abs_eta", ABSETA_EDGES, MIN_COUNT),
        max_r_dev_1d(cd["mc"], cd["data"], "rho",     RHO_EDGES,    MIN_COUNT),
    )
