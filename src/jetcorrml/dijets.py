"""
The dijet pairs of the in-situ step (Sec. VII), the same for both methods.

The leading two jets of each event, back to back (dphi > pair_dphi_min), with the
third-jet veto pT3 / pT_pair < pair_pt3_frac, and at least one jet outside the
central region (a central+central pair says nothing about the relative scale).
jet1 is the inner (smaller |eta|) jet, jet2 the outer probe.

How a pass chooses its pairs (settings.Insitu.pair_rule):
  "both"       both jets above pair_pt_min, pair pT = their average (pass 1)
  "inner"      only the inner jet must pass, pair pT = its pT (pass 2)
  "inner-avg"  the inner jet must pass, pair pT = the average (passes 3 and 4)
With ``selection_pt`` (a file with the same events and jets: the pass before) the
selection and the pair's pT are taken from its jet pT; the balance (pt1, pt2)
always comes from ``path``.
"""

import awkward as ak
import numpy as np
import pandas as pd

from jetcorrml import settings

CUTS = settings.PAPER.insitu        # pair_pt_min, pair_dphi_min, pair_pt3_frac (the same in the reduced run)


def _event_scalar(field, default=0.0):
    padded = ak.pad_none(field, 1, axis=1)
    filled = ak.fill_none(padded, default)
    return ak.to_numpy(filled[:, 0]).astype(float)


def dijet_pairs(path, rule="both", selection_pt=None):
    """The pairs of one file: pt1, eta1, pt2, eta2, pt_avg (the pair's pT), the event's rho
    and the event's index (evt)."""
    matches = ak.from_parquet(path)
    n_jets = ak.num(matches.jet.pt)
    order = ak.argsort(-matches.jet.pt, axis=1)
    srt_pt = matches.jet.pt[order]
    srt_eta = matches.jet.eta[order]
    srt_phi = matches.jet.phi[order]

    def pad(a, n):
        return ak.pad_none(a, n, axis=1)

    def fill(a):
        return ak.to_numpy(ak.fill_none(a, 0.0))

    pt1 = fill(pad(srt_pt, 3)[:, 0])
    eta1 = fill(pad(srt_eta, 3)[:, 0])
    phi1 = fill(pad(srt_phi, 3)[:, 0])
    pt2 = fill(pad(srt_pt, 3)[:, 1])
    eta2 = fill(pad(srt_eta, 3)[:, 1])
    phi2 = fill(pad(srt_phi, 3)[:, 1])
    pt3 = fill(pad(srt_pt, 3)[:, 2])
    if selection_pt:
        sel_pt = ak.from_parquet(selection_pt).jet.pt
        if not bool(ak.all(ak.num(sel_pt) == n_jets)):
            raise SystemExit(f"--selection-pt {selection_pt}: not the same events and jets as {path}")
        sel_pt = sel_pt[order]
        spt1 = fill(pad(sel_pt, 3)[:, 0])
        spt2 = fill(pad(sel_pt, 3)[:, 1])
        spt3 = fill(pad(sel_pt, 3)[:, 2])
        print(f"    selection pT taken from {selection_pt}")
    else:
        spt1, spt2, spt3 = pt1, pt2, pt3

    rho_ev = _event_scalar(matches.event.rho)
    # the pair's pT: its pT_avg cell, the scale of the third-jet veto and the reference pT
    spt_inner = np.where(np.abs(eta1) <= np.abs(eta2), spt1, spt2)
    pt_avg = spt_inner if rule == "inner" else (spt1 + spt2) / 2.0

    dphi = np.abs(phi1 - phi2) % (2 * np.pi)
    dphi = np.where(dphi > np.pi, 2 * np.pi - dphi, dphi)

    has2 = ak.to_numpy(n_jets >= 2)
    if rule in ("inner", "inner-avg"):
        cut_pt = spt_inner > CUTS.pair_pt_min
    else:
        cut_pt = (spt1 > CUTS.pair_pt_min) & (spt2 > CUTS.pair_pt_min)
    cut_dphi = dphi > CUTS.pair_dphi_min
    cut_pt3 = np.where(pt_avg > 0, spt3 / pt_avg < CUTS.pair_pt3_frac, True)
    c1 = np.abs(eta1) < settings.CENTRAL_ETA
    c2 = np.abs(eta2) < settings.CENTRAL_ETA
    cut_topo = ~(c1 & c2)  # at least one jet forward

    sel = has2 & cut_pt & cut_dphi & cut_pt3 & cut_topo

    eta1_s = eta1[sel]
    eta2_s = eta2[sel]
    pt1_s = pt1[sel]
    pt2_s = pt2[sel]
    j1_is_inner = np.abs(eta1_s) <= np.abs(eta2_s)

    df = pd.DataFrame(
        {
            "pt1": np.where(j1_is_inner, pt1_s, pt2_s),
            "eta1": np.where(j1_is_inner, eta1_s, eta2_s),
            "pt2": np.where(j1_is_inner, pt2_s, pt1_s),
            "eta2": np.where(j1_is_inner, eta2_s, eta1_s),
            "pt_avg": pt_avg[sel],
            "rho": rho_ev[sel],
            "evt": np.where(sel)[0],          # the event, for the resampling of a retraining
        }
    )
    print(f"    -> {len(df):,} pairs (|eta1| <= |eta2|, rule: {rule})")
    return df
