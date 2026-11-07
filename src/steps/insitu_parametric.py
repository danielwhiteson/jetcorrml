"""
The parametric in-situ correction (Sec. VII B), the baseline to the NN in situ.
================================================================================
Two steps, as in the ATLAS in-situ chain:
  1. eta intercalibration : matrix method on the reco dijet balance, per pT_avg
                            bin, with the central region |eta| < 0.8 fixed to 1
  2. absolute scale       : MC/data double ratio of the central response vs pT,
                            smoothed along pT

The corrections are derived against the reference pT and applied to each data
jet by numerical inversion (the look-up is made at the corrected pT and
iterated), so no truth enters at application.  The MC sample is copied as is.

Usage
-----
uv run python -m steps.insitu_parametric \\
    --mc   data/division/lo_rho05/quad/0_MC.pq \\
    --data data/division/lo_rho05/quad/0_DATA.pq \\
    --output-mc data/insitu/parametric/pass1/lo_rho05/quad/0_MC.pq \\
    --output-data data/insitu/parametric/pass1/lo_rho05/quad/0_DATA.pq \\
    [--pair-selection inner --selection-pt <pass-1 DATA parquet>]
"""

import argparse
import os
import warnings

import awkward as ak
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

from jetcorrml import settings
from jetcorrml.dijets import dijet_pairs

warnings.filterwarnings("ignore")   # empty pT bins give all-NaN slices

# ─────────────────────────────────────────────────────────────────────────────
# 1.  Binning and fixed settings
# ─────────────────────────────────────────────────────────────────────────────

ETA_EDGES = np.array(
    [
        -4.5,
        -3.2,
        -2.8,
        -2.4,
        -2.1,
        -1.8,
        -1.5,
        -1.2,
        -0.8,
        -0.5,
        -0.2,
        0.0,
        0.2,
        0.5,
        0.8,
        1.2,
        1.5,
        1.8,
        2.1,
        2.4,
        2.8,
        3.2,
        4.5,
    ]
)
ETA_CENTERS = 0.5 * (ETA_EDGES[:-1] + ETA_EDGES[1:])
N_ETA = len(ETA_CENTERS)
ABS_ETA_EDGES = np.unique(np.abs(ETA_EDGES))

PT_EDGES = np.array(
    [17, 25, 35, 45, 60, 80, 110, 160, 210, 300, 400, 600, 800, 1200], dtype=float
)
PT_CENTERS = 0.5 * (PT_EDGES[:-1] + PT_EDGES[1:])
N_PT = len(PT_CENTERS)

ETA_CENTRAL = settings.CENTRAL_ETA        # |eta| below which the intercalibration is fixed to 1
PT_AVG_EDGES = [40.0, 80.0]  # interior pT_avg edges of the intercalibration (3 bins)
MIN_PAIRS = 200             # pairs needed in a (|eta1|, |eta2|, pT_avg) cell
SIGMA_PT = 1.0              # Gaussian smoothing of the absolute scale, in pT bins
N_INVERSION = 10            # iterations of the look-up at the corrected pT

# ─────────────────────────────────────────────────────────────────────────────
# 2.  Data loading
# ─────────────────────────────────────────────────────────────────────────────


def load_parquet(path):
    """Load a matched-pair parquet and return flat pandas DataFrame."""
    matches = ak.from_parquet(path)

    def flat(f):
        return ak.to_numpy(ak.flatten(f, axis=None)).astype(float)

    df = pd.DataFrame(
        {
            "jet_pt": flat(matches.jet.pt),
            "jet_eta": flat(matches.jet.eta),
            "gen_pt": flat(matches.gen.pt),
        }
    )
    df["response"] = df["jet_pt"] / df["gen_pt"]
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Look-up helpers
# ─────────────────────────────────────────────────────────────────────────────


def lookup_correction(eta_val, pt_val, corr_grid, eta_edges, pt_edges):
    """Nearest-bin look-up. Clips to grid boundaries, substitutes 1 for NaN."""
    i = np.clip(np.searchsorted(eta_edges, eta_val) - 1, 0, corr_grid.shape[0] - 1)
    j = np.clip(np.searchsorted(pt_edges, pt_val) - 1, 0, corr_grid.shape[1] - 1)
    c = corr_grid[i, j]
    return np.where(np.isnan(c), 1.0, c)


def apply_eta_correction(df, corr_grid, eta_edges, pt_edges, pt_col):
    """Apply the eta intercalibration to every jet in df, looked up at
    ``pt_col``.  The correction multiplies jet_pt."""
    c = lookup_correction(
        df["jet_eta"].values, df[pt_col].values, corr_grid, eta_edges, pt_edges
    )
    df = df.copy()
    df["jet_pt_etacorr"] = df["jet_pt"] * c
    df["response_etacorr"] = df["jet_pt_etacorr"] / df["gen_pt"]
    return df


def response_1d(df, pt_edges, pt_col, col="response"):
    """Mean of ``col`` in bins of ``pt_col``.  Returns (R, N)."""
    pt = df[pt_col].values
    r = df[col].values
    w = np.ones(len(df))
    R = np.full(len(pt_edges) - 1, np.nan)
    N = np.zeros(len(pt_edges) - 1)
    for j in range(len(pt_edges) - 1):
        m = (pt >= pt_edges[j]) & (pt < pt_edges[j + 1])
        sw = w[m].sum()
        if sw > 0:
            R[j] = np.average(r[m], weights=w[m])
            N[j] = sw
    return R, N


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Dijet eta intercalibration (reco only)
# ─────────────────────────────────────────────────────────────────────────────


def _solve_intercalib(cells, n_bins, central_mask):
    """Weighted least-squares solve of x_b - x_a = delta over dijet cells, with
    central bins (central_mask True) fixed to x=0.  The central anchor
    propagates outward through the pair connectivity (forward+forward links
    included).  ``cells``: list of (a, b, delta, weight)."""
    x = np.zeros(n_bins)
    free = np.where(~central_mask)[0]
    if len(free) == 0 or len(cells) == 0:
        return x
    pos = {b: i for i, b in enumerate(free)}
    rows, rhs, wts = [], [], []
    for a, b, delta, w in cells:
        row = np.zeros(len(free))
        if not central_mask[b]:
            row[pos[b]] += 1.0
        if not central_mask[a]:
            row[pos[a]] -= 1.0
        if not np.any(row):
            continue  # both central -- excluded by the pair selection anyway
        rows.append(row)
        rhs.append(delta)
        wts.append(w)
    if not rows:
        return x
    M, d, sw = np.array(rows), np.array(rhs), np.sqrt(np.array(wts))
    sol, *_ = np.linalg.lstsq(M * sw[:, None], d * sw, rcond=None)
    x[free] = sol
    return x


def derive_eta_intercalibration(dj_mc, dj_data):
    """Matrix-method eta intercalibration from the reco dijet balance.

    Per pT_avg bin, solve for per-|eta|-bin log-corrections x(|eta|) so that the
    corrected data dijet log-balance matches the simulation's, with the central
    region fixed (x=0 -> F=1); its scale is set by the absolute step.  The cell
    equation is ``x_b - x_a = <log(pt2/pt1)>_mc - <log(pt2/pt1)>_data``
    (jet1 = inner a, jet2 = outer b), √N_data-weighted.  Cells with a
    non-finite log balance (a jet at pT <= 0) leave the pair out.  Returns
    F (n_abs_eta, n_ptavg), the correction to apply to data jets."""
    n_ae = len(ABS_ETA_EDGES) - 1
    centers = 0.5 * (ABS_ETA_EDGES[:-1] + ABS_ETA_EDGES[1:])
    central_mask = centers < ETA_CENTRAL
    pa_edges = np.asarray(PT_AVG_EDGES, dtype=float)
    n_pa = len(pa_edges) + 1

    def assign(dj):
        a = np.clip(np.searchsorted(ABS_ETA_EDGES, np.abs(dj["eta1"].values)) - 1, 0, n_ae - 1)
        b = np.clip(np.searchsorted(ABS_ETA_EDGES, np.abs(dj["eta2"].values)) - 1, 0, n_ae - 1)
        pa = np.clip(np.searchsorted(pa_edges, dj["pt_avg"].values), 0, n_pa - 1)
        with np.errstate(divide="ignore", invalid="ignore"):
            ell = np.log(dj["pt2"].values / dj["pt1"].values)
        return a, b, pa, ell

    a_mc, b_mc, pa_mc, l_mc = assign(dj_mc)
    a_da, b_da, pa_da, l_da = assign(dj_data)
    F = np.ones((n_ae, n_pa))

    for pj in range(n_pa):
        def agg(a, b, pa, ell):
            m = (pa == pj) & np.isfinite(ell)
            cid = a[m] * n_ae + b[m]
            s = np.zeros(n_ae * n_ae)
            c = np.zeros(n_ae * n_ae)
            np.add.at(s, cid, ell[m])
            np.add.at(c, cid, 1.0)
            return s, c

        s_mc, c_mc = agg(a_mc, b_mc, pa_mc, l_mc)
        s_da, c_da = agg(a_da, b_da, pa_da, l_da)
        cells = []
        for cid in range(n_ae * n_ae):
            a, b = divmod(cid, n_ae)
            if a == b or c_mc[cid] < MIN_PAIRS or c_da[cid] < MIN_PAIRS:
                continue
            delta = s_mc[cid] / c_mc[cid] - s_da[cid] / c_da[cid]  # = x_b - x_a
            cells.append((a, b, delta, np.sqrt(c_da[cid])))
        F[:, pj] = np.exp(_solve_intercalib(cells, n_ae, central_mask))
    return F


def eta_corr_grid_from_F(F):
    """Map F(|eta|, pT_avg) onto the signed (eta, pT) grid of the look-up.

    A jet's pT picks the pT_avg column (pT_avg ~ jet pT for balanced dijets),
    its |eta| the row."""
    grid = np.ones((N_ETA, N_PT))
    pa_edges = np.asarray(PT_AVG_EDGES, dtype=float)
    n_pa = F.shape[1]
    for i, ec in enumerate(ETA_CENTERS):
        ae = int(np.clip(np.searchsorted(ABS_ETA_EDGES, abs(ec)) - 1, 0, F.shape[0] - 1))
        for j, pc in enumerate(PT_CENTERS):
            pj = int(np.clip(np.searchsorted(pa_edges, pc), 0, n_pa - 1))
            grid[i, j] = F[ae, pj]
    return grid


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Calibration
# ─────────────────────────────────────────────────────────────────────────────


def run_calibration(df_mc, df_data, dj_mc, dj_data):
    """Derive both corrections against the reference pT.

    Returns (corr_eta_smooth (N_ETA, N_PT), corr_abs_smooth (N_PT,))."""
    print("Step 1: eta intercalibration...")
    F_eta = derive_eta_intercalibration(dj_mc, dj_data)
    corr_eta_smooth = eta_corr_grid_from_F(F_eta)
    print(
        f"  Dijet intercalibration solved over {len(ABS_ETA_EDGES) - 1} |eta| "
        f"bins x {len(PT_AVG_EDGES) + 1} pT_avg bins "
        f"(central |eta|<{ETA_CENTRAL} fixed to 1)."
    )

    df_data_corr = apply_eta_correction(
        df_data, corr_eta_smooth, ETA_EDGES, PT_EDGES, pt_col="gen_pt"
    )
    df_mc_corr = df_mc.copy()
    df_mc_corr["response_etacorr"] = df_mc_corr["response"]

    print("Step 2: absolute calibration...")
    central_mc = df_mc_corr[np.abs(df_mc_corr["jet_eta"]) < ETA_CENTRAL].copy()
    central_data = df_data_corr[np.abs(df_data_corr["jet_eta"]) < ETA_CENTRAL].copy()
    central_mc["response"] = central_mc["response_etacorr"]
    central_data["response"] = central_data["response_etacorr"]

    R_mc_abs, _ = response_1d(central_mc, PT_EDGES, pt_col="gen_pt")
    R_data_abs, _ = response_1d(central_data, PT_EDGES, pt_col="gen_pt")
    double_ratio_abs = R_mc_abs / R_data_abs

    valid = ~np.isnan(double_ratio_abs)
    dr_filled = np.where(valid, double_ratio_abs, np.nanmedian(double_ratio_abs))
    corr_abs_smooth = gaussian_filter1d(dr_filled, sigma=SIGMA_PT)
    corr_abs_smooth[~valid] = np.nan
    return corr_eta_smooth, corr_abs_smooth


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Parquet export (MC unchanged; data jets scaled by eta x absolute)
# ─────────────────────────────────────────────────────────────────────────────


def _per_jet_scale_regular(jet_eta, jet_pt, corr_eta_smooth, corr_abs_smooth):
    """Combined eta x absolute scale of each data jet.

    The corrections were derived against the reference pT, and the corrected
    pT estimates it, so solve s = c(eta, pT_reco * s) by iterating the
    nearest-bin look-up.  Uses the jet's reco pT and eta only."""
    s = np.ones(len(jet_pt), dtype=float)
    for _ in range(N_INVERSION):
        p_ref = jet_pt * s
        c_eta = lookup_correction(jet_eta, p_ref, corr_eta_smooth, ETA_EDGES, PT_EDGES)
        idx = np.clip(np.searchsorted(PT_EDGES, p_ref) - 1, 0, len(corr_abs_smooth) - 1)
        s = c_eta * np.where(np.isnan(corr_abs_smooth[idx]), 1.0, corr_abs_smooth[idx])
    return s


def write_insitu(
    mc_path, data_path, output_mc, output_data, corr_eta_smooth, corr_abs_smooth
):
    """Write the MC parquet (a copy) and the corrected data parquet."""
    parent = os.path.dirname(output_mc)
    if parent:
        os.makedirs(parent, exist_ok=True)
    ak.to_parquet(ak.from_parquet(mc_path), output_mc)
    print(f"\n  Saved MC parquet: {output_mc}")

    parent = os.path.dirname(output_data)
    if parent:
        os.makedirs(parent, exist_ok=True)
    matches = ak.from_parquet(data_path)
    jet_pt = ak.flatten(matches.jet.pt)
    jet_eta = ak.flatten(matches.jet.eta)
    jet_E = ak.flatten(matches.jet.E)
    scales = _per_jet_scale_regular(
        ak.to_numpy(jet_eta).astype(float),
        ak.to_numpy(jet_pt).astype(float),
        corr_eta_smooth,
        corr_abs_smooth,
    )
    pt_corr = ak.to_numpy(jet_pt).astype(float) * scales
    E_corr = ak.to_numpy(jet_E).astype(float) * scales
    counts = ak.num(matches.jet.pt)
    data_fields = {
        "jet": ak.zip(
            {
                "pt": ak.unflatten(pt_corr, counts),
                "eta": matches.jet.eta,
                "phi": matches.jet.phi,
                "mass": matches.jet.mass,
                "E": ak.unflatten(E_corr, counts),
            }
        ),
        "gen": matches.gen,
        "event": matches.event,
        "deltaR": matches.deltaR,
    }
    ak.to_parquet(ak.zip(data_fields), output_data)
    print(f"  Saved corrected Data parquet: {output_data}")


# ─────────────────────────────────────────────────────────────────────────────
# 7.  CLI
# ─────────────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Parametric in-situ correction: eta intercalibration + absolute scale.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--mc", "-m", required=True, metavar="FILE", help="MC parquet file")
    parser.add_argument("--data", "-d", required=True, metavar="FILE", help="Data parquet file")
    parser.add_argument(
        "--pair-selection", choices=["both", "inner"], default="both",
        help="How the data dijet pairs are chosen: 'both' (pass 1) both jets above "
        "the pT cut, pair pT = their average; 'inner' (pass 2) only the inner jet "
        "must pass, pair pT = its pT.  'inner' needs --selection-pt.",
    )
    parser.add_argument(
        "--selection-pt", default=None, metavar="FILE",
        help="Parquet with the same events and jets as --data (the pass-1 output); "
        "its jet pT chooses the data pairs and gives the pair's pT.  The balance "
        "always uses the pT in --data.",
    )
    parser.add_argument("--output-mc", required=True, metavar="PATH", help="MC parquet path")
    parser.add_argument("--output-data", required=True, metavar="PATH", help="DATA parquet path")
    args = parser.parse_args()

    print(f"Loading MC  : {args.mc}")
    df_mc = load_parquet(args.mc)
    print(f"Loading data: {args.data}")
    df_data = load_parquet(args.data)
    print(f"  MC rows  : {len(df_mc):,}")
    print(f"  Data rows: {len(df_data):,}")

    if args.pair_selection == "inner" and not args.selection_pt:
        raise SystemExit("--pair-selection inner needs --selection-pt "
                         "(a first-pass in-situ output of the data).")
    print("Building reco dijet pairs (MC)…")
    dj_mc = dijet_pairs(args.mc, rule=args.pair_selection)
    print("Building reco dijet pairs (data)…")
    dj_data = dijet_pairs(args.data, rule=args.pair_selection, selection_pt=args.selection_pt)

    corr_eta_smooth, corr_abs_smooth = run_calibration(df_mc, df_data, dj_mc, dj_data)

    write_insitu(
        args.mc,
        args.data,
        args.output_mc,
        args.output_data,
        corr_eta_smooth,
        corr_abs_smooth,
    )


if __name__ == "__main__":
    main()
