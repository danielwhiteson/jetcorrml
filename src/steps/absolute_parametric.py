"""
Parametric absolute energy scale, after ATLAS (Sec. V B of the paper).

  1. Profile the response R = E_reco / E_true in bins of (E_true, |eta|), as
     the mean of an iterative Gaussian fit to its core (the ATLAS definition).
  2. Fit a smoothing spline to R(log E_true) per |eta| slice to filter noise.
  3. Enforce monotonicity by PCHIP interpolation of the smoothed points.
  4. Numerical inversion, as for the NN absolute scale: a 2000-point E_true
     grid, E_reco_grid = E_true * R made strictly increasing, and the lookup
     table E_reco -> 1/R.
  5. Apply: the energy is scaled, and pT = sqrt(E_corr^2 - m^2) / cosh(eta).

    uv run python -m steps.absolute_parametric \
        -i data/pileup/nn/lo_rho01.pq --output data/absolute/parametric-cross/lo_rho01.pq
"""

import argparse

import numpy as np
from scipy.interpolate import PchipInterpolator, UnivariateSpline

from jetcorrml import settings
from jetcorrml.absolute_closure import closure_floor
from jetcorrml.jets import absolute_columns, pt_at_fixed_mass, read_jets, write_jets
from jetcorrml.response import gaussian_core_peak

DR_MAX = settings.MATCH_DR          # matching distance: every pair as written


def derive_calibration(data, n_e_bins=40, min_count=50):
    """Derive JES via smoothing splines + monotonic PCHIP + numerical inversion.

    Quality cuts (those of the NN absolute scale):
      - deltaR < DR_MAX
      - E_gen >= E_GEN_MIN(eta)   (avoids reco pT threshold bias)
      - 0.05 < response < 3.0     (removes catastrophic mis-matches)
    """
    e_truth = data["e_gen"]
    e_reco = data["e_reco"]
    eta = data["eta_det"]
    dr = data["dr"]
    resp = e_reco / np.maximum(e_truth, 1.0)

    E_GEN_MIN = closure_floor(eta)
    quality = (resp > 0.05) & (resp < 3.0) & (e_truth >= E_GEN_MIN) & (dr < DR_MAX)
    n_total, n_pass = len(quality), quality.sum()
    print(
        f"  Quality cut: keeping {n_pass:,} / {n_total:,} jets ({100 * n_pass / n_total:.1f}%)"
    )

    eta_edges = np.array(settings.ABSOLUTE_ETA_EDGES)
    n_eta_bins = len(eta_edges) - 1
    abs_eta = np.abs(eta)

    inv_tables = []

    for i in range(n_eta_bins):
        eta_mask = (abs_eta >= eta_edges[i]) & (abs_eta < eta_edges[i + 1])
        mask = eta_mask & quality

        if mask.sum() < min_count * 10:
            inv_tables.append(None)
            continue

        e_lo = max(e_truth[mask].min(), 15.0)
        e_hi = e_truth[mask].max()
        e_edges = np.geomspace(e_lo, e_hi, n_e_bins + 1)
        pts_truth, pts_resp, pts_err = [], [], []

        for j in range(n_e_bins):
            m = mask & (e_truth >= e_edges[j]) & (e_truth < e_edges[j + 1])
            if m.sum() >= min_count:
                pts_truth.append(np.sqrt(e_edges[j] * e_edges[j + 1]))
                mu, err = gaussian_core_peak(resp[m])
                pts_resp.append(mu)
                pts_err.append(err)

        if len(pts_truth) < 5:
            inv_tables.append(None)
            continue

        idx = np.argsort(pts_truth)
        x_fit = np.log(np.array(pts_truth)[idx])
        y_fit = np.array(pts_resp)[idx]
        w_fit = 1.0 / np.array(pts_err)[idx]

        # Step 1: smoothing spline to filter noise
        smooth_spline = UnivariateSpline(x_fit, y_fit, w=w_fit, s=None, k=3)

        # Step 2: monotonic PCHIP on smoothed points
        x_fine = np.linspace(x_fit.min(), x_fit.max(), 100)
        y_fine = smooth_spline(x_fine)
        monotonic_fit = PchipInterpolator(x_fine, y_fine)

        # Step 3: numerical inversion (as for the NN absolute scale)
        n_grid = 2000
        log_e_grid = np.linspace(x_fit.min(), x_fit.max(), n_grid)
        t_grid = np.exp(log_e_grid)
        r_grid = monotonic_fit(log_e_grid)
        r_grid = np.clip(r_grid, 0.01, None)

        reco_grid = t_grid * r_grid
        scale_grid = 1.0 / r_grid

        # Enforce monotonicity: E_reco must be strictly increasing
        reco_mono = np.maximum.accumulate(reco_grid)
        good_mask = reco_grid == reco_mono
        if not good_mask.all() and good_mask.sum() >= 2:
            scale_grid = np.interp(
                reco_mono, reco_grid[good_mask], scale_grid[good_mask]
            )
        reco_grid = reco_mono

        # Remove duplicate E_reco values
        unique_mask = np.diff(reco_grid, prepend=-np.inf) > 0
        reco_sorted = reco_grid[unique_mask]
        scale_sorted = scale_grid[unique_mask]

        inv_tables.append((reco_sorted, scale_sorted))

    return eta_edges, inv_tables


def apply_calibration(data, eta_edges, inv_tables):
    """Apply per-eta inversion tables; full 4-vector scaling for pt."""
    e_reco = data["e_reco"]
    abs_eta = np.abs(data["eta_det"])
    i_eta = np.clip(np.digitize(abs_eta, eta_edges) - 1, 0, len(inv_tables) - 1)
    scale = np.ones_like(e_reco)

    for ie, table in enumerate(inv_tables):
        if table is None:
            continue
        mask = i_eta == ie
        reco_grid, scale_grid = table
        scale[mask] = np.interp(
            e_reco[mask],
            reco_grid,
            scale_grid,
            left=scale_grid[0],
            right=scale_grid[-1],
        )

    e_corr = e_reco * scale
    pt_corr = pt_at_fixed_mass(e_corr, data["eta_det"], data["mass"])
    return scale, e_corr, pt_corr


def main():
    parser = argparse.ArgumentParser(description="Derive and apply the parametric absolute energy scale.")
    parser.add_argument("--input", "-i", nargs="+", required=True, metavar="FILE")
    parser.add_argument("--output", required=True, metavar="FILE",
                        help="where the corrected jets are written")
    args = parser.parse_args()

    print(f"Loading {len(args.input)} file(s)...")
    data, raw = read_jets(args.input, absolute_columns)

    print("\nDeriving the parametric absolute scale...")
    eta_edges, inv_tables = derive_calibration(data)
    n_ok = sum(1 for t in inv_tables if t is not None)
    print(f"  {n_ok}/{len(inv_tables)} eta slices have valid inversion tables")

    _, e_corr, pt_corr = apply_calibration(data, eta_edges, inv_tables)

    print(f"\nWriting corrected parquet -> {args.output}")
    write_jets(raw, pt_corr, e_corr, args.output)


if __name__ == "__main__":
    main()
