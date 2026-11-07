"""
Parametric pile-up correction, after ATLAS (Sec. IV B of the paper).

Corrected jet pT:
    p_T^corr = p_T^reco - rho*A - alpha*(N_PV - 1) - beta*mu

Derivation:
  1. Subtract rho*A to isolate the residual pile-up dependence.
  2. In bins of (pT^true, |eta|), fit alpha as the slope of the residual
     against N_PV - 1 in quantile sub-bins of mu, and beta as the slope against
     mu in sub-bins of N_PV, each combined over the sub-bins by inverse variance.
  3. Fit each against log(pT) per |eta| bin with inverse-variance weighting,
     and evaluate at nominal_pt (25 GeV).
  4. Fit a straight line in |eta|, which is what is applied.

    uv run python -m steps.pileup_parametric \
        -i data/injected/lo_rho01.pq --output data/pileup/parametric/lo_rho01.pq
"""

import argparse

import numpy as np
from scipy.stats import linregress

from jetcorrml import settings
from jetcorrml.jets import energy_at_fixed_mass, flat, pileup_columns, read_jets, write_jets

NOMINAL_PT = 25.0          # GeV: the pT at which the log fits are evaluated
FIT_SUBSTRATA = 16         # quantile sub-bins of the controlling variable


def _weighted_mean_slope(slopes, errs):
    """Inverse-variance combination of per-sub-bin slopes."""
    slopes = np.asarray(slopes, dtype=float)
    errs = np.asarray(errs, dtype=float)
    ok = np.isfinite(slopes) & np.isfinite(errs) & (errs > 0)
    if ok.sum() == 0:
        return np.nan, np.nan
    if ok.sum() == 1:
        return float(slopes[ok][0]), float(errs[ok][0])
    w = 1.0 / errs[ok] ** 2
    mean = float(np.sum(w * slopes[ok]) / np.sum(w))
    err = float(np.sqrt(1.0 / np.sum(w)))
    return mean, err


def _fit_ab_stratified(residual, npv, mu, min_jets=10, n_sub=16):
    """Fit alpha and beta as ATLAS does: each in bins of the other variable.

    Ref. [ATLAS pile-up]: "The dependence on N_PV is evaluated in bins of
    <mu>, and vice versa."  Sub-binning in the other variable holds it
    approximately fixed, so each 1D slope is a partial derivative.  The
    sub-bin slopes are combined by inverse-variance weighting.

    beta is fit on the original residual, not on the alpha-subtracted one:
    binning in N_PV already controls for it.

    Sub-bins are quantiles, so they are populated by construction.  Falls back
    to a single bin (i.e. the marginal fit) where the sub-binning leaves too
    few jets or too little spread.

    The sub-binning must be fine enough to hold the other variable genuinely
    fixed: on a closure test with known coefficients and corr(N_PV, mu) = 0.97,
    four sub-bins recover alpha = 0.95 against a true 0.60, sixteen recover
    0.645 and thirty-two recover 0.610.  The default is therefore 16, and
    n_sub trades bias against the statistics available per sub-bin.

    Returns (alpha, alpha_err, beta, beta_err, n_jets).
    """
    n = len(residual)
    if n < min_jets:
        return np.nan, np.nan, np.nan, np.nan, n

    def slope_controlling_for(x, control):
        """slope of residual vs x, in quantile sub-bins of `control`."""
        k = max(1, min(n_sub, n // max(min_jets, 2)))
        if k == 1:
            if np.ptp(x) <= 0:
                return np.nan, np.nan
            r = linregress(x, residual)
            return r.slope, r.stderr
        edges = np.quantile(control, np.linspace(0.0, 1.0, k + 1))
        edges[-1] = np.nextafter(edges[-1], np.inf)
        slopes, errs = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (control >= lo) & (control < hi)
            if int(m.sum()) < min_jets or np.ptp(x[m]) <= 0:
                continue
            r = linregress(x[m], residual[m])
            if np.isfinite(r.slope) and np.isfinite(r.stderr) and r.stderr > 0:
                slopes.append(r.slope)
                errs.append(r.stderr)
        if not slopes:
            if np.ptp(x) <= 0:
                return np.nan, np.nan
            r = linregress(x, residual)
            return r.slope, r.stderr
        return _weighted_mean_slope(slopes, errs)

    a, a_err = slope_controlling_for(npv - 1.0, mu)
    b, b_err = slope_controlling_for(mu, npv)
    return a, a_err, b, b_err, n


def _print_summary(
    eta_bins, eta_centres, alpha_nom, beta_nom, alpha_eta_fit, beta_eta_fit, nominal_pt
):
    print("\n" + "=" * 65)
    print(f"  Parametric pile-up correction constants  (nominal at pT={nominal_pt} GeV)")
    print("=" * 65)
    print(f"  {'|eta| bin':>18}  {'alpha':>10}  {'beta':>10}")
    print("  " + "-" * 43)
    for i, ec in enumerate(eta_centres):
        lo, hi = eta_bins[i], eta_bins[i + 1]
        a = f"{alpha_nom[i]:+.4f}" if np.isfinite(alpha_nom[i]) else "  N/A  "
        b = f"{beta_nom[i]:+.4f}" if np.isfinite(beta_nom[i]) else "  N/A  "
        print(f"  [{lo:.1f}, {hi:.1f})          {a:>10}  {b:>10}")
    print()
    sa, ia = alpha_eta_fit
    sb, ib = beta_eta_fit
    if np.isfinite(sa):
        print(f"  alpha(|eta|) = {ia:+.4f}  +  {sa:+.4f} * |eta|")
    if np.isfinite(sb):
        print(f"  beta (|eta|) = {ib:+.4f}  +  {sb:+.4f} * |eta|")
    print("=" * 65 + "\n")


def derive_pileup_constants(data, pt_bins, eta_bins, nominal_pt, fit_substrata):
    """Derive the alpha and beta pile-up coefficients (see the module docstring)."""
    pt_bins = np.asarray(pt_bins, dtype=float)
    eta_bins = np.asarray(eta_bins, dtype=float)

    pt_centres = 0.5 * (pt_bins[:-1] + pt_bins[1:])
    eta_centres = 0.5 * (eta_bins[:-1] + eta_bins[1:])
    n_pt = len(pt_centres)
    n_eta = len(eta_centres)

    pt_reco = data["pt_reco"]
    pt_gen = data["pt_gen"]
    abs_eta = np.abs(data["eta_det"])
    rho = data["rho"]
    npv = data["npv"]
    mu = data["mu"]
    area_pt = data["area_pt"]

    residual = pt_reco - pt_gen - rho * area_pt

    binned_alpha = np.full((n_eta, n_pt), np.nan)
    binned_beta = np.full((n_eta, n_pt), np.nan)
    binned_alpha_err = np.full((n_eta, n_pt), np.nan)
    binned_beta_err = np.full((n_eta, n_pt), np.nan)

    ieta = np.digitize(abs_eta, eta_bins) - 1
    ipt = np.digitize(pt_gen, pt_bins) - 1


    for ei in range(n_eta):
        for pi in range(n_pt):
            mask = (ieta == ei) & (ipt == pi)
            if mask.sum() < 10:
                continue
            a, ae, b, be, _n = _fit_ab_stratified(
                residual[mask], npv[mask], mu[mask], n_sub=fit_substrata
            )
            binned_alpha[ei, pi] = a
            binned_alpha_err[ei, pi] = ae
            binned_beta[ei, pi] = b
            binned_beta_err[ei, pi] = be

    alpha_logfit = np.full((n_eta, 2), np.nan)
    beta_logfit = np.full((n_eta, 2), np.nan)
    log_pt = np.log(pt_centres)

    for ei in range(n_eta):
        valid = (
            np.isfinite(binned_alpha[ei])
            & (binned_alpha[ei] > -0.05)
            & np.isfinite(binned_alpha_err[ei])
            & (binned_alpha_err[ei] > 0)
        )
        if valid.sum() >= 2:
            w = 1.0 / binned_alpha_err[ei, valid] ** 2
            c1, c0 = np.polyfit(log_pt[valid], binned_alpha[ei, valid], 1, w=w)
            alpha_logfit[ei] = [c0, c1]

        valid = (
            np.isfinite(binned_beta[ei])
            & (binned_beta[ei] > -0.05)
            & np.isfinite(binned_beta_err[ei])
            & (binned_beta_err[ei] > 0)
        )
        if valid.sum() >= 2:
            w = 1.0 / binned_beta_err[ei, valid] ** 2
            c1, c0 = np.polyfit(log_pt[valid], binned_beta[ei, valid], 1, w=w)
            beta_logfit[ei] = [c0, c1]

    log_nom = np.log(nominal_pt)
    alpha_nominal = np.where(
        np.isfinite(alpha_logfit[:, 0]),
        alpha_logfit[:, 0] + alpha_logfit[:, 1] * log_nom,
        np.nan,
    )
    beta_nominal = np.where(
        np.isfinite(beta_logfit[:, 0]),
        beta_logfit[:, 0] + beta_logfit[:, 1] * log_nom,
        np.nan,
    )

    valid_a = np.isfinite(alpha_nominal)
    valid_b = np.isfinite(beta_nominal)

    if valid_a.sum() >= 2:
        res_a = linregress(eta_centres[valid_a], alpha_nominal[valid_a])
        alpha_eta_fit = (res_a.slope, res_a.intercept)
    else:
        alpha_eta_fit = (np.nan, np.nan)

    if valid_b.sum() >= 2:
        res_b = linregress(eta_centres[valid_b], beta_nominal[valid_b])
        beta_eta_fit = (res_b.slope, res_b.intercept)
    else:
        beta_eta_fit = (np.nan, np.nan)

    _print_summary(
        eta_bins,
        eta_centres,
        alpha_nominal,
        beta_nominal,
        alpha_eta_fit,
        beta_eta_fit,
        nominal_pt,
    )
    return dict(alpha_eta_fit=alpha_eta_fit, beta_eta_fit=beta_eta_fit)


def apply_pileup_correction_flat(data, result):
    """Apply the parametric pile-up correction to the flat data dict. Returns pt_corr."""
    pt = data["pt_reco"]
    area_pt = data["area_pt"]
    eta = data["eta_det"]
    rho = data["rho"]
    npv = data["npv"]
    mu = data["mu"]

    abs_eta = np.abs(eta)
    slope_a, intercept_a = result["alpha_eta_fit"]
    slope_b, intercept_b = result["beta_eta_fit"]
    alpha = slope_a * abs_eta + intercept_a
    beta = slope_b * abs_eta + intercept_b

    pt_corr = pt - rho * area_pt - alpha * (npv - 1) - beta * mu
    return pt_corr


def main():
    # the |eta| and pT bins of the pile-up step, shared with the NN correction
    ETA_BINS = list(settings.PILEUP_ETA_BINS)
    PT_BINS = list(settings.PILEUP_PT_BINS)

    parser = argparse.ArgumentParser(description="Derive and apply the parametric pile-up correction.")
    parser.add_argument("--input", "-i", nargs="+", required=True, metavar="FILE")
    parser.add_argument("--output", required=True, metavar="FILE",
                        help="where the corrected jets are written")
    args = parser.parse_args()

    print(f"Loading {len(args.input)} file(s)...")
    data, raw = read_jets(args.input, pileup_columns)

    result = derive_pileup_constants(
        data,
        pt_bins=PT_BINS,
        eta_bins=np.array(ETA_BINS),
        nominal_pt=NOMINAL_PT,
        fit_substrata=FIT_SUBSTRATA,
    )

    pt_full = apply_pileup_correction_flat(data, result)

    print(f"\nWriting corrected parquet -> {args.output}")
    pt_full = np.asarray(pt_full).ravel()
    write_jets(raw, pt_full, energy_at_fixed_mass(pt_full, flat(raw.jet.eta), flat(raw.jet.mass)), args.output)


if __name__ == "__main__":
    main()
