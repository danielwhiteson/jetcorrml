"""
The parametric method's measure of the jet response: the peak of a Gaussian fit
to the core of the response distribution (the NN targets the median instead).
Used by the parametric absolute scale and by scripts/study_absolute_norm.py.
"""

import numpy as np
from scipy.optimize import curve_fit


def gaussian_core_peak(vals, n_sigma=2.0, max_iter=3, min_count=50):
    """Mean of an iterative Gaussian fit to the *core* of the response.

    This is the parametric (ATLAS) definition of the response: a Gaussian is fit to the
    histogram of ``vals`` within ``±n_sigma`` of the running mean, iterated a
    few times so the fit settles on the Gaussian core and is insensitive to the
    asymmetric (low-side, out-of-cone) tail.  Returns ``(mu, err)`` where ``mu``
    is the fitted peak and ``err`` its fit uncertainty.

    Falls back to the median (with the ``1.253·σ/√N`` error) when there are too
    few entries, or if the fit fails / does not converge / lands outside the
    data range -- this keeps sparse or pathological (η, E) bins robust.
    """
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    n = len(vals)
    if n == 0:
        return np.nan, np.nan

    med = float(np.median(vals))
    med_err = float(1.253 * np.std(vals) / np.sqrt(n)) if n > 0 else np.nan

    # Too sparse to fit -> robust median.
    if n < min_count:
        return med, med_err

    # Seed sigma from the IQR (16-84 percentile half-width): less tail-sensitive
    # than the plain std for initialising the core window.
    q16, q84 = np.percentile(vals, [16.0, 84.0])
    sigma = max(0.5 * (q84 - q16), 1e-6)

    def _gauss(x, a, mu, sg):
        return a * np.exp(-0.5 * ((x - mu) / sg) ** 2)

    mu, sg = med, sigma
    core = vals
    try:
        for _ in range(max_iter):
            lo, hi = mu - n_sigma * sg, mu + n_sigma * sg
            core = vals[(vals >= lo) & (vals <= hi)]
            if len(core) < min_count:
                return med, med_err
            nb = int(np.clip(round(np.sqrt(len(core))), 10, 50))
            counts, edges = np.histogram(core, bins=nb)
            centres = 0.5 * (edges[:-1] + edges[1:])
            popt, pcov = curve_fit(
                _gauss,
                centres,
                counts,
                p0=[float(counts.max()), mu, sg],
                maxfev=5000,
            )
            new_mu, new_sg = float(popt[1]), abs(float(popt[2]))
            if not (np.isfinite(new_mu) and np.isfinite(new_sg) and new_sg > 0):
                return med, med_err
            converged = abs(new_mu - mu) < 1e-4 * max(abs(mu), 1e-6)
            mu, sg = new_mu, new_sg
            if converged:
                break

        # Fitted mean must sit inside the data range to be trusted.
        if not (vals.min() <= mu <= vals.max()):
            return med, med_err

        # Error on the fitted mean: prefer the fit covariance, else σ_core/√N.
        perr = np.nan
        if pcov is not None and np.all(np.isfinite(pcov)):
            perr = float(np.sqrt(np.diag(pcov))[1])
        if not np.isfinite(perr) or perr <= 0:
            perr = float(sg / np.sqrt(len(core)))
        return float(mu), perr
    except (RuntimeError, ValueError, TypeError, np.linalg.LinAlgError):
        # curve_fit failure / degenerate histogram -> robust median.
        return med, med_err
