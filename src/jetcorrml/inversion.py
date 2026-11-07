"""
Unified response inversion — a single exact root-finder used by BOTH the
sub-jet-ON and sub-jet-OFF large-R corrections, so that an ON/OFF comparison
reflects only the network's *feature inputs*, never the inversion method.

Design principle
----------------
The inverter must depend on the trained network **only through the value of R**,
never through the *shape* of R (its slope, smoothness, or off-range behaviour).
Two common alternatives do not:

  * a table of R sampled on a grid, smoothed and made monotonic (shape
    operations, and it queries R far below the trained energy range, where the
    net extrapolates), and
  * a fixed number of fixed-point steps, a *truncated* iteration whose result
    depends on the local slope of R (its contraction factor), so the same jet
    converges differently for a 4- and a 7-feature net.

``invert_bisect`` is instead an exact, deterministic solve of

        E * R(E; features) = E_reco

by bisection in log-E over a bracket **fixed to the trained energy range**, with
no smoothing and no monotonisation.  The network is called as a pure black box
via ``feature_fn(E)``; the *only* thing that changes between ON and OFF is the
feature vector that closure builds.  The bracket is identical for both, so
neither net is ever queried where it is untrustworthy.

Where E*R(E) is genuinely ill-conditioned (flat or folded — the crack's
boost-collimation slope), bisection still returns one deterministic root, and it
does so *identically* for ON and OFF: the ill-conditioning then reads as a
property of the problem, surfaced equally, rather than an artefact of the
inverter.  Any regularisation of that tail belongs downstream of this solve and
must be applied identically to both corrections.
"""
from __future__ import annotations
import numpy as np
import torch


def invert_bisect(
    e_reco,
    feature_fn,
    model,
    norm,
    e_lo,
    e_hi,
    n_iter=32,
    r_floor=0.01,
    r_ceil=None,
    batch=65536,
):
    """Solve ``E * R(E) = e_reco`` for E by vectorised bisection in log-E.

    Parameters
    ----------
    e_reco : array
        Reconstructed energies to correct.
    feature_fn : callable
        ``feature_fn(E) -> (N, n_features) float array`` — builds the network
        inputs at trial energy ``E``.  This closure is the *only* place ON and
        OFF differ (7- vs 4-feature).
    model, norm :
        Trained ``ResponseNet`` and its ``Normaliser`` (``norm.transform``).
    e_lo, e_hi : float
        Bracket, fixed to the trained energy range and identical for ON/OFF.
    n_iter : int
        Bisection steps (log-E interval shrinks by 2**-n_iter).
    r_floor, r_ceil : float or None
        Pointwise clip on R (a value operation, not a shape operation): the
        floor keeps E*R finite; an optional ceiling bounds any residual
        out-of-range extrapolation.  Applied identically to both networks.

    Returns
    -------
    e_est : array
        Estimate of E_gen (the energy at which the response solves the equation).
        Roots outside the bracket clamp to ``e_lo``/``e_hi`` by construction.
    """
    e_reco = np.asarray(e_reco, dtype=float)
    log_lo = np.full_like(e_reco, np.log(float(e_lo)))
    log_hi = np.full_like(e_reco, np.log(float(e_hi)))

    model.eval()

    def _R(E):
        X = np.asarray(feature_fn(E), dtype=np.float32)
        Xn = norm.transform(X).astype(np.float32)
        out = np.empty(len(E), dtype=np.float32)
        with torch.no_grad():
            for i0 in range(0, len(E), batch):
                i1 = min(i0 + batch, len(E))
                out[i0:i1] = model(torch.from_numpy(Xn[i0:i1])).numpy().ravel()
        return np.clip(out.astype(float), r_floor, r_ceil)

    for _ in range(int(n_iter)):
        log_mid = 0.5 * (log_lo + log_hi)
        E = np.exp(log_mid)
        g = E * _R(E)  # E * R(E), monotone-increasing where well-conditioned
        below = g < e_reco  # target root is at higher E
        log_lo = np.where(below, log_mid, log_lo)
        log_hi = np.where(below, log_hi, log_mid)

    return np.exp(0.5 * (log_lo + log_hi))
