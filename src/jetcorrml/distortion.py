"""
The distortion of the data half (Sec. VII A): every data jet's pT and E are scaled by
f(pT, |eta|, rho), normalised so that f = 1 + d0 at pT = 45 GeV, |eta| = 0 and
rho = 15 GeV/area.  The forms and their coefficients of the paper are
jetcorrml.settings.DISTORTIONS.
"""

import numpy as np

PT_REF = 45.0  # GeV — reference pT for log centring; ln(PT_REF/PT_REF) = 0
RHO_REF = 15.0  # GeV/area — reference rho of the rhoeta form: near the median rho of the
                # in-situ dataset, so that the rho term vanishes at typical pile-up


# All form functions accept (pt, abs_eta, rho, d0, d1, d2) for a uniform
# dispatch signature.  The four kinematic-only forms (linear, quadratic,
# threshold, expo) ignore `rho`; only `_resid_rhoeta` uses it.


def _resid_linear(pt, abs_eta, rho, d0, d1, d2):
    """
    f = 1 + d0 + d1*ln(pT/PT_REF) + d2*|eta|

    Linear in both log-pT and eta.  The simplest possible shape and the
    natural first test: does the correction pipeline recover a constant
    offset plus a slope in each variable independently?
    """
    del rho  # unused
    return 1.0 + d0 + d1 * np.log(pt / PT_REF) + d2 * abs_eta


def _resid_quadratic(pt, abs_eta, rho, d0, d1, d2):
    """
    f = 1 + d0 + d1*ln(pT/PT_REF) + d2*|eta|^2

    Quadratic eta dependence — flat in the barrel, rising in the endcap.
    Tests whether a correction algorithm can handle eta curvature, e.g.
    when forward calorimeter inter-calibration drifts relative to the barrel.
    """
    del rho  # unused
    return 1.0 + d0 + d1 * np.log(pt / PT_REF) + d2 * abs_eta**2


def _resid_threshold(pt, abs_eta, rho, d0, d1, d2):
    """
    f = 1 + d0 + d1*ln(pT/PT_REF) + d2*sign(|eta| - 1.5)

    Step discontinuity at |eta| = 1.5 (barrel / endcap boundary).
    Barrel (|eta| < 1.5): f = 1 + d0 - d2
    Forward (|eta| > 1.5): f = 1 + d0 + d2
    Tests robustness to non-smooth eta structure, e.g. a dead material
    region or a miscalibrated calorimeter layer that kicks in at the crack.
    """
    del rho  # unused
    step = np.sign(abs_eta - 1.5)  # -1 barrel, 0 at boundary, +1 forward
    return 1.0 + d0 + d1 * np.log(pt / PT_REF) + d2 * step


def _resid_expo(pt, abs_eta, rho, d0, d1, d2):
    """
    f = 1 + d0 + d1*ln(pT/PT_REF) + d2*(exp(|eta|) - 1) / (exp(2.5) - 1)

    Exponentially rising eta dependence, normalised to [0, 1] across
    |eta| in [0, 2.5].  At eta=0: eta-term = 0.  At eta=2.5: eta-term = d2.
    Tests recovery of a smoothly but nonlinearly growing forward distortion,
    harder to fit than a simple slope or step.
    """
    del rho  # unused
    eta_term = (np.exp(abs_eta) - 1.0) / (np.exp(2.5) - 1.0)
    return 1.0 + d0 + d1 * np.log(pt / PT_REF) + d2 * eta_term


def _resid_rhoeta(pt, abs_eta, rho, d0, d1, d2, d3=0.0):
    """
    f = 1 + d0 + d1*ln(pT/PT_REF) + d2*|eta|^2 + d3*(rho/RHO_REF - 1)*(PT_REF/pT)

    Combined form: the |eta|^2 endcap term of 'quadratic' (d2) plus a
    rho/pT pile-up term (d3), with independent coefficients.  This is the realistic test
    of an ADDITIONAL dependence to learn (eta AND rho AND pT) rather than a
    substitute one: ATLAS's eta-intercalibration can absorb the |eta|^2
    part but its area subtraction cannot model the per-jet rho/pT part,
    while the NN -- with --abs-features pt,rho feeding the rho effect into
    s_abs and the chain handling the |eta|^2 part via s_rel -- closes both.

    Anchors: f -> 1 + d0 at (pT, |eta|, rho) = (PT_REF, 0, RHO_REF).
    Set d3=0 to recover 'quadratic'.
    """
    return (
        1.0
        + d0
        + d1 * np.log(pt / PT_REF)
        + d2 * abs_eta**2
        + d3 * (rho / RHO_REF - 1.0) * (PT_REF / pt)
    )


_RESIDUAL_FORMS = {
    "linear": _resid_linear,
    "quadratic": _resid_quadratic,
    "threshold": _resid_threshold,
    "expo": _resid_expo,
    "rhoeta": _resid_rhoeta,
}


def residual_distortion(pt, abs_eta, rho, d0, d1, d2, d3=0.0, form="linear"):
    """
    Compute per-jet residual scale factors for the data sub-sample.

    This is the single public entry point for the residual distortion.
    It validates the form name, dispatches to the appropriate sub-function,
    and returns the per-jet multiplicative factor.

    Parameters
    ----------
    pt      : 1-D float array, jet pT [GeV]
    abs_eta : 1-D float array, |eta_det|
    rho     : 1-D float array, per-jet pileup density [GeV/area].  Event-
              level rho must be broadcast to per-jet before calling.  Only
              the rhoeta form uses it.
    d0      : overall offset  (f → 1+d0 at pT=PT_REF=45 GeV, eta=0,
              rho=RHO_REF=15)
    d1      : log-pT slope coefficient
    d2      : the coefficient of the eta term
    d3      : the coefficient of the rho term (rhoeta only)
    form    : one of 'linear', 'quadratic', 'threshold', 'expo', 'rhoeta'

    Returns
    -------
    scale : 1-D float array, per-jet multiplicative factor applied to pT and E
    """
    if form not in _RESIDUAL_FORMS:
        raise ValueError(
            f"Unknown residual form '{form}'. Choose from: {sorted(_RESIDUAL_FORMS)}"
        )
    if form == "rhoeta":
        return _resid_rhoeta(pt, abs_eta, rho, d0, d1, d2, d3)
    return _RESIDUAL_FORMS[form](pt, abs_eta, rho, d0, d1, d2)
