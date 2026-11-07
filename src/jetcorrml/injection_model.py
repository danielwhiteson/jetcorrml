"""
Shared mathematical primitives of the injection (small-R and large-R jets).

The same alpha/beta/gamma functional forms describe the per-jet pile-up
coefficients in both pipelines.  The response model is shared too, with
an optional barrel-endcap step term used by the large-R injection.

These functions are intentionally tiny pure-numpy helpers so the small-jet
injection (jagged awkward arrays) and the large-R injection (flat parquet) can
both use the same physics with their own data-format specific I/O.
"""

from __future__ import annotations

import numpy as np


def alpha_true(pt, abs_eta, a0, a1, a2):
    """alpha(pt, |eta|) = (a0 + a1 * ln(pt)) * (1 + a2 * |eta|)"""
    return (a0 + a1 * np.log(pt)) * (1.0 + a2 * abs_eta)


def beta_true(pt, abs_eta, b0, b1, b2):
    """beta(pt, |eta|) = (b0 + b1 * ln(pt)) * (1 + b2 * |eta|)"""
    return (b0 + b1 * np.log(pt)) * (1.0 + b2 * abs_eta)


def gamma_true(pt, abs_eta, g0, g1, g2):
    """gamma(pt, |eta|) = (g0 + g1 * ln(pt)) * (1 + g2 * |eta|)

    Quadratic NPV saturation coefficient — same functional form as alpha/beta.
    Negative g0 causes the pile-up correction to saturate at high NPV.
    """
    return (g0 + g1 * np.log(pt)) * (1.0 + g2 * abs_eta)


def eta_density_excess(abs_eta, depth, eta_c=2.0, width=0.15):
    """Fractional excess of the TRUE pile-up density over the reported rho.

        f(|eta|) - 1 = depth * sigmoid( (|eta| - eta_c) / width )

    Motivation (ATLAS arXiv:1510.03823, Fig. 3): rho is the median pT-density of
    kt patches built from topo-clusters, and the measured rho falls to nearly
    zero beyond |eta| ~ 2 because coarser forward cells plus significance-based
    noise suppression stop soft pile-up from seeding clusters.  The TRUE forward
    pile-up density is not zero, so a central-anchored rho UNDER-reports it, and
    rho*A therefore UNDER-subtracts forward jets.

    We inject only the EXCESS (f-1), not the full density: the Delphes pile-up
    is already present in jet.pt and already described by event.rho, so adding
    rho*A*(f-1) before the subtraction means the downstream `pt - rho*A` removes
    the baseline and leaves exactly

        residual(|eta|) = rho*A * depth * sigmoid((|eta|-eta_c)/width)

    which is ~0 centrally and grows forward.  The residual is therefore a
    CONSEQUENCE of the estimator mismatch rather than an additive term chosen by
    hand, and its shape has no reason to match the parametrised alpha/beta form
    an ATLAS-style residual fit assumes.

    Parameters
    ----------
    abs_eta : |eta| of the jet
    depth   : fractional under-report far forward.  depth=0.5 means the true
              forward density is 1.5x the reported rho.
    eta_c   : sigmoid midpoint (2.0 follows Fig. 3)
    width   : sigmoid width in eta units (smaller = sharper turn-on)
    """
    if depth == 0:
        return np.zeros_like(np.asarray(abs_eta, dtype=float))
    return depth / (1.0 + np.exp(-(abs_eta - eta_c) / width))


def response_true(
    E,
    abs_eta,
    r0,
    r1,
    pt0,
    r2,
    eta0,
    eta_crack=None,
    c_depth=0.25,
    c_width=0.02,
):
    """
    Synthetic calorimeter energy response.

        response(E, |eta|) = r0
                           * (1 - r1 * exp(-E/pt0))
                           * (1 - r2 * max(|eta|-eta0, 0)^2)
                           * step_term(|eta|, eta_crack, c_depth, c_width)

    The step term is only active when ``eta_crack`` is provided and ``c_depth``
    is positive.  It models a sharp barrel-endcap response drop at
    |eta| = eta_crack of fractional depth ``c_depth`` with sigmoid sharpness
    ``c_width`` (in eta units).  Used by the large-R injection; kept off by
    default for the small-jet pipeline.

    Parameters
    ----------
    E         : jet energy [GeV]  (use E_gen = pT_gen * cosh(eta_gen))
    r0, r1    : overall scale and low-E non-linearity depth
    pt0       : characteristic energy scale [GeV]
    r2        : forward |eta|^2 degradation coefficient (above eta0)
    eta0      : eta hinge — degradation only kicks in for |eta|>eta0
    eta_crack : optional |eta| of a step-down (None disables)
    c_depth   : fractional depth of the step (default 0.25 = 25%)
    c_width   : sigmoid sharpness in eta units (default 0.02)
    """
    E_term = 1.0 - r1 * np.exp(-E / pt0)
    eta_hinge = np.maximum(abs_eta - eta0, 0.0)
    eta_term = 1.0 - r2 * eta_hinge**2

    if eta_crack is not None and c_depth > 0:
        step_term = 1.0 - c_depth / (1.0 + np.exp(-(abs_eta - eta_crack) / c_width))
    else:
        step_term = 1.0

    return r0 * E_term * eta_term * step_term


def make_mu(npv_per_event, mu_min, mu_max, rho, rng):
    """
    Generate a per-event mu with tunable correlation to NPV.

    Physical model: mu is the true mean pileup, NPV ~ Poisson(mu), so
    mu ~ NPV with width sqrt(NPV).  We interpolate between:

      rho=0: mu ~ U(mu_min, mu_max)         — completely independent of NPV
      rho=1: mu ~ N(2*NPV, sqrt(2*NPV))     — physically correlated with NPV

    Intermediate rho blends the two linearly:
      mu = rho * mu_phys + (1-rho) * mu_indep
    """
    n = len(npv_per_event)
    sigma = np.sqrt(2.0 * np.maximum(npv_per_event, 1.0))
    mu_phys = rng.normal(2.0 * npv_per_event, sigma)
    mu_indep = rng.uniform(mu_min, mu_max, size=n)
    mu = rho * mu_phys + (1.0 - rho) * mu_indep
    return np.clip(mu, 1.0, None)
