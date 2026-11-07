"""
Injection for large-R jets: pile-up and an absolute response bias.

Add synthetic pile-up and JES distortion to matched large-R jet parquet files
that use the flat schema produced by the fjetcorrml extractor.

Input schema (one row per matched reco-gen large-R jet pair, all flat fields):
    jet_pt, jet_eta, jet_phi, jet_mass, jet_E, jet_areaE, ...
    jet_sjA_pt, jet_sjA_eta, jet_sjA_phi, jet_sjA_mass, ...
    jet_sjB_pt, jet_sjB_eta, jet_sjB_phi, jet_sjB_mass, ...
    gen_pt, gen_eta, gen_phi, gen_E, gen_mass, ...
    gen_sjA_pt, gen_sjA_eta, ..., gen_sjB_pt, gen_sjB_eta, ...
    rho, npv, mu

Forward injection model (matches the small-R injection closely so the same
fitting machinery downstream is exercised):

    pt_reco_injected = pt_reco
                     + alpha(pt_gen, |eta|) * (NPV - 1)
                     + beta (pt_gen, |eta|) * mu_injected

Subjet propagation (faithful to the fjetcorrml source):
- Pile-up additive shift is distributed across the two subjets in proportion
  to their pT fraction of the large-R jet.
- The JES applies a multiplicative response per object: each subjet uses
  its own |eta| (and matching gen-level subjet energy), the residual
  large-R energy (E_jet - E_sjA - E_sjB) uses the large-R |eta| with
  E_gen_jet, and the new large-R 4-vector is rebuilt from the parts.

Detector boundary (the crack):
    step_term = 1 - c_depth * sigmoid((|eta| - eta_crack) / c_width)

drops the response by c_depth (e.g. 25%) for |eta| > eta_crack with sharpness
c_width.  This is what creates the alpha-dependent large-R response when the
splitting axis straddles the boundary.

The settings (those of the small-R injection, and the crack's eta_crack,
c_depth and c_width) are jetcorrml.settings.LARGE_R.
"""

import argparse
from pathlib import Path

import numpy as np
import awkward as ak

from jetcorrml import settings
from jetcorrml.injection_model import (
    alpha_true,
    beta_true,
    make_mu,
    response_true,
)


def print_true_values(
    a0,
    a1,
    a2,
    b0,
    b1,
    b2,
    r0,
    r1,
    pt0,
    r2,
    eta0,
    eta_crack,
    c_depth,
    c_width,
    nominal_pt=25.0,
):
    print("\n" + "=" * 65)
    print(f"  True injection coefficients  (nominal at pT={nominal_pt} GeV)")
    print(f"  alpha(pt,|eta|) = ({a0:+.3f} + {a1:+.3f}*ln(pt)) * (1 + {a2:+.3f}*|eta|)")
    print(f"  beta (pt,|eta|) = ({b0:+.3f} + {b1:+.3f}*ln(pt)) * (1 + {b2:+.3f}*|eta|)")
    print("=" * 65)
    print(f"  {'|eta|':>8}  {'alpha@25GeV':>14}  {'beta@25GeV':>14}")
    print("  " + "-" * 40)
    for eta in [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]:
        a = alpha_true(nominal_pt, eta, a0, a1, a2)
        b = beta_true(nominal_pt, eta, b0, b1, b2)
        print(f"  {eta:>8.1f}  {a:>+14.4f}  {b:>+14.4f}")
    print()
    print("  JES response(E,|eta|):")
    print(
        f"    base = {r0:.3f}"
        f" * (1 - {r1:.3f}*exp(-E/{pt0:.0f}))"
        f" * (1 - {r2:.3f}*max(|eta|-{eta0:.1f},0)^2)"
    )
    print(
        f"    step = 1 - {c_depth:.3f}*sigmoid((|eta|-{eta_crack:.2f})/{c_width:.3f})"
        f"   [{c_depth * 100:.0f}% drop at |eta|={eta_crack:.2f}]"
    )
    print(
        f"  {'|eta|':>8}  {'resp@E=25':>12}  {'resp@E=60':>12}  {'resp@E=250':>13}"
    )
    print("  " + "-" * 50)
    etas = [0.0, 0.5, 1.0, 1.2, 1.3, 1.37, 1.4, 1.5, 1.6, 1.8, 2.0, 2.5]
    kw = dict(r0=r0, r1=r1, pt0=pt0, r2=r2, eta0=eta0, eta_crack=eta_crack, c_depth=c_depth, c_width=c_width)
    for eta in etas:
        r25 = response_true(25.0 * np.cosh(eta), eta, **kw)
        r60 = response_true(60.0 * np.cosh(eta), eta, **kw)
        r250 = response_true(250.0 * np.cosh(eta), eta, **kw)
        print(f"  {eta:>8.2f}  {r25:>12.4f}  {r60:>12.4f}  {r250:>13.4f}")
    print("=" * 65 + "\n")


def _apply_jes_object(
    pt,
    eta,
    mass,
    e_gen_for_resp,
    eta_for_resp,
    r0,
    r1,
    pt0,
    r2,
    eta0,
    eta_crack=None,
    c_depth=0.25,
    c_width=0.02,
):
    """Multiplicative energy response on a single jet/subjet.

    Returns updated (pt_new, E_new, response_factor)."""
    cosh_eta = np.cosh(eta)
    e_before = np.sqrt((pt * cosh_eta) ** 2 + mass**2)
    resp = response_true(
        e_gen_for_resp,
        np.abs(eta_for_resp),
        r0,
        r1,
        pt0,
        r2,
        eta0,
        eta_crack=eta_crack,
        c_depth=c_depth,
        c_width=c_width,
    )
    e_new = e_before * resp
    e2_m2 = np.maximum(e_new**2 - mass**2, 0.0)
    pt_new = np.sqrt(e2_m2) / cosh_eta
    return pt_new, e_new, resp


def inject(
    paths,
    a0,
    a1,
    a2,
    b0,
    b1,
    b2,
    noise_sigma,
    seed,
    mu_range,
    mu_rho,
    r0,
    r1,
    pt0,
    r2,
    eta0,
    eta_crack,
    c_depth,
    c_width,
):
    """Load flat large-R jet parquet files, inject pileup + JES, concatenate.

    The schema is one row per matched large-R/gen-jet pair, so npv and mu are
    already per-row.  Subjets are stored as flat fields (jet_sjA_*, jet_sjB_*,
    gen_sjA_*, gen_sjB_*).
    """
    rng = np.random.default_rng(seed)
    parts = []

    for path in paths:
        print(f"  Loading {path}")
        d = ak.from_parquet(path)
        n = len(d)

        pt_gen = np.asarray(d.gen_pt, dtype=float)
        pt_reco = np.asarray(d.jet_pt, dtype=float)
        jet_eta = np.asarray(d.jet_eta, dtype=float)
        abs_eta = np.abs(jet_eta)
        jet_mass = np.asarray(d.jet_mass, dtype=float)
        npv = np.asarray(d.npv, dtype=float)

        # ── mu generation ────────────────────────────────────────────────────
        mu_min, mu_max = mu_range
        mu = make_mu(npv, mu_min, mu_max, mu_rho, rng)
        actual_corr = float(np.corrcoef(npv, mu)[0, 1])
        print(
            f"    mu generated: rho={mu_rho:.2f}  "
            f"mean={mu.mean():.1f}  actual corr(NPV,mu)={actual_corr:+.3f}"
        )

        # ── pile-up: additive pT shift on the large-R jet ────────────────────────
        a = alpha_true(pt_gen, abs_eta, a0, a1, a2)
        b = beta_true(pt_gen, abs_eta, b0, b1, b2)
        delta = a * (npv - 1.0) + b * mu

        if noise_sigma > 0:
            delta = delta + rng.normal(0.0, noise_sigma, n)

        pt_after_pu = pt_reco + delta

        # ── distribute pile-up to subjets proportionally to their pT fraction
        safe_jet_pt = np.maximum(pt_reco, 1e-6)
        sjA_pt = np.asarray(d.jet_sjA_pt, dtype=float)
        sjB_pt = np.asarray(d.jet_sjB_pt, dtype=float)
        frac_A = sjA_pt / safe_jet_pt
        frac_B = sjB_pt / safe_jet_pt
        sjA_pt_after_pu = sjA_pt + delta * frac_A
        sjB_pt_after_pu = sjB_pt + delta * frac_B

        print(
            f"    -> {n:,} pairs  mean alpha={a.mean():.4f}  "
            f"mean beta={b.mean():.4f}  mean delta={delta.mean():.3f} GeV"
        )

        # ── JES response ─────────────────────────────────────────────────────
        # The large-R jet energy is decomposed:
        #     E_jet = E_sjA + E_sjB + E_residual
        # Each subjet gets its own response; the residual gets the large-R
        # response.  The large-R jet is then rebuilt from the parts.
        sjA_eta = np.asarray(d.jet_sjA_eta, dtype=float)
        sjA_mass = np.asarray(d.jet_sjA_mass, dtype=float)
        sjB_eta = np.asarray(d.jet_sjB_eta, dtype=float)
        sjB_mass = np.asarray(d.jet_sjB_mass, dtype=float)

        cosh_eta = np.cosh(jet_eta)

        # Per-subjet response uses E_gen of the matching subjet
        gen_sjA_E = np.asarray(d.gen_sjA_pt, dtype=float) * np.cosh(
            np.asarray(d.gen_sjA_eta, dtype=float)
        )
        sjA_pt_new, sjA_E_new, resp_A = _apply_jes_object(
            sjA_pt_after_pu,
            sjA_eta,
            sjA_mass,
            gen_sjA_E,
            sjA_eta,
            r0,
            r1,
            pt0,
            r2,
            eta0,
            eta_crack=eta_crack,
            c_depth=c_depth,
            c_width=c_width,
        )

        gen_sjB_E = np.asarray(d.gen_sjB_pt, dtype=float) * np.cosh(
            np.asarray(d.gen_sjB_eta, dtype=float)
        )
        sjB_pt_new, sjB_E_new, resp_B = _apply_jes_object(
            sjB_pt_after_pu,
            sjB_eta,
            sjB_mass,
            gen_sjB_E,
            sjB_eta,
            r0,
            r1,
            pt0,
            r2,
            eta0,
            eta_crack=eta_crack,
            c_depth=c_depth,
            c_width=c_width,
        )

        # Residual large-R energy gets the large-R response
        E_after_pu = np.sqrt((pt_after_pu * cosh_eta) ** 2 + jet_mass**2)
        sjA_E_before = np.sqrt(
            (sjA_pt_after_pu * np.cosh(sjA_eta)) ** 2 + sjA_mass**2
        )
        sjB_E_before = np.sqrt(
            (sjB_pt_after_pu * np.cosh(sjB_eta)) ** 2 + sjB_mass**2
        )
        E_residual = np.maximum(E_after_pu - sjA_E_before - sjB_E_before, 0.0)

        e_gen_jet = pt_gen * np.cosh(np.asarray(d.gen_eta, dtype=float))
        resp_jet = response_true(
            e_gen_jet,
            abs_eta,
            r0,
            r1,
            pt0,
            r2,
            eta0,
            eta_crack=eta_crack,
            c_depth=c_depth,
            c_width=c_width,
        )

        E_jet_new = sjA_E_new + sjB_E_new + E_residual * resp_jet
        e2_m2 = np.maximum(E_jet_new**2 - jet_mass**2, 0.0)
        pt_jet_new = np.sqrt(e2_m2) / cosh_eta

        resid_frac = float((E_residual / np.maximum(E_after_pu, 1.0)).mean())
        jes_str = (
            f"  JES: resp_jet={resp_jet.mean():.4f} "
            f"(on residual {resid_frac:.1%})  "
            f"resp_sjA={resp_A.mean():.4f}  resp_sjB={resp_B.mean():.4f}"
        )
        print(f"    {jes_str}")

        # ── Build output record ──────────────────────────────────────────────
        # Copy every field, then overwrite the ones we modified
        fields = {f: np.asarray(d[f]) for f in d.fields}
        fields["jet_pt"] = pt_jet_new
        fields["jet_E"] = E_jet_new
        fields["jet_sjA_pt"] = sjA_pt_new
        fields["jet_sjB_pt"] = sjB_pt_new
        fields["mu"] = mu
        fields["npv"] = npv

        parts.append(ak.Array(fields))

    combined = ak.concatenate(parts, axis=0)
    print(f"\n  Combined: {len(combined):,} matched pairs across {len(paths)} file(s)")
    return combined


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Inject pile-up dependence and the absolute-scale "
                                                 "distortion with a crack into the large-R sample.")
    parser.add_argument("--input", "-i", nargs="+", required=True, metavar="FILE",
                        help="the raw large-R files, in the order they are read")
    parser.add_argument("--output", "-o", required=True, metavar="FILE")
    args = parser.parse_args()
    inj = settings.LARGE_R                      # the one large-R dataset
    pu, ab = inj.pileup, inj.absolute
    mu_range, mu_rho = (inj.mu[0], inj.mu[1]), inj.mu[2]
    print(f"mu mode: correlated   U({mu_range[0]:.0f}, {mu_range[1]:.0f})  rho={mu_rho:.2f}")
    print(f"JES mode: r0={ab.r0:.3f}  r1={ab.r1:.3f}  pt0={ab.pt0:.0f} GeV  r2={ab.r2:.3f}  eta0={ab.eta0:.1f}")
    print(f"  Detector boundary: {inj.c_depth * 100:.0f}% drop at |eta|={inj.eta_crack:.2f}  (width={inj.c_width:.3f})")
    print("Nonlinear mode: disabled (purely linear pile-up)")

    terms = dict(r0=ab.r0, r1=ab.r1, pt0=ab.pt0, r2=ab.r2, eta0=ab.eta0,
                 eta_crack=inj.eta_crack, c_depth=inj.c_depth, c_width=inj.c_width)
    print_true_values(pu.a0, pu.a1, pu.a2, pu.b0, pu.b1, pu.b2, **terms)

    print(f"Loading {len(args.input)} file(s)...")
    combined = inject(
        args.input,
        a0=pu.a0, a1=pu.a1, a2=pu.a2, b0=pu.b0, b1=pu.b1, b2=pu.b2,
        noise_sigma=inj.noise,
        seed=inj.seed,
        mu_range=mu_range,
        mu_rho=mu_rho,
        **terms,
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\nWriting to {out_path} ...")
    ak.to_parquet(combined, str(out_path))
    print("Done.")


if __name__ == "__main__":
    main()
