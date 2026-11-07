"""
Injection of pile-up dependence and an absolute-scale distortion into the
dijet sample (paper Sec. IV A and V A; App. A-C).

    pt_reco -> pt_reco + alpha(pt_gen, |eta|) (NPV - 1) + beta(pt_gen, |eta|) mu'
               [ + epsilon ( gamma(pt_gen, |eta|) (NPV - 1)^2 + kappa (NPV - 1)(mu' - <mu'>) ) ]

    alpha(pt, |eta|) = (a0 + a1 ln pt)(1 + a2 |eta|), beta and gamma likewise,

with Gaussian noise on each jet's shift, then E -> E R(E_gen, |eta|) with
R = r0 (1 - r1 exp(-E/pt0)) (1 - r2 max(|eta| - eta0, 0)^2), and pT recomputed
from E at fixed mass.  The mock mu' = rho mu_phys(NPV) + (1 - rho) mu_indep is
written back into the event record.  The forward-density variant (App. A)
adds rho A (f(|eta|) - 1) instead of the alpha and beta terms.

Every dataset's settings are in jetcorrml.settings (DATASETS):

    uv run python -m steps.inject \
        -i data/raw/jj*.pq -o data/injected/lo_rho05.pq --dataset lo_rho05
"""

import argparse
from pathlib import Path

import numpy as np
import awkward as ak

from jetcorrml import settings
from jetcorrml.injection_model import (
    alpha_true,
    beta_true,
    eta_density_excess,
    gamma_true,
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
    nominal_pt=25.0,
    r0=None,
    r1=None,
    pt0=None,
    r2=None,
    eta0=None,
    epsilon=0.0,
    g0=0.0,
    g1=0.0,
    g2=0.0,
    kappa=0.0,
):
    print("\n" + "=" * 65)
    print(f"  True injection coefficients  (nominal at pT={nominal_pt} GeV)")
    print(f"  alpha(pt,|eta|) = ({a0:+.3f} + {a1:+.3f}*ln(pt)) * (1 + {a2:+.3f}*|eta|)")
    print(f"  beta (pt,|eta|) = ({b0:+.3f} + {b1:+.3f}*ln(pt)) * (1 + {b2:+.3f}*|eta|)")
    if epsilon > 0:
        print(
            f"  gamma(pt,|eta|) = ({g0:+.4f} + {g1:+.4f}*ln(pt)) * (1 + {g2:+.3f}*|eta|)"
        )
        print(f"  kappa           = {kappa:+.4f}")
        print(f"  epsilon         = {epsilon:.2f}")
    print("=" * 65)
    header = f"  {'|eta|':>8}  {'alpha@25GeV':>14}  {'beta@25GeV':>14}"
    if epsilon > 0:
        header += f"  {'gamma@25GeV':>14}"
    print(header)
    print("  " + "-" * (40 if epsilon == 0 else 56))
    for eta in [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]:
        a = alpha_true(nominal_pt, eta, a0, a1, a2)
        b = beta_true(nominal_pt, eta, b0, b1, b2)
        line = f"  {eta:>8.1f}  {a:>+14.4f}  {b:>+14.4f}"
        if epsilon > 0:
            g = gamma_true(nominal_pt, eta, g0, g1, g2)
            line += f"  {g:>+14.6f}"
        print(line)
    if r0 is not None:
        print()
        print(
            f"  JES response(E,|eta|) = {r0:.3f}"
            f" * (1 - {r1:.3f}*exp(-E/{pt0:.0f}))"
            f" * (1 - {r2:.3f}*max(|eta| - {eta0:g}, 0)^2)"
        )
        print("  (E_gen = pT_gen * cosh(eta); pt0 acts as energy scale)")
        print(
            f"  {'|eta|':>8}  {'resp@E=25':>12}  {'resp@E=60':>12}  {'resp@E=250':>13}"
        )
        print("  " + "-" * 50)
        for eta in [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]:
            e25 = 25.0 * np.cosh(eta)
            e60 = 60.0 * np.cosh(eta)
            e250 = 250.0 * np.cosh(eta)
            r25 = response_true(e25, eta, r0, r1, pt0, r2, eta0)
            r60 = response_true(e60, eta, r0, r1, pt0, r2, eta0)
            r250 = response_true(e250, eta, r0, r1, pt0, r2, eta0)
            print(f"  {eta:>8.1f}  {r25:>12.4f}  {r60:>12.4f}  {r250:>13.4f}")
    print("=" * 65 + "\n")


# -----------------------------------------------------------------------------
# Main injection function
# -----------------------------------------------------------------------------


def inject(
    paths,
    a0,
    a1,
    a2,
    b0,
    b1,
    b2,
    noise_sigma=0.0,
    seed=None,
    mu_range=None,
    mu_rho=None,
    r0=None,
    r1=0.10,
    pt0=30.0,
    r2=0.02,
    eta0=1.0,
    epsilon=0.0,
    g0=0.0,
    g1=0.0,
    g2=0.0,
    kappa=0.0,
    eta_density_depth=0.0,
    eta_density_etac=2.0,
    eta_density_width=0.15,
):
    """
    Load parquet files, inject pT- and eta-dependent pile-up, concatenate.

    Parameters
    ----------
    paths       : list of str
    a0,a1,a2    : alpha parameterisation coefficients
    b0,b1,b2    : beta  parameterisation coefficients
    noise_sigma : Gaussian noise sigma [GeV] added on top (0 = no noise)
    seed        : random seed for reproducibility
    mu_range    : (mu_min, mu_max) — if given, generate synthetic per-event mu
    mu_rho      : float in [0,1] — NPV-mu correlation for synthetic mu.
                  0 = independent (default when mu_range given without mu_rho)
                  1 = fully correlated with NPV
                  None = use fixed per-file mu from DELPHES
    r0          : overall JES response scale (None = no JES effect injected).
                  e.g. 0.95 means reco pT is ~5% below gen at high pT, eta=0.
    r1          : low-pT non-linearity depth (default 0.10)
    pt0         : characteristic pT [GeV] of the non-linearity (default 30)
    r2          : forward degradation coefficient on |eta|^2 (default 0.02)
    epsilon     : nonlinearity strength in [0,1]. 0 = purely linear (default).
    g0,g1,g2    : gamma parameterisation coefficients (quadratic NPV term)
    kappa       : NPV*mu interaction strength
    eta_density_depth : if >0, inject an eta-dependent pile-up DENSITY excess
                  (rho*A*(f(|eta|)-1)) INSTEAD of the alpha/beta residual.
                  Mutually exclusive with alpha/beta by design, so the two
                  mechanisms can be studied in isolation.
    eta_density_etac, eta_density_width : sigmoid midpoint / width of f.

    Returns
    -------
    combined : awkward array with jet.pt (and event.mu) replaced
    """
    rng = np.random.default_rng(seed)
    parts = []

    for path in paths:
        print(f"  Loading {path}")
        matches = ak.from_parquet(path)

        def flat(arr):
            return ak.to_numpy(ak.flatten(arr, axis=None)).astype(float)

        pt_gen = flat(matches.gen.pt)
        pt_reco = flat(matches.jet.pt)  # original reco pT before injection
        abs_eta = np.abs(flat(matches.jet.eta))
        npv_jet = flat(matches.event.npv)  # per-jet NPV
        counts = ak.num(matches)

        if mu_range is not None:
            mu_min, mu_max = mu_range
            rho = mu_rho if mu_rho is not None else 0.0

            # Need per-EVENT NPV to generate correlated mu.
            # matches.event.npv is jagged (one value per pair per event),
            # so take the first entry per event — they're all identical within
            # an event. ak.firsts gives the first element of each inner list.
            npv_per_event = ak.to_numpy(ak.firsts(matches.event.npv)).astype(float)

            mu_per_event = make_mu(npv_per_event, mu_min, mu_max, rho, rng)
            mu_per_jet = np.repeat(mu_per_event, ak.to_numpy(counts))
            mu = mu_per_jet

            # Write new mu back into the event record using the same
            # jagged structure as the original (one value per jet, same
            # as the other event fields like npv).
            ev_fields = {f: matches.event[f] for f in ak.fields(matches.event)}
            ev_fields["mu"] = ak.unflatten(mu_per_jet, counts)
            new_event = ak.zip(ev_fields)

            actual_corr = np.corrcoef(npv_per_event, mu_per_event)[0, 1]
            print(
                f"    mu generated: N(NPV, sqrt(NPV)) rho={rho:.2f}  "
                f"mean={mu_per_event.mean():.1f}  "
                f"actual corr(NPV,mu)={actual_corr:+.3f}"
            )
        else:
            mu = flat(matches.event.mu)
            new_event = matches.event
            print(
                f"    mu from file: mean={mu.mean():.1f}  "
                f"corr(NPV,mu)={np.corrcoef(npv_jet, mu)[0, 1]:+.3f}"
            )

        if eta_density_depth > 0:
            # ── eta-dependent DENSITY injection (mutually exclusive with
            #    the alpha/beta residual — see the module docstring) ──────────
            # Inject the EXCESS of the true forward pile-up density over what
            # the central-anchored rho reports, scaled by the SAME jet area the
            # the pile-up correction subtracts (areaE / cosh eta).  Because this term is
            # proportional to rho*A, the downstream `pt - rho*A` acts on it, and
            # what survives is the estimator mismatch itself.
            area_pt = flat(matches.jet.areaE) / np.cosh(abs_eta)
            rho_jet = flat(matches.event.rho)  # per-jet (event scalar repeated)
            excess = eta_density_excess(
                abs_eta, eta_density_depth, eta_density_etac, eta_density_width
            )
            delta = rho_jet * area_pt * excess
            fwd = abs_eta >= eta_density_etac
            fwd_txt = (f"{delta[fwd].mean():.3f}" if fwd.any() else "n/a")
            print(
                f"    eta-density injection: depth={eta_density_depth:g} "
                f"eta_c={eta_density_etac:g} width={eta_density_width:g}\n"
                f"      <rho*A>={np.mean(rho_jet * area_pt):.2f} GeV  "
                f"<delta>={delta.mean():.3f} GeV  "
                f"<delta | |eta|>={eta_density_etac:g}> = {fwd_txt} GeV  "
                f"(alpha/beta terms disabled)"
            )
        else:
            # Evaluate true coefficients per jet
            a = alpha_true(pt_gen, abs_eta, a0, a1, a2)
            b = beta_true(pt_gen, abs_eta, b0, b1, b2)

            # Synthetic pile-up delta (linear terms)
            delta = a * (npv_jet - 1) + b * mu

            # Nonlinear terms: quadratic NPV saturation + NPV*mu interaction
            if epsilon > 0:
                g = gamma_true(pt_gen, abs_eta, g0, g1, g2)
                mu_mean = mu.mean()
                delta += epsilon * (
                    g * (npv_jet - 1) ** 2 + kappa * (npv_jet - 1) * (mu - mu_mean)
                )

        if noise_sigma > 0:
            delta += rng.normal(0, noise_sigma, len(delta))

        # ── pile-up: additive pT shift ────────────────────────────────────────
        # Pile-up adds transverse momentum at roughly fixed eta/phi, so the
        # correction lives entirely in pT.  eta, phi, mass are unchanged.
        #
        # We also compute the pile-up-shifted E here (needed by JES below)
        # using the exact massive relation:
        #   E_after_pu = sqrt((pT_after_pu * cosh(eta))^2 + mass^2)
        # This must be done before building jet_fields so JES can use it.
        jet_mass_flat = flat(matches.jet.mass)
        cosh_eta = np.cosh(abs_eta)
        pt_after_pu = pt_reco + delta  # flat array
        e_after_pu = np.sqrt((pt_after_pu * cosh_eta) ** 2 + jet_mass_flat**2)

        delta_jagged = ak.unflatten(delta, counts)
        new_jet_pt = matches.jet.pt + delta_jagged
        jet_fields = {f: matches.jet[f] for f in ak.fields(matches.jet)}
        jet_fields["pt"] = new_jet_pt
        # jet_fields["E"] left as original here; JES block below updates both
        # pt and E together using e_after_pu as the base.  If JES is disabled,
        # E remains the original reco value (pile-up is a pT-only effect).

        # ── JES: multiplicative energy response ───────────────────────────────
        # Calorimeter mis-measurement is fundamentally an ENERGY effect:
        #   E_final = E_after_pu * response(E_gen, |eta|)
        #
        # pT is then derived from the scaled E using the exact massive relation:
        #   pT = sqrt(E_final^2 - mass^2) / cosh(eta)
        #
        # The response multiplies e_after_pu (the pile-up-shifted E), not the
        # original reco E, which would discard the pile-up shift.
        jes_str = ""
        if r0 is not None:
            e_gen = flat(matches.gen.pt) * np.cosh(flat(matches.gen.eta))
            resp = response_true(e_gen, abs_eta, r0, r1, pt0, r2, eta0)

            e_reco_new = e_after_pu * resp  # ← correct base: after pile-up

            e2_minus_m2 = np.maximum(e_reco_new**2 - jet_mass_flat**2, 0.0)
            pt_reco_new_jes = np.sqrt(e2_minus_m2) / cosh_eta

            pt_jes_jagged = ak.unflatten(pt_reco_new_jes, counts)
            e_jes_jagged = ak.unflatten(e_reco_new, counts)

            jet_fields["pt"] = pt_jes_jagged
            jet_fields["E"] = e_jes_jagged

            mean_de = (e_reco_new - e_after_pu).mean()
            mean_dp = (pt_reco_new_jes - pt_after_pu).mean()
            jes_str = (
                f"  JES: mean_resp={resp.mean():.4f}  "
                f"mean_dE={mean_de:.2f} GeV  mean_dpT={mean_dp:.2f} GeV"
            )

        new_matches = ak.zip(
            {
                "jet": ak.zip(jet_fields),
                "gen": matches.gen,
                "event": new_event,
                "deltaR": matches.deltaR,
            }
        )

        if eta_density_depth > 0:
            coef_str = f"mean excess={excess.mean():.4f}  "
        else:
            coef_str = f"mean alpha={a.mean():.4f}  mean beta={b.mean():.4f}  "
        print(
            f"    -> {len(delta):,} pairs  "
            f"{coef_str}"
            f"mean delta={delta.mean():.3f} GeV{jes_str}"
        )
        parts.append(new_matches)

    combined = ak.concatenate(parts, axis=0)
    total = int(ak.sum(ak.num(combined)))
    print(f"\n  Combined: {total:,} matched pairs across {len(paths)} file(s)")
    return combined


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Inject pile-up dependence and the absolute-scale "
                                                 "distortion into the dijet sample.")
    parser.add_argument("--input", "-i", nargs="+", required=True, metavar="FILE",
                        help="the raw dijet files, in the order they are read")
    parser.add_argument("--output", "-o", required=True, metavar="FILE")
    parser.add_argument("--dataset", required=True, choices=sorted(settings.DATASETS),
                        help="the dataset to make (its settings: jetcorrml.settings.DATASETS)")
    args = parser.parse_args()
    inj = settings.DATASETS[args.dataset].injection
    pu, nl, ed, ab = inj.pileup, inj.nonlinear, inj.eta_density, inj.absolute

    # mu': correlated with N_PV by rho_muN, or the file's own mu
    if inj.mu is not None:
        mu_range, mu_rho = (inj.mu[0], inj.mu[1]), inj.mu[2]
        print(f"mu mode: correlated   U({mu_range[0]:.0f}, {mu_range[1]:.0f})  rho={mu_rho:.2f}")
    else:
        mu_range = mu_rho = None
        print("mu mode: fixed per-file (from DELPHES, fully correlated with NPV)")

    if ab is not None:
        print(f"JES mode: r0={ab.r0:.3f}  r1={ab.r1:.3f}  pt0={ab.pt0:.0f} GeV  r2={ab.r2:.3f}")
    else:
        print("JES mode: disabled (no absolute energy deficit injected)")
    if nl.epsilon > 0:
        print(f"Nonlinear mode: epsilon={nl.epsilon:.2f}  g0={nl.g0}  g1={nl.g1}  g2={nl.g2}  kappa={nl.kappa}")
    else:
        print("Nonlinear mode: disabled (purely linear pile-up)")
    jes = dict(r0=ab.r0, r1=ab.r1, pt0=ab.pt0, r2=ab.r2, eta0=ab.eta0) if ab is not None else {}
    print_true_values(pu.a0, pu.a1, pu.a2, pu.b0, pu.b1, pu.b2, **jes,
                      epsilon=nl.epsilon, g0=nl.g0, g1=nl.g1, g2=nl.g2, kappa=nl.kappa)

    print(f"Loading {len(args.input)} file(s)...")
    combined = inject(
        args.input,
        a0=pu.a0, a1=pu.a1, a2=pu.a2, b0=pu.b0, b1=pu.b1, b2=pu.b2,
        noise_sigma=inj.noise,
        seed=inj.seed,
        mu_range=mu_range,
        mu_rho=mu_rho,
        **jes,
        epsilon=nl.epsilon, g0=nl.g0, g1=nl.g1, g2=nl.g2, kappa=nl.kappa,
        eta_density_depth=ed.depth,
        eta_density_etac=ed.etac,
        eta_density_width=ed.width,
    )

    print(f"\nWriting to {args.output} ...")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    ak.to_parquet(combined, args.output)


if __name__ == "__main__":
    main()
