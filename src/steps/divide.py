"""
The division of the sample into simulation and data, and the distortion of
the data half (paper Sec. VII A).

The corrected sample is divided at the event level (all jets of an event go
to the same half) by a seeded uniform draw; the simulation half is left as it
is and the data half's jets are scaled by f, normalised so that f = 1 + d0 at
pT = 45 GeV, |eta| = 0 (and rho = 15 GeV/area):

  linear     f = 1 + d0 + d1 ln(pT/45) + d2 |eta|
  quadratic  f = 1 + d0 + d1 ln(pT/45) + d2 |eta|^2
  threshold  f = 1 + d0 + d1 ln(pT/45) + d2 sign(|eta| - 1.5)
  expo       f = 1 + d0 + d1 ln(pT/45) + d2 (exp|eta| - 1) / (exp(2.5) - 1)
  rhoeta     f = 1 + d0 + d1 ln(pT/45) + d2 |eta|^2 + d3 (rho/15 - 1)(45/pT)

The distortions and their coefficients are in jetcorrml.settings (DISTORTIONS):

    uv run python -m steps.divide -i JETS.pq \
        --output-mc MC.pq --output-data DATA.pq --dist expo --division 0
"""

import argparse
from pathlib import Path

import awkward as ak
import numpy as np

from jetcorrml import settings
from jetcorrml.distortion import PT_REF, RHO_REF, residual_distortion


def _print_residual_table(d0, d1, d2, d3, form):
    """Print a sanity-check table of scale factors at key (pT, x) points.

    For the kinematic-only forms (linear/quadratic/threshold/expo) the
    second axis is |eta|.  For 'rhoeta' (eta AND rho) two tables are printed: an
    |eta|-axis table at rho=RHO_REF (where the rho-term vanishes, isolating
    d2*|eta|^2) and a rho-axis table at |eta|=0 (isolating the d3 rho term).
    """
    pts = [20.0, 45.0, 100.0, 300.0]
    col_w = 10

    def _val(pt, eta, rho):
        return residual_distortion(
            np.full(1, pt), np.full(1, eta), np.full(1, rho),
            d0, d1, d2, d3=d3, form=form,
        )[0]

    def _eta_table(rho_fixed):
        etas = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]
        header = (f"  {'|eta|':>5}  "
                  + "  ".join(f"pT={p:.0f}".rjust(col_w) for p in pts))
        print(header)
        print("  " + "-" * (len(header) - 2))
        for eta in etas:
            row = f"  {eta:>5.1f}  "
            for p in pts:
                row += f"{_val(p, eta, rho_fixed):>{col_w}.5f}  "
            print(row)

    def _rho_table(eta_fixed):
        rhos = [5.0, 10.0, 15.0, 20.0, 25.0, 30.0]
        header = (f"  {'rho':>5}  "
                  + "  ".join(f"pT={p:.0f}".rjust(col_w) for p in pts))
        print(header)
        print("  " + "-" * (len(header) - 2))
        for rho in rhos:
            row = f"  {rho:>5.1f}  "
            for p in pts:
                row += f"{_val(p, eta_fixed, rho):>{col_w}.5f}  "
            print(row)

    if form == "rhoeta":
        print("  [|eta|-axis at rho=RHO_REF -> rho-term=0, shows d2*|eta|^2]")
        _eta_table(rho_fixed=RHO_REF)
        print("  [rho-axis at |eta|=0 -> eta-term=0, shows d3*(rho/15-1)*(45/pT)]")
        _rho_table(eta_fixed=0.0)
    else:
        _eta_table(rho_fixed=RHO_REF)


# =============================================================================
# Load input files
# =============================================================================


def load_files(paths):
    """Load and concatenate parquet files without modifying any fields."""
    parts = []
    for path in paths:
        print(f"  Loading {path}")
        matches = ak.from_parquet(path)
        n_jets = int(ak.sum(ak.num(matches)))
        print(f"    -> {n_jets:,} matched pairs")
        parts.append(matches)

    combined = ak.concatenate(parts, axis=0)
    total = int(ak.sum(ak.num(combined)))
    print(f"\n  Combined: {total:,} matched pairs across {len(paths)} file(s)")
    return combined


# =============================================================================
# MC / data split and residual distortion injection
# =============================================================================


def split_and_distort(combined, d0, d1, d2, d3=0.0, form="linear", split_fraction=0.5, seed=0):
    """
    Split a combined sample into MC and data sub-samples, then apply
    the residual distortion to the data half only.

    Split mechanics
    ---------------
    The split is done at the EVENT level: a uniform random draw in [0, 1) is
    made once per event, and events below split_fraction go to MC.  This
    guarantees that all jets in a given event end up in the same output file —
    there is no information leak between the two halves.  The seed is separate
    from any upstream seeds so it can be varied independently.

    Residual distortion (data only)
    --------------------------------
    After splitting, every jet in the data sub-sample is scaled by the factor
    returned by residual_distortion(pT, |eta|, d0, d1, d2, form).

    The distortion is applied as:
        pT_data = pT * scale
        E_data  = E  * scale          (same factor — residual JES interpretation)
    The jet mass and eta are unchanged.

    Parameters
    ----------
    combined        : awkward array (concatenated parquet records)
    d0, d1, d2      : residual distortion coefficients
    form            : 'linear', 'quadratic', 'threshold', or 'expo'
    split_fraction  : fraction of events assigned to MC (default 0.5)
    seed            : RNG seed for the event split (default 0)

    Returns
    -------
    mc_sample   : awkward array — unchanged jets
    data_sample : awkward array — jets with residual distortion applied
    """
    n_events = len(combined)
    rng = np.random.default_rng(seed)

    # One uniform draw per event; below split_fraction → MC
    draw = rng.uniform(0.0, 1.0, n_events)
    is_mc = draw < split_fraction
    is_data = ~is_mc

    n_mc = int(is_mc.sum())
    n_data = int(is_data.sum())
    print(
        f"\n  Split: {n_mc:,} MC events  /  {n_data:,} data events  "
        f"(fraction_mc = {n_mc / n_events:.3f})"
    )

    mc_sample = combined[is_mc]
    data_sample = combined[is_data]

    # ── Residual distortion on data only ─────────────────────────────────────
    def flat(arr):
        return ak.to_numpy(ak.flatten(arr, axis=None)).astype(float)

    counts_data = ak.num(data_sample)
    pt_flat = flat(data_sample.jet.pt)
    abs_eta_flat = np.abs(flat(data_sample.jet.eta))
    # event.rho is event-level scalar; broadcast across each event's jets so
    # we get one rho value per jet (matches pt_flat / abs_eta_flat lengths).
    # The four kinematic-only forms ignore rho; only rhoeta uses it.
    rho_per_jet = ak.broadcast_arrays(data_sample.event.rho, data_sample.jet.pt)[0]
    rho_flat = flat(rho_per_jet)

    # Single call to the isolated distortion function — this is the only place
    # in the codebase that applies the residual scale factor to data jets.
    # A jet at pT <= 0 has no log pT (and the rho term divides by pT): its factor is taken
    # at the reference pT, so it stays finite.
    _nonpos = ~(pt_flat > 0)
    scale = residual_distortion(
        np.where(_nonpos, PT_REF, pt_flat), abs_eta_flat, rho_flat, d0, d1, d2, d3=d3, form=form
    )

    pt_distorted = pt_flat * scale
    e_distorted = flat(data_sample.jet.E) * scale  # same factor → residual JES

    jet_fields = {f: data_sample.jet[f] for f in ak.fields(data_sample.jet)}
    jet_fields["pt"] = ak.unflatten(pt_distorted, counts_data)
    jet_fields["E"] = ak.unflatten(e_distorted, counts_data)

    data_sample = ak.zip(
        {
            "jet": ak.zip(jet_fields),
            "gen": data_sample.gen,
            "event": data_sample.event,
            "deltaR": data_sample.deltaR,
        }
    )

    n_jets_mc = int(ak.sum(ak.num(mc_sample)))
    n_jets_data = int(ak.sum(ak.num(data_sample)))
    print(f"  MC jets: {n_jets_mc:,}   data jets: {n_jets_data:,}")
    print(f"  Residual form='{form}'  d0={d0:+.4f}  d1={d1:+.4f}  d2={d2:+.4f}")
    print(
        f"  Scale factor stats:  mean={scale.mean():.5f}  "
        f"min={scale.min():.4f}  max={scale.max():.4f}  "
        f"std={scale.std():.5f}"
    )

    return mc_sample, data_sample


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Divide the corrected sample into simulation and data, and distort the data half.")
    parser.add_argument("--input", "-i", nargs="+", required=True, metavar="FILE")
    parser.add_argument("--output-mc", required=True, metavar="FILE")
    parser.add_argument("--output-data", required=True, metavar="FILE")
    parser.add_argument("--dist", required=True, choices=sorted(settings.DISTORTIONS),
                        help="the distortion of the data half (jetcorrml.settings.DISTORTIONS)")
    parser.add_argument("--division", type=int, required=True, metavar="N",
                        help="the division: its seed chooses which events become the simulation")
    args = parser.parse_args()
    dist = settings.DISTORTIONS[args.dist]

    print(f"Loading {len(args.input)} file(s)...")
    combined = load_files(args.input)

    _d3_str = f"  d3={dist.d3:+.4f}" if dist.form == "rhoeta" else ""
    print(
        f"\nResidual distortion  form='{dist.form}'  "
        f"d0={dist.d0:+.4f}  d1={dist.d1:+.4f}  d2={dist.d2:+.4f}{_d3_str}"
    )
    print(f"Scale factor table  (reference pT={PT_REF:.0f} GeV):")
    _print_residual_table(dist.d0, dist.d1, dist.d2, dist.d3, dist.form)

    mc_sample, data_sample = split_and_distort(
        combined,
        d0=dist.d0,
        d1=dist.d1,
        d2=dist.d2,
        d3=dist.d3,
        form=dist.form,
        split_fraction=settings.SPLIT_FRACTION,
        seed=args.division,
    )

    print(f"\nWriting MC   sample -> {args.output_mc}")
    for _out in (args.output_mc, args.output_data):
        Path(_out).parent.mkdir(parents=True, exist_ok=True)
    ak.to_parquet(mc_sample, args.output_mc)
    print(f"Writing data sample -> {args.output_data}")
    ak.to_parquet(data_sample, args.output_data)
    print(f"\nTruth: form='{dist.form}' d0={dist.d0:+.4f} d1={dist.d1:+.4f} d2={dist.d2:+.4f}{_d3_str}")


if __name__ == "__main__":
    main()
