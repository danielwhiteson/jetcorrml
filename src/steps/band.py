#!/usr/bin/env python3
"""
Nuisance-parameter band of the NN in-situ correction (paper Sec. VIII).

Retrains the in-situ step B times on resampled events (--retraining b), each
writing its scale s(pt, |eta|) on a grid (a sidecar .npz), and decomposes the
bundle by principal components:
    s(x, nu) = z0(x) * exp( sum_i nu_i sqrt(lam_i) phi_i(x) )
It writes the share of the variance the leading components carry (the paper's
numbers) to output/paper/tables/band_modes.csv and .txt.  The decomposition is
jetcorrml.band's; the closure figures with the band are drawn by
jetcorrml.paper.fig_band from the retrainings.

Usage (the number of retrainings and their seeds are in jetcorrml.settings.Band):
    python -m steps.band --dist expo [--refits-only] [--jobs N] \
        [--pair-selection RULE --selection-pt FILE]
"""
import argparse
import os
import subprocess
import sys

import numpy as np

from jetcorrml import paths, settings
from jetcorrml.band import decompose
from jetcorrml.paper import tables


def retrain(ds, d, b, extra, quiet=False):
    """Retraining b of the in-situ correction on division 0 of distortion d."""
    cmd = [sys.executable, "-m", "steps.insitu_nn",
           "--mc", paths.division(ds, d, 0, "MC"), "--data", paths.division(ds, d, 0, "DATA"),
           "--dist", d, "--retraining", str(b),
           "--output-data", paths.retraining(ds, d, b, "DATA"),
           "--output-mc", paths.retraining(ds, d, b, "MC"),
           "--grid-out", paths.retraining_grid(ds, d, b)] + extra
    if quiet:      # several at once: keep each run's output in its own file
        with open(paths.retraining_log(ds, d, b), "w") as log:
            subprocess.run(cmd, check=True, stdout=log, stderr=subprocess.STDOUT)
    else:
        subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dist", required=True, choices=sorted(settings.DISTORTIONS),
                    help="the distortion of the data half")
    ap.add_argument("--jobs", type=int, default=1,
                    help="retrainings to run at the same time (default 1). Each is its own process "
                         "with its own seed, so the result does not depend on this; with more than "
                         "one, each run's output goes to a .log file next to its parquets.")
    ap.add_argument("--refits-only", action="store_true",
                    help="run the retrainings and stop (for distortions whose retrainings are only "
                         "read by the band figures)")
    ap.add_argument("--pair-selection", default=None,
                    help="how the retrainings choose their pairs (as the nominal last pass)")
    ap.add_argument("--selection-pt", default=None, metavar="FILE",
                    help="the nominal pass before the last, whose jets the pairs are chosen on")
    ap.add_argument("--mini", action="store_true", help="the reduced run of run_paper.py --mini")
    args = ap.parse_args()
    run = settings.get(args.mini)
    ds, d, nboot = run.insitu.dataset, args.dist, run.band.retrainings
    extra = ((["--pair-selection", args.pair_selection, "--selection-pt", args.selection_pt]
              if args.pair_selection else []) + (["--mini"] if args.mini else []))
    os.makedirs(paths.retraining_dir(d), exist_ok=True)

    print(f"=== {nboot} bootstrap retrainings ===", flush=True)
    if args.jobs <= 1:
        for b in range(nboot):
            retrain(ds, d, b, extra)
    else:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            futs = {pool.submit(retrain, ds, d, b, extra, quiet=True): b
                    for b in range(nboot)}
            failed = []
            for n_done, f in enumerate(as_completed(futs), 1):
                b = futs[f]
                try:
                    f.result()
                    print(f"  bootstrap {b:>3} done ({n_done}/{nboot})", flush=True)
                except Exception as e:                      # noqa: BLE001
                    failed.append(b)
                    print(f"  bootstrap {b:>3} FAILED: {e}", flush=True)
        if failed:
            raise SystemExit(f"{len(failed)} bootstrap run(s) failed: {sorted(failed)}; "
                             f"see the .log files in {paths.retraining_dir(d)}")

    if args.refits_only:
        print(f"  {nboot} retrainings of {ds}, {d} are in {paths.retraining_dir(d)}")
        return

    sidecars = paths.retraining_grids(ds, d)
    if len(sidecars) < 2:
        raise SystemExit(f"fewer than two retrainings in {paths.retraining_dir(d)}: "
                         "no band without at least two retrainings.")
    pt, eta, z0, lam, phi, C, B = decompose(sidecars)
    tot = max(lam.sum(), 1e-12)
    share, cumulative = lam / tot, np.cumsum(lam) / tot
    modes95 = int(np.sum(cumulative < 0.95) + 1)
    rows = [{"mode": i + 1, "share": float(share[i]), "cumulative": float(cumulative[i])}
            for i in range(min(6, len(lam)))]
    tables.write("band_modes", rows,
                 [f"  PCA frac var: {', '.join(f'{v:.2f}' for v in share[:6])}  modes95={modes95}"], ext="txt")


if __name__ == "__main__":
    main()
