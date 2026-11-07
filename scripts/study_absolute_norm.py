#!/usr/bin/env python3
"""study_absolute_norm.py -- is the absolute-scale comparison like for like?  READ-ONLY.

The NN absolute correction ends with one overall factor that sets the median
response of the corrected jets (E_gen >= 50 GeV) to unity.  The parametric
correction has no such factor.  This study gives the parametric result the
same factor, computed by the same rule from its own corrected jets, and
recomputes the closure of the paper's table.  Nothing is trained.

    uv run python scripts/study_absolute_norm.py

Run from the top of a pipeline folder in which the paper run has finished.
It READS the corrected files of the absolute-scale step (data/absolute/nn,
data/absolute/parametric-cross) and the factor the NN step recorded
(data/absolute/nn/<dataset>_factor.json).  It WRITES only
study_absolute_norm/REPORT.txt and study_absolute_norm/numbers.json.

Rows, per dataset and averaged over the seven rho_muN values of each regime:

    NN                     as in the paper (with its overall factor)
    NN, no factor          the same correction with the factor taken out
    parametric             as in the paper (no factor)
    parametric, factor     the paper's parametric correction times its own overall factor

Closure: RMS over bins of median(E_corr/E_gen) - 1, as scripts/table_absolute.py
(same jets, same bins).  A second block scores every row by the PEAK of a Gaussian
fit to the core in each bin, the quantity the parametric method targets, in place
of the median, which the network targets.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

from jetcorrml import paths, settings
from jetcorrml.absolute_closure import DEFAULT_E_EDGES, DEFAULT_ETA_EDGES, closure_jets, closure_metrics

MIN_COUNT = 50                                       # as the absolute-scale table
REGIMES = (("lo-pT", "lo"), ("hi-pT", "hi"))
FILES = {"NN": paths.nn_absolute, "parametric": lambda ds: paths.parametric_absolute(ds, "nn")}
ROWS = ("NN", "NN, no factor", "parametric", "parametric, factor")


def peak_closure(resp, e_gen, eta, sel, fit):
    """As closure_metrics, with the peak of a Gaussian fit to the core in place of the median."""
    ae = np.abs(eta)
    i_e = np.digitize(e_gen, DEFAULT_E_EDGES) - 1
    i_eta = np.digitize(ae, DEFAULT_ETA_EDGES) - 1
    n_e, n_eta = len(DEFAULT_E_EDGES) - 1, len(DEFAULT_ETA_EDGES) - 1
    ok = sel & (i_e >= 0) & (i_e < n_e) & (i_eta >= 0) & (i_eta < n_eta) & np.isfinite(resp)

    def dev(m):
        return fit(resp[m])[0] - 1.0 if int(m.sum()) >= MIN_COUNT else np.nan

    rms = lambda v: float(np.sqrt(np.nanmean(np.square(v)))) if np.isfinite(v).any() else float("nan")
    d_e = [dev(ok & (i_e == a)) for a in range(n_e)]
    d_eta = [dev(ok & (i_eta == b)) for b in range(n_eta)]
    d_2d = [dev(ok & (i_e == a) & (i_eta == b)) for a in range(n_e) for b in range(n_eta)]
    return {"peak_2d": rms(np.array(d_2d)), "peak_e": rms(np.array(d_e)), "peak_eta": rms(np.array(d_eta))}


def score(resp, e_gen, eta, sel, fit):
    m = closure_metrics(resp, e_gen, eta, sel=sel, e_edges=DEFAULT_E_EDGES, eta_edges=DEFAULT_ETA_EDGES,
                        min_count=MIN_COUNT)
    out = {"rms_2d": m["closure_rms_2d"], "rms_e": m["closure_rms_e"], "rms_eta": m["closure_rms_eta"],
           "median": float(np.median(resp[sel])), "jets": int(sel.sum())}
    if fit is not None:
        out |= peak_closure(resp, e_gen, eta, sel, fit)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default="study_absolute_norm")
    ap.add_argument("--no-peak", action="store_true", help="leave out the peak-based score (faster)")
    a = ap.parse_args()
    fit = None
    if not a.no_peak:
        from jetcorrml.response import gaussian_core_peak as fit   # the parametric step's own fit

    rows, missing = [], []
    for regime, reg in REGIMES:
        for rho in settings.RHO_MUN:
            ds = settings.pileup_dataset(rho, reg).name
            files = {k: Path(path_of(ds)) for k, path_of in FILES.items()}
            files["NN factor"] = Path(paths.nn_absolute_factor(ds))
            if not all(p.is_file() for p in files.values()):
                missing.append(ds)
                continue
            r_nn, eg, eta, sel = closure_jets(files["NN"])
            r_at, eg2, eta2, sel2 = closure_jets(files["parametric"])
            if not (len(eg) == len(eg2) and np.allclose(eg, eg2, rtol=0, atol=1e-6)):
                sys.exit(f"{ds}: the NN and parametric files do not hold the same jets")
            f_at = float(np.median(r_at[sel2]))                 # the NN's rule, applied to the parametric output
            f_nn = json.loads(files["NN factor"].read_text())["median_response_before"]
            cases = {"NN": r_nn, "NN, no factor": r_nn * f_nn, "parametric": r_at, "parametric, factor": r_at / f_at}
            for name in ROWS:
                if name in cases:
                    rows.append({"regime": regime, "rho_muN": rho, "dataset": ds, "row": name,
                                 "factor_removed_or_applied": {"NN, no factor": f_nn, "parametric, factor": f_at}.get(name),
                                 **score(cases[name], eg, eta, sel, fit)})
            print(f"  {ds}: parametric median {f_at:.4f}, NN median before its factor {f_nn:.4f}", flush=True)
    if not rows:
        sys.exit("no corrected files found (run from the top of a pipeline folder, after the paper run)")

    L = []
    say = L.append
    say("Absolute scale: the NN's overall factor given to the parametric correction as well")
    say("Closure in %: RMS over bins of median(E_corr/E_gen) - 1 (E_gen >= 50 GeV; the paper's table).")
    if missing:
        say(f"NOT FOUND, left out: {', '.join(missing)}")
    say("")

    def mean(regime, row, key):
        v = [r[key] for r in rows if r["regime"] == regime and r["row"] == row and key in r]
        return (float(np.mean(v)), len(v)) if v else (float("nan"), 0)

    for title, keys in (("MEDIAN-based closure (the paper's)", ("rms_e", "rms_eta", "rms_2d")),
                        ("PEAK-based closure (Gaussian fit to the core in each bin)", ("peak_e", "peak_eta", "peak_2d"))):
        if keys[0].startswith("peak") and fit is None:
            continue
        say(f"{title}, mean over the rho_muN values of each regime")
        say(f"  {'':8s} {'':22s} {'E_gen':>7s} {'|eta|':>7s} {'2D':>7s} {'median response':>17s}  datasets")
        for regime, _ in REGIMES:
            for row in ROWS:
                e, n = mean(regime, row, keys[0])
                if n == 0:
                    continue
                say(f"  {regime:8s} {row:22s} {100 * e:7.2f} {100 * mean(regime, row, keys[1])[0]:7.2f} "
                    f"{100 * mean(regime, row, keys[2])[0]:7.2f} {mean(regime, row, 'median')[0]:17.4f}  {n}")
            nn, at, atf = (mean(regime, r, keys[2])[0] for r in ("NN", "parametric", "parametric, factor"))
            say(f"  {regime:8s} NN / parametric (2D): as in the paper {nn / at:.2f}; with the factor for both {nn / atf:.2f}")
            say("")

    say("Per dataset, 2D closure in % (median-based):")
    say(f"  {'':8s} {'rho_muN':>7s} " + " ".join(f"{r:>20s}" for r in ROWS) + "   parametric median")
    for regime, _ in REGIMES:
        for rho in settings.RHO_MUN:
            get = lambda row, k="rms_2d": next((r[k] for r in rows if r["regime"] == regime and r["rho_muN"] == rho
                                                 and r["row"] == row), None)
            if get("NN") is None:
                continue
            say(f"  {regime:8s} {rho:7.1f} "
                + " ".join(f"{100 * get(r):20.2f}" if get(r) is not None else f"{'-':>20s}" for r in ROWS)
                + f"   {get('parametric', 'median'):.4f}")
    say("")
    say("parametric, factor = the paper's parametric correction divided by the median response of its corrected jets")
    say("(E_gen >= max(50, 21 cosh eta) GeV), the rule the NN step applies to itself.  NN, no factor = the NN's")
    say("correction with that step taken out again (by the median the NN step recorded before it).")
    text = "\n".join(L)
    print("\n" + text)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "REPORT.txt").write_text(text + "\n")
    (out / "numbers.json").write_text(json.dumps(rows, indent=1))
    print(f"\nwritten: {out}/REPORT.txt, {out}/numbers.json")


if __name__ == "__main__":
    main()
