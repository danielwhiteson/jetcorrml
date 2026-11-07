#!/usr/bin/env python3
"""Table IV: agreement of corrected data with simulation, NN against parametric in-situ.

For each distortion of the in-situ dataset and each division of the sample
(jetcorrml.settings: dists, divisions), both methods correct the same data half;
the table gives the Kolmogorov-Smirnov distance between corrected data and
simulation of the response and of the dijet asymmetry, mean and spread over the
divisions.  The metrics come from jetcorrml.insitu_metrics, as in the in-situ figures.

Dijet asymmetry test: pairs are selected AFTER correction, from each sample's own
jets (MC: MC jets; data: the method's corrected jets), with the pT cut on the
CENTRAL jet only and the third-jet veto relative to the central jet; the
asymmetry uses corrected pT throughout.  A perfect correction gives the best score.

Writes output/paper/tables/insitu.csv and .tex:

    uv run python scripts/table_insitu.py [--mini]
"""
import argparse
from pathlib import Path

import numpy as np

from jetcorrml import paths, settings
from jetcorrml.insitu_metrics import DR_MAX, GEN_PT_MIN, cell_data, metrics_from
from jetcorrml.paper import tables

NAME = "insitu"
METHODS = ("nn", "parametric")
TESTS = ("r", "a")                  # the response and the dijet asymmetry: metrics_from's first two
UNIT = 1e-3                         # the table is printed in units of 10^-3
N_WORDS = {2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine",
           10: "ten", 20: "twenty"}


def fmt(v, pct=False):
    if v is None:
        return r"---"
    return f"{100*v:.2f}" if pct else f"{v:.4f}"


def ks_distances(dataset, dists, divisions):
    """{(division, dist, method): (D_KS of the response, of the asymmetry)}; both methods
    correct the same division (the NN chain's output)."""
    out = {}
    for s in divisions:
        for d in dists:
            mc, data = Path(paths.division(dataset, d, s, "MC")), Path(paths.division(dataset, d, s, "DATA"))
            for method in METHODS:
                f = lambda kind: Path(paths.insitu(dataset, d, s, method, kind))
                m = metrics_from(cell_data(mc, data, f("MC"), f("DATA"), GEN_PT_MIN, DR_MAX))
                out[(s, d, method)] = (m[0], m[1])
    return out


def rows(ks, dists, divisions):
    """One row per distortion: for each method and test the mean and the spread over the
    divisions (None if missing everywhere), then the value in each division."""
    out = []
    for d in dists:
        r = {"distortion": d}
        for method in METHODS:
            for i, t in enumerate(TESTS):
                vals = [ks[(s, d, method)][i] for s in divisions]
                vals = [v for v in vals if v is not None]
                mean, sd = (float(np.mean(vals)), float(np.std(vals))) if vals else (None, None)
                r[f"{method}_{t}_mean"], r[f"{method}_{t}_spread"] = mean, sd
        for s in divisions:
            for method in METHODS:
                for i, t in enumerate(TESTS):
                    r[f"{method}_{t}_division{s}"] = ks[(s, d, method)][i]
        out.append(r)
    return out


def cell(mean, sd, n_divisions):
    """One entry in units of 10^-3: the spread to one significant figure, the mean to the
    same decimal place (whole numbers whenever the spread is 10^-3 or more).  The spread
    over a handful of divisions is itself known to about a third, so more digits say nothing;
    the unrounded numbers of every division are written under the table."""
    if mean is None:
        return r"---"
    m, e = mean / UNIT, sd / UNIT
    if n_divisions == 1 or not e > 0:
        return f"{m:.0f}"
    nd = max(0, -int(np.floor(np.log10(e))))
    return f"{m:.{nd}f} $\\pm$ {e:.{nd}f}"


def latex(rows, dataset, divisions):
    n = len(divisions)
    L = [r"% dataset: " + dataset,
         r"% D_KS: Kolmogorov-Smirnov distance (corrected data vs MC); "
         r"r = response p_T^reco/p_T^gen, A = dijet asymmetry. "
         r"delta R_x metrics are shown in the figures, not tabulated.",
         r"% asymmetry test: pairs selected after correction, pT cut on the central jet only",
         r"% parametric outputs: data/insitu/parametric"]
    if n > 1:
        L.append(r"% divisions: " + " ".join(str(s) for s in divisions)
                 + r"; entries are mean +- spread (standard deviation) over divisions")
    for name, t in (("response", "r"), ("asymmetry", "a")):
        better = total = 0
        for s in divisions:
            for r in rows:
                nn, par = r[f"nn_{t}_division{s}"], r[f"parametric_{t}_division{s}"]
                if r["distortion"] == "nom" or nn is None or par is None:
                    continue
                total += 1
                better += nn < par
        L.append(f"% NN lower than parametric on the {name} test in {better} of {total} "
                 f"(distortion, division) cases (the undistorted 'nom' row not counted)")
    dists = [r["distortion"] for r in rows]
    L += [r"\begin{table}",
          r"\centering",
          r"\caption{Agreement between corrected data and simulation for the NN and parametric in-situ "
          r"corrections: the Kolmogorov--Smirnov distance between the two distributions of the response "
          r"($D_\text{KS}^{r}$) and of the dijet asymmetry ($D_\text{KS}^{A}$), in units of $10^{-3}$; "
          r"smaller is better. "
          + (rf"Values are the mean and spread over {N_WORDS.get(n, str(n))} divisions of the sample into "
             rf"simulation and data. " if n > 1 else "")
          + (r"The row ``nom'' has no distortion." if "nom" in dists else "") + "}",
          r"\label{tab:mcdata}",
          r"\begin{tabular}{lcccc}",
          r"\hline\hline",
          r" & \multicolumn{2}{c}{NN (combined-QM)}"
          r" & \multicolumn{2}{c}{Param.} \\",
          r"\cline{2-3} \cline{4-5}"]
    head = r" & $D_\text{KS}^{r}$ & $D_\text{KS}^{A}$"
    L += [r"distortion" + head + head + r" \\", r"\hline"]
    for r in rows:
        cells = [r["distortion"].replace("_", r"\_")] + [
            cell(r[f"{m}_{t}_mean"], r[f"{m}_{t}_spread"], n) for m in METHODS for t in TESTS]
        L.append(" & ".join(cells) + r" \\")
    L += [r"\hline\hline", r"\end{tabular}", r"\end{table}"]
    if n > 1:
        L.append(r"% mean and spread, unrounded:  distortion   NN D_KS^r   NN D_KS^A   Param. D_KS^r   Param. D_KS^A")
        for r in rows:
            st = [(r[f"{m}_{t}_mean"], r[f"{m}_{t}_spread"]) for m in METHODS for t in TESTS]
            L.append(f"% {r['distortion']:10s} "
                     + "   ".join("---" if m is None else f"{m:.5f} +- {e:.5f}" for m, e in st))
        L.append(r"% per division:  distortion   NN D_KS^r  NN D_KS^A   Param. D_KS^r  Param. D_KS^A")
        for s in divisions:
            for r in rows:
                v = [r[f"{m}_{t}_division{s}"] for m in METHODS for t in TESTS]
                L.append(f"% division {s}: {r['distortion']:10s} {fmt(v[0])} {fmt(v[1])}   {fmt(v[2])} {fmt(v[3])}")
    return L


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mini", action="store_true", help="the reduced run of run_paper.py --mini")
    args = p.parse_args()
    run = settings.get(args.mini)
    dataset, dists, divisions = run.insitu.dataset, list(run.dists), list(run.divisions)
    ks = ks_distances(dataset, dists, divisions)
    rs = rows(ks, dists, divisions)
    tables.write(NAME, rs, latex(rs, dataset, divisions))


if __name__ == "__main__":
    main()
