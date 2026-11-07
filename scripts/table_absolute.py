#!/usr/bin/env python3
"""Table III: closure of the absolute energy-scale correction, NN vs the parametric one.

Reads the corrected jets of both absolute-scale steps (both on the NN pile-up correction).

Metric (jetcorrml.absolute_closure): in each bin the deviation of the median
response E_corr/E_gen from unity is measured, and the figure of merit is the RMS
of those deviations across bins -- the analogue of the pile-up figure of merit,
which is the RMS across |eta| bins of the residual slope.  Zero is ideal.

Three binnings are reported: in E_gen, in |eta_det|, and over the grid of both.
The two-dimensional one is the most complete, since a correction can be right
when averaged over eta at each energy and still wrong cell by cell.

sigma_68 is reported alongside but should not be read as a quality measure.  For
a deterministic correction c(E_reco,|eta|), applying it is a change of variable,
so the width transforms by the local derivative of the map and sigma_68 responds
to the SLOPE of the correction rather than its accuracy.  A correction which
deviates from the truth can be narrower than the exact one.

Writes output/paper/tables/absolute.csv and .tex:
    uv run python scripts/table_absolute.py
"""
import argparse

from jetcorrml.absolute_closure import SERIES, closure_of_scan, regime_mean
from jetcorrml.paper import tables

NAME = "absolute"
REGIMES = (("lo", r"lo-$p_T$"), ("hi", r"hi-$p_T$"))
BINNINGS = (("e", "closure_rms_e"), ("eta", "closure_rms_eta"), ("2d", "closure_rms_2d"))


def rows(res):
    """One row per pT regime: the closure of each series in each binning, averaged over rho_muN."""
    return [{"regime": regime, **{f"{lab}_{b}": regime_mean(res, regime, lab, key)
                                  for lab, _ in SERIES for b, key in BINNINGS}}
            for regime, _ in REGIMES]


def latex(rows):
    shown = dict(REGIMES)
    L = [r"\begin{table}[ht]",
         r"\centering",
         r"\caption{Closure of the absolute energy-scale correction, NN versus "
         r"the parametric baseline, averaged over the seven $\rho_{\mu N}$ "
         r"values in each $p_T$ regime. Entries are the RMS across bins of "
         r"$\mathrm{median}(E^{\mathrm{corr}}/E^{\mathrm{gen}}) - 1$; zero is "
         r"ideal. The two-dimensional binning is the most complete, a "
         r"correction being able to close on average in $\eta$ at each energy "
         r"and still fail cell by cell.}",
         r"\label{tab:abs_closure}",
         r"\begin{tabular}{l ccc ccc}",
         r"\toprule",
         r" & \multicolumn{3}{c}{NN} & \multicolumn{3}{c}{Param.} \\",
         r"\cmidrule(lr){2-4} \cmidrule(lr){5-7}",
         r"$p_T$ regime & $E_{\rm gen}$ & $|\eta|$ & 2D"
         r" & $E_{\rm gen}$ & $|\eta|$ & 2D \\",
         r"\midrule"]
    for r in rows:
        cells = [f"{v:.4f}" if v == v else "--"
                 for v in (r[f"{lab}_{b}"] for lab, _ in SERIES for b, _ in BINNINGS)]
        L.append(f"{shown[r['regime']]} & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return L


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.parse_args()

    res, missing = closure_of_scan()
    if not res:
        raise SystemExit("no absolute-scale outputs found: run the NN and the parametric "
                         "absolute scale first")
    head = []
    if missing:
        head.append(f"# note: {len(missing)} expected file(s) absent")
        head += [f"#   {p}" for p in missing[:5]]
        if len(missing) > 5:
            head.append(f"#   ... and {len(missing)-5} more")
    rs = rows(res)
    tables.write(NAME, rs, head + latex(rs))


if __name__ == "__main__":
    main()
