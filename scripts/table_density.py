#!/usr/bin/env python3
"""Table V: the pile-up step on the forward pile-up density sample (App. A).

Table II averages the RMS_eta slopes over the seven rho_muN datasets in each pT
regime.  The eta-density injection has no such sweep -- mu enters only through
its correlation with rho -- so this reports the same metric for one dataset,
read from the corrected jets:
  * bin jets in |eta|,
  * in each bin, linear-regress the CORRECTED pT against N_PV and against mu,
  * take the RMS across bins of those per-bin slopes.
Units are GeV per unit N_PV (or mu).  Zero is ideal; smaller is better.  RMS is
used rather than a signed mean so that slopes of opposite sign in different |eta|
regions cannot cancel.

Rows: 'no correction' (the injected jets with only rho*A removed, i.e. what the
correction stages are asked to fix), then NN and parametric.  The uncorrected row
is what gives a single-dataset table its scale, since there is no sweep to
compare against.  Writes output/paper/tables/density.csv and .tex:

    uv run python scripts/table_density.py
"""
import argparse
from pathlib import Path
import numpy as np

from jetcorrml import paths, settings
from jetcorrml.paper import tables
from jetcorrml.pileup_slopes import load_jets, rms_slopes

NAME = "density"
CAPTION = (
    r"Residual pile-up dependence after the pile-up correction for the $\eta$-density injection,"
    "\n"
    r"NN versus the parametric baseline.  Entries are the RMS across $|\eta|$ bins"
    "\n"
    r"of the post-correction slope of $p_T$ against each pile-up variable (GeV per"
    "\n"
    r"unit $N_{\mathrm{PV}}$ or $\mu$; smaller is better, zero is ideal).  Unlike"
    "\n"
    r"Table~II there is no $\rho_{\mu N}$ sweep to average over: the residual here"
    "\n"
    r"arises from a central-anchored $\rho$ under-subtracting forward jets, so $\mu$"
    "\n"
    r"enters only through its correlation with $\rho$.  The uncorrected row is the"
    "\n"
    r"injected sample with only $\rho \times A$ removed."
)


SHOWN = {"none": r"$\rho A$ only", "nn": "NN", "parametric": "Param.", "ratio": "NN/Param."}
METRICS = ("rms_npv", "rms_mu", "rms_npv_fwd", "rms_mu_fwd")


def rows(slopes):
    """slopes: [(correction, rms_slopes result)].  One row per correction, then the ratio
    NN/parametric (None where it is undefined)."""
    out = [{"correction": c, **{k: m[k] for k in METRICS}} for c, m in slopes]
    d = dict(slopes)
    if "nn" in d and "parametric" in d:
        def ratio(k):
            a, b = d["nn"][k], d["parametric"][k]
            return a / b if (b == b and abs(b) > 1e-12 and a == a) else None
        out.append({"correction": "ratio", **{k: ratio(k) for k in METRICS}})
    return out


def latex(rows, label="tab:etadens", forward_from=2.0, of="r"):
    """of: what was regressed, 'r' (the residual) or 'p_T' (the corrected pT; the figure's choice)."""
    q = r"p_{\textrm{T}}" if of == "p_T" else "r"
    L = [r"\begin{table}[ht]",
         r"\centering",
         r"\caption{" + CAPTION + "}",
         r"\label{" + label + "}",
         r"\begin{tabular}{l cc cc}",
         r"\toprule",
         r" & \multicolumn{2}{c}{all $|\eta|$ bins}"
         r" & \multicolumn{2}{c}{$|\eta| \geq " + f"{forward_from:g}" + r"$ only} \\",
         r"\cmidrule(lr){2-3} \cmidrule(lr){4-5}",
         rf" & $\partial {q}/\partial N_{{\mathrm{{PV}}}}$"
         rf" & $\partial {q}/\partial \mu$"
         rf" & $\partial {q}/\partial N_{{\mathrm{{PV}}}}$"
         rf" & $\partial {q}/\partial \mu$ \\",
         r"\midrule"]

    def c(v):
        return f"{v:.3f}" if v == v else "--"

    def ratio(v):
        return "--" if v is None else f"{v:.2f}"

    for r in rows:
        if r["correction"] == "ratio":
            L.append(r"\midrule")
            L.append(f"{SHOWN['ratio']:<16} & " + " & ".join(ratio(r[k]) for k in METRICS) + " \\\\")
        else:
            L.append(f"{SHOWN[r['correction']]:<16} & " + " & ".join(c(r[k]) for k in METRICS) + " \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return L


DATASET = settings.DENSITY.name

ETA_BINS = np.array(settings.DENSITY_ETA_BINS)  # the |eta| bins of the table: those of its figure
FORWARD_FROM = settings.DENSITY.injection.eta_density.etac    # |eta| from which a bin counts as forward


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.parse_args()

    ds = DATASET
    srcs = [
        ("none", Path(paths.injected(ds)), True),
        ("nn", Path(paths.nn_pileup(ds)), False),
        ("parametric", Path(paths.parametric_pileup(ds)), False),
    ]

    head = ["% parametric baseline: the stratified fit",
            "% slopes of: the corrected pT (pT_gen not subtracted)"]
    slopes = []
    for name, path, sub in srcs:
        if not path.exists():
            print(f"# WARNING: missing {name}: {path}")
            continue
        d = load_jets(path, subtract_rho_a=sub)
        m = rms_slopes(d["pt"], d["npv"], d["mu"], d["eta"], ETA_BINS, outside="drop",
                       forward_from=FORWARD_FROM)
        slopes.append((name, m))

    if not slopes:
        raise SystemExit("no inputs found: run the injection and both pile-up steps for this dataset first")
    rs = rows(slopes)
    tables.write(NAME, rs, head + latex(rs, forward_from=FORWARD_FROM, of="p_T"))


if __name__ == "__main__":
    main()
