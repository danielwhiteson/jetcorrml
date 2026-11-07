"""
Table II: the pile-up step's performance, NN against the parametric correction.

The figure of merit is the RMS over |eta| bins of the slope of the residual
r = pT_reco - pT_gen against N_PV and against mu (GeV per unit; 0 is ideal),
averaged over the seven rho_muN of each pT regime.  Reads the corrected jets
of both pile-up steps; writes output/paper/tables/pileup.csv and .tex.

    uv run python scripts/table_pileup.py
"""
import argparse

import numpy as np

from jetcorrml import settings
from jetcorrml.paper import tables
from jetcorrml.pileup_slopes import residual_slopes_of_scan

NAME = "pileup"
CAPTION = (
    r"Residual pile-up dependence after the pile-up correction, NN versus the parametric"
    "\n"
    r"baseline, averaged over the seven $\rho_{\mu N}$ values in each $p_T$ regime."
    "\n"
    r"Entries are the mean across datasets of the RMS$_\eta$ slope of the"
    "\n"
    r"post-correction residual $r = p_T^{\mathrm{reco}} - p_T^{\mathrm{gen}}$ against"
    "\n"
    r"each pile-up variable (GeV per unit $N_{\mathrm{PV}}$ or $\mu$; smaller is"
    "\n"
    r"better, zero is ideal). The NN/Param. ratio is the residual dependence the NN"
    "\n"
    r"leaves relative to the baseline; the NN advantage is largest in the hi-$p_T$"
    "\n"
    r"regime. The parametric coefficients are derived in bins of the other pile-up"
    "\n"
    r"variable, following Ref.~\cite{ATLAS:2015ull}."
)
LABEL = "tab:pu"
REGIMES = (("lo", "lo-$p_T$"), ("hi", "hi-$p_T$"), ("all", "all"))
ETA_BINS = np.array(settings.PILEUP_ETA_BINS)                # the |eta| bins of the slopes


def _avg(data, dss, key):
    vals = [data[d][key] for d in dss if isinstance(data[d].get(key), (int, float))]
    return sum(vals) / len(vals) if vals else None


def rows(data):
    """One row per pT regime (lo, hi, all): the means over its datasets and their ratios."""
    lo = sorted(d for d in data if data[d]["regime"] == "lo")
    hi = sorted(d for d in data if data[d]["regime"] == "hi")
    out = []
    for regime, dss in zip(("lo", "hi", "all"), (lo, hi, lo + hi)):
        r = {"regime": regime, "datasets": len(dss)}
        for v in ("npv", "mu"):
            nn, par = _avg(data, dss, f"nn_{v}"), _avg(data, dss, f"parametric_{v}")
            r[f"nn_{v}"], r[f"parametric_{v}"] = nn, par
            r[f"ratio_{v}"] = nn / par if (nn is not None and par) else float("nan")
        out.append(r)
    return out


def latex(rows, incomplete):
    shown = dict(REGIMES)

    def line(r):
        return (f"{shown[r['regime']]:<8} & {r['nn_npv']:.3f} & {r['parametric_npv']:.3f} & {r['ratio_npv']:.2f} "
                f"& {r['nn_mu']:.3f} & {r['parametric_mu']:.3f} & {r['ratio_mu']:.2f} \\\\")

    L = [f"%   {shown[r['regime']]}: n_datasets={r['datasets']}" for r in rows]
    if incomplete:
        L.append(f"% WARNING: incomplete (missing NN or parametric metric): {', '.join(incomplete)}")
    L += [r"\begin{table}[ht]",
          r"\centering",
          r"\caption{" + CAPTION + "}",
          r"\label{" + LABEL + "}",
          r"\begin{tabular}{l ccc ccc}",
          r"\hline",
          r" & \multicolumn{3}{c}{$\partial r / \partial N_{\mathrm{PV}}$}",
          r" & \multicolumn{3}{c}{$\partial r / \partial \mu$} \\",
          r"\cmidrule(lr){2-4} \cmidrule(lr){5-7}",
          r"$p_T$ regime & NN & Param. & NN/Param. & NN & Param. & NN/Param. \\",
          r"\hline",
          line(rows[0]),
          line(rows[1]),
          r"\hline",
          line(rows[2]),
          r"\hline",
          r"\end{tabular}",
          r"\end{table}"]
    return L


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.parse_args()

    head = ["% parametric baseline: the stratified fit (source: corrected parquets under data/)",
            "% slopes of: the residual pT - pT_gen"]
    data, missing = residual_slopes_of_scan(ETA_BINS)
    if missing:
        head.append(f"% NOTE: {len(missing)} expected parquet(s) absent")
        head += [f"%   {q}" for q in missing[:5]]
        if len(missing) > 5:
            head.append(f"%   ... and {len(missing)-5} more")
    data = {k: v for k, v in data.items()
            if v["nn_npv"] is not None or v["parametric_npv"] is not None}
    if not data:
        raise SystemExit("No corrected pile-up files found: run both pile-up steps first.")
    in_order = (sorted(d for d in data if data[d]["regime"] == "lo")
                + sorted(d for d in data if data[d]["regime"] == "hi"))
    incomplete = [d for d in in_order if any(data[d].get(k) is None
                  for k in ("nn_npv", "nn_mu", "parametric_npv", "parametric_mu"))]
    rs = rows(data)
    tables.write(NAME, rs, head + latex(rs, incomplete))


if __name__ == "__main__":
    main()
