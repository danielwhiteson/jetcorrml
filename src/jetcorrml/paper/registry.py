"""
The paper's items: every figure and table the code makes (``items``, in paper
order, with the paper's labels), and how scripts/figures.py makes each figure
(``figures``).

    name      file name of the figure (figures/<subdir>/<name>.pdf)
    subdir    folder under figures/, as in the paper's \\includegraphics
    compute   "module:function" (key, **args): pipeline outputs -> saved numbers
    draw      "module:function" (key, subdir): saved numbers -> PDF
              key = "<subdir>/<name>": the saved numbers are data/<subdir>/<name>.npz
    args      what compute is given (the dataset)
    dargs     what draw is given beyond name and subdir (e.g. the width)

Nothing heavy is imported here, so run_paper.py can read the lists.
"""

from collections import namedtuple
from importlib import import_module

from jetcorrml import settings

class Fig(namedtuple("Fig", "name subdir compute draw args dargs", defaults=({},))):
    @property
    def key(self):
        return f"{self.subdir}/{self.name}"

RUN = settings.PAPER                         # what the figures are (the same in the reduced run)
PU_DS = settings.PILEUP_EXAMPLE              # the example dataset of the pile-up and absolute sections
DENSITY_DS = settings.DENSITY.name           # forward pile-up density study
MU_NPV_RHO = (0.0, 0.1, 1.0)                 # rho_muN of the three mu' vs N_PV figures


def function(spec):
    """'fig_pileup:draw_mu_vs_npv' -> the function."""
    mod, fn = spec.split(":")
    return getattr(import_module(f"jetcorrml.paper.{mod}"), fn)


def figures():
    F = []
    P, A = "fig_pileup", "fig_absolute"
    for rho in MU_NPV_RHO:
        F.append(Fig(f"mu_vs_npv_rho{settings.rho_code(rho)}", "pu", f"{P}:compute_mu_vs_npv",
                     f"{P}:draw_mu_vs_npv", {"ds": settings.pileup_dataset(rho).name}))
    F.append(Fig("nn_residual_vs_pileup", "pu", f"{P}:compute_residual_vs_pileup",
                 f"{P}:draw_residual_vs_pileup", {"ds": PU_DS}))
    F.append(Fig("nn_residual_vs_eta", "pu", f"{P}:compute_residual_vs_eta", f"{P}:draw_residual_vs_eta",
                 {"ds": PU_DS}))
    F.append(Fig("nn_slopes_vs_eta", "pu", f"{P}:compute_slopes_vs_eta", f"{P}:draw_slopes_vs_eta",
                 {"ds": PU_DS}))
    F.append(Fig("slopes_vs_eta", "pu-etadens", f"{P}:compute_slopes_vs_eta",
                 f"{P}:draw_slopes_vs_eta", {"ds": DENSITY_DS, "fine": True}))
    F.append(Fig("nn_closure_vs_egen", "abs", f"{A}:compute_closure_vs_egen", f"{A}:draw_closure_vs_egen",
                 {"ds": PU_DS}))
    F.append(Fig("nn_closure_vs_eta", "abs", f"{A}:compute_closure_vs_eta", f"{A}:draw_closure_vs_eta",
                 {"ds": PU_DS}))
    # ── the samples, the injection, the pile-up scan and the absolute scale ──
    R = "fig_raw"
    for which in ("jet_pt", "jet_y", "jet_mass", "event_mu", "event_npv", "event_rho"):
        F.append(Fig(which, "raw", f"{R}:compute_hist", f"{R}:draw_hist", {"which": which},
                     {"which": which}))
    # the detector response in energy, the variable of the absolute correction
    F.append(Fig("response_E", "raw", f"{R}:compute_response_e", f"{R}:draw_response_e", {}))
    F.append(Fig("response_E_vs_eta", "raw", f"{R}:compute_response_e_vs_eta", f"{R}:draw_response_e_vs_eta", {}))
    F.append(Fig("response_E_vs_E_regions", "raw", f"{R}:compute_response_e_regions",
                 f"{R}:draw_response_e_regions", {}))
    F.append(Fig("residual_vs_pileup", "pu", f"{P}:compute_injection", f"{P}:draw_injection", {}))
    # slopes of the residual r = pT - pT_gen, as the paper defines the metric
    for var in ("mu", "npv"):
        F.append(Fig(f"slope_{var}_vs_rho", "pu", f"{P}:compute_slope_vs_corr",
                     f"{P}:draw_slope_vs_corr", {"var": var}, {"var": var}))
    F.append(Fig("injection", "pu-etadens", f"{P}:compute_etadens_validation",
                 f"{P}:draw_etadens_validation", {"ds": DENSITY_DS}))
    nonlin, cross = (d.name for d in settings.VARIANTS)
    for sub, ds, width in (("pu-etadens", DENSITY_DS, "wide"),
                           ("pu-nonlin", nonlin, "full"),
                           ("pu-cross", cross, "full")):
        F.append(Fig("residual_vs_pileup", sub, f"{P}:compute_nn_vs_parametric",
                     f"{P}:draw_nn_vs_parametric", {"ds": ds}, {"width": width}))
    F.append(Fig("closure_vs_rho", "abs", f"{A}:compute_closure_vs_corr", f"{A}:draw_closure_vs_corr",
                 {}))
    F.append(Fig("resolution", "abs", f"{A}:compute_resolution", f"{A}:draw_resolution",
                 {}))

    # ── in-situ: one distortion at a time (its files are read once) ──
    I = "fig_insitu"
    for dist in RUN.figure_dists:
        for kind, what in (("r", "response"), ("a", "asymmetry")):
            F.append(Fig(f"{dist}_{what}", "insitu", f"{I}:compute_dist", f"{I}:draw_dist",
                         {"dist": dist, "kind": kind}, {"kind": kind}))
        for key in ("pt", "eta", "rho"):
            F.append(Fig(f"{dist}_ratio_{key}", "insitu", f"{I}:compute_ratio", f"{I}:draw_ratio",
                         {"dist": dist, "key": key}, {"key": key}))
        F.append(Fig(f"{dist}_asymmetry_eta", "insitu", f"{I}:compute_asym_eta", f"{I}:draw_asym_eta",
                     {"dist": dist}))

    # ── the uncertainty band ──
    B = "fig_band"
    for var in ("pt", "eta"):
        F.append(Fig(f"band_closure_{var}", "np",
                     f"{B}:compute_closure", f"{B}:draw_closure", {"var": var}, {"var": var}))
    F.append(Fig("band_regions", "np", f"{B}:compute_regions", f"{B}:draw_regions", {}))

    # ── large-R jets (the sub-jet section) ──
    J = "fig_large_r"
    for what, fn in (("uncorrected_response", "uncorrected"), ("network_probe", "probe"),
                     ("corrected_response", "corrected"), ("resolution", "resolution")):
        F.append(Fig(what, "subjet-y", f"{J}:compute_{fn}", f"{J}:draw_{fn}", {}))
    return F


# ── the paper's items, in paper order ────────────────────────────────────────
# (the paper's number, its \label, the files under output/paper).  The figures
# with several files are panels a, b, c, ... in the order given.  Not made here:
# Fig. 1 (the strategy drawing), Table I (the injection settings) and Fig. 12
# (the sub-jet drawing).
_INSITU = [f"insitu/{d}_{what}" for what in ("response", "ratio_pt", "ratio_eta", "ratio_rho")
           for d in RUN.figure_dists]
_INSITU_ASYM = [f"insitu/{d}_{what}" for what in ("asymmetry", "asymmetry_eta") for d in RUN.figure_dists]
PAPER = [
    ("Fig. 2", "fig:jet_kin", ["raw/jet_pt", "raw/jet_y", "raw/jet_mass"]),
    ("Fig. 3", "fig:jet_pu", ["raw/event_mu", "raw/event_npv", "raw/event_rho"]),
    ("Fig. 4", "fig:jet_res", ["raw/response_E", "raw/response_E_vs_E_regions", "raw/response_E_vs_eta"]),
    ("Fig. 5", "fig:pu_inject", ["pu/residual_vs_pileup"]),
    ("Fig. 6", "fig:pu_corr_sim", [f"pu/mu_vs_npv_rho{settings.rho_code(r)}" for r in MU_NPV_RHO]),
    ("Fig. 7", "fig:pu_corr", ["pu/nn_residual_vs_pileup", "pu/nn_residual_vs_eta", "pu/nn_slopes_vs_eta"]),
    ("Table II", "tab:pu", ["table:pileup"]),
    ("Fig. 8", "fig:pu_corr_scan", ["pu/slope_mu_vs_rho", "pu/slope_npv_vs_rho"]),
    ("Fig. 9", "fig:abs", ["abs/nn_closure_vs_egen", "abs/nn_closure_vs_eta"]),
    ("Table III", "tab:abs_closure", ["table:absolute"]),
    ("Fig. 10", "fig:abs_closure", ["abs/closure_vs_rho"]),
    ("Fig. 11", "fig:abs_sigma68", ["abs/resolution"]),
    ("Fig. 13", "fig:subjet-uncorr", ["subjet-y/uncorrected_response", "subjet-y/network_probe"]),
    ("Fig. 14", "fig:subjet-corr", ["subjet-y/corrected_response"]),
    ("Fig. 15", "fig:subjet-sigma68", ["subjet-y/resolution"]),
    ("Fig. 16", "fig:insitu_resp", _INSITU),
    ("Fig. 17", "fig:insitu_asym", _INSITU_ASYM),
    ("Table IV", "tab:mcdata", ["table:insitu"]),
    ("Fig. 18", "fig:np_tests", ["np/band_closure_pt", "np/band_closure_eta"]),
    ("Sec. VIII", "sec:np", ["numbers:band_modes"]),
    ("Fig. 19", "fig:np_spread", ["np/band_regions"]),
    ("Fig. 20", "fig:etadens_validation", ["pu-etadens/injection"]),
    ("Fig. 21", "fig:etadens_perf", ["pu-etadens/residual_vs_pileup", "pu-etadens/slopes_vs_eta"]),
    ("Table V", "tab:etadens", ["table:density"]),
    ("Fig. 22", "fig:pu_nonlin_resid", ["pu-nonlin/residual_vs_pileup"]),
    ("Fig. 23", "fig:pu_cross_resid", ["pu-cross/residual_vs_pileup"]),
]


class Item(namedtuple("Item", "label tex_label kind path")):
    """One item of the paper.  label: as the paper numbers it ("Fig. 7b", "Table II");
    kind: "figure", "table" or "numbers" (numbers quoted in the text); path: its file
    under output/paper."""


def items():
    """Every item the code makes, in paper order (and every figure of ``figures`` is one)."""
    out = []
    for label, tex, files in PAPER:
        for i, f in enumerate(files):
            panel = "abcdefghijklmnopqrstuvwxyz"[i] if len(files) > 1 else ""
            if f.startswith("table:"):
                out.append(Item(label, tex, "table", f"tables/{f[6:]}.tex"))
            elif f.startswith("numbers:"):
                out.append(Item(label, tex, "numbers", f"tables/{f[8:]}.txt"))
            else:
                out.append(Item(label + panel, tex, "figure", f"figures/{f}.pdf"))
    drawn = {f"figures/{f.key}.pdf" for f in figures()}
    listed = {it.path for it in out if it.kind == "figure"}
    if drawn != listed:
        raise ValueError(f"the paper's items and the figures made differ: {sorted(drawn ^ listed)}")
    return out
