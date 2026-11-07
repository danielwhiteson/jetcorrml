"""
Paper figures of the pile-up section (Sec. IV) and of Apps. A-C: the injection,
the residual after the NN and parametric corrections against N_PV, mu and |eta|,
and the slopes of Table II against rho_muN.  The numbers are computed from the
files before and after each correction.
"""

import numpy as np
from scipy.stats import linregress

from jetcorrml import settings
from jetcorrml.pileup_slopes import residual_slopes_of_scan
from jetcorrml.profiles import binned_mean, profile, profile_inclusive

from . import inputs, store
from . import style as S

STAGES = ("none", "rhoA", "nn")
SLOPE_STAGES = ("none", "rhoA", "parametric", "nn")      # with the parametric correction beside the NN
PARAMETRIC_FIT = "stratified"                        # the parametric coefficients are fit in strata of the other variable


def _stage_pt(d):
    rhoA = d["rho"] * d["area_pt"]
    return {"none": d["pt_reco"], "rhoA": d["pt_reco"] - rhoA, "nn": d["pt_nn"]}


# ───────────────────── mu against N_PV ──────────────────────────────────
def compute_mu_vs_npv(name, ds):
    d, src = inputs.injected(ds)
    mu_edges = np.arange(0.0, 202.0, 2.0)
    n_edges = np.arange(-0.5, 150.5, 1.0)            # N_PV is a whole number: one bin each
    h, _, _ = np.histogram2d(d["mu"], d["npv"] - 1, bins=[mu_edges, n_edges])
    store.save(name, {"counts": h, "mu_edges": mu_edges, "npv1_edges": n_edges}, src,
               {"dataset": ds, "jets": int(len(d["mu"]))})


def draw_mu_vs_npv(name, subdir):
    import matplotlib.colors as mcolors

    d = store.load(name)
    # fixed places for the pane and the colour scale, so the three maps of the row line up;
    # the pane is 3 wide by 2 high
    fig, ax = S.figure("third", height=1.2, layout="none")
    ax.set_position([0.24, 0.35, 0.48, 0.56])            # 1.01 x 0.67 inches
    cax = fig.add_axes([0.74, 0.35, 0.035, 0.56], label="<colorbar>")
    h = np.ma.masked_less(d["counts"].T, 1)
    im = ax.pcolormesh(d["mu_edges"], d["npv1_edges"], h, cmap=S.CMAP,
                       norm=mcolors.LogNorm(vmin=1, vmax=max(10.0, float(d["counts"].max()))),
                       rasterized=True)
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("Jets")
    cb.outline.set_linewidth(0.7)
    ax.set_xlim(0, 85)
    ax.set_ylim(0, 85)
    ax.set_xticks([0, 40, 80])
    ax.set_yticks([0, 40, 80])
    ax.set_xlabel(S.LAB["mu"])
    ax.set_ylabel(S.LAB["npv1"])
    return S.save(fig, store.fig_path(subdir, name))


# ─────────────── residual against N_PV and mu ────────────────────────────
def compute_residual_vs_pileup(name, ds):
    d, src = inputs.pileup(ds)
    pt_par, p_par = inputs.pileup_parametric(ds)              # the same jets, after rho*A and the parametric fit
    src = list(src) + [p_par]
    # in the order the diagnostics subtract, so the numbers agree to the last digit
    resid = {"rhoA": d["pt_reco"] - d["pt_gen"] - d["rho"] * d["area_pt"], "parametric": pt_par - d["pt_gen"],
             "nn": d["pt_nn"] - d["pt_gen"]}
    out = {}
    for key, x in (("npv", d["npv"] - 1), ("mu", d["mu"])):
        for st in ("rhoA", "parametric", "nn"):
            y = resid[st]
            xc, ym, ys = profile(x, y, n_bins=30)
            out[f"{key}_{st}_x"] = store.pad(xc, 30)
            out[f"{key}_{st}_y"] = store.pad(ym, 30)
            out[f"{key}_{st}_e"] = store.pad(ys, 30)
            out[f"{key}_{st}_slope"] = linregress(x, y).slope
    store.save(name, out, src, {"dataset": ds, "jets": int(len(d["mu"])), "parametric_fit": PARAMETRIC_FIT})


def draw_residual_vs_pileup(name, subdir):
    d = store.load(name)
    fig, axes = S.figure("wide", height=2.45, ncols=2)
    # short names, as in the NN-against-parametric figures: the long ones with a slope do not fit a pane
    lab = {"rhoA": S.LAB["st_rhoA"], "parametric": S.LAB["parametric"], "nn": "NN"}
    for ax, key, xl in ((axes[0], "npv", S.LAB["npv1"]), (axes[1], "mu", S.LAB["mu"])):
        for st in ("rhoA", "parametric", "nn"):
            sl = float(d[f"{key}_{st}_slope"])
            ax.errorbar(d[f"{key}_{st}_x"], d[f"{key}_{st}_y"], yerr=d[f"{key}_{st}_e"],
                        fmt=S.MARK[st], color=S.COL[st], lw=0.8,
                        label=f"{lab[st]}: slope ${sl:+.3f}$")
        ax.axhline(0, color="black", lw=0.6, ls=":")
        ax.set_xlabel(xl)
        ax.set_ylabel(S.LAB["resid"])
        S.legend_top(ax, "upper left")
    return S.save(fig, store.fig_path(subdir, name))


# ─────────────── relative residual against eta ────────────────────────────
def compute_residual_vs_eta(name, ds):
    d, src = inputs.pileup(ds)
    pt = _stage_pt(d)
    pt["parametric"], p_par = inputs.pileup_parametric(ds)    # the same jets, after rho*A and the parametric fit
    src = list(src) + [p_par]
    edge = float(inputs.pileup_eta_bins()[-1])
    out = {}
    for st in SLOPE_STAGES:
        xc, ym, ys = profile(d["eta_det"], (pt[st] - d["pt_gen"]) / d["pt_gen"],
                             n_bins=40, x_range=(-edge, edge))
        out[f"{st}_x"], out[f"{st}_y"], out[f"{st}_e"] = (store.pad(v, 40) for v in (xc, ym, ys))
    store.save(name, out, src, {"dataset": ds, "parametric_fit": PARAMETRIC_FIT})


def draw_residual_vs_eta(name, subdir):
    d = store.load(name)
    fig, ax = S.figure("wide", height=2.7)
    for st in SLOPE_STAGES:
        ax.errorbar(d[f"{st}_x"], d[f"{st}_y"], yerr=d[f"{st}_e"], fmt=S.MARK[st],
                    color=S.COL[st], lw=0.8, label=S.LAB[f"st_{st}"])
    ax.axhline(0, color="black", lw=0.6, ls=":")
    ax.set_xlabel(S.LAB["eta"])
    ax.set_ylabel(S.LAB["relresid"])
    S.legend_top(ax, "upper center", ncols=4)
    return S.save(fig, store.fig_path(subdir, name))


# ───────── slope of pT against N_PV and mu, in |eta| bins ──────────────────────
FINE_ETA_BINS = np.array(settings.DENSITY_ETA_BINS)   # for the forward-density figure (and Table V)


def compute_slopes_vs_eta(name, ds, fine=False):
    """fine: ten |eta| bins of 0.25 up to 2.5 in place of the bins of the pile-up diagnostics
    (those of the tables), so that a change over a short range of |eta| can be seen."""
    d, src = inputs.pileup(ds)
    pt = _stage_pt(d)
    pt["parametric"], p_par = inputs.pileup_parametric(ds)    # the same jets, after rho*A and the parametric fit
    src = list(src) + [p_par]
    bins = FINE_ETA_BINS if fine else inputs.pileup_eta_bins()
    n = len(bins) - 1
    abs_eta = np.abs(d["eta_det"])
    idx = np.clip(np.digitize(abs_eta, bins) - 1, 0, n - 1)
    # the bins of the diagnostics count a jet beyond the last edge in the last bin (as the
    # diagnostics do); the fine bins hold only the jets inside them
    inside = abs_eta < bins[-1] if fine else np.ones(len(abs_eta), dtype=bool)
    out = {"eta_edges": bins, "eta_centres": 0.5 * (bins[:-1] + bins[1:])}
    for key, x in (("npv", d["npv"]), ("mu", d["mu"])):
        for st in SLOPE_STAGES:
            sl, er = np.full(n, np.nan), np.full(n, np.nan)
            for i in range(n):
                m = (idx == i) & inside
                if m.sum() < 10:
                    continue
                r = linregress(x[m], pt[st][m])
                sl[i], er[i] = r.slope, r.stderr
            out[f"{key}_{st}_slope"], out[f"{key}_{st}_err"] = sl, er
    store.save(name, out, src, {"dataset": ds, "parametric_fit": PARAMETRIC_FIT,
                                "eta_bins": "fine (0.25 wide, to 2.5)" if fine else "those of the diagnostics",
                                "jets_outside_bins": int((~inside).sum())})


def draw_slopes_vs_eta(name, subdir):
    d = store.load(name)
    fig, axes = S.figure("wide", height=2.6, ncols=2)
    for ax, key in ((axes[0], "npv"), (axes[1], "mu")):
        for st in SLOPE_STAGES:
            ok = np.isfinite(d[f"{key}_{st}_slope"])
            ax.errorbar(d["eta_centres"][ok], d[f"{key}_{st}_slope"][ok], yerr=d[f"{key}_{st}_err"][ok],
                        fmt=S.MARK[st], ms=S.ms(4.5), color=S.COL[st], lw=0.8, label=S.LAB[f"st_{st}"])
        ax.axhline(0, color="black", lw=0.6, ls=":")
        seen = d["eta_centres"][np.isfinite(d[f"{key}_none_slope"])]
        ax.set_xlim(0, (seen.max() if len(seen) else d["eta_edges"][-1]) + 0.15)
        ax.set_xlabel(S.LAB["abseta"])
        ax.set_ylabel(S.LAB["d" + key])
    S.legend_top(axes[0], "upper left", also=(axes[1],))
    return S.save(fig, store.fig_path(subdir, name))


# ═════════ figures that had their own script: same numbers, new drawing ═════════

# ───── residual against pile-up, before and after the injection (Delphes) ─────
INJECTED_EXAMPLE = settings.INJECTION_EXAMPLE


def compute_injection(name):
    """The residual against N_PV - 1 and mu, before and after the injection, with and without
    the rho*A subtraction; profiles and straight-line fits."""
    raw = inputs.raw_files()
    after = inputs.injected_file(INJECTED_EXAMPLE)
    b, a = inputs.injection_residuals(raw), inputs.injection_residuals([str(after)])
    out = {}
    for key, fx in (("npv", lambda v: v - 1), ("mu", lambda v: v)):
        for ser, d, y in (("raw_nosub", b, "residual_nosub"), ("raw", b, "residual"), ("inj", a, "residual")):
            x = fx(d[key])
            xc, ym, ye = profile_inclusive(x, d[y], n_bins=30)
            lr = linregress(x, d[y])
            xx = np.array([xc.min(), xc.max()]) if len(xc) else np.array([x.min(), x.max()])
            k = f"{key}_{ser}"
            out[k + "_x"], out[k + "_y"], out[k + "_e"] = (store.pad(v, 30) for v in (xc, ym, ye))
            out[k + "_slope"], out[k + "_intercept"], out[k + "_line_x"] = lr.slope, lr.intercept, xx
    store.save(name, out, list(raw) + [after], {"injected": INJECTED_EXAMPLE, "jets_raw": int(b["n"])})


def draw_injection(name, subdir):
    d = store.load(name)
    fig, axes = S.figure("full", height=2.6, ncols=2)
    # black: the sample before the injection; green: with the injection.  Open: no subtraction.
    look = (("raw_nosub", "Raw", S.COL["raw"], "o", "none", ":"),
            ("raw", r"Raw $-\,\rho A$", S.COL["raw"], "o", S.COL["raw"], "--"),
            ("inj", r"Raw $-\,\rho A$ + injection", S.COL["rhoA"], "s", S.COL["rhoA"], "-"))
    for ax, key, xl in ((axes[0], "npv", S.LAB["npv1"]), (axes[1], "mu", S.LAB["mu"])):
        for ser, lab, col, mk, mfc, ls in look:
            k = f"{key}_{ser}"
            ax.errorbar(d[k + "_x"], d[k + "_y"], yerr=d[k + "_e"], fmt=mk, color=col, mfc=mfc, lw=0.8)
            xx = d[k + "_line_x"]
            ax.plot(xx, float(d[k + "_intercept"]) + float(d[k + "_slope"]) * xx, ls, color=col, lw=1.1,
                    marker=mk, mfc=mfc, markevery=[], label=lab)
        ax.axhline(0, color="black", lw=0.6, ls=":")
        ax.set_xlabel(xl)
        ax.set_ylabel(S.LAB["resid"])
    # (markevery=[] above: the fitted line carries its marker in the legend only)
    S.legend_top(axes[0], "upper left", also=(axes[1],))
    return S.save(fig, store.fig_path(subdir, name))


# ───────────── forward pile-up density: what the injection does ─────────────
_DENSITY = settings.DENSITY.injection.eta_density
ETADENS = dict(depth=_DENSITY.depth, eta_c=_DENSITY.etac, width=_DENSITY.width,   # the injected density, and
               eta_max=2.5, dr_max=0.2)                                           # the figure's own range and cut


def compute_etadens_validation(name, ds):
    """What the forward pile-up density injection does: the residual against N_PV, mu and |eta|,
    before and after, with the injected excess predicted from the mean rho*A."""
    raw = inputs.raw_files()
    after = inputs.injected_file(ds)
    b, a = inputs.density_residuals(raw, ETADENS["dr_max"]), inputs.density_residuals([str(after)], ETADENS["dr_max"])
    npv_hi = np.nanpercentile(np.concatenate([b["npv"], a["npv"]]), 99.5)
    allmu = np.concatenate([b["mu"], a["mu"]])
    edges = {"npv": np.linspace(0, max(npv_hi - 1, 5), 26),
             "mu": np.linspace(float(np.nanpercentile(allmu, 0.5)), float(np.nanpercentile(allmu, 99.5)), 26),
             "abs_eta": np.linspace(0, ETADENS["eta_max"], 21)}
    out = {}
    for key, ed in edges.items():
        shift = 1.0 if key == "npv" else 0.0
        for ser, d, y in (("raw", b, "resid"), ("inj", a, "resid"), ("sub", a, "resid_sub")):
            cx, cy, ce = binned_mean(d[key] - shift, d[y], ed, 20)
            n = len(ed) - 1
            out[f"{key}_{ser}_x"], out[f"{key}_{ser}_y"], out[f"{key}_{ser}_e"] = (
                store.pad(v, n) for v in (cx, cy, ce))
    ed = edges["abs_eta"]
    ctr = 0.5 * (ed[:-1] + ed[1:])
    pred = float(b["rho_A"].mean()) * ETADENS["depth"] / (1.0 + np.exp(-(ctr - ETADENS["eta_c"]) / ETADENS["width"]))
    _, base_y, _ = binned_mean(b["abs_eta"], b["resid"] - b["rho_A"], ed, 20)
    with_base = base_y.size == pred.size
    out["pred_x"], out["pred_y"] = ctr, (base_y + pred if with_base else pred)
    out["pred_on_baseline"] = with_base
    store.save(name, out, list(raw) + [after], {"dataset": ds, **ETADENS})


def draw_etadens_validation(name, subdir):
    d = store.load(name)
    fig, axes = S.figure("full", height=2.35, ncols=3)
    # black: the sample before the injection; green: with the injection.  Open: no subtraction.
    look = (("raw", "Raw", S.COL["raw"], "o", "none"),
            ("inj", "Raw + injection", S.COL["rhoA"], "^", "none"),
            ("sub", r"Raw + injection $-\,\rho A$", S.COL["rhoA"], "s", S.COL["rhoA"]))
    for ax, key, xl in ((axes[0], "npv", S.LAB["npv1"]), (axes[1], "mu", S.LAB["mu"]),
                        (axes[2], "abs_eta", S.LAB["abseta"])):
        for ser, lab, col, mk, mfc in look:
            ok = np.isfinite(d[f"{key}_{ser}_y"])
            ax.errorbar(d[f"{key}_{ser}_x"][ok], d[f"{key}_{ser}_y"][ok], yerr=d[f"{key}_{ser}_e"][ok],
                        fmt=mk, ms=S.ms(3.0), color=col, mfc=mfc, lw=0.8, label=lab)
        ax.axhline(0, color="black", lw=0.6, ls=":")
        ax.set_xlabel(xl)
    axes[0].set_ylabel(S.LAB["resid"])
    pred = axes[2].plot(d["pred_x"], d["pred_y"], "--", color=S.COL["ref"], lw=1.0)[0]
    h, l = axes[0].get_legend_handles_labels()
    S.legend_top(axes[0], "upper left", also=(axes[1],), handles=h, labels=l)
    S.legend_top(axes[2], "upper left", handles=[pred], labels=["Predicted"])
    return S.save(fig, store.fig_path(subdir, name))


# ───────── residual against pile-up: before, after the parametric correction, after NN ─────────
def compute_nn_vs_parametric(name, ds):
    """The residual against N_PV and mu after the NN and the parametric pile-up corrections."""
    p_in, p_nn, p_par = inputs.injected_file(ds), inputs.nn_pileup_file(ds), inputs.parametric_pileup_file(ds)
    for p in (p_in, p_nn, p_par):
        if not p.is_file():
            raise FileNotFoundError(f"{p} is not there")
    pt_nn, gen_nn, _, _, _ = inputs.pileup_jets(p_nn)
    pt_par, gen_par, _, _, _ = inputs.pileup_jets(p_par)
    pt_raw, gen_raw, rhoA_raw, npv, mu = inputs.pileup_jets(p_in)
    if not (len(pt_nn) == len(pt_par) == len(pt_raw)) or not np.allclose(gen_nn, gen_par, rtol=0, atol=1e-6):
        raise ValueError(f"{ds}: the NN, parametric and injected files do not hold the same jets")
    ys = {"rhoA": pt_raw - gen_raw - rhoA_raw, "parametric": pt_par - gen_par, "nn": pt_nn - gen_nn}
    out = {}
    for key, x in (("npv", npv - 1), ("mu", mu)):
        for st, y in ys.items():
            cx, cy, ce = profile(x, y)
            out[f"{key}_{st}_x"], out[f"{key}_{st}_y"], out[f"{key}_{st}_e"] = (
                store.pad(v, 30) for v in (cx, cy, ce))
            out[f"{key}_{st}_slope"] = linregress(x, y).slope
    store.save(name, out, [p_in, p_nn, p_par], {"dataset": ds, "parametric_fit": PARAMETRIC_FIT})


def draw_nn_vs_parametric(name, subdir, width="full"):
    d = store.load(name)
    fig, axes = S.figure(width, height=2.7, ncols=2)
    lab = {"rhoA": S.LAB["st_rhoA"], "parametric": S.LAB["parametric"], "nn": "NN"}
    for ax, key, xl in ((axes[0], "npv", S.LAB["npv1"]), (axes[1], "mu", S.LAB["mu"])):
        for st in ("rhoA", "parametric", "nn"):
            ok = np.isfinite(d[f"{key}_{st}_y"])
            ax.errorbar(d[f"{key}_{st}_x"][ok], d[f"{key}_{st}_y"][ok], yerr=d[f"{key}_{st}_e"][ok],
                        fmt=S.MARK[st], color=S.COL[st], lw=0.8,
                        label=f"{lab[st]}: slope ${float(d[f'{key}_{st}_slope']):+.3f}$")
        ax.axhline(0, color="black", lw=0.6, ls=":")
        ax.set_xlabel(xl)
        ax.set_ylabel(S.LAB["resid"])
        S.legend_top(ax, "upper left")
    return S.save(fig, store.fig_path(subdir, name))


# ───── residual slope left after correction, against the injected correlation ─────
_PERF = {}


def _performance():
    """The slopes of Table II (of the residual r = pT - pT_gen), dataset by dataset."""
    if "data" not in _PERF:
        data, _missing = residual_slopes_of_scan(np.array(settings.PILEUP_ETA_BINS))
        _PERF["data"] = {k: v for k, v in data.items() if v["nn_npv"] is not None or v["parametric_npv"] is not None}
        if not _PERF["data"]:
            raise FileNotFoundError("no pile-up files under data/pileup")
    return _PERF["data"]


def compute_slope_vs_corr(name, var):
    data = _performance()
    out, src = {}, []
    for chain in ("nn", "parametric"):
        for reg in ("lo", "hi"):
            pts = sorted((data[d]["rho"], data[d][f"{chain}_{var}"]) for d in data
                         if data[d]["regime"] == reg and data[d]["rho"] is not None
                         and isinstance(data[d].get(f"{chain}_{var}"), (int, float)))
            out[f"{chain}_{reg}_x"] = np.array([p[0] for p in pts])
            out[f"{chain}_{reg}_y"] = np.array([p[1] for p in pts])
    for ds in sorted(data):
        for p in (inputs.nn_pileup_file(ds), inputs.parametric_pileup_file(ds)):
            if p.is_file():
                src.append(p)
    store.save(name, out, src, {"variable": var, "parametric_fit": PARAMETRIC_FIT, "slope_of": "residual"})


def draw_slope_vs_corr(name, subdir, var):
    d = store.load(name)
    fig, ax = S.figure("half", height=2.5)
    for chain, col, who in (("nn", S.COL["nn"], "NN"), ("parametric", S.COL["parametric"], S.LAB["parametric"])):
        for reg, ls, mk in (("lo", "-", "o"), ("hi", "--", "s")):
            if len(d[f"{chain}_{reg}_x"]):
                ax.plot(d[f"{chain}_{reg}_x"], d[f"{chain}_{reg}_y"], ls, marker=mk, color=col,
                        label=f"{who}, {S.LAB[reg]}")
    ax.axhline(0, color=S.COL["aux"], lw=0.6, ls=":")
    ax.set_xlabel(S.LAB["corr"])
    ax.set_ylabel(r"RMS$_\eta$ of $\partial r/\partial " + (r"N_{\rm PV}" if var == "npv" else r"\mu") + "$ [GeV]")
    ax.set_ylim(bottom=min(0, ax.get_ylim()[0]))
    S.legend_top(ax, "upper left", ncols=2)
    return S.save(fig, store.fig_path(subdir, name))
