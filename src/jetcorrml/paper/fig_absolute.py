"""
Paper figures of the absolute-scale section (Sec. V): the closure of the NN
absolute scale against E_gen and |eta| (Fig. 10), the closure of both methods
against rho_muN (with Table III), and the resolution after the correction (Fig. 11).
The numbers are computed from the files before and after each correction.
"""

from pathlib import Path

import numpy as np

from jetcorrml import absolute_closure, settings
from jetcorrml.profiles import profile_xy

from . import inputs, store
from . import style as S

E_GEN_MIN_CLOSURE = settings.CLOSURE_E_MIN
E_BINS = np.array([50.0, 80.0, 150.0, 300.0, 600.0])
N_BINS = 25
MIN_JETS = 50


def _selection(d):
    floor = absolute_closure.closure_floor(d["eta_det"])
    return (d["e_gen"] >= E_GEN_MIN_CLOSURE) & (d["dr"] < settings.MATCH_DR) & (d["e_gen"] >= floor)


def _curves(d, slices, x, x_log):
    """Median response before and after, against x, one curve per slice."""
    resp = {"before": d["e_before"] / d["e_gen"], "after": d["e_after"] / d["e_gen"]}
    out = {}
    for stage, r in resp.items():
        for k in ("x", "y", "e"):
            out[f"{stage}_{k}"] = np.full((len(slices), N_BINS), np.nan)
        for i, m in enumerate(slices):
            if m.sum() < MIN_JETS:
                continue
            cx, my, sy, _ = profile_xy(x[m], r[m], n_bins=N_BINS, x_log=x_log)
            for k, v in (("x", cx), ("y", my), ("e", sy)):
                out[f"{stage}_{k}"][i] = store.pad(v, N_BINS)
    return out


# ───────── response against E_gen, in |eta| slices ──────────────────────
def compute_closure_vs_egen(name, ds):
    d, src = inputs.absolute(ds)
    edges = np.array(settings.ABSOLUTE_ETA_EDGES, dtype=float)
    idx = np.clip(np.digitize(np.abs(d["eta_det"]), edges) - 1, 0, len(edges) - 2)
    ok = _selection(d)
    out = _curves(d, [(idx == i) & ok for i in range(len(edges) - 1)], d["e_gen"], True)
    out["slice_edges"] = edges
    store.save(name, out, src, {"dataset": ds})


# ───────── response against |eta|, in E_gen slices ──────────────────────
def compute_closure_vs_eta(name, ds):
    d, src = inputs.absolute(ds)
    ok = _selection(d)
    sl = [(d["e_gen"] >= E_BINS[i]) & (d["e_gen"] < E_BINS[i + 1]) & ok for i in range(len(E_BINS) - 1)]
    out = _curves(d, sl, np.abs(d["eta_det"]), False)
    out["slice_edges"] = E_BINS
    store.save(name, out, src, {"dataset": ds})


def _draw(name, subdir, xlabel, legend_title, log_ticks, legend_cols):
    d = store.load(name)
    edges = d["slice_edges"]
    drawn = [i for i in range(len(edges) - 1) if np.isfinite(d["before_y"][i]).any()
             or np.isfinite(d["after_y"][i]).any()]
    cols = dict(zip(drawn, S.ramp(len(drawn))))
    fig, axes = S.figure("wide", height=2.6, ncols=2, sharey=True)
    for ax, stage, text in ((axes[0], "before", "Before correction"),
                            (axes[1], "after", "After correction")):
        for i in drawn:
            ok = np.isfinite(d[f"{stage}_y"][i])
            if not ok.any():
                continue
            ax.errorbar(d[f"{stage}_x"][i][ok], d[f"{stage}_y"][i][ok], yerr=d[f"{stage}_e"][i][ok],
                        fmt="o-", ms=S.ms(2.6), lw=0.9, color=cols[i], label=S.rng(edges[i], edges[i + 1]))
        ax.axhline(1.0, color="black", lw=0.6, ls="--")
        if log_ticks:
            S.plain_log_x(ax, log_ticks)
        ax.set_ylim(0.1, 1.8)
        ax.set_xlabel(xlabel)
        S.pane_label(ax, text, "upper left")
    # each pane names its energy (before: reconstructed; after: corrected); same scale
    axes[0].set_ylabel(S.LAB["resp_med_reco"])
    axes[1].set_ylabel(S.LAB["resp_med_corr"])
    axes[1].tick_params(labelleft=True)
    axes[1].legend(title=legend_title, ncols=legend_cols, loc="lower center")
    return S.save(fig, store.fig_path(subdir, name))


def draw_closure_vs_egen(name, subdir):
    return _draw(name, subdir, S.LAB["egen"], S.LAB["abseta"], (50, 100, 200, 500), 2)


def draw_closure_vs_eta(name, subdir):
    return _draw(name, subdir, S.LAB["abseta"], S.LAB["egen"], None, 2)


# ═════════ figures that had their own script: same numbers, new drawing ═════════

# ───────── closure of the absolute scale against the injected correlation ─────────
def compute_closure_vs_corr(name):
    """The closure of Table III against rho_muN, in each of its three binnings."""
    res, _missing = absolute_closure.closure_of_scan()
    if not res:
        raise FileNotFoundError("no absolute-scale files under data/absolute")
    out = {"corr": np.array(settings.RHO_MUN)}
    src = []
    for regime in settings.REGIMES:
        for lab, _ in absolute_closure.SERIES:
            for b in ("2d", "e", "eta"):
                out[f"{regime}_{lab}_{b}"] = np.array(
                    [res[(regime, r, lab)][f"closure_rms_{b}"] if (regime, r, lab) in res else np.nan
                     for r in settings.RHO_MUN])
        for b in ("2d", "e", "eta"):
            out[f"{regime}_floor_{b}"] = np.array(
                [res[(regime, r, "nn")][f"closure_floor_{b}"] if (regime, r, "nn") in res else np.nan
                 for r in settings.RHO_MUN])
        for r in settings.RHO_MUN:
            for _lab, path_of in absolute_closure.SERIES:
                p = Path(path_of(settings.pileup_dataset(r, regime).name))
                if p.is_file():
                    src.append(p)
    store.save(name, out, src)


def draw_closure_vs_corr(name, subdir, binning="2d"):
    d = store.load(name)
    fig, axes = S.figure("full", height=2.6, ncols=2, sharey=True)
    for ax, regime in zip(axes, ("lo", "hi")):
        ax.plot(d["corr"], d[f"{regime}_nn_{binning}"], "-", marker="o", color=S.COL["nn"], label="NN")
        ax.plot(d["corr"], d[f"{regime}_parametric_{binning}"], "--", marker="s", color=S.COL["parametric"],
                label=S.LAB["parametric"])
        # (the statistical floor is saved with the numbers but not drawn)
        ax.set_xlabel(S.LAB["corr"])
    axes[0].set_ylabel(r"RMS of median$(E^{\rm corr}/E^{\rm gen})-1$")
    S.plain_log(axes[0], "y")                            # 0.005, 0.01, 0.02 rather than one power of ten
    S.legend_top(axes[0], "upper left")
    for ax, regime in zip(axes, ("lo", "hi")):           # in the room the legend made, on the other side
        S.pane_label(ax, S.LAB[regime], "upper right")
    return S.save(fig, store.fig_path(subdir, name))


# ───────────── resolution after the absolute correction ─────────────
def compute_resolution(name):
    """sigma_68 of the corrected response against E_gen and |eta|, for the three chains."""
    rho = inputs.RUN.resolution_rho
    loaded, src = {}, []
    for key, path in absolute_closure.resolution_series(rho):
        if path.exists():
            loaded[key] = absolute_closure.resolution_jets(path)     # (R, |eta|, E_gen)
            src.append(path)
    if not loaded:
        raise FileNotFoundError(f"no absolute-scale files for rho_muN = {rho}")
    all_egen = np.concatenate([v[2] for v in loaded.values()])
    all_eta = np.concatenate([v[1] for v in loaded.values()])
    e_edges = np.logspace(np.log10(30.0), np.log10(float(np.quantile(all_egen, 0.99))), 15)
    eta_edges = np.linspace(0.0, max(2.5, float(np.quantile(all_eta, 0.999))), 11)
    out = {"e_edges": e_edges, "eta_edges": eta_edges}
    for key, (R, aeta, egen) in loaded.items():
        out[f"{key}_e_x"], out[f"{key}_e_y"] = absolute_closure.profile_sigma68(egen, R, e_edges, log_x=True)
        out[f"{key}_eta_x"], out[f"{key}_eta_y"] = absolute_closure.profile_sigma68(aeta, R, eta_edges, log_x=False)
    store.save(name, out, src, {"rho_muN_code": settings.rho_code(rho)})


def draw_resolution(name, subdir):
    from matplotlib.lines import Line2D

    d = store.load(name)
    fig, axes = S.figure("full", height=2.6, ncols=2)
    who = (("nn", "NN", S.COL["nn"], "o"), ("cross", f"{S.LAB['parametric']} (cross)", S.COL["parametric"], "s"),
           ("full", f"{S.LAB['parametric']} (full)", S.COL["parametric_full"], "^"))
    for ax, v in zip(axes, ("e", "eta")):
        for key, _lab, col, mk in who:
            for reg, ls in (("lo", "-"), ("hi", "--")):
                if f"{key}_{reg}_{v}_x" in d:
                    ax.plot(d[f"{key}_{reg}_{v}_x"], d[f"{key}_{reg}_{v}_y"], ls, marker=mk, color=col)
        ax.set_ylim(bottom=0)
        ax.set_ylabel(S.LAB["sigma68"])
    S.plain_log_x(axes[0])
    axes[0].set_xlabel(S.LAB["egen"])
    axes[1].set_xlabel(S.LAB["abseta"])
    # colour says the method, the line says the pT regime
    S.legend_top(axes[0], "upper right", also=(axes[1],),
                 handles=[Line2D([], [], color=c, marker=m, ls="-") for _k, _l, c, m in who],
                 labels=[l for _k, l, _c, _m in who])
    axes[1].legend(loc="upper left", ncols=2,
                   handles=[Line2D([], [], color="black", ls="-"), Line2D([], [], color="black", ls="--")],
                   labels=[S.LAB["lo"], S.LAB["hi"]])
    return S.save(fig, store.fig_path(subdir, name))
