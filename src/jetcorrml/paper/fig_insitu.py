"""
Paper figures of the in-situ section: corrected data against simulation, NN and
parametric on the same pane, for one distortion each.

The numbers are computed by the functions of Table IV (jetcorrml.insitu_metrics),
on the first division of the sample.
"""

from pathlib import Path

import numpy as np

from jetcorrml import insitu_metrics as M
from jetcorrml import paths, settings

from . import inputs, store
from . import style as S

GEN_PT_MIN = M.GEN_PT_MIN                # all jets, as in Table IV
_CELLS = {}

# what is drawn: key, colour, line, marker, label.  The legend is drawn in the two
# distributions only (response, asymmetry); the "versus" panes under them use the same code.
SERIES = (("raw", S.COL["none"], ":", "d", "Before"),
          ("mc", S.COL["ref"], "-", "o", "MC"),
          ("nn", S.COL["nn"], "--", "o", "NN"),
          ("parametric", S.COL["parametric"], "-.", "s", S.LAB["parametric"]))
RATIO = "$R_{\\rm data}/R_{\\rm MC}$"


def _cells(dist):
    """Everything Table IV derives for one distortion (division 0), NN and parametric."""
    if dist not in _CELLS:
        ds = inputs.RUN.insitu.dataset
        s4 = [Path(paths.division(ds, dist, 0, k)) for k in ("MC", "DATA")]
        f5 = [Path(paths.insitu(ds, dist, 0, m, k))
              for m in ("nn", "parametric") for k in ("MC", "DATA")]
        for p in s4 + f5:
            if not p.is_file():
                raise FileNotFoundError(f"{p} is not there")
        cdn = M.cell_data(s4[0], s4[1], f5[0], f5[1], GEN_PT_MIN, settings.MATCH_DR)
        cdp = M.cell_data(s4[0], s4[1], f5[2], f5[3], GEN_PT_MIN, settings.MATCH_DR)
        _CELLS.clear()                       # one distortion in memory at a time
        _CELLS[dist] = (cdn, cdp, M.metrics_from(cdn), M.metrics_from(cdp), s4 + f5)
    return _CELLS[dist]


def _note(dist, **more):
    insitu = inputs.RUN.insitu
    return {"dataset": insitu.dataset, "distortion": dist, "division": 0, **more}


# ───────────── distributions: response and dijet asymmetry ─────────────
def compute_dist(name, dist, kind):
    cdn, cdp, mn, mp, src = _cells(dist)
    if kind == "r":
        x = {"mc": cdn["mc"]["response"], "nn": cdn["data"]["response"], "parametric": cdp["data"]["response"],
             "raw": cdn["data_raw"]["response"] if cdn.get("data_raw") is not None else None}
        bins = np.linspace(0.5, 1.5, 60)
    else:
        x = {"mc": cdn["a_mc"], "nn": cdn["a_data"], "parametric": cdp["a_data"],
             "raw": cdn["a_data_raw"] if cdn["a_data_raw"].size else None}
        bins = np.linspace(float(np.percentile(x["mc"], 0.5)), float(np.percentile(x["mc"], 99.5)), 60)
    out = {"edges": bins}
    for k, v in x.items():
        if v is not None:
            out[k] = np.histogram(v, bins=bins, density=True)[0]
    i = 0 if kind == "r" else 1
    out["ks_nn"], out["ks_parametric"] = (np.nan if v is None else v for v in (mn[i], mp[i]))
    store.save(name, out, src, _note(dist, quantity="response" if kind == "r" else "asymmetry"))


def draw_dist(name, subdir, kind):
    d = store.load(name)
    fig, ax = S.figure("third", height=2.05)
    for key, col, ls, _mk, lab in SERIES:
        if key in d:
            ax.stairs(d[key], d["edges"], color=col, ls=ls, lw=1.0 if key != "mc" else 1.2, label=lab)
    ax.set_xlabel(r"$p_T/p_T^{\rm gen}$" if kind == "r" else r"$A$")
    ax.set_ylabel(S.LAB["norm"])
    ax.set_ylim(bottom=0)
    S.legend_top(ax, "upper center", ncols=2, handlelength=1.0, columnspacing=0.5, handletextpad=0.3)
    return S.save(fig, store.fig_path(subdir, name))


# ───────── data over simulation against pT, |eta| and rho ─────────
def _ratio_spec(key):
    return {"pt": ("gen_pt", M.PT_EDGES, True, 2), "eta": ("abs_eta", M.ABSETA_EDGES, False, 3),
            "rho": ("rho", M.RHO_EDGES, False, 4)}[key]


def compute_ratio(name, dist, key):
    cdn, cdp, mn, mp, src = _cells(dist)
    var, edges, central, i = _ratio_spec(key)
    out = {}
    for k, mc, da in (("nn", cdn["mc"], cdn["data"]), ("parametric", cdp["mc"], cdp["data"]),
                      ("raw", cdn["mc"], cdn.get("data_raw"))):
        if da is None:
            continue
        x, r, ok = M.r_ratio_1d(mc, da, var, edges, M.MIN_COUNT, central=central)
        out[f"{k}_x"], out[f"{k}_y"] = np.asarray(x)[ok], np.asarray(r)[ok]
    out["maxdev_nn"], out["maxdev_parametric"] = (np.nan if v is None else v for v in (mn[i], mp[i]))
    store.save(name, out, src, _note(dist, against=var, central_only=bool(central)))


def draw_ratio(name, subdir, key):
    d = store.load(name)
    fig, ax = S.figure("third", height=1.4)              # 3 wide by 2 high
    vals = []
    for k, col, ls, mk, lab in SERIES:
        if f"{k}_y" in d:
            ax.plot(d[f"{k}_x"], d[f"{k}_y"], ls if k == "raw" else "-", marker=mk, ms=S.ms(3.0), color=col, label=lab)
            vals.append(d[f"{k}_y"])
    ax.axhline(1.0, ls=":", color="black", lw=0.6)
    for y in (1.01, 0.99):
        ax.axhline(y, ls="--", color=S.COL["aux"], lw=0.5)
    # at least 0.9-1.1; wider when the uncorrected curve needs it
    allv = np.concatenate(vals) if vals else np.array([1.0])
    lo, hi = min(0.9, float(np.nanmin(allv))), max(1.1, float(np.nanmax(allv)))
    ax.set_ylim(lo - 0.03 * (hi - lo), hi + 0.03 * (hi - lo))
    if key == "pt":
        S.plain_log_x(ax)
    ax.set_xlabel({"pt": r"$p_T^{\rm gen}$ [GeV]", "eta": r"$|\eta|$", "rho": S.LAB["rho"]}[key])
    ax.set_ylabel(RATIO)
    # no legend: these panes sit under the response distributions, which carry it
    return S.save(fig, store.fig_path(subdir, name))


# ───────── median dijet asymmetry against the forward jet's |eta| ─────────
def compute_asym_eta(name, dist):
    cdn, cdp, _mn, _mp, src = _cells(dist)
    edges = M.FWD_ETA_EDGES
    have_raw = cdn["a_data_raw"].size > 0

    def med(a, eta, elo, ehi):
        m = (eta >= elo) & (eta < ehi)
        return float(np.median(a[m])) if m.sum() >= M.MIN_COUNT else np.nan

    out = {"x": np.array([0.5 * (edges[i] + edges[i + 1]) for i in range(len(edges) - 1)]),
           "edges": np.array(edges, dtype=float)}
    pairs = list(zip(edges[:-1], edges[1:]))
    out["mc"] = np.array([med(cdn["a_mc"], cdn["a_eta_mc"], lo, hi) for lo, hi in pairs])
    out["nn"] = np.array([med(cdn["a_data"], cdn["a_eta_data"], lo, hi) for lo, hi in pairs])
    out["parametric"] = np.array([med(cdp["a_data"], cdp["a_eta_data"], lo, hi) for lo, hi in pairs])
    if have_raw:
        out["raw"] = np.array([med(cdn["a_data_raw"], cdn["a_eta_data_raw"], lo, hi) for lo, hi in pairs])
    store.save(name, out, src, _note(dist))


def draw_asym_eta(name, subdir):
    d = store.load(name)
    fig, ax = S.figure("third", height=1.4)              # 3 wide by 2 high
    vals = []
    for k, col, ls, mk, lab in SERIES:
        if k in d:
            ax.plot(d["x"], d[k], ls, marker=mk, ms=S.ms(3.2), color=col, label=lab)
            vals += [v for v in d[k] if np.isfinite(v)]
    ax.axhline(0, ls=":", color=S.COL["aux"], lw=0.6)
    vals = np.array(vals or [0.0])
    ax.set_ylim(min(-0.1, float(vals.min()) - 0.02), max(0.1, float(vals.max()) + 0.02))
    ax.set_xlabel(r"Forward jet $|\eta|$")
    ax.set_ylabel(r"Median $A$")
    # no legend: these panes sit under the asymmetry distributions, which carry it
    return S.save(fig, store.fig_path(subdir, name))
