"""
Paper figures of the uncertainty band (nuisance-parameter section).

  closure with its band   the closure of each retraining, their median, and their spread
  band against scatter    the band compared with the spread over the divisions of the sample

Both read files the pipeline has written (the retrainings and the divisions);
no fit is run.  The numbers are computed by jetcorrml.band.
"""

import os
import warnings
from pathlib import Path

import numpy as np

from jetcorrml import band, paths

from . import inputs, store
from . import style as S

JOBS = max(1, min(6, os.cpu_count() or 1))
DIST_NAME = {"nom": "Undistorted", "expo": "Expo", "quad": "Quadratic", "linear": "Linear",
             "threshold": "Threshold", "rhoeta": r"$\rho\eta$"}
RATIO = "$R_{\\rm data}/R_{\\rm MC}$"
BAND = S.COL["nn"]            # the band is the NN's


# ───────────── closure of the corrected data, with the band ─────────────
_JETS = {}


def _flat_cols(path, cols):
    """Columns of a jet file, flattened, every jet kept (the files of one sample line up jet by jet)."""
    import awkward as ak

    m = ak.from_parquet(path, columns=[c.replace("/", ".") for c in cols])
    out = []
    for c in cols:
        rec, field = c.split("/")
        x = ak.to_numpy(ak.flatten(m[rec][field], axis=None)).astype(float)
        out.append(np.abs(x) if field == "eta" else x)
    return out


def _bin_medians(r, groups):
    return np.array([np.median(r[g]) if g is not None else np.nan for g in groups])


def _model_band(sidecars, data_files, mc_file, input_file, edges, var, n_keep):
    """The band the parameter model predicts in each closure bin.

    Each parameter is moved by one standard deviation: every data jet's corrected pT is multiplied by
    exp(+-delta(x)), delta the parameter's shape on the grid, interpolated at the jet's pT before the
    correction and its |eta| (clipped to the grid, 25-250 GeV and 0.2-2.4); the closure of the bin is
    recomputed as band.closure_vs_gen / closure_vs_eta compute it (median of R, at least 50 jets) and the
    shift is (log C+ - log C-)/2.  The shifts of the parameters are added in quadrature.  The jets
    moved are the retrainings' mean corrected pT, jet by jet.  Nothing is retrained."""
    from scipy.interpolate import RegularGridInterpolator

    pt, eta, z0, lam, phi, C, B = band.decompose(sidecars)
    npt, neta = len(pt), len(eta)
    shapes, sig_res, _ = band.residual_model(lam, phi, C, n_keep)
    keep = lam > 1e-12 * max(lam[0], 1e-300)                     # the components the retrainings span
    all_shapes = np.sqrt(lam[keep])[:, None] * phi[:, keep].T
    stack = np.concatenate([all_shapes, sig_res[None, :]])      # [K + 1, P]: every component, then the residual
    grid = stack.T.reshape(npt, neta, -1)

    key = (input_file, tuple(data_files))
    if key not in _JETS:                                         # read once for the two figures
        (p_in, e_in, g_in) = _flat_cols(input_file, ["jet/pt", "jet/eta", "gen/pt"])
        acc = np.zeros_like(p_in)
        for f in data_files:
            p_b, g_b = _flat_cols(f, ["jet/pt", "gen/pt"])
            if p_b.shape != p_in.shape or not np.array_equal(g_b, g_in, equal_nan=True):
                raise ValueError(f"{f}: its jets are not those of {input_file}")
            acc += p_b
        p0 = acc / len(data_files)
        ok = np.isfinite(p0) & np.isfinite(g_in) & (g_in > 0)   # as the band's jets_of
        _JETS[key] = (p_in[ok], e_in[ok], g_in[ok], p0[ok])
    p_in, e_in, g_in, p0 = _JETS[key]
    x_in = np.stack([np.clip(np.nan_to_num(p_in, nan=pt[0]), pt[0], pt[-1]), np.clip(e_in, eta[0], eta[-1])], axis=1)
    delta = RegularGridInterpolator((pt, eta), grid, method="linear")(x_in)      # [jets, K + 1]

    _, etam, genm, Rm = band.jets_of(mc_file)
    nb = len(edges) - 1
    key_d, key_m = (g_in, genm) if var == "pt" else (e_in, etam)
    bd = np.clip(np.digitize(key_d, edges) - 1, 0, nb - 1)
    bm = np.clip(np.digitize(key_m, edges) - 1, 0, nb - 1)
    groups, med_mc = [], np.full(nb, np.nan)
    for k in range(nb):
        gd, gm = np.flatnonzero(bd == k), np.flatnonzero(bm == k)
        use = gd.size >= 50 and gm.size >= 50
        groups.append(gd if use else None)
        if use:
            med_mc[k] = np.median(Rm[gm])
    R0 = p0 / g_in
    base = _bin_medians(R0, groups) / med_mc
    shift = np.array([0.5 * (np.log(_bin_medians(R0 * np.exp(delta[:, j]), groups))
                             - np.log(_bin_medians(R0 * np.exp(-delta[:, j]), groups)))
                      for j in range(delta.shape[1])])                                  # [K + 1, nb]
    n = int(min(n_keep, keep.sum()))
    s_np = np.sqrt(np.sum(shift[:n] ** 2, axis=0))
    s_all = np.sqrt(np.sum(shift[:-1] ** 2, axis=0))
    s_res = np.abs(shift[-1])
    return {"sigma_np": s_np, "sigma_np_res": np.hypot(s_np, s_res), "sigma_all": s_all,
            "sigma_res": s_res, "shift_each": shift, "model_base": base, "n_np": n,
            "below_grid": float(np.mean(p_in < pt[0])), "above_grid": float(np.mean(p_in > pt[-1]))}


def compute_closure(name, var):
    ds, d = inputs.RUN.insitu.dataset, inputs.RUN.band.dist
    sidecars = paths.retraining_grids(ds, d)
    if len(sidecars) < 2:
        raise FileNotFoundError(f"fewer than two retrainings ({paths.retraining_grid(ds, d, '*')})")
    edges = np.linspace(20, 250, 13) if var == "pt" else np.linspace(0.0, 2.5, 9)      # as jetcorrml.band
    fn = band.closure_vs_gen if var == "pt" else band.closure_vs_eta
    curves, src = [], []
    for s in sidecars:
        b = paths.retraining_of_grid(s)
        data, mc = (paths.retraining(ds, d, b, k) for k in ("DATA", "MC"))
        if not (os.path.exists(data) and os.path.exists(mc)):
            raise FileNotFoundError(f"{data} or its MC file is not there")
        curves.append(fn(data, mc, edges))
        src.append(data)
    CL = np.array(curves)
    with warnings.catch_warnings():                      # bins no retraining fills stay empty
        warnings.simplefilter("ignore", RuntimeWarning)
        nom = np.nanmedian(CL, axis=0)
        sig = np.nanstd(np.log(CL), axis=0, ddof=1)
    out = {"edges": edges, "x": 0.5 * (edges[:-1] + edges[1:]), "replicas": CL, "nominal": nom,
           "sigma_log": sig, "lo": nom * np.exp(-sig), "hi": nom * np.exp(sig)}
    note = {"dataset": inputs.RUN.insitu.dataset, "distortion": d, "retrainings": len(curves), "against": var}

    # the band the parameter model predicts, next to the retrainings' spread
    inp = paths.division(ds, d, 0, "DATA")          # the jets the retrainings corrected
    if os.path.exists(inp):
        mc0 = paths.retraining(ds, d, paths.retraining_of_grid(sidecars[0]), "MC")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            m = _model_band(sidecars, src, mc0, inp, edges, var, inputs.RUN.band.keep)
        n = m["n_np"]
        out |= {k: m[k] for k in ("sigma_np", "sigma_np_res", "sigma_all", "sigma_res", "shift_each", "model_base")}
        out |= {"n_np": n, "lo_np": nom * np.exp(-m["sigma_np"]), "hi_np": nom * np.exp(m["sigma_np"]),
                "lo_np_res": nom * np.exp(-m["sigma_np_res"]), "hi_np_res": nom * np.exp(m["sigma_np_res"])}
        note |= {"model_input": inp, "model_jets_below_grid": m["below_grid"], "model_jets_above_grid": m["above_grid"]}
        src = src + [inp]
        f = lambda a: " ".join(f"{100 * v:5.2f}" if np.isfinite(v) else "    -" for v in a)
        print(f"    band per bin [%]  retrainings  {f(sig)}\n"
              f"                      {n} NP         {f(m['sigma_np'])}\n"
              f"                      {n} NP + res   {f(m['sigma_np_res'])}\n"
              f"                      all comp.    {f(m['sigma_all'])}\n"
              f"    (check) closure of the jets moved, minus the retrainings' median [%]: {f(m['model_base'] - nom)}\n"
              f"    jets before the correction below the grid's 25 GeV: {100 * m['below_grid']:.0f}% "
              f"(given the shapes at 25 GeV)", flush=True)
    else:
        print(f"    {inp} not there: the model's band is not drawn", flush=True)
    store.save(name, out, src, note)


def draw_closure(name, subdir, var):
    d = store.load(name)
    ok = np.isfinite(d["nominal"])
    fig, ax = S.figure("half", height=2.5)
    model = "sigma_np" in d
    ax.fill_between(d["x"][ok], d["lo"][ok], d["hi"][ok], color=BAND, alpha=0.25, lw=0,
                    label="Retrainings" if model else r"$\pm1\sigma$ band")
    ax.plot(d["x"][ok], d["nominal"][ok], "-", marker="o", color=BAND, label="Corrected data")
    if model:                                            # the band of the parameter model
        n = int(d["n_np"])
        for key, ls, col, lab in (("np", "--", S.COL["aux"], f"{n} NP"),
                                  ("np_res", "-", S.COL["ref"], f"{n} NP + residual")):
            for j, side in enumerate(("lo", "hi")):
                ax.plot(d["x"][ok], d[f"{side}_{key}"][ok], ls=ls, color=col, lw=0.8, label=lab if j == 0 else None)
    ax.axhline(1.0, color="black", ls=":", lw=0.6)
    ax.set_ylim(0.90, 1.10)
    ax.set_xlabel(r"$p_T^{\rm gen}$ [GeV]" if var == "pt" else r"$|\eta|$")
    ax.set_ylabel(RATIO)
    ax.legend(loc="upper right", ncols=2 if model else 1)
    return S.save(fig, store.fig_path(subdir, name))


# ───────────── the band against the scatter over divisions ─────────────
_VAL = {}


def _map(fn, jobs):
    """fn over jobs, side by side when that works (each job is independent)."""
    if JOBS > 1 and len(jobs) > 1:
        try:
            from concurrent.futures import ProcessPoolExecutor

            with ProcessPoolExecutor(max_workers=JOBS) as ex:
                return list(ex.map(fn, jobs))
        except Exception as e:                           # fall back to one at a time
            print(f"    (side-by-side reading failed: {type(e).__name__}; one at a time)", flush=True)
    return [fn(j) for j in jobs]


def _validation():
    """Divisions and retrainings summarised (jetcorrml.band)."""
    if not _VAL:
        ds = inputs.RUN.insitu.dataset
        dists = list(dict.fromkeys([inputs.RUN.band.dist] + list(inputs.RUN.band.dists)))
        div_jobs = [(ds, d, s) for d in dists for s in range(inputs.RUN.band.divisions)]
        ref_jobs = [(ds, d, 200) for d in dists]          # 200: its default "at most"
        div = _map(band.one_division, div_jobs)
        ref = _map(band.all_refits, ref_jobs)
        D = {d: [r for (_ds, dd, _s), r in zip(div_jobs, div) if dd == d and r is not None] for d in dists}
        R = dict(zip(dists, ref))
        src = [p for d in dists for s in range(inputs.RUN.band.divisions) for p in [Path(paths.insitu(ds, d, s, "nn", "DATA"))] if p.is_file()]
        _VAL.update(D=D, R=R, src=src)
    return _VAL["D"], _VAL["R"], _VAL["src"]


def compute_regions(name):
    D, R, src = _validation()
    out, ratios, used = {}, [], []
    for d in inputs.RUN.band.dists:
        if len(D[d]) >= 3 and len(R[d]) >= 3:
            g = band.region_numbers(D[d], R[d])
            out[f"{d}_spread"], out[f"{d}_band"] = g["scat"][g["ok"]], g["band"][g["ok"]]
            out[f"{d}_mean"], out[f"{d}_jets"] = g["mean"][g["ok"]], g["n"][g["ok"]]
            out[f"{d}_inside"] = g["inside"]
            ratios += list(g["band"][g["ok"]] / g["scat"][g["ok"]])
            used.append(d)
    if not used:
        raise FileNotFoundError("no distortion with three divisions and three retrainings")
    out["median_band_over_spread"], out["regions"] = float(np.median(ratios)), len(ratios)
    store.save(name, out, src, {"dataset": inputs.RUN.insitu.dataset, "distortions": used})


def draw_regions(name, subdir):
    d = store.load(name)
    dists = d["__meta__"]["notes"]["distortions"]
    fig, ax = S.figure("half", height=3.0)
    allv = np.concatenate([np.concatenate([d[f"{k}_spread"], d[f"{k}_band"]]) for k in dists])
    lo, hi = 0.6 * np.nanmin(allv), 1.9 * np.nanmax(allv)
    ax.plot([lo, hi], [lo, hi], "--", color="black", lw=0.7, label="Band = spread")
    for k, mk in zip(dists, "osD^v<"):
        ax.plot(d[f"{k}_spread"], d[f"{k}_band"], ls="", marker=mk, ms=S.ms(4.0), mfc=BAND, mec=BAND, mew=0.3,
                label=DIST_NAME.get(k, k))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ticks = [t for t in (0.03, 0.1, 0.3, 1, 3, 10) if lo < t < hi]
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_ticks(ticks)
        axis.set_ticklabels([f"{t:g}" for t in ticks])
        axis.set_ticks([], minor=True)
    ax.set_xlabel("Spread over divisions [%]")
    ax.set_ylabel("Band width [%]")
    ax.legend(loc="upper left")
    ax.text(0.97, 0.04, f"Median band/spread {float(d['median_band_over_spread']):.2f}",
            transform=ax.transAxes, ha="right", va="bottom")
    return S.save(fig, store.fig_path(subdir, name))
