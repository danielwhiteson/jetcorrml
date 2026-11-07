"""
Paper figures of the large-R (sub-jet) section.

The numbers are computed from three files (the same jets in the same order):

    the pile-up-corrected file          response before the energy-scale step
    the corrected file, no sub-jet inputs   ("without")
    the corrected file, sub-jet inputs      ("with")

and, for the network probe, the two saved networks beside the corrected files
(they are evaluated on a scan of 160 points; nothing is trained).  The jets are
binned in their own rapidity |y| and split by the angle of the sub-jet axis.
"""

import contextlib
import io
from pathlib import Path

import numpy as np

from jetcorrml import paths, settings
from jetcorrml.absolute_closure import closure_floor
from jetcorrml.large_r import build_response_features, load_files, set_crack_feature
from jetcorrml.networks import Normaliser, ResponseNet
from jetcorrml.profiles import profile_xy

from . import store
from . import style as S

DS = settings.LARGE_R_DATASET
FAT_COORD = "jet_y"                 # the jet's own rapidity
PROBE_ENERGY = 400.0                # GeV: the energy of the network probe
CRACK_WINDOW = (1.15, 1.6)          # |y| range whose sub-jet separation the probe uses
PROBE_MAX = 1.6
CRACK = settings.LARGE_R.eta_crack

CLASSES = (("eta", r"Sub-jets along $\eta$", "alpha_eta", "o"),
           ("phi", r"Sub-jets along $\phi$", "alpha_phi", "s"))
WITHOUT, WITH = "Without sub-jet inputs", "With sub-jet inputs"
XLAB = r"Jet $|y|$"
_J = {}


# the crack window of the figures and their binning
ETA_LO, ETA_HI = 1.07, 1.67
N_ETA_BINS = 15
MIN_COUNT = 50


def _alpha_slices(alpha):
    """The two classes of the sub-jet axis angle alpha = atan2(dphi, deta) in [-pi, pi]:
    along eta (alpha ~ 0 or +-pi) and along phi (alpha ~ +-pi/2), each the union of two of
    four quarter slices."""
    pi4 = np.pi / 4
    quarters = [
        (alpha >= -pi4) & (alpha < pi4),
        (alpha >= pi4) & (alpha < 3 * pi4),
        (alpha >= 3 * pi4) | (alpha < -3 * pi4),
        (alpha >= -3 * pi4) & (alpha < -pi4),
    ]
    return [quarters[0] | quarters[2], quarters[1] | quarters[3]]


def _sigma68(y):
    """Half the 16-84% range (NaN below 20 jets)."""
    if y.size < 20:
        return np.nan
    q16, q84 = np.percentile(y, [16.0, 84.0])
    return 0.5 * (q84 - q16)


def _sigma68_profile(x, y, edges, log=False):
    cx, s68 = [], []
    for i in range(len(edges) - 1):
        m = (x >= edges[i]) & (x < edges[i + 1])
        if int(m.sum()) >= MIN_COUNT:
            c = (
                np.sqrt(edges[i] * edges[i + 1])
                if log
                else 0.5 * (edges[i] + edges[i + 1])
            )
            cx.append(c)
            s68.append(_sigma68(y[m]))
    return np.array(cx), np.array(s68)


def _load_probe_model(path):
    """A saved large-R network and its input standardisation; sets the crack input as the
    network was trained with."""
    import torch

    ck = torch.load(str(path), map_location="cpu", weights_only=False)
    net = ResponseNet(ck["n_features"], n_hidden=ck["n_hidden"], width=ck["width"], dropout=ck["dropout"])
    net.load_state_dict(ck["state_dict"])
    net.eval()
    norm = Normaliser()
    norm.mean_ = ck["norm_mean"]
    norm.std_ = ck["norm_std"]
    if bool(ck["use_subjet_eta"]):
        set_crack_feature(ck["crack_ref"], ck["crack_signed"])
    return net, norm, bool(ck["use_subjet_eta"])


def _jets():
    """The three files, and what the figures derive from them."""
    if not _J:
        files = [Path(paths.large_r_pileup()), Path(paths.large_r_absolute(subjets=True)),
                 Path(paths.large_r_absolute(subjets=False))]
        for p in files:
            if not p.is_file():
                raise FileNotFoundError(f"{p} is not there")
        with contextlib.redirect_stdout(io.StringIO()):
            d_in, d_on, d_off = (load_files([str(p)]) for p in files)
        n = len(d_in["e_reco"])
        if not (len(d_on["e_reco"]) == n == len(d_off["e_reco"])):
            raise ValueError("the three large-R files do not hold the same jets")
        e_gen = d_in["e_gen"]
        masks = _alpha_slices(d_in["alpha"])
        e_gen_min = closure_floor(d_in["eta_geom"])
        _J.update(d_in=d_in, files=files, masks=masks,
                  x=np.abs(d_in["eta_det"]),                      # |y| of the jet with --fat-coord jet_y
                  good=(d_in["dr"] < settings.LARGE_R_MATCH_DR) & (e_gen >= e_gen_min), e_gen=e_gen,
                  unc=d_in["e_reco"] / np.maximum(e_gen, 1.0),
                  on=d_on["e_reco"] / np.maximum(e_gen, 1.0),
                  off=d_off["e_reco"] / np.maximum(e_gen, 1.0))
    return _J


def _class_curves(J, y):
    """Median of y against jet |y| in the crack window, one curve per alpha class."""
    out = {}
    for (key, _lab, _col, _mk), mask in zip(CLASSES, J["masks"]):
        sel = J["good"] & mask
        x = yv = np.array([])
        if int(sel.sum()) >= MIN_COUNT:
            x, yv, _, _ = profile_xy(J["x"][sel], y[sel], n_bins=N_ETA_BINS,
                                     x_range=(ETA_LO, ETA_HI))
        out[f"{key}_x"], out[f"{key}_y"] = store.pad(x, N_ETA_BINS), store.pad(yv, N_ETA_BINS)
    return out


def _note(J, **more):
    return {"dataset": DS, "fat_coord": FAT_COORD, "jets": int(len(J["x"])), "pass_gate": int(J["good"].sum()),
            "window": [ETA_LO, ETA_HI], **more}


def _draw_classes(ax, d, prefix=""):
    for key, lab, col, mk in CLASSES:
        ok = np.isfinite(d[f"{prefix}{key}_y"])
        ax.plot(d[f"{prefix}{key}_x"][ok], d[f"{prefix}{key}_y"][ok], "-", marker=mk, color=S.COL[col], label=lab)
    ax.axhline(1.0, color="black", ls=":", lw=0.6)
    ax.set_xlabel(XLAB)


# ───────────── response before the energy-scale step (Fig. 16 left) ─────────────
def compute_uncorrected(name):
    J = _jets()
    store.save(name, _class_curves(J, J["unc"]), J["files"][:1], _note(J))


def draw_uncorrected(name, subdir):
    d = store.load(name)
    fig, ax = S.figure("half", height=2.5)
    _draw_classes(ax, d)
    ax.set_ylabel(r"Median $E^{\rm reco}/E^{\rm gen}$")
    S.legend_top(ax, "upper right")
    return S.save(fig, store.fig_path(subdir, name))


# ───────────── response after it, without and with sub-jet inputs (Fig. 17) ─────────────
def compute_corrected(name):
    J = _jets()
    out = {}
    for which in ("off", "on"):
        out |= {f"{which}_{k}": v for k, v in _class_curves(J, J[which]).items()}
    store.save(name, out, J["files"], _note(J))


def draw_corrected(name, subdir):
    d = store.load(name)
    fig, axes = S.figure("full", height=2.6, ncols=2, sharey=True)
    for ax, which, text in ((axes[0], "off", WITHOUT), (axes[1], "on", WITH)):
        _draw_classes(ax, d, f"{which}_")
    axes[0].set_ylabel(r"Median $E^{\rm corr}/E^{\rm gen}$")
    S.legend_top(axes[1], "upper right")
    for ax, text in ((axes[0], WITHOUT), (axes[1], WITH)):
        S.pane_label(ax, text, "upper left")
    return S.save(fig, store.fig_path(subdir, name))


# ───────────── resolution after it (Fig. 18) ─────────────
def compute_resolution(name):
    J = _jets()
    g = J["good"]
    eg, yg = J["e_gen"][g], J["x"][g]
    e_edges = np.geomspace(max(100.0, float(np.percentile(eg, 1))), float(np.percentile(eg, 99)), 21)
    y_edges = np.linspace(0.0, 2.5, 26)
    out = {"e_edges": e_edges, "y_edges": y_edges}
    for which in ("off", "on"):
        r = J[which][g]
        out[f"{which}_e_x"], out[f"{which}_e_y"] = _sigma68_profile(eg, r, e_edges, log=True)
        out[f"{which}_y_x"], out[f"{which}_y_y"] = _sigma68_profile(yg, r, y_edges, log=False)
    store.save(name, out, J["files"], _note(J))


def draw_resolution(name, subdir):
    d = store.load(name)
    fig, axes = S.figure("full", height=2.6, ncols=2)
    look = (("off", WITHOUT, S.COL["ref"], "--", "s"), ("on", WITH, S.COL["nn"], "-", "o"))
    for ax, v in zip(axes, ("e", "y")):
        for which, lab, col, ls, mk in look:
            ax.plot(d[f"{which}_{v}_x"], d[f"{which}_{v}_y"], ls, marker=mk, color=col, label=lab)
        ax.set_ylabel(S.LAB["sigma68"])
    S.plain_log_x(axes[0])
    axes[0].set_xlabel(S.LAB["egen"])
    axes[1].set_xlabel(XLAB)
    S.legend_top(axes[0], "upper left", also=(axes[1],))
    return S.save(fig, store.fig_path(subdir, name))


# ───────────── what each network learned (Fig. 16 right) ─────────────
def compute_probe(name):
    """The correction 1/R of each network against jet |y| at a fixed energy, for sub-jets
    apart in eta (by the median separation in the crack window) or at the same eta."""
    import torch

    J = _jets()
    d_in = J["d_in"]
    models = [Path(paths.large_r_model(subjets=True)), Path(paths.large_r_model(subjets=False))]
    for p in models:
        if not p.is_file():
            raise FileNotFoundError(f"{p} is not there (the network saved by the large-R absolute scale)")
    lo, hi = CRACK_WINDOW
    crackm = J["good"] & (J["x"] >= lo) & (J["x"] < hi)
    dsep = float(np.median(np.abs(d_in["sjA_eta"][crackm] - d_in["sjB_eta"][crackm])))
    scan = np.linspace(0.0, PROBE_MAX, 160).astype(np.float32)
    log_e = np.log(np.full_like(scan, PROBE_ENERGY))
    configs = {"eta": (scan + dsep / 2, np.abs(scan - dsep / 2)), "phi": (scan.copy(), scan.copy())}
    out = {"x": scan, "separation": dsep}
    for which, path in (("on", models[0]), ("off", models[1])):
        # (loading a network also sets how the crack-distance input is built for it)
        net, norm, use_sj = _load_probe_model(path)
        for key, (sj_hi, sj_lo) in configs.items():
            X = (build_response_features(log_e, scan, sj_hi.astype(np.float32), sj_lo.astype(np.float32))
                 if use_sj else build_response_features(log_e, scan))
            with torch.no_grad():
                R = net(torch.from_numpy(norm.transform(X).astype(np.float32))).numpy().ravel()
            out[f"{which}_{key}"] = 1.0 / np.clip(R, 0.01, None)
    store.save(name, out, J["files"][:1] + models, _note(J, probe_energy=PROBE_ENERGY))


def draw_probe(name, subdir):
    from matplotlib.lines import Line2D

    d = store.load(name)
    fig, ax = S.figure("half", height=2.5)          # the height of its companion, draw_uncorrected
    for key, _lab, col, _mk in CLASSES:
        ax.plot(d["x"], d[f"on_{key}"], "-", color=S.COL[col], lw=1.3)
        ax.plot(d["x"], d[f"off_{key}"], "--", color=S.COL["ref"], lw=1.0)
    ax.axvline(CRACK, color=S.COL["aux"], ls=":", lw=0.7)
    ax.set_xlim(0.0, float(d["x"][-1]))
    ax.set_xlabel(XLAB)
    ax.set_ylabel(r"Correction $1/\mathcal{R}$")
    # colour says how the sub-jets lie, the line says which network
    S.legend_top(ax, "upper left",
                 handles=[Line2D([], [], color=S.COL[c], ls="-") for _k, _l, c, _m in CLASSES]
                 + [Line2D([], [], color=S.COL["ref"], ls="--")],
                 labels=[r"With sub-jet inputs, along $\eta$", r"With sub-jet inputs, along $\phi$", WITHOUT])
    return S.save(fig, store.fig_path(subdir, name))
