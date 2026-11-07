"""
The one figure style of the paper.

The rules (checked by `save`, which refuses a figure that breaks one):

  * no text above a pane: no titles; what a title said goes inside the pane
    (`pane_label`) or into the caption
  * all text in ONE size and ONE font, mathematics included
  * no grid lines
  * the frame is drawn on all four sides
  * legends have no border
  * no white space around the figure (it is cut to its contents on saving)

Figures are made at the size they are printed, so TEXT_PT is the size of the
text on the page.  A figure is given the width it has in the paper
("third", "half", "wide", "full" of the line width).  To change the look of
every paper figure, change the constants below and run

    uv run python scripts/figures.py --draw-only
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# ─────────────────────────────── the knobs ───────────────────────────────
TEXT_PT = 10.0            # every piece of text, in points on the printed page
FONT = "STIXGeneral"      # text font (ships with matplotlib; Times-like)
MATH = "stix"             # the same font for mathematics
PAGE_IN = 7.0             # \linewidth of the paper, inches (revtex4 aps, one column)
WIDTHS = {"third": 0.30, "half": 0.45, "wide": 0.85, "full": 0.95}   # as in \includegraphics

# The same colour and marker for the same thing in every figure.
COL = {
    "none": "#666666",    # injected sample, no correction; data before the in-situ correction
    "rhoA": "#00AA00",    # the injected sample (after the area subtraction): green
    "nn": "#0040FF",      # NN: bright blue
    "parametric": "#FF0000",       # the parametric correction (after ATLAS): bright red
    "parametric_full": "#AA00FF",  # parametric on parametric pile-up input, the full parametric chain: purple
    "ref": "#000000",     # reference (particle level, simulation, the exact answer)
    "reco": "#8B0000",    # reconstructed jets of the raw sample (dark red)
    "raw": "#000000",     # the Delphes sample before any injection (black; green = injected)
    "alpha_eta": "#AA00FF",   # large-R jets whose two sub-jets lie along eta (purple)
    "alpha_phi": "#00AA00",   # ... along phi (green)
    "aux": "#7f7f7f",     # guide lines (statistical floor, ...)
}
MARK = {"none": "o", "rhoA": "s", "nn": "D", "parametric": "^", "ref": "o", "raw": "o", "reco": "o",
        "parametric_full": "v"}
MARKER_SCALE = 0.5        # every marker, relative to the first version of the style
PAD_IN = 0.01             # white space left around a figure, inches
CMAP = "viridis"          # for maps and for families of curves (slices)

# The same symbol for the same quantity in every figure.
LAB = {
    "npv": r"$N_{\rm PV}$",
    "npv1": r"$N_{\rm PV}-1$",
    "mu": r"$\mu$",
    "eta": r"$\eta_{\rm det}$",
    "abseta": r"$|\eta_{\rm det}|$",
    "egen": r"$E^{\rm gen}$ [GeV]",
    "resid": r"$p_T-p_T^{\rm gen}$ [GeV]",
    "relresid": r"$(p_T-p_T^{\rm gen})\,/\,p_T^{\rm gen}$",
    "dnpv": r"$\partial p_T/\partial N_{\rm PV}$ [GeV]",
    "dmu": r"$\partial p_T/\partial\mu$ [GeV]",
    "resp_med": r"median $E/E^{\rm gen}$",
    "resp_med_reco": r"median $E^{\rm reco}/E^{\rm gen}$",   # before / after the correction
    "resp_med_corr": r"median $E^{\rm corr}/E^{\rm gen}$",
    "corr": r"$\rho_{\mu N}$",
    "rho": r"$\rho$ [GeV/area]",
    "norm": "Normalized / bin",
    "reco": "Reconstructed",
    "gen": "Particle level",
    "ptresp": r"$p_T^{\rm reco}/p_T^{\rm gen}$",
    "ptresp_mean": r"$\langle p_T^{\rm reco}/p_T^{\rm gen}\rangle$",
    "eresp": r"$E^{\rm reco}/E^{\rm gen}$",
    "eresp_mean": r"$\langle E^{\rm reco}/E^{\rm gen}\rangle$",
    "lo": r"lo-$p_T$",
    "hi": r"hi-$p_T$",
    "sigma68": r"$\sigma_{68}(E^{\rm corr}/E^{\rm gen})$",
    "st_parametric": r"After $\rho A$ and parametric",
    "parametric": "Parametric",     # the baseline: it follows ATLAS
    "st_none": "No correction",
    "st_rhoA": r"After $\rho A$",
    "st_nn": r"After $\rho A$ and NN",
}

_LW = 0.7                 # frame and tick lines


def ms(size):
    """A marker size, scaled as every marker of the paper is."""
    return size * MARKER_SCALE


def apply():
    """Set the style.  Call once before drawing."""
    plt.rcdefaults()
    s = TEXT_PT
    plt.rcParams.update({
        "font.family": FONT, "mathtext.fontset": MATH, "font.size": s,
        "axes.labelsize": s, "axes.titlesize": s, "xtick.labelsize": s, "ytick.labelsize": s,
        "legend.fontsize": s, "legend.title_fontsize": s, "figure.titlesize": s,
        "axes.grid": False,
        "axes.spines.left": True, "axes.spines.right": True,
        "axes.spines.top": True, "axes.spines.bottom": True,
        "axes.linewidth": _LW, "axes.labelpad": 3.0,
        "xtick.major.width": _LW, "ytick.major.width": _LW,
        "xtick.minor.width": 0.5, "ytick.minor.width": 0.5,
        "xtick.major.size": 3.0, "ytick.major.size": 3.0,
        "xtick.minor.size": 1.8, "ytick.minor.size": 1.8,
        "xtick.major.pad": 2.5, "ytick.major.pad": 2.5,
        "lines.linewidth": 1.1, "lines.markersize": ms(3.5),
        "errorbar.capsize": 0.0,
        "legend.frameon": False, "legend.handlelength": 1.3, "legend.handletextpad": 0.4,
        "legend.labelspacing": 0.25, "legend.borderpad": 0.2, "legend.borderaxespad": 0.4,
        "legend.columnspacing": 0.9,
        "pdf.fonttype": 42, "savefig.dpi": 300,
        "figure.constrained_layout.use": True,
        "figure.constrained_layout.h_pad": 0.03, "figure.constrained_layout.w_pad": 0.03,
    })


def figure(width="full", height=2.6, ncols=1, nrows=1, **kw):
    """A figure of the width it has in the paper; height in inches."""
    w = PAGE_IN * WIDTHS[width]
    fig, axes = plt.subplots(nrows, ncols, figsize=(w, height), **kw)
    return fig, axes


def pane_label(ax, text, loc="upper left"):
    """What a title would have said, inside the pane."""
    v, h = loc.split()
    x = {"left": 0.04, "center": 0.5, "right": 0.96}[h]
    y = {"upper": 0.95, "lower": 0.05}[v]
    return ax.text(x, y, text, transform=ax.transAxes, ha=h,
                   va={"upper": "top", "lower": "bottom"}[v])


def legend_top(ax, loc="upper left", also=(), **kw):
    """A legend along the top of the pane, with the y axis extended upward so that
    it covers no point.  If it is wider than the pane it is given fewer columns.
    `also`: other panes given the same room, to match."""
    ncols = kw.pop("ncols", 1)
    while True:
        leg = ax.legend(loc=loc, ncols=ncols, **kw)
        ax.figure.canvas.draw()
        if ncols == 1 or leg.get_window_extent().width <= 0.98 * ax.get_window_extent().width:
            break
        ncols -= 1
    frac = min(0.6, leg.get_window_extent().height / ax.get_window_extent().height + 0.05)
    for a in (ax, *also):
        lo, hi = a.get_ylim()
        if a.get_yscale() == "log":
            a.set_ylim(lo, lo * (hi / lo) ** (1.0 / (1.0 - frac)))
        else:
            a.set_ylim(lo, lo + (hi - lo) / (1.0 - frac))
    return leg


def plain_log(ax, which="x", ticks=None):
    """A logarithmic axis with ordinary numbers (20, 50, 100, ...) instead of powers of
    ten: at the given places, or at the 1-2-5 values inside the range drawn."""
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter

    (ax.set_xscale if which == "x" else ax.set_yscale)("log")
    axis = ax.xaxis if which == "x" else ax.yaxis
    get, put = (ax.get_xlim, ax.set_xlim) if which == "x" else (ax.get_ylim, ax.set_ylim)
    if ticks is None:
        lo, hi = get()
        nice = [m * 10.0 ** k for k in range(int(np.floor(np.log10(lo))) - 1, int(np.ceil(np.log10(hi))) + 2)
                for m in (1, 2, 5)]
        below = max(v for v in nice if v <= lo)          # reach out to a round number when it is close
        above = min(v for v in nice if v >= hi)
        lo = below if lo / below < 1.3 else lo
        hi = above if above / hi < 1.3 else hi
        ticks = [v for v in nice if lo <= v <= hi]
        put(lo / 1.04, hi * 1.04)
    axis.set_major_locator(FixedLocator(list(ticks)))
    axis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v:g}"))
    axis.set_minor_formatter(NullFormatter())


def plain_log_x(ax, ticks=None):
    plain_log(ax, "x", ticks)


def ramp(n, lo=0.0, hi=0.85):
    """n colours for a family of curves."""
    return [matplotlib.colormaps[CMAP](v) for v in np.linspace(lo, hi, max(n, 1))]


def rng(lo, hi):
    """'0.3-0.8' with an en dash and no needless zeros."""
    f = lambda v: f"{v:g}"
    return f"{f(lo)}–{f(hi)}"


def _broken_rules(fig):
    bad = []
    if fig._suptitle is not None and fig._suptitle.get_text():
        bad.append("figure title")
    texts = []
    for ax in fig.get_axes():
        for t in (ax.title, ax._left_title, ax._right_title):
            if t.get_text():
                bad.append(f"pane title '{t.get_text()}'")
        is_bar = ax.get_label() == "<colorbar>"
        if not is_bar:
            if not all(ax.spines[k].get_visible() for k in ("left", "right", "top", "bottom")):
                bad.append("frame not drawn on all four sides")
            if any(ln.get_visible() for ln in ax.get_xgridlines() + ax.get_ygridlines()):
                bad.append("grid lines")
        leg = ax.get_legend()
        if leg is not None:
            if leg.get_frame_on():
                bad.append("legend with a border")
            texts += list(leg.get_texts()) + [leg.get_title()]
        texts += [ax.xaxis.label, ax.yaxis.label, ax.xaxis.get_offset_text(),
                  ax.yaxis.get_offset_text()] + list(ax.texts)
        texts += ax.get_xticklabels() + ax.get_yticklabels()
        texts += ax.get_xticklabels(minor=True) + ax.get_yticklabels(minor=True)
    for t in texts:
        if not t.get_text():
            continue
        if abs(t.get_fontsize() - TEXT_PT) > 0.01:
            bad.append(f"text '{t.get_text()}' in {t.get_fontsize():g} pt")
        if t.get_fontname() != FONT:
            bad.append(f"text '{t.get_text()}' in font {t.get_fontname()}")
    return bad


def save(fig, path):
    """Write the figure, or refuse it if it breaks a rule of the style."""
    fig.canvas.draw()                      # tick labels exist only after a draw
    bad = _broken_rules(fig)
    if bad:
        plt.close(fig)
        raise ValueError(f"{Path(path).name} breaks the paper style: " + "; ".join(sorted(set(bad))))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # a fixed creation time: the same numbers give the same file, byte for byte
    fig.savefig(path, bbox_inches="tight", pad_inches=PAD_IN,       # no white space around the figure
                metadata={"CreationDate": None, "ModDate": None} if path.suffix == ".pdf" else None)
    plt.close(fig)
    return path
