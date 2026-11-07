#!/usr/bin/env python3
"""study_band_regions.py -- the share of the band's variance the leading components carry, region by region.  READ-ONLY.

    uv run python scripts/study_band_regions.py               # the paper's distortion (expo)
    uv run python scripts/study_band_regions.py --dist quad   # another one, if its retrainings were made

Run from the top of a pipeline folder in which the paper run has finished.  It reads the saved retrainings
of the nuisance-parameter band (jetcorrml.paths.retraining_grids: the in-situ scale
of each of the 30 retrainings on a grid of 24 pT x 7 |eta| points) and repeats the decomposition of the band
(jetcorrml.band.decompose): logarithm of the scale, centred over the retrainings, covariance C
over the 168 grid points, eigenvalues lambda_i and shapes phi_i.  Then, for every grid point x,

    f_k(x) = sum_{i<=k} lambda_i phi_i(x)^2 / C(x, x)

is the share of the variance AT THAT POINT carried by the first k components (k = 1, 2, 3).  The paper's
number (first component 84%, three for 95% on the ATLAS-card pt-10 run) is the same ratio taken over all
points at once, sum_x numerator / sum_x C(x, x), so it is weighted by where the variance is.

It WRITES only study_np_regions/REPORT.txt, numbers.json and regions.pdf.  Nothing is trained.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

from jetcorrml import paths, settings
from jetcorrml.band import decompose

DS = settings.PAPER.insitu.dataset                    # the in-situ dataset of the paper
KS = (1, 2, 3)
PT_REGIONS = (("pT < 60", 0.0, 60.0), ("60-120", 60.0, 120.0), ("pT > 120", 120.0, 1e9))
ETA_REGIONS = (("|eta| < 0.8", 0.0, 0.8), ("0.8-2.0", 0.8, 2.0), ("|eta| > 2.0", 2.0, 9.0))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dist", default=settings.PAPER.band.dist,
                    help=f"distortion whose band is read (default {settings.PAPER.band.dist}, the paper's)")
    ap.add_argument("--out", default="study_np_regions")
    a = ap.parse_args()
    grids = paths.retraining_grids(DS, a.dist)
    if len(grids) < 2:
        sys.exit(f"found {len(grids)} retrainings ({paths.retraining_grid(DS, a.dist, '*')}); need at least 2. "
                 f"Run from the top of a pipeline folder after the paper run.")

    pt, eta, _z0, lam, phi, C, B = decompose(grids)
    npt, neta = len(pt), len(eta)
    var = np.diag(C).copy()                                   # variance at each grid point (of log s)
    tot = max(lam.sum(), 1e-300)
    num = {k: np.sum(lam[None, :k] * phi[:, :k] ** 2, axis=1) for k in KS}
    with np.errstate(invalid="ignore", divide="ignore"):
        f = {k: np.where(var > 0, num[k] / var, np.nan) for k in KS}
    P, E = np.meshgrid(pt, eta, indexing="ij")
    P, E = P.ravel(), E.ravel()

    L, say = [], None
    say = L.append
    say(f"Share of the band's variance carried by the leading components, point by point ({a.dist}, {B} retrainings)")
    say(f"Grid: {npt} pT points {pt[0]:.0f}-{pt[-1]:.0f} GeV x {neta} |eta| points {', '.join(f'{e:g}' for e in eta)}")
    say("")
    say("Global shares (the paper's numbers; must match the band program's output/paper/tables/band_modes.txt):")
    say("  " + ", ".join(f"{l / tot:.2f}" for l in lam[:6])
        + f"   components for 95%: {int(np.sum(np.cumsum(lam) / tot < 0.95) + 1)}")
    say("")
    say("Where the variance is (share of the total, by region; the global number is weighted this way):")
    head = f"  {'':12s}" + "".join(f"{n:>14s}" for n, _, _ in PT_REGIONS) + f"{'all pT':>14s}"
    say(head)
    for en, elo, ehi in ETA_REGIONS + (("all |eta|", 0.0, 9.0),):
        row = f"  {en:12s}"
        for pn, plo, phi_ in PT_REGIONS + (("all", 0.0, 1e9),):
            m = (E >= elo) & (E < ehi) & (P >= plo) & (P < phi_)
            row += f"{100 * var[m].sum() / var.sum():13.1f}%"
        say(row)
    say("")
    for k in KS:
        say(f"Share carried by the first {k} component(s), within each region (sum of numerator / sum of variance):")
        say(head)
        for en, elo, ehi in ETA_REGIONS + (("all |eta|", 0.0, 9.0),):
            row = f"  {en:12s}"
            for pn, plo, phi_ in PT_REGIONS + (("all", 0.0, 1e9),):
                m = (E >= elo) & (E < ehi) & (P >= plo) & (P < phi_)
                row += f"{100 * num[k][m].sum() / var[m].sum():13.1f}%"
            say(row)
        fk = f[k]
        i = int(np.nanargmin(fk))
        say(f"  point by point: lowest {100 * fk[i]:.1f}% (pT {P[i]:.0f} GeV, |eta| {E[i]:g}); "
            f"median {100 * np.nanmedian(fk):.1f}%; points above 90%: {int(np.sum(fk > 0.9))} of {fk.size}")
        say("")

    say("Point by point, share carried by the first 1 / 2 / 3 components (%), and the band width sqrt(C) (% of s):")
    say(f"  {'pT':>5s} | " + " | ".join(f"|eta| {e:<4g}      " for e in eta))
    for ip in range(npt):
        cells = []
        for ie in range(neta):
            x = ip * neta + ie
            cells.append(f"{100 * f[1][x]:3.0f}/{100 * f[2][x]:3.0f}/{100 * f[3][x]:3.0f} {100 * np.sqrt(var[x]):4.2f}")
        say(f"  {pt[ip]:5.0f} | " + " | ".join(cells))
    say("")
    say("f_k(x) = sum_{i<=k} lambda_i phi_i(x)^2 / C(x,x), the decomposition of src/steps/band.py.")
    text = "\n".join(L)
    print(text)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "REPORT.txt").write_text(text + "\n")
    (out / "numbers.json").write_text(json.dumps({
        "dist": a.dist, "n_retrainings": B, "pt": pt.tolist(), "eta": eta.tolist(),
        "global_share": (lam / tot).tolist(),
        "band_width": np.sqrt(var).reshape(npt, neta).tolist(),
        **{f"share_first_{k}": f[k].reshape(npt, neta).tolist() for k in KS},
        "files": [os.path.basename(p) for p in grids]}, indent=1))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        def edges(c):
            m = 0.5 * (c[1:] + c[:-1])
            return np.concatenate([[c[0] - (m[0] - c[0])], m, [c[-1] + (c[-1] - m[-1])]])

        fig, axes = plt.subplots(2, 2, figsize=(9, 6.5), constrained_layout=True)
        panels = [(100 * np.sqrt(var), "band width sqrt(C) [% of s]", "viridis", None)] + \
                 [(100 * f[k], f"share carried by the first {k} [%]", "magma", (0, 100)) for k in KS]
        for ax, (z, title, cmap, lim) in zip(axes.ravel(), panels):
            m = ax.pcolormesh(edges(pt), edges(eta), z.reshape(npt, neta).T, cmap=cmap,
                              vmin=None if lim is None else lim[0], vmax=None if lim is None else lim[1])
            fig.colorbar(m, ax=ax)
            ax.set_title(title, fontsize=10)
            ax.set_xlabel("pT [GeV]")
            ax.set_ylabel("|eta|")
        fig.suptitle(f"In-situ band ({a.dist}, {B} retrainings): global shares "
                     + ", ".join(f"{l / tot:.2f}" for l in lam[:3]), fontsize=11)
        fig.savefig(out / "regions.pdf")
        plt.close(fig)
        figs = f", {out}/regions.pdf"
    except Exception as err:                                  # noqa: BLE001
        figs = f" (no figure: {err})"
    print(f"\nwritten: {out}/REPORT.txt, {out}/numbers.json{figs}")


if __name__ == "__main__":
    main()
