"""
The nuisance-parameter band of the NN in-situ correction (Sec. VIII), and its check
against the scatter over divisions of the sample (Fig. 19).

The band comes from retrainings on resampled events, all on ONE division of the
sample into simulation and data (division 0): each writes its scale on a fixed grid
(steps.band runs them).  `decompose` takes the principal components of
the bundle,
    s(x, nu) = z0(x) * exp( sum_i nu_i sqrt(lam_i) phi_i(x) ),
and the closure figures draw the band from the retrainings.

The check (Fig. 19) compares the band with something observed directly: the in-situ
step repeated on other divisions of the same sample.  In each (|eta|, pT) region it
compares the spread of the divisions with that of the retrainings, each jet measured
against its own pT before the distortion (`regions`); every retraining corrects the
same data jets, so only the correction varies.
"""
import dataclasses
import os
import warnings

import awkward as ak
import numpy as np

from jetcorrml import paths, settings
from jetcorrml.jets import flat, per_jet
from jetcorrml.distortion import residual_distortion

# the distortion of each data half: form, d0 ... d3
DIST = {name: dataclasses.asdict(d) for name, d in settings.DISTORTIONS.items()}


# ─────────────────────────── the band ───────────────────────────
def jets_of(path):
    """(pt, |eta|, pt_gen, R = pt/pt_gen) of the jets of an in-situ output with a finite R."""
    m = ak.from_parquet(path)
    pt = flat(m.jet.pt); eta = np.abs(flat(m.jet.eta)); gen = flat(m.gen.pt)
    ok = np.isfinite(pt) & np.isfinite(gen) & (gen > 0)
    return pt[ok], eta[ok], gen[ok], (pt / gen)[ok]


def closure_vs_eta(corr_data_pq, mc_pq, eedges):
    """double ratio med(R_corr_data)/med(R_mc) per |eta| bin (all pt)."""
    ptc, etac, genc, Rc = jets_of(corr_data_pq)        # etac is |eta|
    ptm, etam, genm, Rm = jets_of(mc_pq)
    nb = len(eedges) - 1
    out = np.full(nb, np.nan)
    bd = np.clip(np.digitize(etac, eedges) - 1, 0, nb - 1)
    bm = np.clip(np.digitize(etam, eedges) - 1, 0, nb - 1)
    for k in range(nb):
        dd = Rc[bd == k]; mm = Rm[bm == k]
        if dd.size >= 50 and mm.size >= 50:
            out[k] = np.median(dd) / np.median(mm)
    return out


def closure_vs_gen(corr_data_pq, mc_pq, gedges):
    """double ratio med(R_corr_data)/med(R_mc) per gen-pt bin."""
    ptc, etac, genc, Rc = jets_of(corr_data_pq)        # corrected data (jet.pt already corrected)
    ptm, etam, genm, Rm = jets_of(mc_pq)
    nb = len(gedges) - 1
    out = np.full(nb, np.nan)
    bd = np.clip(np.digitize(genc, gedges) - 1, 0, nb - 1)
    bm = np.clip(np.digitize(genm, gedges) - 1, 0, nb - 1)
    for k in range(nb):
        dd = Rc[bd == k]; mm = Rm[bm == k]
        if dd.size >= 50 and mm.size >= 50:
            out[k] = np.median(dd) / np.median(mm)
    return out


def residual_model(lam, phi, C, n):
    """The band reduced to its n leading components plus ONE residual nuisance parameter.

    The components after the n-th are combined, point by point, into a single shape
        sigma_r(x) = sqrt( C(x,x) - sum_{i<=n} lambda_i phi_i(x)^2 )
    taken as one parameter (fully correlated over the grid), so that
        s(x, nu) = z0(x) * exp( sum_{i<=n} nu_i sqrt(lambda_i) phi_i(x) + nu_r sigma_r(x) )
    has, at every grid point, exactly the variance of the full band. What is given up is the
    correlation of the remainder between points. Returns (shapes [n, P], sigma_r [P], C of the model)."""
    n = int(min(n, len(lam)))
    shapes = (np.sqrt(np.clip(lam[:n], 0, None))[:, None] * phi[:, :n].T)          # [n, P]
    lead = np.sum(shapes ** 2, axis=0)
    sigma_r = np.sqrt(np.clip(np.diag(C) - lead, 0, None))
    C_model = shapes.T @ shapes + np.outer(sigma_r, sigma_r)
    return shapes, sigma_r, C_model


def decompose(sidecars):
    """The principal-component decomposition of the bundle.  Returns pt, eta (the grid),
    z0 [P], lam (descending), phi [P, P], C [P, P], B; P = npt * neta, the flattened grid
    (pt-major)."""
    grids = [np.load(s) for s in sidecars]
    pt = grids[0]["pt"]; eta = grids[0]["eta"]
    S = np.stack([g["s"] for g in grids]); B, npt, neta = S.shape
    flat = S.reshape(B, -1)
    z0 = np.exp(np.mean(np.log(np.clip(flat, 1e-6, None)), axis=0))
    rho = np.log(np.clip(flat, 1e-6, None)) - np.log(z0); rho -= rho.mean(0, keepdims=True)
    C = (rho.T @ rho) / (B - 1)
    lam, phi = np.linalg.eigh(C); o = np.argsort(lam)[::-1]
    lam, phi = np.clip(lam[o], 0, None), phi[:, o]
    return pt, eta, z0, lam, phi, C, B


# ─────────────────────────── the check against the divisions ───────────────────────────
ETA_ROWS = [(0.0, 0.8), (0.8, 1.5), (1.5, 2.0), (2.0, 2.5)]
PT_COLS = [(20, 30), (30, 45), (45, 70), (70, 120)]
MIN_BIN, MIN_JETS = 50, 500


def undistorted_pt(pt_d, abs_eta, rho, p):
    """The pT of each data jet before its distortion p (a DIST entry): solve pT * f(pT, |eta|, rho) = pT_data."""
    kw = dict(d3=p["d3"], form=p["form"])
    pt = pt_d / residual_distortion(np.full_like(pt_d, 45.0), abs_eta, rho, p["d0"], p["d1"], p["d2"], **kw)
    for _ in range(200):
        f = residual_distortion(np.clip(pt, 1e-3, None), abs_eta, rho, p["d0"], p["d1"], p["d2"], **kw)
        new = pt_d / np.where(np.abs(f) > 1e-6, f, np.nan)
        pt = np.where(np.isfinite(new), 0.5 * (pt + new), pt)
    return pt


def base(ds, d, s):
    """the data jets of one division, with their pT before the distortion"""
    da = ak.from_parquet(paths.division(ds, d, s, "DATA"))
    rho = per_jet(da, "rho")
    eta = np.abs(flat(da.jet.eta))
    return dict(eta=eta, pt0=undistorted_pt(flat(da.jet.pt), eta, rho, DIST[d]))


def regions(b, ptc):
    """median of corrected pT over the pT before the distortion, minus 1, in %, per (|eta|, pT) region"""
    m = np.full((len(ETA_ROWS), len(PT_COLS)), np.nan)
    n = np.zeros_like(m)
    r = ptc / b["pt0"]
    for i, (elo, ehi) in enumerate(ETA_ROWS):
        for j, (plo, phi) in enumerate(PT_COLS):
            k = (b["eta"] >= elo) & (b["eta"] < ehi) & (b["pt0"] >= plo) & (b["pt0"] < phi)
            n[i, j] = k.sum()
            if k.sum() >= MIN_BIN:
                m[i, j] = 100 * (np.median(r[k]) - 1)
    return m, n


def one_division(job):
    ds, d, s = job
    if not all(os.path.exists(p) for p in (paths.division(ds, d, s, "DATA"), paths.insitu(ds, d, s, "nn", "DATA"))):
        return None
    ptc = flat(ak.from_parquet(paths.insitu(ds, d, s, "nn", "DATA")).jet.pt)
    m, n = regions(base(ds, d, s), ptc)
    return dict(s=s, map=m, n=n)


def all_refits(job):
    """every retraining of one distortion (they share the test sample of division 0)"""
    ds, d, nboot = job
    if not os.path.exists(paths.division(ds, d, 0, "DATA")):
        return []
    b = base(ds, d, 0)
    out = []
    for i in range(nboot):
        p = paths.retraining(ds, d, i, "DATA")
        if not os.path.exists(p):
            continue
        m, _ = regions(b, flat(ak.from_parquet(p).jet.pt))
        out.append(dict(b=i, map=m))
    return out


def sd(x):
    """spread over the first axis and its own statistical error (Gaussian estimate)"""
    x = np.asarray(x, dtype=float)
    n = np.sum(np.isfinite(x), axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s = np.nanstd(x, axis=0, ddof=1)
    return s, s / np.sqrt(2.0 * np.maximum(n - 1, 1))


def region_numbers(D, R):
    S = np.array([x["map"] for x in D])
    N = np.array([x["n"] for x in D]).mean(axis=0)
    B = np.array([x["map"] for x in R])
    ok = (N >= MIN_JETS) & (np.sum(np.isfinite(S), axis=0) >= 0.75 * len(S)) & (np.sum(np.isfinite(B), axis=0) >= 0.75 * len(B))
    scat, band = sd(S)[0], sd(B)[0]
    mean = np.nanmean(S, axis=0)
    inside = np.nanmean((np.abs(S - mean) <= band)[:, ok])
    return dict(ok=ok, scat=scat, band=band, mean=mean, n=N, inside=float(inside))
