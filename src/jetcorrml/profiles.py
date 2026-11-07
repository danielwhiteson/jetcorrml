"""
Binned profiles of one variable against another, as the paper's figures draw them.

They differ in how the bins are chosen and which bins are kept, and each figure
uses the one its numbers were made with.
"""
import numpy as np


def profile(x, y, n_bins=30, x_range=None):
    """Mean of y and its standard error in n_bins equal bins of x over x_range (default: the
    range of x); bins with 5 jets or fewer are left out.  Returns (centres, means, errors)."""
    x = np.asarray(x)
    y = np.asarray(y)
    lo = x.min() if x_range is None else x_range[0]
    hi = x.max() if x_range is None else x_range[1]
    edges = np.linspace(lo, hi, n_bins + 1)
    centres, means, stds = [], [], []
    for i in range(n_bins):
        mask = (x >= edges[i]) & (x < edges[i + 1])
        if mask.sum() > 5:
            centres.append(0.5 * (edges[i] + edges[i + 1]))
            means.append(y[mask].mean())
            stds.append(y[mask].std() / np.sqrt(mask.sum()))
    return np.array(centres), np.array(means), np.array(stds)


def profile_xy(x, y, n_bins=30, x_range=None, x_log=False):
    """Median profile of y against x in n_bins bins (equal in log x with x_log), bins with 10
    jets or fewer left out.  Returns (centres, median, its error 1.253 std/sqrt(N), std)."""
    x = np.asarray(x)
    y = np.asarray(y)
    if len(x) == 0:
        return np.array([]), np.array([]), np.array([]), np.array([])
    lo = x.min() if x_range is None else x_range[0]
    hi = x.max() if x_range is None else x_range[1]
    if x_log:
        lo = max(lo, 1.0)
        edges = np.geomspace(lo, hi, n_bins + 1)
    else:
        edges = np.linspace(lo, hi, n_bins + 1)
    cx, my, sy, stdy = [], [], [], []
    for i in range(n_bins):
        m = (x >= edges[i]) & (x < edges[i + 1])
        if m.sum() > 10:
            c = (
                np.sqrt(edges[i] * edges[i + 1])
                if x_log
                else 0.5 * (edges[i] + edges[i + 1])
            )
            cx.append(c)
            s = y[m].std()
            stdy.append(s)
            my.append(np.median(y[m]))
            sy.append(1.253 * s / np.sqrt(m.sum()))
    return np.array(cx), np.array(my), np.array(sy), np.array(stdy)


def profile_inclusive(x, y, n_bins=30, x_range=None, min_count=5):
    """As profile, over the full finite range of x with the largest value in the top bin,
    keeping bins with at least min_count jets (the bounded pile-up observables, whose
    highest bins are wanted)."""
    if x_range is None:
        x_range = (np.nanmin(x), np.nanmax(x))
    edges = np.linspace(x_range[0], x_range[1], n_bins + 1)
    edges[-1] = np.nextafter(edges[-1], np.inf)  # make top bin inclusive of max
    xc, ym, ye = [], [], []
    for i in range(n_bins):
        sel = (x >= edges[i]) & (x < edges[i + 1])
        k = int(sel.sum())
        if k >= min_count:
            yy = y[sel]
            xc.append(0.5 * (edges[i] + edges[i + 1]))
            ym.append(yy.mean())
            ye.append(yy.std() / np.sqrt(k))
    return np.array(xc), np.array(ym), np.array(ye)


def binned_mean(x, y, edges, min_count):
    """Mean of y and its standard error in the bins `edges` of x, keeping bins with at least
    min_count jets.  Returns (centres, means, errors)."""
    xc, m, e = [], [], []
    for i in range(len(edges) - 1):
        sel = (x >= edges[i]) & (x < edges[i + 1])
        n = int(sel.sum())
        if n >= min_count:
            yy = y[sel]
            xc.append(0.5 * (edges[i] + edges[i + 1]))
            m.append(yy.mean())
            e.append(yy.std() / np.sqrt(n))
    return np.array(xc), np.array(m), np.array(e)
