"""Loss terms of the NN in-situ correction (steps.insitu_nn, paper Sec. VII C).

The trained objective is the sum of two quantile-matching terms:
  * response_quantile_match_loss      -- the reference term: the central response
    (|eta| < 0.8) of corrected data matched to the simulation, 19 quantiles per
    (|eta|, pT_ref) bin;
  * dijet_forward_balance_median_loss -- the balance term: the median dijet
    log-balance of corrected data matched to the simulation per
    (|eta1|, |eta2|, pT_avg) cell, which relays the central scale outward.
"""

import torch

# The quantile ladder of the reference term: 19 levels, 5%..95%.
RESPONSE_QM_LEVELS = torch.linspace(0.05, 0.95, 19)

MIN_COUNT = 50   # bins with fewer data or simulation entries are left out


def response_quantile_match_loss(r_corr, r_mc, levels, *, data_bin, mc_bin, n_bins):
    """Mean squared difference between the corrected-data and simulation
    quantiles at ``levels``, per bin, summed over bins with weight sqrt(N_data).

    ``r_corr`` carries the gradient; the simulation quantiles are a fixed
    target.  ``data_bin``, ``mc_bin`` are per-entry bin ids in [0, n_bins)
    (the same edges for both).  Bins with fewer than MIN_COUNT data or
    simulation entries are skipped.  Returns a scalar tensor (graph-connected
    even when every bin is skipped)."""
    levels = levels.to(device=r_corr.device, dtype=r_corr.dtype)

    def _bin_loss(rc, rm):
        if rc.numel() < MIN_COUNT or rm.numel() < MIN_COUNT:
            return None
        qc = torch.quantile(rc, levels)
        qm = torch.quantile(rm, levels).detach()
        return torch.mean((qc - qm) ** 2)

    total = r_corr.sum() * 0.0  # graph-connected zero
    wsum = 0.0
    for b in range(int(n_bins)):
        rc = r_corr[data_bin == b]
        rm = r_mc[mc_bin == b]
        bl = _bin_loss(rc, rm)
        if bl is None:
            continue
        w = float(rc.numel()) ** 0.5
        total = total + w * bl
        wsum += w
    return total / wsum if wsum > 0 else total


def make_response_bins(abs_eta, gen_pt, eta_edges, pt_edges):
    """Bin id over (|eta|, pT_ref) for the reference term.

    ``eta_edges`` / ``pt_edges`` are interior edges (torch.bucketize); the
    number of bins is (len(eta_edges)+1) * (len(pt_edges)+1).
    Returns (bin_id LongTensor, n_bins int)."""
    eta_edges = eta_edges.to(device=abs_eta.device, dtype=abs_eta.dtype)
    pt_edges = pt_edges.to(device=gen_pt.device, dtype=gen_pt.dtype)
    ei = torch.bucketize(abs_eta.contiguous(), eta_edges)
    pi = torch.bucketize(gen_pt.contiguous(), pt_edges)
    n_pt = int(pt_edges.numel()) + 1
    n_eta = int(eta_edges.numel()) + 1
    return (ei * n_pt + pi).long(), n_eta * n_pt


def make_eta_pair_pt_bins(abs_eta1, abs_eta2, pt_avg, eta_edges, pt_avg_edges):
    """Cell id over (|eta1|, |eta2|, pT_avg) for the balance term.

    One interior |eta| edge set serves both jets (jet1 the inner one, so only
    the i <= j half fills).  Central jets fall in |eta| bin 0, where s_rel == 1,
    so any pair touching it anchors the chain and forward+forward pairs relay
    the anchor outward.  Returns (bin_id LongTensor, n_bins int) with
    n_bins = n_eta * n_eta * n_pt."""
    eta_edges = eta_edges.to(device=abs_eta1.device, dtype=abs_eta1.dtype)
    i = torch.bucketize(abs_eta1.contiguous(), eta_edges)
    j = torch.bucketize(abs_eta2.contiguous(), eta_edges)
    n_eta = int(eta_edges.numel()) + 1
    pt_avg_edges = pt_avg_edges.to(device=pt_avg.device, dtype=pt_avg.dtype)
    pi = torch.bucketize(pt_avg.contiguous(), pt_avg_edges)
    n_pt = int(pt_avg_edges.numel()) + 1
    return (i * n_eta * n_pt + j * n_pt + pi).long(), n_eta * n_eta * n_pt


def dijet_forward_balance_median_loss(logbal_corr, logbal_mc, *, fwd_bin, mc_fwd_bin, n_bins):
    """Match the median log-balance beta = log(pT2*s2) - log(pT1*s1) of
    corrected data to the simulation's (log(pT2) - log(pT1)) per cell.

    The log-balance, not the asymmetry, so nothing is divided by a quantity
    that crosses zero; the caller floors the pT*s products before the log.
    The single-level (q = 0.5) case of response_quantile_match_loss, so the
    median match, the fixed simulation target, the sqrt(N) weights and the
    MIN_COUNT skip are the reference term's."""
    median_level = torch.tensor([0.5], device=logbal_corr.device, dtype=logbal_corr.dtype)
    return response_quantile_match_loss(
        logbal_corr,
        logbal_mc,
        median_level,
        data_bin=fwd_bin,
        mc_bin=mc_fwd_bin,
        n_bins=n_bins,
    )
