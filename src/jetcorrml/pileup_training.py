"""
Shared NN pile-up correction training core.

This module contains the architecture, normaliser, feature builder,
covariance loss, stratified sampler, training loop, and apply helper used
by both the small-R (jagged awkward array) and large-R (flat parquet)
pile-up correction steps.  Each step's program owns its own
loader/writer because the data formats differ; the rest is identical.

The training loop minimises the higher-order covariance loss with the
per-|eta| and per-pT bin terms, Gram-Schmidt projection of mu against NPV and
the soft positivity penalty, and stops early on the physics metric.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from jetcorrml.networks import Normaliser


# ─────────────────────────────────────────────────────────────────────────────
# Feature engineering and normalisation
# ─────────────────────────────────────────────────────────────────────────────


def build_features(data):
    """
    Construct the 4 NN input features from a flat numpy data dict.

    Features
    --------
    f0 : log(pT_reco - rho*A)   log of area-subtracted reco pT
    f1 : eta_det                 signed detector eta
    f2 : NPV                     number of primary vertices
    f3 : mu                      mean interactions per crossing

    Using log(pT) rather than pT directly matches the logarithmic pT dependence
    of the correction observed in the ATLAS parametric fits.
    Using pT_reco - rho*A rather than raw pT_reco means the network input
    already has the dominant pile-up energy removed, leaving only the residual
    dependence for the network to learn.
    """
    pt_rhoA = data["pt_reco"] - data["rho"] * data["area_pt"]
    pt_rhoA = np.clip(pt_rhoA, 1.0, None)
    X = np.column_stack(
        [
            np.log(pt_rhoA),
            data["eta_det"],
            data["npv"],
            data["mu"],
        ]
    )
    return X.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Network definition
# ─────────────────────────────────────────────────────────────────────────────


class PileUpNet(nn.Module):
    """
    4-input MLP predicting the per-jet pile-up correction delta in GeV.

    ELU between hidden layers; no activation on the output, so the correction
    is unbounded in GeV.
    """

    def __init__(self, n_hidden, width):
        super().__init__()
        layers = [nn.Linear(4, width), nn.ELU()]
        for _ in range(n_hidden - 1):
            layers.append(nn.Linear(width, width))
            layers.append(nn.ELU())

        layers.append(nn.Linear(width, 1))
        self.net = nn.Sequential(*layers)

        nn.init.zeros_(self.net[-1].bias)
        nn.init.normal_(self.net[-1].weight, std=1e-3)

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ─────────────────────────────────────────────────────────────────────────────
# Loss function
# ─────────────────────────────────────────────────────────────────────────────


def full_loss(delta, residual, npv, mu, lam, abs_eta, eta_bins, abs_pt, pt_bins):
    """
    Higher-order covariance loss with L2 regularisation and Gram-Schmidt
    orthogonalisation of mu w.r.t. NPV.  See the small-R NN pile-up program (steps.pileup_nn)
    docstring for full detail; this implementation is identical.
    """

    def _cov(a, b):
        a = a - a.mean()
        b = b - b.mean()
        return (a * b).mean()

    npv_s = (npv - npv.mean()) / (npv.std() + 1e-6)
    mu_s = (mu - mu.mean()) / (mu.std() + 1e-6)

    npv_s_zm = npv_s - npv_s.mean()
    mu_s_zm = mu_s - mu_s.mean()
    proj = (mu_s_zm * npv_s_zm).mean() / ((npv_s_zm**2).mean() + 1e-10)
    mu_perp = mu_s - proj * npv_s

    c_npv = _cov(residual, npv_s)
    c_mu = _cov(residual, mu_perp)

    c_npv2 = _cov(residual, npv_s**2)
    c_mu2 = _cov(residual, mu_perp**2)

    c_int = _cov(residual, npv_s * mu_perp)

    positivity = torch.nn.functional.softplus(-delta, beta=1).mean()

    loss = c_npv**2 + c_mu**2 + c_npv2**2 + c_mu2**2 + c_int**2 + lam * positivity

    # the same linear terms in bins of |eta|
    eta_bins = eta_bins.to(residual.device)
    n_bins = eta_bins.shape[0] - 1

    in_range = (abs_eta >= eta_bins[0]) & (abs_eta < eta_bins[-1])
    abs_eta_ = abs_eta[in_range]
    residual_ = residual[in_range]
    npv_ = npv[in_range]
    mu_perp_ = mu_perp[in_range]

    bin_idx = (
        torch.searchsorted(eta_bins[1:], abs_eta_, right=False)
        .clamp(0, n_bins - 1)
        .long()
    )

    ones = torch.ones_like(residual_)
    counts = torch.zeros(n_bins, device=residual.device, dtype=residual.dtype)
    counts.scatter_add_(0, bin_idx, ones)

    sum_r = torch.zeros(n_bins, device=residual.device, dtype=residual.dtype)
    sum_npv = torch.zeros(n_bins, device=residual.device, dtype=residual.dtype)
    sum_mu = torch.zeros(n_bins, device=residual.device, dtype=residual.dtype)
    sum_r_npv = torch.zeros(n_bins, device=residual.device, dtype=residual.dtype)
    sum_r_mu = torch.zeros(n_bins, device=residual.device, dtype=residual.dtype)
    sum_npv2 = torch.zeros(n_bins, device=residual.device, dtype=residual.dtype)
    sum_mu2 = torch.zeros(n_bins, device=residual.device, dtype=residual.dtype)

    sum_r.scatter_add_(0, bin_idx, residual_)
    sum_npv.scatter_add_(0, bin_idx, npv_)
    sum_mu.scatter_add_(0, bin_idx, mu_perp_)
    sum_r_npv.scatter_add_(0, bin_idx, residual_ * npv_)
    sum_r_mu.scatter_add_(0, bin_idx, residual_ * mu_perp_)
    sum_npv2.scatter_add_(0, bin_idx, npv_ * npv_)
    sum_mu2.scatter_add_(0, bin_idx, mu_perp_ * mu_perp_)

    n = counts.clamp(min=1)
    n_unbiased = (counts - 1).clamp(min=1)
    mean_r = sum_r / n
    mean_npv = sum_npv / n
    mean_mu = sum_mu / n
    std_npv = (sum_npv2 / n - mean_npv**2).clamp(min=0) * (n / n_unbiased)
    std_npv = std_npv.sqrt() + 1e-6
    std_mu = (sum_mu2 / n - mean_mu**2).clamp(min=0) * (n / n_unbiased)
    std_mu = std_mu.sqrt() + 1e-6

    cov_r_npv = sum_r_npv / n - mean_r * mean_npv
    cov_r_mu = sum_r_mu / n - mean_r * mean_mu
    cov_npv_norm = cov_r_npv / std_npv
    cov_mu_norm = cov_r_mu / std_mu

    valid = counts >= 20
    n_valid = valid.float().sum().clamp(min=1.0)
    eta_weights = valid.float() * counts.sqrt()
    eta_weights = eta_weights / eta_weights.sum().clamp(min=1e-6) * n_valid
    loss = loss + (eta_weights * (cov_npv_norm**2 + cov_mu_norm**2)).sum()

    # and in bins of pT
    pt_bins = pt_bins.to(residual.device)
    n_bins = pt_bins.shape[0] - 1

    in_range = (abs_pt >= pt_bins[0]) & (abs_pt < pt_bins[-1])
    abs_pt_ = abs_pt[in_range]
    residual_ = residual[in_range]
    npv_ = npv[in_range]
    mu_perp_ = mu_perp[in_range]

    bin_idx = (
        torch.searchsorted(pt_bins[1:], abs_pt_, right=False)
        .clamp(0, n_bins - 1)
        .long()
    )

    ones = torch.ones_like(residual_)
    counts = torch.zeros(n_bins, device=residual.device, dtype=residual.dtype)
    counts.scatter_add_(0, bin_idx, ones)

    sum_r = torch.zeros(n_bins, device=residual.device, dtype=residual.dtype)
    sum_npv = torch.zeros(n_bins, device=residual.device, dtype=residual.dtype)
    sum_mu = torch.zeros(n_bins, device=residual.device, dtype=residual.dtype)
    sum_r_npv = torch.zeros(n_bins, device=residual.device, dtype=residual.dtype)
    sum_r_mu = torch.zeros(n_bins, device=residual.device, dtype=residual.dtype)
    sum_npv2 = torch.zeros(n_bins, device=residual.device, dtype=residual.dtype)
    sum_mu2 = torch.zeros(n_bins, device=residual.device, dtype=residual.dtype)

    sum_r.scatter_add_(0, bin_idx, residual_)
    sum_npv.scatter_add_(0, bin_idx, npv_)
    sum_mu.scatter_add_(0, bin_idx, mu_perp_)
    sum_r_npv.scatter_add_(0, bin_idx, residual_ * npv_)
    sum_r_mu.scatter_add_(0, bin_idx, residual_ * mu_perp_)
    sum_npv2.scatter_add_(0, bin_idx, npv_ * npv_)
    sum_mu2.scatter_add_(0, bin_idx, mu_perp_ * mu_perp_)

    n = counts.clamp(min=1)
    n_unbiased = (counts - 1).clamp(min=1)
    mean_r = sum_r / n
    mean_npv = sum_npv / n
    mean_mu = sum_mu / n
    std_npv = (sum_npv2 / n - mean_npv**2).clamp(min=0) * (n / n_unbiased)
    std_npv = std_npv.sqrt() + 1e-6
    std_mu = (sum_mu2 / n - mean_mu**2).clamp(min=0) * (n / n_unbiased)
    std_mu = std_mu.sqrt() + 1e-6

    cov_r_npv = sum_r_npv / n - mean_r * mean_npv
    cov_r_mu = sum_r_mu / n - mean_r * mean_mu
    cov_npv_norm = cov_r_npv / std_npv
    cov_mu_norm = cov_r_mu / std_mu

    valid = counts >= 20
    n_valid = valid.float().sum().clamp(min=1.0)
    pt_weights = valid.float() * counts.sqrt()
    pt_weights = pt_weights / pt_weights.sum().clamp(min=1e-6) * n_valid
    loss = loss + (pt_weights * (cov_npv_norm**2 + cov_mu_norm**2)).sum()

    return loss, c_npv.detach(), c_mu.detach()


# ─────────────────────────────────────────────────────────────────────────────
# Stratified sampler
# ─────────────────────────────────────────────────────────────────────────────


def make_stratified_sampler(npv, mu, n_npv_bins=10, n_mu_bins=10, min_count=1000):
    """
    Build a WeightedRandomSampler that samples uniformly over (NPV, mu) bins.

    Without stratification, low-pileup events dominate and the covariance
    signal for high-NPV / high-mu is weak, slowing convergence.

    Each jet receives weight = 1 / max(count of jets in its (NPV, mu) bin, min_count).

    min_count: a bin with fewer jets than this is weighted as if it had this
    many.  With min_count=1 every occupied bin gets
    the same share of each batch, however few jets it holds.  When N_PV and mu
    are strongly correlated the jets lie on a narrow diagonal and the bins at
    its edge are almost empty: at rho_muN = 1.0, 17 bins holding 0.1% of the
    jets made up 35% of every batch, a single jet being repeated ~85 times per
    batch.  Training then chased a few dozen jets and did not settle (hi-pT,
    rho_muN = 0.8 and 1.0).  Sparse bins are still boosted, up to the weight
    of a bin of min_count jets.
    """
    npv_edges = np.linspace(npv.min(), npv.max() + 1, n_npv_bins + 1)
    mu_edges = np.linspace(mu.min(), mu.max() + 1, n_mu_bins + 1)

    i_npv = np.clip(np.digitize(npv, npv_edges) - 1, 0, n_npv_bins - 1)
    i_mu = np.clip(np.digitize(mu, mu_edges) - 1, 0, n_mu_bins - 1)
    cell = i_npv * n_mu_bins + i_mu

    counts = np.bincount(cell, minlength=n_npv_bins * n_mu_bins).astype(float)
    n_small = int(((counts > 0) & (counts < min_count)).sum())
    print(f"  sampler: {int((counts > 0).sum())} occupied (N_PV, mu) bins; {n_small} with fewer "
          f"than {int(min_count)} jets are weighted as if they had {int(min_count)}")
    counts = np.maximum(counts, max(1, min_count))
    weights = 1.0 / counts[cell]

    return WeightedRandomSampler(
        weights=torch.from_numpy(weights).float(),
        num_samples=len(weights),
        replacement=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────


def train(
    data,
    epochs,
    batch_size,
    lr,
    lam,
    val_frac,
    seed,
    n_hidden,
    width,
    eta_bins,
    pt_bins,
    early_stop_patience,
    sampler_min_count,
):
    """
    Train PileUpNet on the provided data dict; returns (model, normaliser).

    Training stops early once |Cov(r, N_PV)| + |Cov(r, mu)| on the held-out jets
    has not improved for early_stop_patience epochs, counted from the first
    third of the budget on (the first restart of the learning-rate schedule).
    """
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Training on: {device}")

    X_raw = build_features(data)
    n = len(X_raw)

    residual_base = (
        data["pt_reco"] - data["pt_gen"] - data["rho"] * data["area_pt"]
    ).astype(np.float32)
    npv_arr = data["npv"].astype(np.float32)
    mu_arr = data["mu"].astype(np.float32)
    abs_eta_arr = np.abs(data["eta_det"]).astype(np.float32)

    pt_rhoA_arr = np.clip(
        (data["pt_reco"] - data["rho"] * data["area_pt"]).astype(np.float32),
        1.0,
        None,
    )

    eta_bins_t = torch.from_numpy(eta_bins.astype(np.float32)).to(device)
    pt_bins_t = torch.from_numpy(pt_bins.astype(np.float32)).to(device)

    idx = rng.permutation(n)
    n_val = int(n * val_frac)
    val_idx = idx[:n_val]
    trn_idx = idx[n_val:]

    norm = Normaliser().fit(X_raw[trn_idx])
    X_all = norm.transform(X_raw)

    def to_t(arr):
        return torch.from_numpy(arr).to(device)

    X_trn = to_t(X_all[trn_idx])
    r_trn = to_t(residual_base[trn_idx])
    npv_trn = to_t(npv_arr[trn_idx])
    mu_trn = to_t(mu_arr[trn_idx])
    eta_trn = to_t(abs_eta_arr[trn_idx])
    pt_rhoA_trn = to_t(pt_rhoA_arr[trn_idx])

    X_val = to_t(X_all[val_idx])
    r_val = to_t(residual_base[val_idx])
    npv_val = to_t(npv_arr[val_idx])
    mu_val = to_t(mu_arr[val_idx])
    eta_val = to_t(abs_eta_arr[val_idx])
    pt_rhoA_val = to_t(pt_rhoA_arr[val_idx])

    sampler = make_stratified_sampler(npv_arr[trn_idx], mu_arr[trn_idx],
                                      min_count=sampler_min_count)
    dataset = TensorDataset(X_trn, r_trn, npv_trn, mu_trn, eta_trn, pt_rhoA_trn)
    loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, drop_last=True)

    model = PileUpNet(n_hidden=n_hidden, width=width).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        opt, T_0=max(1, epochs // 3), T_mult=1, eta_min=lr * 1e-2
    )

    print(
        f"\n  {'Epoch':>5}  {'Loss':>10}  {'ValCov(r,NPV)':>13}  "
        f"{'ValCov(r,mu)':>12}  {'ValLoss':>10}  {'LR':>8}"
    )
    print("  " + "-" * 67)

    _best_phys = float("inf")
    _epochs_no_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        ep_loss_t = None
        n_batches = 0

        for X_b, r_b, npv_b, mu_b, eta_b, pt_rhoA_b in loader:
            opt.zero_grad()
            delta = model(X_b)
            resid = r_b - delta
            loss, _cn, _cm = full_loss(delta, resid, npv_b, mu_b, lam,
                                       eta_b, eta_bins_t, pt_rhoA_b, pt_bins_t)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            opt.step()

            ep_loss_t = (
                loss.detach() if ep_loss_t is None else ep_loss_t + loss.detach()
            )
            n_batches += 1

        ep_loss = ep_loss_t.item() / n_batches

        sched.step()

        model.eval()
        with torch.no_grad():
            delta_v = model(X_val)
            resid_v = r_val - delta_v
            vl, vcn, vcm = full_loss(delta_v, resid_v, npv_val, mu_val, lam,
                                     eta_val, eta_bins_t, pt_rhoA_val, pt_bins_t)

        lr_now = sched.get_last_lr()[0]
        if epoch % max(1, epochs // 20) == 0 or epoch == 1:
            print(
                f"  {epoch:>5}  {ep_loss:>10.4f}  {vcn.item():>+13.4f}  "
                f"{vcm.item():>+12.4f}  {vl.item():>10.4f}  {lr_now:>8.2e}"
            )

        warmup = max(1, epochs // 3)
        if epoch >= warmup:
            phys = abs(vcn.item()) + abs(vcm.item())
            if phys < _best_phys:
                _best_phys = phys
                _epochs_no_improve = 0
            else:
                _epochs_no_improve += 1
            if _epochs_no_improve >= early_stop_patience:
                print(
                    f"\n  Early stop at epoch {epoch}: |cov_npv|+|cov_mu| has "
                    f"not improved for {early_stop_patience} consecutive epochs "
                    f"(best={_best_phys:.4f})"
                )
                break

    model.eval()
    return model.cpu(), norm


# ─────────────────────────────────────────────────────────────────────────────
# Apply correction
# ─────────────────────────────────────────────────────────────────────────────


def apply_nn_correction(data, model, norm):
    """
    Apply the trained network correction to all jets in the flat data dict.

    Returns
    -------
    delta   : (N,) NN per-jet correction in GeV
    pt_corr : (N,) area-subtracted corrected pT (pt_reco - rho*A - delta)
    """
    X_raw = build_features(data)
    X = norm.transform(X_raw)
    X_t = torch.from_numpy(X.astype(np.float32))

    pt_rhoA = np.clip(
        (data["pt_reco"] - data["rho"] * data["area_pt"]).astype(np.float32),
        1.0,
        None,
    )

    model.eval()
    with torch.no_grad():
        delta = model(X_t).numpy()

    pt_corr = pt_rhoA - delta
    return delta, pt_corr
