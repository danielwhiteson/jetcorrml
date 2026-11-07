"""
NN absolute energy scale (Sec. V C of the paper).

The network learns the median jet energy response R = E_reco / E_true as a
continuous function of the true energy and |eta_det| (a quantile loss at
tau = 0.5), and the correction is obtained by the numerical inversion of the
parametric method: for each |eta|, R is evaluated on a grid of true energies,
each point is carried to the reconstructed energy it implies,
E_reco = E_true * R, and the calibration factor 1/R is tabulated against it.
Settings: jetcorrml.settings (absolute_nn, ABSOLUTE_ETA_EDGES).

    uv run python -m steps.absolute_nn \
        -i data/pileup/nn/lo_rho01.pq --output data/absolute/nn/lo_rho01.pq [--mini]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from jetcorrml import settings
from jetcorrml.absolute_closure import closure_floor
from jetcorrml.jets import absolute_columns, pt_at_fixed_mass, read_jets, write_jets
from jetcorrml.networks import Normaliser, ResponseNet, flat_sampling_weights

DR_MAX = settings.MATCH_DR          # matching distance: every pair as written
EBUFFER = settings.ABSOLUTE_E_BUFFER

# The response fit uses jets with E_gen >= max(TRAIN_FLOOR, EBUFFER * 20 * cosh(eta)); at
# application, below that floor the response is held at its value at the floor.  The
# closure metric and the global normalisation keep their own 50 GeV floor.
TRAIN_FLOOR = 40.0

# Blow-up guard of the response training (see train_response_nn): an epoch whose
# validation loss ends above BLOWUP_FACTOR x the best so far is undone.  Normal
# epoch-to-epoch changes are below 10%; the blow-ups seen were 6-9x.
BLOWUP_FACTOR = 2.0
MAX_ROLLBACKS = 10

# |eta| range of the inversion tables, their slice width, and their grid of true energies
ETA_EDGES = np.array(settings.ABSOLUTE_ETA_EDGES)
FINE_ETA_STEP = 0.05
APPLY_N_GRID = 500
APPLY_RANGE_FACTOR = 1.5

N_RESPONSE_FEATURES = 6  # keep in sync with build_response_features


def build_response_features(log_e, eta):
    """
    The feature vector of the response NN (numpy, float32):
      0: log(E)         — primary energy scale
      1: |η|            — absolute pseudorapidity (response is η-symmetric)
      2: 1/E            — captures the ~1/E threshold/noise term at low E
      3: 1/E²           — captures higher-order low-E effects
      4: |η|²           — the degradation in the forward region
      5: log(E) * |η|   — energy-eta interaction term

    The last four are functions of the first two, supplied to aid convergence.
    """
    E = np.exp(log_e)
    abs_eta = np.abs(eta)
    inv_e = 1.0 / np.maximum(E, 1.0)
    cols = [
        log_e,
        abs_eta,
        inv_e,
        inv_e**2,
        abs_eta**2,
        log_e * abs_eta,
    ]
    return np.column_stack(cols).astype(np.float32)


def train_response_nn(
    data,
    epochs=20,
    batch_size=4096,
    lr=1e-3,
    val_frac=0.15,
    seed=42,
    n_hidden=5,
    width=192,
    dropout=0.05,
):
    """
    Train a NN to predict the *median* response R = E_reco / E_true
    as a function of (log E_true, eta_det).

    Key design choices
    ------------------
    1. **Quantile (pinball) loss for the median (τ = 0.5):**
       The response distribution E_reco/E_true is asymmetric with a
       left-side tail from out-of-cone losses, dead material, neutrinos,
       etc.  The MSE loss learns the conditional *mean*, which is pulled
       low by these tails.  But the ATLAS calibration targets the
       *Gaussian core peak* (≈ median), not the mean, because the tails
       represent irrecoverable losses that the calibration shouldn't try
       to compensate for.

       The pinball loss  L = mean[ ρ_τ(R_obs - R_pred) ]  where
       ρ_τ(u) = u*(τ - 1{u<0})  directly targets the conditional median
       when τ = 0.5.  It's naturally robust to tails — a single outlier
       at R = 0.01 doesn't drag the prediction the way MSE does.

    2. **Stratified sampling** with sqrt-flattening of the (log E, |eta|)
       distribution.  Boosts high-E jets without over-amplifying
       low-statistics outlier-heavy cells.

    3. **Hard quality cuts only.**  No soft downweighting — the quantile
       loss already handles tails gracefully.

    Returns
    -------
    model   : trained ResponseNet (cpu, eval mode)
    norm    : fitted Normaliser (for the *truth-space* features)
    history : dict with per-epoch training metrics
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    # ── Build truth-space features: (log E_true, eta_det) ─────────────
    e_gen = data["e_gen"].astype(np.float32)
    e_reco = data["e_reco"].astype(np.float32)
    eta = data["eta_det"].astype(np.float32)
    dr = data["dr"].astype(np.float32)

    # Target: per-jet response
    response = (e_reco / np.maximum(e_gen, 1.0)).astype(np.float32)

    # ── Hard quality cut ──────────────────────────────────────────────
    # Keep E_reco threshold low (just the existing reco pT > 20 cut)
    # so that downward-fluctuating jets are visible.  But require
    # E_gen > E_GEN_MIN so that even jets with R ~ 0.5 still have
    # E_reco ~ E_GEN_MIN * 0.5 above the reco threshold, giving an
    # unbiased response distribution in each E_true bin.
    #    E_GEN_MIN = 50.0
    E_GEN_MIN = closure_floor(eta, e_min=TRAIN_FLOOR)

    good = (response > 0.05) & (response < 3.0) & (e_gen >= E_GEN_MIN) & (dr < DR_MAX)
    print(
        f"  Quality cut (E_gen >= {E_GEN_MIN}): keeping {good.sum():,} / "
        f"{len(good):,} jets ({100 * good.mean():.1f}%)"
    )

    e_gen = e_gen[good]
    e_reco = e_reco[good]
    eta = eta[good]
    response = response[good]

    log_e_gen = np.log(np.clip(e_gen, 1.0, None))
    X_raw = build_response_features(log_e_gen, eta)

    norm = Normaliser()
    X = norm.fit_transform(X_raw).astype(np.float32)

    # ── Train / val split ─────────────────────────────────────────────
    n = len(e_gen)
    idx = np.random.permutation(n)
    n_val = int(n * val_frac)
    val_idx, trn_idx = idx[:n_val], idx[n_val:]

    def to_t(arr):
        return torch.from_numpy(arr).to(device)

    X_trn, X_val = to_t(X[trn_idx]), to_t(X[val_idx])
    R_trn, R_val = to_t(response[trn_idx]), to_t(response[val_idx])

    # ── Stratified sampling ───────────────────────────────────────────
    print("  Computing stratified sampling weights...")
    samp_probs = flat_sampling_weights(
        log_e_gen[trn_idx], np.abs(eta[trn_idx])
    )
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=samp_probs,
        num_samples=len(trn_idx),
        replacement=True,
    )

    trn_ds = torch.utils.data.TensorDataset(X_trn, R_trn)
    trn_loader = torch.utils.data.DataLoader(
        trn_ds, batch_size=batch_size, sampler=sampler, drop_last=True
    )

    # ── Model + optimiser ─────────────────────────────────────────────
    n_features = N_RESPONSE_FEATURES
    model = ResponseNet(
        n_features=n_features, n_hidden=n_hidden, width=width, dropout=dropout
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-3)

    # OneCycleLR with moderate peak.  The previous version peaked at
    # 3×lr which let the large network memorize noise.  1.5× is enough
    # to explore without overfitting.
    steps_per_epoch = max(len(trn_idx) // batch_size, 1)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt,
        max_lr=lr * 1.5,  # moderate peak
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.1,
        anneal_strategy="cos",
        div_factor=1.5,
        final_div_factor=20.0,
    )

    history = {"loss": [], "val_loss": [], "mean_resp": [], "val_mean_resp": []}

    TAU = 0.5  # median quantile

    # Guard against a blow-up.  Now and then the loss jumps several-fold within
    # one epoch near the learning-rate peak.  Usually it comes back within ten
    # epochs; if the output has been driven to zero it never does, because the
    # softplus output is flat there (lo-pT rho_muN = 0.8 in one run: loss
    # 0.050 -> 0.50, <R_pred> = 0 for the remaining 90 epochs, unnoticed).
    # After every epoch that ends normally the weights and optimiser state are
    # kept; an epoch that ends with the validation loss above BLOWUP_FACTOR
    # times the best so far is undone by restoring them, and training goes on
    # with the next batches and the scheduled learning rate.  A training without
    # such an epoch is not altered in any way.
    import copy as _copy

    _good = None            # (model state, optimiser state) after the last normal epoch
    _best_vl = float("inf")
    _n_rollback = 0

    for epoch in range(1, epochs + 1):
        model.train()
        for X_b, R_b in trn_loader:
            R_pred = model(X_b).squeeze(-1)
            # Pinball loss for median (τ = 0.5)
            # ρ_τ(u) = u * (τ - 1{u < 0})
            # For τ=0.5 this simplifies to 0.5 * |u|, but we keep
            # the general form so TAU can be tuned if desired.
            residual = R_b - R_pred
            loss = torch.where(
                residual >= 0,
                TAU * residual,
                (TAU - 1.0) * residual,
            ).mean()

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()  # OneCycleLR steps per batch, not per epoch

        # ── Epoch metrics ─────────────────────────────────────────────
        model.eval()
        with torch.no_grad():
            Rp_trn = model(X_trn).squeeze(-1)
            Rp_val = model(X_val).squeeze(-1)
            res_trn = R_trn - Rp_trn
            res_val = R_val - Rp_val
            tl = (
                torch.where(res_trn >= 0, TAU * res_trn, (TAU - 1.0) * res_trn)
                .mean()
                .item()
            )
            vl = (
                torch.where(res_val >= 0, TAU * res_val, (TAU - 1.0) * res_val)
                .mean()
                .item()
            )
            _blown = (_good is not None) and not (vl <= BLOWUP_FACTOR * _best_vl)   # also catches NaN
            if _blown:
                _n_rollback += 1
                print(
                    f"  Epoch {epoch:3d}/{epochs}  BLOW-UP: val_loss={vl:.6f} "
                    f"(best so far {_best_vl:.6f}), <R_pred>={Rp_val.mean().item():.4f} "
                    f"-> weights of the previous good epoch restored ({_n_rollback} so far)"
                )
                if _n_rollback > MAX_ROLLBACKS:
                    raise RuntimeError(
                        f"absolute-scale training blew up {_n_rollback} times; giving up "
                        f"(dataset unusable with these settings)"
                    )
                model.load_state_dict(_good[0])
                opt.load_state_dict(_good[1])
                Rp_trn = model(X_trn).squeeze(-1)
                Rp_val = model(X_val).squeeze(-1)
                res_trn = R_trn - Rp_trn
                res_val = R_val - Rp_val
                tl = torch.where(res_trn >= 0, TAU * res_trn, (TAU - 1.0) * res_trn).mean().item()
                vl = torch.where(res_val >= 0, TAU * res_val, (TAU - 1.0) * res_val).mean().item()
            else:
                _good = (_copy.deepcopy(model.state_dict()), _copy.deepcopy(opt.state_dict()))
                _best_vl = min(_best_vl, vl)
            history["loss"].append(tl)
            history["val_loss"].append(vl)
            history["mean_resp"].append(Rp_trn.mean().item())
            history["val_mean_resp"].append(Rp_val.mean().item())

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"  Epoch {epoch:3d}/{epochs}  "
                f"loss={tl:.6f}  val_loss={vl:.6f}  "
                f"<R_pred>={Rp_val.mean().item():.4f}"
            )

    if _n_rollback:
        print(f"  {_n_rollback} blown-up epoch(s) undone during training")
    # Last line of defence: never hand on a dead network.
    _final_R = history["val_mean_resp"][-1]
    if not (history["val_loss"][-1] <= BLOWUP_FACTOR * _best_vl) or not (0.2 < _final_R < 5.0):
        raise RuntimeError(
            f"absolute-scale training ended in a bad state: val_loss={history['val_loss'][-1]:.6f} "
            f"(best {_best_vl:.6f}), <R_pred>={_final_R:.4f}"
        )

    model.eval()
    return model.cpu(), norm, history


def apply_calibration(data, model, norm):
    """
    Apply the numerically inverted calibration to all jets.

    The inversion tables are built from the network at the centre of each
    |eta| slice of width FINE_ETA_STEP, on a grid of APPLY_N_GRID true energies
    spanning the particle-level energies of the data (up to APPLY_RANGE_FACTOR
    times the highest).  Below the training floor the response is held at its
    value at the floor, so the network is not evaluated where it was not trained.
    A Savitzky-Golay filter removes small wiggles before the inversion.

    Then a global scale factor, the inverse of the median corrected response of
    the jets above 50 GeV, is applied to all jets.
    """
    from scipy.signal import savgol_filter

    e_reco = data["e_reco"]
    abs_eta = np.abs(data["eta_det"])
    scale = np.ones_like(e_reco)

    fine_edges = np.arange(
        ETA_EDGES[0], ETA_EDGES[-1] + FINE_ETA_STEP * 0.5, FINE_ETA_STEP
    )
    _eg = np.asarray(data["e_gen"], dtype=float)
    _eg = _eg[np.isfinite(_eg) & (_eg > 0)]
    e_lo = float(np.min(_eg))
    e_hi = float(np.max(_eg))
    n_grid = APPLY_N_GRID

    model.eval()

    for j in range(len(fine_edges) - 1):
        eta_c = 0.5 * (fine_edges[j] + fine_edges[j + 1])
        mask = (abs_eta >= fine_edges[j]) & (abs_eta < fine_edges[j + 1])
        if mask.sum() == 0:
            continue

        log_e_grid = np.linspace(
            np.log(max(e_lo, 5.0)), np.log(e_hi * APPLY_RANGE_FACTOR), n_grid
        )
        e_true_grid = np.exp(log_e_grid)
        X_grid = build_response_features(log_e_grid, np.full(n_grid, eta_c))
        X_grid_norm = norm.transform(X_grid)

        with torch.no_grad():
            R_grid = (
                model(torch.from_numpy(X_grid_norm.astype(np.float32)))
                .numpy()
                .ravel()
            )

        # Hold the response at its value at the training floor for all
        # energies below it: the network is not used where it was not trained.
        _floor = max(TRAIN_FLOOR, EBUFFER * 20.0 * np.cosh(eta_c))
        _k = min(int(np.searchsorted(e_true_grid, _floor)), n_grid - 1)
        R_grid = R_grid.copy()
        R_grid[:_k] = R_grid[_k]

        # Light smoothing
        sg_w = max(int(n_grid * 0.05) | 1, 5)              # odd, and at least 5
        R_grid = savgol_filter(R_grid, window_length=sg_w, polyorder=3)
        R_grid = np.clip(R_grid, 0.01, None)

        # Inversion
        reco_grid = e_true_grid * R_grid
        scale_grid = 1.0 / R_grid
        reco_grid = np.maximum.accumulate(reco_grid)
        umask = np.diff(reco_grid, prepend=-np.inf) > 0
        rg, sg = reco_grid[umask], scale_grid[umask]

        scale[mask] = np.interp(e_reco[mask], rg, sg, left=sg[0], right=sg[-1])

    # ── Global scale correction ───────────────────────────────────────
    e_corr_test = e_reco * scale
    e_gen = data["e_gen"]
    eta_det = data["eta_det"]
    dr = data["dr"]

    E_GEN_MIN = closure_floor(eta_det)
    quality = (e_gen >= E_GEN_MIN) & (dr < DR_MAX)
    resp_test = e_corr_test[quality] / np.maximum(e_gen[quality], 1.0)
    median_resp = np.median(resp_test)

    global_corr = 1.0 / median_resp
    scale = scale * global_corr

    print("\n  Global scale correction:")
    print(f"    Median response before: {median_resp:.6f}")
    print(f"    Global scale factor: {global_corr:.6f}")

    e_corr = e_reco * scale
    pt_corr = pt_at_fixed_mass(e_corr, data["eta_det"], data["mass"])
    return scale, e_corr, pt_corr, median_resp


def run_from_config(cfg, input_path, output, factor_out):
    """Train the response network on one dataset, apply the correction and write the corrected
    jets, and the overall factor it ends with (scripts/study_absolute_norm.py takes it out)."""
    t = cfg.training
    m = cfg.model

    print(f"\n{'=' * 65}")
    print(f"  NN absolute scale — {input_path}")
    print(
        f"  Config: {m.n_hidden}x{m.width} do={m.dropout} ep={t.epochs} lr={t.lr:.0e}"
    )
    print(f"  Seed: {cfg.seed}")
    print(f"{'=' * 65}\n")

    print(f"Loading {input_path}...")
    data, raw = read_jets([input_path], absolute_columns)

    print(f"\nTraining response NN (epochs={t.epochs})...")
    model, norm, history = train_response_nn(
        data,
        epochs=t.epochs,
        batch_size=t.batch_size,
        lr=t.lr,
        n_hidden=m.n_hidden,
        width=m.width,
        dropout=m.dropout,
        seed=cfg.seed,
        val_frac=t.val_frac,
    )

    scale, e_corr, pt_corr, median_resp = apply_calibration(data, model=model, norm=norm)

    print(f"\nWriting corrected parquet -> {output}")
    write_jets(raw, pt_corr, e_corr, output)
    print(
        f"  mean scale = {scale.mean():.4f}  "
        f"mean E_reco = {data['e_reco'].mean():.1f} GeV  "
        f"mean E_corr = {e_corr.mean():.1f} GeV"
    )
    Path(factor_out).write_text(json.dumps({"median_response_before": float(median_resp),
                                            "factor": float(1.0 / median_resp)}, indent=1) + "\n")
    print(f"  overall factor -> {factor_out}")


def main():
    parser = argparse.ArgumentParser(description="NN absolute energy scale of one dataset.")
    parser.add_argument("-i", "--input", required=True, metavar="FILE",
                        help="the jets after the NN pile-up correction")
    parser.add_argument("--output", required=True, metavar="FILE",
                        help="where the corrected jets are written")
    parser.add_argument("--factor-out", required=True, metavar="FILE",
                        help="where the overall factor is written (json)")
    parser.add_argument("--mini", action="store_true", help="the reduced run of run_paper.py --mini")
    args = parser.parse_args()

    print(f"  absolute scale: training floor {TRAIN_FLOOR:g} GeV; below the floor: constant")
    cfg = settings.get(args.mini).absolute_nn

    run_from_config(cfg=cfg, input_path=args.input, output=args.output, factor_out=args.factor_out)


if __name__ == "__main__":
    main()
