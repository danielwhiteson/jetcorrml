"""
NN absolute energy scale of large-R jets, without and with sub-jet inputs
(Sec. VI of the paper).

The response network of the small-R absolute scale, a residual MLP trained on
the median of E_reco / E_true, here as a function of log E_true and the jet's
own rapidity |y|, and with --use-subjet-eta also of the two sub-jet positions
|eta_sj,hi|, |eta_sj,lo| and the distance of the nearer sub-jet from the crack.
The correction is found jet by jet by solving E * R(E) = E_reco by bisection,
the same for both networks, so they differ only in their inputs.

Settings: jetcorrml.settings (large_r_absolute_nn; the crack of the sub-jet input is LARGE_R.eta_crack).

    uv run python -m steps.absolute_nn_large_r -i PILEUP_CORRECTED.pq --output FILE \
        --model-out MODEL.pt [--use-subjet-eta] [--mini]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import awkward as ak
import numpy as np
import torch
import torch.nn as nn

from jetcorrml.inversion import invert_bisect
from jetcorrml import large_r
from jetcorrml.absolute_closure import closure_floor
from jetcorrml.large_r import build_response_features, load_files_with_raw, set_crack_feature
from jetcorrml.networks import Normaliser, ResponseNet, flat_sampling_weights
from jetcorrml import settings

DR_MAX = settings.LARGE_R_MATCH_DR
R_CEIL = 1.0            # ceiling on R for the bisection (R above ~1 is unphysical here)


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
    use_subjet_eta=False,
):
    """Train the response NN with pinball loss at tau=0.5 (median).

    Returns (model_cpu, fitted_normaliser).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    e_gen = data["e_gen"].astype(np.float32)
    e_reco = data["e_reco"].astype(np.float32)
    eta = data["eta_det"].astype(np.float32)
    dr = data["dr"].astype(np.float32)
    sj_hi = data["abs_eta_sj_hi"].astype(np.float32) if use_subjet_eta else None
    sj_lo = data["abs_eta_sj_lo"].astype(np.float32) if use_subjet_eta else None

    response = (e_reco / np.maximum(e_gen, 1.0)).astype(np.float32)

    E_GEN_MIN = closure_floor(data["eta_geom"])
    good = (response > 0.05) & (response < 3.0) & (e_gen >= E_GEN_MIN) & (dr < DR_MAX)
    print(
        f"  Quality cut: keeping {good.sum():,} / {len(good):,} "
        f"large-R jets ({100 * good.mean():.1f}%)"
    )

    e_gen = e_gen[good]
    e_reco = e_reco[good]
    eta = eta[good]
    response = response[good]
    if use_subjet_eta:
        sj_hi = sj_hi[good]
        sj_lo = sj_lo[good]

    log_e_gen = np.log(np.clip(e_gen, 1.0, None))
    X_raw = build_response_features(log_e_gen, eta, sj_hi, sj_lo)
    n_feat = X_raw.shape[1]
    print(
        f"  Feature count: {n_feat} ({'with' if use_subjet_eta else 'without'}"
        f" subjet eta)"
    )

    norm = Normaliser()
    X = norm.fit_transform(X_raw).astype(np.float32)

    n = len(e_gen)
    idx = np.random.permutation(n)
    n_val = int(n * val_frac)
    val_idx, trn_idx = idx[:n_val], idx[n_val:]

    def to_t(arr):
        return torch.from_numpy(arr).to(device)

    X_trn, X_val = to_t(X[trn_idx]), to_t(X[val_idx])
    R_trn, R_val = to_t(response[trn_idx]), to_t(response[val_idx])

    print("  Computing stratified sampling weights...")
    samp_probs = flat_sampling_weights(
        log_e_gen[trn_idx], np.abs(eta[trn_idx])
    )
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=samp_probs, num_samples=len(trn_idx), replacement=True
    )
    trn_ds = torch.utils.data.TensorDataset(X_trn, R_trn)
    trn_loader = torch.utils.data.DataLoader(
        trn_ds, batch_size=batch_size, sampler=sampler, drop_last=True
    )

    model = ResponseNet(
        n_features=n_feat, n_hidden=n_hidden, width=width, dropout=dropout
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-3)
    steps_per_epoch = max(len(trn_idx) // batch_size, 1)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt,
        max_lr=lr * 1.5,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.1,
        anneal_strategy="cos",
        div_factor=1.5,
        final_div_factor=20.0,
    )

    TAU = 0.5
    for epoch in range(1, epochs + 1):
        model.train()
        for X_b, R_b in trn_loader:
            R_pred = model(X_b).squeeze(-1)
            residual = R_b - R_pred
            loss = torch.where(
                residual >= 0, TAU * residual, (TAU - 1.0) * residual
            ).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()

        if epoch % 10 == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                Rp_trn = model(X_trn).squeeze(-1)
                Rp_val = model(X_val).squeeze(-1)
                res_trn = R_trn - Rp_trn
                res_val = R_val - Rp_val
                tl = torch.where(res_trn >= 0, TAU * res_trn, (TAU - 1.0) * res_trn).mean().item()
                vl = torch.where(res_val >= 0, TAU * res_val, (TAU - 1.0) * res_val).mean().item()
            print(
                f"  Epoch {epoch:3d}/{epochs}  "
                f"loss={tl:.6f}  val_loss={vl:.6f}  <R_pred>={Rp_val.mean().item():.4f}"
            )

    model.eval()
    return model.cpu(), norm


def apply_calibration(data, model, norm, use_subjet_eta, e_bracket):
    """The calibration scale C = E_true / E_reco of every jet, from solving
    E * R(E) = E_reco by bisection over e_bracket; the two networks differ only
    in the inputs the features are built from."""
    e_reco = data["e_reco"]
    coord = data["eta_det"].astype(np.float32)
    if use_subjet_eta:
        _sj_hi = data["abs_eta_sj_hi"].astype(np.float32)
        _sj_lo = data["abs_eta_sj_lo"].astype(np.float32)

        def _feat(E):
            le = np.log(np.clip(E, 1.0, None)).astype(np.float32)
            return build_response_features(le, coord, _sj_hi, _sj_lo)
    else:

        def _feat(E):
            le = np.log(np.clip(E, 1.0, None)).astype(np.float32)
            return build_response_features(le, coord)

    e_lo_b, e_hi_b = float(e_bracket[0]), float(e_bracket[1])
    e_est = invert_bisect(e_reco, _feat, model, norm, e_lo_b, e_hi_b, r_ceil=R_CEIL)
    scale = (e_est / np.maximum(e_reco, 1e-9)).astype(np.float64)

    e_corr = e_reco * scale
    pt_corr = np.sqrt(np.maximum(e_corr**2 - data["mass"] ** 2, 0.0)) / np.cosh(
        data["eta_geom"]
    )
    return scale, e_corr, pt_corr


def write_corrected_parquet(raw, data, scale, e_corr, pt_corr, outpath):
    """Write the corrected large-R jets in the flat schema, applying the same
    multiplicative scale to both sub-jets (right for a multiplicative
    energy-scale correction).
    """
    sjA_pt_corr = data["sjA_pt"] * scale
    sjB_pt_corr = data["sjB_pt"] * scale

    fields = {f: np.asarray(raw[f]) for f in raw.fields}
    fields["jet_pt"] = pt_corr
    fields["jet_E"] = e_corr
    fields["jet_sjA_pt"] = sjA_pt_corr
    fields["jet_sjB_pt"] = sjB_pt_corr

    out_path = Path(outpath)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ak.to_parquet(ak.Array(fields), str(out_path))
    n = len(pt_corr)
    print(f"Corrected parquet written: {out_path}")
    print(f"  {n:,} large-R jets")
    print(
        f"  mean scale = {scale.mean():.4f}  "
        f"mean E_reco = {data['e_reco'].mean():.1f} GeV  "
        f"mean E_corr = {e_corr.mean():.1f} GeV"
    )


def save_model_bundle(path, model, norm, use_subjet_eta):
    """Persist enough to rebuild and probe the trained response network:
    weights, architecture (read off the module), normaliser stats, the sub-jet
    flag and, with sub-jet inputs, how the crack input was built.  Written next
    to the corrected jets; the network-probe figure (Fig. 13b) reads it.
    """
    trunk_linears = [layer for layer in model.trunk if isinstance(layer, nn.Linear)]
    bundle = {
        "state_dict": model.state_dict(),
        "n_features": int(model.trunk[0].in_features),
        "n_hidden": len(trunk_linears),
        "width": int(model.head.in_features),
        "dropout": float(model.trunk[2].p),
        "norm_mean": np.asarray(norm.mean_),
        "norm_std": np.asarray(norm.std_),
        "use_subjet_eta": bool(use_subjet_eta),
    }
    if use_subjet_eta:   # how the crack-proximity input was built: the distance from crack_ref, unsigned
        bundle["crack_ref"], bundle["crack_signed"] = float(large_r.CRACK_REF), False
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, str(p))
    print(f"  Model bundle saved: {p}")


def run_from_config(cfg, input_path, output, model_out, use_subjet_eta):
    """Train one network (with or without the sub-jet inputs), apply it and write
    the corrected jets and the model file."""
    t = cfg.training
    m = cfg.model

    print(f"\n{'=' * 65}")
    print(f"  Large-R NN absolute scale — {input_path}")
    print(
        f"  Config: {m.n_hidden}x{m.width} do={m.dropout} "
        f"ep={t.epochs} lr={t.lr:.0e}  use_subjet_eta={use_subjet_eta}"
    )
    print(f"  Seed: {cfg.seed}")
    print(f"{'=' * 65}\n")

    print(f"Loading {input_path}...")
    data, raw = load_files_with_raw([input_path])

    print(f"\nTraining response NN (epochs={t.epochs})...")
    model, norm = train_response_nn(
        data,
        epochs=t.epochs,
        batch_size=t.batch_size,
        lr=t.lr,
        val_frac=t.val_frac,
        seed=cfg.seed,
        n_hidden=m.n_hidden,
        width=m.width,
        dropout=m.dropout,
        use_subjet_eta=use_subjet_eta,
    )

    # Bracket of the bisection: the trained energy range, the same for both networks.
    _eg = np.asarray(data["e_gen"], dtype=float)
    e_bracket = (
        max(5.0, float(np.percentile(_eg, 0.5)) * 0.8),
        float(np.percentile(_eg, 99.9)) * 1.5,
    )
    scale, e_corr, pt_corr = apply_calibration(data, model, norm, use_subjet_eta, e_bracket)

    print(f"\nApplying calibration and writing parquet -> {output}")
    write_corrected_parquet(raw, data, scale, e_corr, pt_corr, output)

    save_model_bundle(model_out, model, norm, use_subjet_eta)


def main():
    parser = argparse.ArgumentParser(description="NN absolute energy scale of one large-R dataset.")
    parser.add_argument("-i", "--input", required=True, metavar="FILE",
                        help="the jets after the large-R pile-up correction")
    parser.add_argument("--output", required=True, metavar="FILE")
    parser.add_argument("--model-out", required=True, metavar="FILE", help="where the trained network is saved")
    parser.add_argument("--use-subjet-eta", action="store_true",
                        help="add the sub-jet inputs (the network 'with sub-jet inputs')")
    parser.add_argument("--mini", action="store_true", help="the reduced run of run_paper.py --mini")
    args = parser.parse_args()
    if args.use_subjet_eta:
        # the crack-proximity input: the sub-jets' distance from the injected crack
        set_crack_feature(settings.LARGE_R.eta_crack)
        print(f"  Crack-proximity input: measured from |eta| = {settings.LARGE_R.eta_crack:g}, unsigned")

    cfg = settings.get(args.mini).large_r_absolute_nn

    run_from_config(cfg=cfg, input_path=args.input, output=args.output,
                    model_out=args.model_out, use_subjet_eta=args.use_subjet_eta)


if __name__ == "__main__":
    main()
