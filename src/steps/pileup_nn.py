"""
NN pile-up correction (Sec. IV C of the paper).

The network learns the per-jet additive correction

    p_T^corr = p_T^reco - rho*A - NN(log(p_T^reco - rho*A), eta_det, N_PV, mu),

trained with the covariance loss of jetcorrml.pileup_training, which removes
the dependence of the residual on N_PV and mu but not its offset (the absolute
scale sets that).  Reads the injected sample, writes the corrected jets.
Settings: jetcorrml.settings (pileup_nn).

    uv run python -m steps.pileup_nn -i data/injected/lo_rho01.pq --output data/pileup/nn/lo_rho01.pq [--mini]
"""

import argparse

import numpy as np

from jetcorrml import settings
from jetcorrml.jets import energy_at_fixed_mass, flat, pileup_columns, read_jets, write_jets
from jetcorrml.pileup_training import apply_nn_correction, train


def write_corrected(raw, data, delta, outpath):
    """The corrected jets, pt_corr = pt_reco - rho*A - delta_NN, with E recomputed at fixed
    mass (the correction changes the energy scale, not the opening angles that set the
    mass); every other field as it was."""
    pt_reco = data["pt_reco"]
    rho_A = data["rho"] * data["area_pt"]
    pt_corr = pt_reco - rho_A - delta.ravel()
    e_corr = energy_at_fixed_mass(pt_corr, data["eta_det"], flat(raw.jet.mass))
    write_jets(raw, pt_corr, e_corr, outpath)
    print(
        f"  mean pt_corr = {pt_corr.mean():.2f} GeV  "
        f"(was {pt_reco.mean():.2f} GeV reco, "
        f"mean rho*A = {rho_A.mean():.2f} GeV, "
        f"mean delta_NN = {delta.ravel().mean():.2f} GeV)"
    )


def run_from_config(cfg, input_path, output):
    """Train the NN pile-up correction on one dataset and write its corrected jets."""
    eta_bins_arr = np.array(cfg.eta_bins)

    print(f"Loading {input_path}...")
    data, raw = read_jets([input_path], pileup_columns)

    t = cfg.training
    m = cfg.model
    print(
        f"\nTraining NN (epochs={t.epochs}, batch={t.batch_size}, "
        f"lr={t.lr}, lam={t.lam}, seed={cfg.seed}, "
        f"arch={m.n_hidden}x{m.width}_elu)..."
    )
    model, norm = train(
        data,
        epochs=t.epochs,
        batch_size=t.batch_size,
        lr=t.lr,
        lam=t.lam,
        val_frac=t.val_frac,
        seed=cfg.seed,
        n_hidden=m.n_hidden,
        width=m.width,
        eta_bins=eta_bins_arr,
        pt_bins=np.array(cfg.pt_bins, dtype=np.float32),
        early_stop_patience=t.early_stop_patience,
        sampler_min_count=t.sampler_min_count,
    )

    delta, _ = apply_nn_correction(data, model, norm)
    print(f"\nApplying correction and writing parquet -> {output}")
    write_corrected(raw, data, delta, output)


def main():
    parser = argparse.ArgumentParser(description="NN pile-up correction of one dataset.")
    parser.add_argument("-i", "--input", required=True, metavar="FILE", help="the injected jets")
    parser.add_argument("--output", required=True, metavar="FILE",
                        help="where the corrected jets are written")
    parser.add_argument("--mini", action="store_true", help="the reduced run of run_paper.py --mini")
    args = parser.parse_args()

    cfg = settings.get(args.mini).pileup_nn

    run_from_config(cfg=cfg, input_path=args.input, output=args.output)


if __name__ == "__main__":
    main()
