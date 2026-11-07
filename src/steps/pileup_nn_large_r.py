"""
NN pile-up correction of large-R jets (Sec. VI of the paper).

The same network, loss and training as the small-R correction
(jetcorrml.pileup_training); only the data layer differs.  The large-R files
are flat (one row per matched jet), and the correction rho*A + delta_NN is
shared between the two sub-jets in proportion to their share of the jet pT,
the same rule the large-R injection uses for the pile-up shift.

Settings: jetcorrml.settings (large_r_pileup_nn).

    uv run python -m steps.pileup_nn_large_r -i INJECTED.pq --output CORRECTED.pq [--mini]
"""

import argparse
from pathlib import Path

import awkward as ak
import numpy as np

from jetcorrml import settings
from jetcorrml.pileup_training import apply_nn_correction, train


def _load_flat(d):
    """
    A flat numpy dict from a large-R awkward record: the keys of the small-R
    loader (pt_reco, pt_gen, eta_det, rho, npv, mu, area_pt), the sub-jet fields
    the correction is shared with, and jet_mass for the corrected energy.
    """
    jet_eta = np.asarray(d.jet_eta, dtype=float)
    area_pt = np.asarray(d.jet_areaE, dtype=float) / np.cosh(jet_eta)

    return dict(
        pt_reco=np.asarray(d.jet_pt, dtype=float),
        pt_gen=np.asarray(d.gen_pt, dtype=float),
        eta_det=jet_eta,
        jet_mass=np.asarray(d.jet_mass, dtype=float),
        rho=np.asarray(d.rho, dtype=float),
        npv=np.asarray(d.npv, dtype=float),
        mu=np.asarray(d.mu, dtype=float),
        area_pt=area_pt,
        sjA_pt=np.asarray(d.jet_sjA_pt, dtype=float),
        sjA_eta=np.asarray(d.jet_sjA_eta, dtype=float),
        sjA_mass=np.asarray(d.jet_sjA_mass, dtype=float),
        sjB_pt=np.asarray(d.jet_sjB_pt, dtype=float),
        sjB_eta=np.asarray(d.jet_sjB_eta, dtype=float),
        sjB_mass=np.asarray(d.jet_sjB_mass, dtype=float),
    )


def load_files_with_raw(paths):
    """The jets of the files, flat, and the concatenated raw awkward array.

    The raw record is needed by the writer so every input field is preserved
    in the output (only the corrected fields are overwritten).
    """
    parts_flat = []
    parts_raw = []
    for path in paths:
        print(f"  Loading {path}")
        d = ak.from_parquet(path)
        parts_flat.append(_load_flat(d))
        parts_raw.append(d)
        print(f"    -> {len(parts_flat[-1]['pt_reco']):,} matched large-R jets")
    keys = parts_flat[0].keys()
    data = {k: np.concatenate([p[k] for p in parts_flat]) for k in keys}
    raw = ak.concatenate(parts_raw, axis=0)
    print(f"  Total: {len(data['pt_reco']):,} large-R jets")
    return data, raw


def propagate_to_subjets(data, delta):
    """Distribute the large-R correction (rho*A + delta_NN) across the sub-jets in
    proportion to each sub-jet's pT fraction of the jet.

    Returns the corrected jet pT and E and the corrected sub-jet pT.
    """
    rhoA = data["rho"] * data["area_pt"]
    pt_corr = data["pt_reco"] - rhoA - delta.ravel()

    cosh_eta = np.cosh(data["eta_det"])
    e_corr = np.sqrt(np.maximum((pt_corr * cosh_eta) ** 2 + data["jet_mass"] ** 2, 0.0))

    total_delta = rhoA + delta.ravel()
    safe_jet_pt = np.maximum(data["pt_reco"], 1e-6)
    frac_A = data["sjA_pt"] / safe_jet_pt
    frac_B = data["sjB_pt"] / safe_jet_pt
    sjA_pt_corr = data["sjA_pt"] - total_delta * frac_A
    sjB_pt_corr = data["sjB_pt"] - total_delta * frac_B
    return pt_corr, e_corr, sjA_pt_corr, sjB_pt_corr


def write_corrected_parquet(raw, data, delta, outpath):
    """Write the corrected large-R jets to a flat parquet with the input's fields,
    replacing only the corrected ones.
    """
    pt_corr, e_corr, sjA_pt_corr, sjB_pt_corr = propagate_to_subjets(data, delta)

    fields = {f: np.asarray(raw[f]) for f in raw.fields}
    fields["jet_pt"] = pt_corr
    fields["jet_E"] = e_corr
    fields["jet_sjA_pt"] = sjA_pt_corr
    fields["jet_sjB_pt"] = sjB_pt_corr

    out_path = Path(outpath)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ak.to_parquet(ak.Array(fields), str(out_path))

    n = len(pt_corr)
    rhoA = data["rho"] * data["area_pt"]
    print(f"Corrected parquet written: {out_path}")
    print(f"  {n:,} large-R jets")
    print(
        f"  mean pt_corr = {pt_corr.mean():.2f} GeV  "
        f"(was {data['pt_reco'].mean():.2f} GeV reco, "
        f"mean rho*A = {rhoA.mean():.2f} GeV, "
        f"mean delta_NN = {delta.ravel().mean():.2f} GeV)"
    )
    print(f"  sjA: mean pt {data['sjA_pt'].mean():.2f} -> {sjA_pt_corr.mean():.2f} GeV")
    print(f"  sjB: mean pt {data['sjB_pt'].mean():.2f} -> {sjB_pt_corr.mean():.2f} GeV")


def run_from_config(cfg, input_path, output):
    """Train the large-R NN pile-up correction on one dataset and write its corrected jets."""
    eta_bins_arr = np.array(cfg.eta_bins, dtype=float)
    pt_bins_arr = np.array(cfg.pt_bins, dtype=float)

    print(f"Loading {input_path}...")
    data, raw = load_files_with_raw([input_path])

    t = cfg.training
    m = cfg.model
    print(
        f"\nTraining the large-R NN (epochs={t.epochs}, batch={t.batch_size}, "
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
        pt_bins=pt_bins_arr,
        early_stop_patience=t.early_stop_patience,
        sampler_min_count=t.sampler_min_count,
    )

    delta, _ = apply_nn_correction(data, model, norm)
    print(f"\nApplying correction and writing parquet -> {output}")
    write_corrected_parquet(raw, data, delta, output)


def main():
    parser = argparse.ArgumentParser(description="NN pile-up correction of one large-R dataset.")
    parser.add_argument("-i", "--input", required=True, metavar="FILE", help="the injected jets")
    parser.add_argument("--output", required=True, metavar="FILE",
                        help="where the corrected jets are written")
    parser.add_argument("--mini", action="store_true", help="the reduced run of run_paper.py --mini")
    args = parser.parse_args()

    cfg = settings.get(args.mini).large_r_pileup_nn

    run_from_config(cfg=cfg, input_path=args.input, output=args.output)


if __name__ == "__main__":
    main()
