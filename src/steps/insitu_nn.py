# steps/insitu_nn.py
# -----------------------------------------------------------------------
# NN in-situ correction (paper Sec. VII C).
#
# The scale s applied to data jets is factorized, s = s_abs * s_rel, with
# s_rel exactly 1 for |eta| < 0.8.  One training loop minimises
#     L = L_reference + L_balance
# where
#   L_reference = quantile match of the central (|eta| < 0.8) response
#                 s * pT_reco / pT_ref of data to simulation, in (|eta|, pT) bins;
#                 it moves s_abs only, since s_rel == 1 there;
#   L_balance   = match of the median dijet log-balance
#                 log(s2 pT2) - log(s1 pT1) of data to simulation in
#                 (|eta1|, |eta2|, pT_avg) cells, with s_abs held fixed; it
#                 moves s_rel and relays the central scale outward.
# The network is a function of the reference pT (the particle-level jet for
# central jets, pT_avg for pairs) and is applied to each data jet by numerical
# inversion.  The simulation is not corrected.
#
# Outputs: the MC and corrected data parquets; with --retraining (a retraining
# of the nuisance-parameter band) also the scale on a fixed grid.
#
# Stages: load the jets (and, for a retraining, resample whole events), choose
# the dijet pairs, build the two terms, train, then invert the scale at each
# data jet's reco pT and write.
# -----------------------------------------------------------------------

import argparse
import os
import warnings

import awkward as ak
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler

from jetcorrml import settings
from jetcorrml.dijets import dijet_pairs

from jetcorrml.insitu_losses import (
    response_quantile_match_loss,
    make_response_bins,
    dijet_forward_balance_median_loss,
    make_eta_pair_pt_bins,
    RESPONSE_QM_LEVELS,
)

warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------
#  Settings: the run's (jetcorrml.settings) and the method's own
# -----------------------------------------------------------------------

QM_ETA_EDGES = [0.5]                 # interior (|eta|, pT_ref) edges of the reference term
QM_PT_EDGES = [30.0, 60.0, 100.0, 200.0]
CHAIN_ETA_EDGES = [0.8, 1.5, 2.0]    # interior |eta| edges of the balance cells (both jets)
CHAIN_PT_EDGES = [40.0, 80.0]        # interior pT_avg edges of the balance cells
N_INVERSION = 8                      # iterations of the numerical inversion

# The scale on a fixed grid, written by the band's retrainings (steps.band reads it).
BOOTGRID_PT = np.linspace(25.0, 250.0, 24)
BOOTGRID_ETA = np.array([0.2, 0.5, 0.9, 1.3, 1.7, 2.1, 2.4])



class Features:
    """The inputs of the scale s and of its absolute term s_abs, in the scaler's column
    order: (pT, |eta|) or, for the distortions of settings.Insitu.rho_dists, (pT, |eta|, rho)."""

    def __init__(self, rho_input):
        self.scale = ["pt", "eta", "rho"] if rho_input else ["pt", "eta"]
        self.abs = ["pt", "rho"] if rho_input else ["pt"]

    def stack(self, feat_map):
        """Column-stack the selected scale features in canonical order (float32).

        ``feat_map`` maps each of {'pt','eta','rho'} to a 1-D array (|eta| and rho
        already in the right form), so every input shares one layout and the one
        fitted scaler applies to all of them."""
        return np.column_stack(
            [feat_map[name] for name in self.scale]
        ).astype(np.float32)


class FactorizedScale(nn.Module):
    """s(pt,|eta|) = s_abs(pt) * s_rel(pt,|eta|).

    s_rel is exactly 1 for |eta| <= eta_central, by a boundary subtraction in
    scaled space:
        log s_rel = r(pt, max(|eta|, eta_c)) - r(pt, eta_c)
    For central jets the two r() calls are identical, so s_rel == 1 with zero
    gradient to r and the reference term trains only s_abs.  With
    detach_abs=True (the balance term) s_abs is held fixed and only s_rel moves.
    """

    def __init__(self, in_dim, hidden, pt_idx, eta_idx, eta_s_floor, abs_idx):
        super().__init__()
        self.pt_idx = int(pt_idx)
        self.eta_idx = int(eta_idx)
        self.abs_idx = [int(i) for i in abs_idx]   # inputs of s_abs (a subset of the scale's)
        self.register_buffer("eta_s_floor", torch.tensor(float(eta_s_floor)))
        self.abs_net = nn.Sequential(
            nn.Linear(len(self.abs_idx), hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        self.rel_net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        # zero-init both heads -> s_abs == 1, s_rel == 1 at the start
        for _net in (self.abs_net, self.rel_net):
            nn.init.zeros_(_net[-1].weight)
            nn.init.zeros_(_net[-1].bias)

    def forward(self, x, detach_abs=False):
        s_abs = 1.0 + self.abs_net(x[:, self.abs_idx]).squeeze(-1)
        if detach_abs:
            s_abs = s_abs.detach()
        x_up = x.clone()
        x_up[:, self.eta_idx] = torch.clamp(x[:, self.eta_idx], min=self.eta_s_floor)
        x_bnd = x.clone()
        x_bnd[:, self.eta_idx] = self.eta_s_floor
        log_srel = self.rel_net(x_up).squeeze(-1) - self.rel_net(x_bnd).squeeze(-1)
        return s_abs * torch.exp(log_srel)


def tt(arr):
    return torch.tensor(arr.astype(np.float32))


# ─────────────────────────────── loading ───────────────────────────────
def load_jets(path):
    """Jets of one file, flat.  Returns (df, number of jets in the file,
    indices of the jets kept); jets with a non-finite key column are dropped."""
    print(f"  Loading flat jets from {path} …")
    matches = ak.from_parquet(path)

    def flat(f):
        return ak.to_numpy(ak.flatten(f, axis=None)).astype(float)

    df = pd.DataFrame(
        {
            "jet_pt": flat(matches.jet.pt),
            "jet_eta": flat(matches.jet.eta),
            "jet_phi": flat(matches.jet.phi),
            "z_pt": flat(matches.gen.pt),
            "PUdensityRho": flat(matches.event.rho),
        }
    )
    df["balance"] = df["jet_pt"] / df["z_pt"]

    _key_cols = ["jet_pt", "jet_eta", "jet_phi", "z_pt", "PUdensityRho", "balance"]
    _n_before = len(df)
    _valid = df[_key_cols].notna().all(axis=1) & np.isfinite(df[_key_cols]).all(axis=1)
    _valid_idx = np.where(_valid)[0]
    df = df[_valid].reset_index(drop=True)
    if len(df) < _n_before:
        print(f"    Dropped {_n_before - len(df)} rows with NaN/Inf ({path})")

    return df, _n_before, _valid_idx


def event_weights(rng, path):
    """A retraining's draw of whole events with replacement: how often each event of
    the file is drawn, and the event of each jet."""
    _n = ak.to_numpy(ak.num(ak.from_parquet(path).jet.pt))
    _w = np.bincount(rng.integers(0, len(_n), len(_n)), minlength=len(_n))
    return _w, np.repeat(np.arange(len(_n)), _n)


# ─────────────────────────────── the two terms ───────────────────────────────
def reference_term(df_mc, df_data, features, seed, val_frac, eta_central):
    """The reference term's tensors: the central training jets in (|eta|, pT_ref) bins.
    Also fits the scaler (on the training simulation jets).  Returns (term, scaler)."""
    jet_pt_mc = tt(df_mc["jet_pt"].values)
    z_pt_mc = tt(df_mc["z_pt"].values)
    jet_pt_data = tt(df_data["jet_pt"].values)
    z_pt_data = tt(df_data["z_pt"].values)

    balance_mc_nom = (jet_pt_mc / z_pt_mc).detach()

    # The network is a function of the REFERENCE pT (z_pt, the particle-level jet);
    # the reco pT is used only at application, through the numerical inversion.
    x_mc_np = features.stack(
        {
            "pt": df_mc["z_pt"].values,
            "eta": np.abs(df_mc["jet_eta"].values),
            "rho": df_mc["PUdensityRho"].values,
        }
    )
    x_data_np = features.stack(
        {
            "pt": df_data["z_pt"].values,
            "eta": np.abs(df_data["jet_eta"].values),
            "rho": df_data["PUdensityRho"].values,
        }
    )

    # ── held-out jets: left out of the reference term and of the scaler fit ──
    n_mc = len(df_mc)
    n_data = len(df_data)
    _split_rng = torch.Generator().manual_seed(seed)
    n_val_mc = int(n_mc * val_frac)
    n_val_data = int(n_data * val_frac)
    perm_mc = torch.randperm(n_mc, generator=_split_rng)
    perm_data = torch.randperm(n_data, generator=_split_rng)
    trn_idx_mc = perm_mc[n_val_mc:]
    trn_idx_data = perm_data[n_val_data:]

    print("Fitting StandardScaler on training MC …")
    scaler = StandardScaler()
    scaler.fit(x_mc_np[trn_idx_mc.numpy()])

    x_data_scaled = torch.tensor(scaler.transform(x_data_np), dtype=torch.float32)

    # The reference term: training jets with |eta| < eta_central, in (|eta|, pT_ref) bins.
    _abs_eta_data_full = tt(np.abs(df_data["jet_eta"].values))
    _abs_eta_mc_full = tt(np.abs(df_mc["jet_eta"].values))
    _z_mask_d = _abs_eta_data_full[trn_idx_data] < eta_central
    _z_mask_m = _abs_eta_mc_full[trn_idx_mc] < eta_central
    _zj_qm_data_idx = trn_idx_data[_z_mask_d]
    _zj_qm_mc_idx = trn_idx_mc[_z_mask_m]
    print(f"  Z+jet QM central restriction |eta|<{eta_central}: "
          f"data {len(_zj_qm_data_idx)}/{len(trn_idx_data)}, "
          f"mc {len(_zj_qm_mc_idx)}/{len(trn_idx_mc)}")

    _qm_pt_edges_t = tt(np.asarray(QM_PT_EDGES, dtype=np.float32))
    _qm_eta_edges_t = tt(np.asarray(QM_ETA_EDGES, dtype=np.float32))
    _qm_data_bin, _qm_n_bins = make_response_bins(
        _abs_eta_data_full[_zj_qm_data_idx],
        z_pt_data[_zj_qm_data_idx],
        _qm_eta_edges_t,
        _qm_pt_edges_t,
    )
    _qm_mc_bin, _ = make_response_bins(
        _abs_eta_mc_full[_zj_qm_mc_idx],
        z_pt_mc[_zj_qm_mc_idx],
        _qm_eta_edges_t,
        _qm_pt_edges_t,
    )
    term = {"x_data": x_data_scaled, "jet_pt_data": jet_pt_data, "z_pt_data": z_pt_data,
            "balance_mc": balance_mc_nom, "data_idx": _zj_qm_data_idx, "mc_idx": _zj_qm_mc_idx,
            "data_bin": _qm_data_bin, "mc_bin": _qm_mc_bin, "n_bins": _qm_n_bins}
    return term, scaler


def dijet_tensors(df, features, scaler):
    # The relative scale is a function of the pair's reference pT (pT_avg) and
    # each jet's own |eta|; the measured balance uses the reco pt1, pt2.
    feat1 = features.stack(
        {
            "pt": df["pt_avg"].values,
            "eta": np.abs(df["eta1"].values),
            "rho": df["rho"].values,
        }
    )
    feat2 = features.stack(
        {
            "pt": df["pt_avg"].values,
            "eta": np.abs(df["eta2"].values),
            "rho": df["rho"].values,
        }
    )
    return {
        "feat1_s": torch.tensor(scaler.transform(feat1), dtype=torch.float32),
        "feat2_s": torch.tensor(scaler.transform(feat2), dtype=torch.float32),
        "pt1": torch.tensor(df["pt1"].values.astype(np.float32)),
        "pt2": torch.tensor(df["pt2"].values.astype(np.float32)),
        "pt_avg": torch.tensor(df["pt_avg"].values.astype(np.float32)),
        "eta1": torch.tensor(np.abs(df["eta1"].values).astype(np.float32)),
        "eta2": torch.tensor(np.abs(df["eta2"].values).astype(np.float32)),
    }


def balance_term(dj_mc, dj_data, features, scaler):
    """The balance term's tensors: the dijet pairs in (|eta1|, |eta2|, pT_avg) cells, with
    the simulation's log-balance as the target (the simulation is not corrected)."""
    dj_t_mc = dijet_tensors(dj_mc, features, scaler)
    dj_t_data = dijet_tensors(dj_data, features, scaler)
    _beta_dj_mc = (
        torch.log(dj_t_mc["pt2"]) - torch.log(dj_t_mc["pt1"])
    ).detach()

    _chain_eta_edges_t = torch.tensor(CHAIN_ETA_EDGES, dtype=torch.float32)
    _dj_pt_edges_t = torch.tensor(CHAIN_PT_EDGES, dtype=torch.float32)
    _dj_data_fwd_bin, _dj_n_bins = make_eta_pair_pt_bins(
        dj_t_data["eta1"], dj_t_data["eta2"], dj_t_data["pt_avg"],
        _chain_eta_edges_t, _dj_pt_edges_t,
    )
    _dj_mc_fwd_bin, _ = make_eta_pair_pt_bins(
        dj_t_mc["eta1"], dj_t_mc["eta2"], dj_t_mc["pt_avg"],
        _chain_eta_edges_t, _dj_pt_edges_t,
    )
    print(f"  Dijet balance cells: n_bins={_dj_n_bins}  (|eta1|,|eta2|,pT_avg), |eta| edges "
          f"{CHAIN_ETA_EDGES} x pT_avg edges {CHAIN_PT_EDGES}")
    return {"data": dj_t_data, "beta_mc": _beta_dj_mc, "data_bin": _dj_data_fwd_bin,
            "mc_bin": _dj_mc_fwd_bin, "n_bins": _dj_n_bins}


# ─────────────────────────────── training ───────────────────────────────
def train(model, optimizer, ref, bal, epochs):
    """Minimise L_reference + L_balance, full batch, for `epochs` epochs."""
    print(f"\n── Training: reference + balance  ({epochs} epochs) ─────")
    dj = bal["data"]
    for epoch in range(epochs):
        optimizer.zero_grad()
        s = model(ref["x_data"])
        # reference term: the central training jets, in (|eta|, pT_ref) bins
        r_corr_qm = (
            ref["jet_pt_data"][ref["data_idx"]] * s[ref["data_idx"]]
            / ref["z_pt_data"][ref["data_idx"]]
        )
        r_mc_qm = ref["balance_mc"][ref["mc_idx"]]
        l_zj = response_quantile_match_loss(
            r_corr_qm, r_mc_qm, RESPONSE_QM_LEVELS,
            data_bin=ref["data_bin"], mc_bin=ref["mc_bin"], n_bins=ref["n_bins"],
        )
        # balance term: s_abs held fixed, so it moves s_rel only.  The inner jet may
        # itself be forward, so its s_rel trains too and relays the gauge outward.
        s1 = model(dj["feat1_s"], detach_abs=True)
        s2 = model(dj["feat2_s"], detach_abs=True)
        _ps1 = torch.clamp(dj["pt1"] * s1, min=1e-6)
        _ps2 = torch.clamp(dj["pt2"] * s2, min=1e-6)
        beta_corr = torch.log(_ps2) - torch.log(_ps1)
        l_fwdbal = dijet_forward_balance_median_loss(
            beta_corr, bal["beta_mc"],
            fwd_bin=bal["data_bin"], mc_fwd_bin=bal["mc_bin"],
            n_bins=bal["n_bins"],
        )
        loss = l_zj + l_fwdbal
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"  Epoch {epoch:3d} | loss = {loss.item():.8f} "
                  f"[l_ref={l_zj.item():.2e}  l_bal={l_fwdbal.item():.2e}]")


# ─────────────────────────────── application ───────────────────────────────
def guarded_inversion(_c_at, reco_pt, n_iter, support):
    """Fixed-point inversion s = c(reco * s) with a numerical guard.

    For each jet the scale is bounded by the range the correction takes over
    the support (lo, hi) (at the jet's own eta and rho), widened by
    0.05 plus one full span on each side, so the bound only acts on a run-away
    iteration; the reference-pT argument is clipped to the support."""
    ref_lo, ref_hi = support
    c_lo = np.full(reco_pt.shape, np.inf)
    c_hi = np.full(reco_pt.shape, -np.inf)
    for q in np.geomspace(ref_lo, ref_hi, 12):
        v = _c_at(np.full(reco_pt.shape, q))
        c_lo, c_hi = np.fmin(c_lo, v), np.fmax(c_hi, v)
    margin = 0.05 + (c_hi - c_lo)
    b_lo, b_hi = np.maximum(c_lo - margin, 0.2), c_hi + margin
    mid = 0.5 * (c_lo + c_hi)

    def _arg(ref):
        ref = np.where(np.isfinite(ref) & (ref > 0), ref, reco_pt)
        return np.clip(ref, ref_lo, ref_hi)

    def _bound(v, prev):
        bad = ~np.isfinite(v)
        v = np.where(bad, prev, v)
        return np.clip(v, b_lo, b_hi), bad, (v < b_lo) | (v > b_hi)

    s, bad, out = _bound(_c_at(_arg(reco_pt)), mid)
    for _ in range(int(n_iter)):
        s, bad, out = _bound(_c_at(_arg(reco_pt * s)), s)
    ref = reco_pt * s
    print(f"  [INV-GUARD] jets={len(s)} clamped={int(out.sum())} nonfinite={int(bad.sum())} "
          f"below_support={int((ref < ref_lo).sum())} above_support={int((ref > ref_hi).sum())} "
          f"support_lo={ref_lo:.2f} support_hi={ref_hi:.2f}")
    return s


def invert_scale_to_reco(model, scaler, features, support, reco_pt, abs_eta, rho):
    """The applied scale s(reco) of each jet: the network c(ref_pt, eta) was
    trained against the reference pT, and a jet measured at reco_pt belongs to
    a reference of about reco_pt * s, so solve s = c(reco_pt * s, eta)."""
    reco_pt = np.asarray(reco_pt, dtype=np.float64)
    abs_eta = np.asarray(abs_eta, dtype=np.float64)
    rho = np.asarray(rho, dtype=np.float64)

    def _c_at(ref_pt):
        feats = features.stack({"pt": ref_pt, "eta": abs_eta, "rho": rho})
        ft = torch.tensor(scaler.transform(feats), dtype=torch.float32)
        with torch.no_grad():
            return model(ft).cpu().numpy().astype(np.float64)

    return guarded_inversion(_c_at, reco_pt, N_INVERSION, support)


# ─────────────────────────────── writing ───────────────────────────────
def write_grid(path, scale_at, rho, seed, w_mc, w_data):
    """A retraining of the band: the applied scale on a fixed grid of reco pT and
    |eta| (at the median rho), with how often each event was drawn."""
    _S = np.zeros((len(BOOTGRID_PT), len(BOOTGRID_ETA)), dtype=np.float64)
    for _i, _ptv in enumerate(BOOTGRID_PT):
        _S[_i] = scale_at(
            np.full(len(BOOTGRID_ETA), _ptv),
            BOOTGRID_ETA,
            np.full(len(BOOTGRID_ETA), rho),
        )
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savez(path, pt=BOOTGRID_PT, eta=BOOTGRID_ETA, s=_S, seed=seed,
             evw_mc=w_mc.astype(np.uint8), evw_da=w_data.astype(np.uint8))
    print(f"  Wrote bootstrap-grid sidecar: {path}")


def write_scaled(orig, scale, out):
    """The jets of `orig` with pt and E multiplied by `scale` (one factor per jet)."""
    jet_pt_orig = ak.flatten(orig.jet.pt)
    jet_pt_corr = jet_pt_orig * scale
    jet_E_orig = ak.flatten(orig.jet.E)
    jet_E_corr = jet_E_orig * scale

    fields = {
        "jet": ak.zip(
            {
                "pt": ak.unflatten(jet_pt_corr, ak.num(orig.jet.pt)),
                "eta": orig.jet.eta,
                "phi": orig.jet.phi,
                "mass": orig.jet.mass,
                "E": ak.unflatten(jet_E_corr, ak.num(orig.jet.E)),
            }
        ),
        "gen": orig.gen,
        "event": orig.event,
        "deltaR": orig.deltaR,
    }
    ak.to_parquet(ak.zip(fields), out)


def main():
    parser = argparse.ArgumentParser(description="NN in-situ correction (Z+jet reference + dijet balance)")
    parser.add_argument("--mc", "-m", required=True, metavar="FILE")
    parser.add_argument("--data", "-d", required=True, metavar="FILE")
    parser.add_argument("--dist", required=True, choices=sorted(settings.DISTORTIONS),
                        help="the distortion of the data half: for those in settings.Insitu.rho_dists "
                             "the scale also takes rho as an input")
    parser.add_argument(
        "--pair-selection", choices=["both", "inner", "inner-avg"], default="both",
        help="How the dijet pairs are chosen (settings.Insitu.pair_rule).  'both' (pass 1): both jets "
        "above the pT cut, pair pT = their average.  'inner' (pass 2): only the inner (smaller-|eta|) "
        "jet must pass, pair pT = its pT.  'inner-avg' (passes 3 and 4): the inner jet "
        "must pass, pair pT = the average.  'inner' and 'inner-avg' need --selection-pt.",
    )
    parser.add_argument(
        "--selection-pt", default=None, metavar="FILE",
        help="Parquet with the same events and jets as --data (the pass before); its jet "
        "pT chooses the data pairs and gives the pair's pT.  The balance is always "
        "computed from the pT in --data.",
    )
    parser.add_argument(
        "--retraining", type=int, default=None, metavar="B",
        help="Retraining B of the nuisance-parameter band: resample whole events of both samples "
        "with replacement (seed B), train with seed band.seed_base + B, and write the scale on "
        "a fixed grid to --grid-out.",
    )
    parser.add_argument("--output-mc", required=True, metavar="PATH")
    parser.add_argument("--output-data", required=True, metavar="PATH")
    parser.add_argument("--grid-out", default=None, metavar="PATH",
                        help="with --retraining: where the scale on the fixed grid is written")
    parser.add_argument("--mini", action="store_true", help="the reduced run of run_paper.py --mini")

    args = parser.parse_args()
    if (args.retraining is not None) != (args.grid_out is not None):
        parser.error("--grid-out goes with --retraining, and only with it")

    run = settings.get(args.mini)
    insitu = run.insitu
    seed = insitu.seed if args.retraining is None else run.band.seed_base + args.retraining
    bootstrap_seed = -1 if args.retraining is None else args.retraining
    eta_central = settings.CENTRAL_ETA   # central region: s_rel == 1, reference term only

    torch.manual_seed(seed)

    features = Features(rho_input=args.dist in insitu.rho_dists)

    if args.pair_selection in ("inner", "inner-avg") and not args.selection_pt:
        raise SystemExit(f"--pair-selection {args.pair_selection} needs --selection-pt (an earlier-pass in-situ output of "
                         "the data): on uncorrected data the inner jet's pT is itself off scale.")

    print(f"Scale-net inputs: s({', '.join(features.scale)}), s_abs({', '.join(features.abs)})")
    print(f"Training: {insitu.epochs} epochs")

    print("Loading MC …")
    df_mc, _n_mc_orig, _valid_idx_mc = load_jets(args.mc)
    print("Loading data …")
    df_data, _n_data_orig, _valid_idx_data = load_jets(args.data)
    print(f"  Flat jets: MC={len(df_mc):,}  data={len(df_data):,}")

    # ── a retraining of the band: resample whole events of both samples ──────────
    #  every jet and every dijet pair of an event comes along
    if bootstrap_seed >= 0:
        _brng = np.random.default_rng(bootstrap_seed)
        _bw_mc, _ev_mc = event_weights(_brng, args.mc)
        _bw_da, _ev_da = event_weights(_brng, args.data)
        _bi_mc = np.repeat(np.arange(len(df_mc)), _bw_mc[_ev_mc[_valid_idx_mc]])
        _bi_da = np.repeat(np.arange(len(df_data)), _bw_da[_ev_da[_valid_idx_data]])
        print(f"  BOOTSTRAP of whole events ({int((_bw_mc > 0).sum())} of {len(_bw_mc)} simulation events, "
              f"{int((_bw_da > 0).sum())} of {len(_bw_da)} data events drawn)")
        df_mc = df_mc.iloc[_bi_mc].reset_index(drop=True)
        df_data = df_data.iloc[_bi_da].reset_index(drop=True)
        print(f"  BOOTSTRAP seed={bootstrap_seed}: resampled MC+data "
              f"(MC={len(df_mc):,} data={len(df_data):,})")

    print(f"  Pair selection: {args.pair_selection}"
          + (f"  (data pairs chosen on the pT in {args.selection_pt})" if args.selection_pt else ""))
    dj_mc = dijet_pairs(args.mc, rule=args.pair_selection)
    dj_data = dijet_pairs(args.data, rule=args.pair_selection, selection_pt=args.selection_pt)
    if bootstrap_seed >= 0:
        # the pairs of the resampled events
        dj_mc = dj_mc.iloc[np.repeat(np.arange(len(dj_mc)), _bw_mc[dj_mc["evt"].values])].reset_index(drop=True)
        dj_data = dj_data.iloc[np.repeat(np.arange(len(dj_data)), _bw_da[dj_data["evt"].values])].reset_index(drop=True)
    print(f"  Dijet pairs: MC={len(dj_mc):,}  data={len(dj_data):,}")

    ref, scaler = reference_term(df_mc, df_data, features, seed, insitu.val_frac, eta_central)
    bal = balance_term(dj_mc, dj_data, features, scaler)

    # The support of the reference pT the correction was derived on: the reference
    # pT of the central jets and the pT_avg of the pairs, in simulation and data.
    support = (float(min(np.min(r) for r in (df_mc["z_pt"].values, df_data["z_pt"].values,
                                             dj_mc["pt_avg"].values, dj_data["pt_avg"].values))),
               float(max(np.max(r) for r in (df_mc["z_pt"].values, df_data["z_pt"].values,
                                             dj_mc["pt_avg"].values, dj_data["pt_avg"].values))))

    _PT_IDX = features.scale.index("pt")
    _ETA_IDX = features.scale.index("eta")
    _ETA_S_FLOOR = (eta_central - scaler.mean_[_ETA_IDX]) / scaler.scale_[_ETA_IDX]
    _ABS_IDX = [features.scale.index(f) for f in features.abs]
    model = FactorizedScale(len(features.scale), insitu.width, _PT_IDX, _ETA_IDX, _ETA_S_FLOOR, abs_idx=_ABS_IDX)
    optimizer = optim.Adam(model.parameters(), lr=insitu.lr, weight_decay=insitu.weight_decay)

    train(model, optimizer, ref, bal, insitu.epochs)

    def scale_at(reco_pt, abs_eta, rho):
        return invert_scale_to_reco(model, scaler, features, support, reco_pt, abs_eta, rho)

    if bootstrap_seed >= 0:
        write_grid(args.grid_out, scale_at, float(np.median(df_data["PUdensityRho"].values)),
                   bootstrap_seed, _bw_mc, _bw_da)

    # The simulation is not corrected.  Its jets are written through the same
    # path as the data's, with a factor of 1, so pt and E have the data file's type.
    print(f"\nSaving corrected MC parquet to {args.output_mc}")
    for _out in (args.output_mc, args.output_data):
        os.makedirs(os.path.dirname(_out) or ".", exist_ok=True)
    mc_orig = ak.from_parquet(args.mc)
    write_scaled(mc_orig, np.ones(len(ak.flatten(mc_orig.jet.pt)), dtype=np.float64), args.output_mc)
    print(f"  Saved corrected MC parquet: {args.output_mc}")

    print(f"\nSaving corrected Data parquet to {args.output_data}")
    data_orig = ak.from_parquet(args.data)
    # The correction is applied to the original (not resampled) jets: the applied
    # scale is a function of each jet's (reco pT, |eta|, rho) and the model alone.
    _jpt_orig = ak.to_numpy(ak.flatten(data_orig.jet.pt, axis=None)).astype(np.float64)
    _jeta_orig = np.abs(ak.to_numpy(ak.flatten(data_orig.jet.eta, axis=None)).astype(np.float64))
    _rho_orig = ak.to_numpy(ak.flatten(
        ak.broadcast_arrays(data_orig.event.rho, data_orig.jet.pt)[0], axis=None)).astype(np.float64)
    write_scaled(data_orig, scale_at(_jpt_orig, _jeta_orig, _rho_orig), args.output_data)
    print(f"  Saved corrected Data parquet: {args.output_data}")


if __name__ == "__main__":
    main()
