"""
The large-R jets (Sec. VI): the flat records of the large-R files as numpy arrays,
and the inputs of the large-R response network.

The large-R files are flat: one row per matched jet, with its two sub-jets.
"""

import awkward as ak
import numpy as np


def _delta_phi(phi1, phi2):
    dp = phi1 - phi2
    return np.arctan2(np.sin(dp), np.cos(dp))


def compute_alpha(sjA_eta, sjA_phi, sjB_eta, sjB_phi):
    """Angle of the subjet splitting axis in the eta-phi plane,
    alpha = atan2(dphi_AB, deta_AB) in [-pi, pi]."""
    deta = sjA_eta - sjB_eta
    dphi = _delta_phi(sjA_phi, sjB_phi)
    return np.arctan2(dphi, deta)


def _load_flat(d):
    """A flat numpy dict from a large-R record.  eta_det is the jet's own rapidity
    (the file's jet_y column); eta_geom, the geometric eta, drives the pT <-> E
    conversions, which are detector quantities."""
    jet_eta = np.asarray(d.jet_eta, dtype=float)
    gen_eta = np.asarray(d.gen_eta, dtype=float)
    gen_pt = np.asarray(d.gen_pt, dtype=float)
    jet_E = np.asarray(d.jet_E, dtype=float)
    gen_E = np.asarray(d.gen_E, dtype=float)

    sjA_eta = np.asarray(d.jet_sjA_eta, dtype=float)
    sjA_phi = np.asarray(d.jet_sjA_phi, dtype=float)
    sjB_eta = np.asarray(d.jet_sjB_eta, dtype=float)
    sjB_phi = np.asarray(d.jet_sjB_phi, dtype=float)
    alpha = compute_alpha(sjA_eta, sjA_phi, sjB_eta, sjB_phi)

    deta_sjA = sjA_eta - jet_eta
    deta_sjB = sjB_eta - jet_eta
    abs_sjA = np.abs(sjA_eta)
    abs_sjB = np.abs(sjB_eta)
    abs_eta_sj_hi = np.maximum(abs_sjA, abs_sjB)
    abs_eta_sj_lo = np.minimum(abs_sjA, abs_sjB)

    dr = np.asarray(d.deltaR, dtype=float)
    eta_det = np.asarray(d.jet_y, dtype=float)
    return dict(
        e_reco=jet_E,
        pt_reco=np.asarray(d.jet_pt, dtype=float),
        eta_det=eta_det,
        eta_geom=jet_eta,
        mass=np.asarray(d.jet_mass, dtype=float),
        e_gen=gen_E,
        pt_gen=gen_pt,
        eta_gen=gen_eta,
        dr=dr,
        alpha=alpha,
        deta_sjA=deta_sjA,
        deta_sjB=deta_sjB,
        sjA_eta=sjA_eta,
        sjB_eta=sjB_eta,
        abs_eta_sj_hi=abs_eta_sj_hi,
        abs_eta_sj_lo=abs_eta_sj_lo,
        sjA_pt=np.asarray(d.jet_sjA_pt, dtype=float),
        sjA_mass=np.asarray(d.jet_sjA_mass, dtype=float),
        sjB_pt=np.asarray(d.jet_sjB_pt, dtype=float),
        sjB_mass=np.asarray(d.jet_sjB_mass, dtype=float),
    )


def load_files(paths):
    parts = []
    for path in paths:
        print(f"  Loading {path}")
        d = ak.from_parquet(path)
        data_part = _load_flat(d)
        parts.append(data_part)
        print(f"    -> {len(data_part['e_reco']):,} matched large-R jets")
    keys = parts[0].keys()
    data = {k: np.concatenate([p[k] for p in parts]) for k in keys}
    print(f"  Total: {len(data['e_reco']):,} large-R jets")
    return data


def load_files_with_raw(paths):
    parts_flat, parts_raw = [], []
    for path in paths:
        print(f"  Loading {path}")
        d = ak.from_parquet(path)
        parts_flat.append(_load_flat(d))
        parts_raw.append(d)
        print(f"    -> {len(parts_flat[-1]['e_reco']):,} matched large-R jets")
    keys = parts_flat[0].keys()
    data = {k: np.concatenate([p[k] for p in parts_flat]) for k in keys}
    raw = ak.concatenate(parts_raw, axis=0)
    print(f"  Total: {len(data['e_reco']):,} large-R jets")
    return data, raw


# The crack-proximity input of the sub-jet network: the |eta| the distance is measured
# from.  One setting per process, made by set_crack_feature(): the absolute scale with
# sub-jet inputs sets the injected crack (settings.LARGE_R.eta_crack), the figure code the
# value saved with the network it probes.  The network without sub-jet inputs saves the
# default.
CRACK_REF = 1.3


def set_crack_feature(ref=1.3, signed=False):
    global CRACK_REF
    if signed:
        raise ValueError("the crack-proximity input is unsigned")
    CRACK_REF = float(ref)


def build_response_features(log_e, eta, abs_eta_sj_hi=None, abs_eta_sj_lo=None):
    """The network inputs: (logE, |eta|, |eta|^2, logE*|eta|), and with the sub-jet
    positions also (|eta_sj_hi|, |eta_sj_lo|, the distance of the nearer sub-jet
    from CRACK_REF, unsigned)."""
    abs_eta = np.abs(eta)
    cols = [log_e, abs_eta, abs_eta**2, log_e * abs_eta]
    if abs_eta_sj_hi is not None and abs_eta_sj_lo is not None:
        cols.append(abs_eta_sj_hi)
        cols.append(abs_eta_sj_lo)
        ETA_CRACK = CRACK_REF
        crack_dist = np.minimum(
            np.abs(abs_eta_sj_hi - ETA_CRACK),
            np.abs(abs_eta_sj_lo - ETA_CRACK),
        )
        cols.append(crack_dist)
    return np.column_stack(cols).astype(np.float32)

