"""
Jets from the pipeline's files as flat numpy arrays.

Every file of the dijet chain holds events with jagged fields jet.*, gen.*,
event.* (one value per jet, as the injection writes them) and deltaR.  The
functions here flatten them in file order, so arrays from one file line up
jet by jet.
"""

from pathlib import Path

import awkward as ak
import numpy as np


def flat(a):
    """An awkward array of any depth as a flat float64 numpy array (a copy)."""
    return ak.to_numpy(ak.flatten(a, axis=None)).astype(float)


def per_jet(ev, field):
    """The event field `field` of the events `ev`, one value per jet."""
    return flat(ak.broadcast_arrays(ev.event[field], ev.jet.pt)[0])


# ─────────────────────────── the correction steps' files ───────────────────────────
def pileup_columns(matches):
    """The columns the pile-up steps read (area_pt = areaE / cosh(eta), formed per jet
    before flattening)."""
    area_pt = matches.jet.areaE / np.cosh(matches.jet.eta)
    return dict(
        pt_reco=flat(matches.jet.pt),
        pt_gen=flat(matches.gen.pt),
        eta_det=flat(matches.jet.eta),
        rho=flat(matches.event.rho),
        npv=flat(matches.event.npv),
        mu=flat(matches.event.mu),
        area_pt=flat(area_pt),
    )


def absolute_columns(matches):
    """The columns the absolute-scale steps read."""
    pg, eg = flat(matches.gen.pt), flat(matches.gen.eta)
    return {
        "e_reco": flat(matches.jet.E),
        "pt_reco": flat(matches.jet.pt),
        "eta_det": flat(matches.jet.eta),
        "mass": flat(matches.jet.mass),
        "pt_gen": pg,
        "e_gen": pg * np.cosh(eg),
        "dr": flat(matches.deltaR),
    }


def read_jets(paths, columns):
    """The jets of the files, flat (`columns` of each file, concatenated), and the files'
    events concatenated, from which the corrected jets are written with the same schema."""
    parts_flat, parts_raw = [], []
    for path in paths:
        print(f"  Loading {path}")
        matches = ak.from_parquet(path)
        parts_flat.append(columns(matches))
        parts_raw.append(matches)
        print(f"    -> {len(parts_flat[-1]['pt_reco']):,} matched jets")
    data = {k: np.concatenate([p[k] for p in parts_flat]) for k in parts_flat[0]}
    raw = ak.concatenate(parts_raw, axis=0)
    print(f"  Total: {len(data['pt_reco']):,} jets")
    return data, raw


def energy_at_fixed_mass(pt, eta, mass):
    """E of a jet of transverse momentum pt, pseudorapidity eta and mass `mass`."""
    return np.sqrt(np.maximum((pt * np.cosh(np.abs(eta))) ** 2 + mass**2, 0.0))


def pt_at_fixed_mass(e, eta, mass):
    """pT of a jet of energy e, pseudorapidity eta and mass `mass`."""
    return np.sqrt(np.maximum(e**2 - mass**2, 0.0)) / np.cosh(eta)


def write_jets(raw, pt, e, outpath):
    """The events `raw` with their jets' pT and E replaced (flat, in file order); every other
    field as it was."""
    counts = ak.num(raw)
    jet_fields = {f: raw.jet[f] for f in ak.fields(raw.jet)}
    jet_fields["pt"] = ak.unflatten(pt, counts)
    jet_fields["E"] = ak.unflatten(e, counts)
    corrected = ak.zip({"jet": ak.zip(jet_fields), "gen": raw.gen, "event": raw.event, "deltaR": raw.deltaR})
    Path(outpath).parent.mkdir(parents=True, exist_ok=True)
    ak.to_parquet(corrected, outpath)
    print(f"Corrected parquet written: {outpath}")
    print(f"  {len(counts):,} events, {len(pt):,} jets")
