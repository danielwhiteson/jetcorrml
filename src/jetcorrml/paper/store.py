"""
The numbers behind each paper figure, saved and read back.

One file per figure, output/paper/data/<subdir>/<name>.npz: the arrays that are
drawn, plus where they came from (the files read, with their time stamps).
Drawing reads only this file.  A figure is known by its key, "<subdir>/<name>"
(jetcorrml.paper.registry).
"""

import json
import time
from pathlib import Path

import numpy as np

from jetcorrml import paths


def data_path(key):
    return Path(paths.paper_data(key))


def fig_path(subdir, key):
    return Path(paths.paper_figure(subdir, key.rsplit("/", 1)[-1]))


def pad(a, n):
    """A curve as a fixed-length row: unused places hold NaN."""
    out = np.full(n, np.nan)
    a = np.asarray(a, dtype=float)
    out[: len(a)] = a
    return out


def save(name, arrays, sources=(), notes=None):
    """Save the numbers of one figure.  `sources`: the files they were computed from."""
    src = []
    for p in sources:
        p = Path(p)
        src.append({"file": str(p), "modified": time.strftime("%Y-%m-%d %H:%M:%S",
                                                             time.localtime(p.stat().st_mtime)),
                    "bytes": p.stat().st_size})
    meta = {"figure": name, "sources": src, "notes": notes or {}}
    path = data_path(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, __meta__=np.array(json.dumps(meta)), **{k: np.asarray(v) for k, v in arrays.items()})
    return path


def load(name):
    """The numbers of one figure, as a dict; d['__meta__'] says where they came from."""
    path = data_path(name)
    if not path.is_file():
        raise FileNotFoundError(f"{path}: no saved numbers for '{name}' — run without --draw-only first")
    with np.load(path, allow_pickle=False) as z:
        d = {k: z[k] for k in z.files}
    d["__meta__"] = json.loads(str(d["__meta__"]))
    return d
