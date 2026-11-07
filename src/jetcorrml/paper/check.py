"""
Gather the paper's items from output/paper into one folder, in paper order, and
compare them with an earlier run's folder.

    python -m jetcorrml.paper.check --out paper_check [--baseline DIR] [--run FILE]

paper_check/NNN_<file> are the items (registry.items: figures, tables, numbers);
paper_check/figure_data/ and table_data/ hold the numbers behind them.  The
comparison is by those numbers where both runs have them: every array of a
figure's npz (and the picture without its creation time), every cell of a table's
csv.  An earlier run without them is compared by the numbers in the text of its
tables.  --run is the runner's record of the steps (which failed, how long they
took).  REPORT.txt says what was found and what differs.
"""

import argparse
import hashlib
import json
import re
import shutil
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from jetcorrml import paths
from jetcorrml.paper import tables
from jetcorrml.paper.registry import items

NUMTOK = re.compile(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")
PDF_NOISE = re.compile(rb"/(CreationDate|ModDate)\s*\((?:\\.|[^\\)])*\)|/ID\s*\[[^\]]*\]")
ITEM = re.compile(r"^\d{3}_")


def human(sec):
    return str(timedelta(seconds=int(sec)))


def picture_hash(p):
    """Checksum of a PDF without its creation time, so equal pictures compare equal."""
    try:
        return hashlib.md5(PDF_NOISE.sub(b"", Path(p).read_bytes())).hexdigest()
    except OSError:
        return None


def numbers_differ(new, old):
    """Compare the saved numbers of a figure in two runs: None if equal, else what differs."""
    with np.load(new, allow_pickle=False) as a, np.load(old, allow_pickle=False) as b:
        bk = {k: k for k in b.files if k != "__meta__"}
        ka, kb = set(a.files) - {"__meta__"}, set(bk)
        if ka != kb:
            return f"different contents: {sorted(ka ^ kb)[:4]}"
        worst, where = 0.0, ""
        for k in sorted(ka):
            x, y = a[k], b[bk[k]]
            if x.shape != y.shape:
                return f"{k}: different size"
            if not np.array_equal(x, y, equal_nan=True):
                d = float(np.nanmax(np.abs(x.astype(float) - y.astype(float))))
                if d >= worst:
                    worst, where = d, k
        return None if not where else f"largest change {worst:.3g} (in {where})"


def cells_differ(new, old):
    """Compare two tables' csv cell by cell: (cells compared, [differences])."""
    a, b = tables.read(new), tables.read(old)
    if len(a) != len(b) or (a and list(a[0]) != list(b[0])):
        return 0, [f"different shape: {len(b)} rows -> {len(a)} rows"]
    n, diffs = 0, []
    for i, (ra, rb) in enumerate(zip(a, b)):
        for k in ra:
            n += 1
            if ra[k] != rb[k]:
                diffs.append(f"row {i + 1} {k}: {rb[k]} -> {ra[k]}")
    return n, diffs


def table_cells(text, plain=False):
    """{(section, row label): [cells]} for every line of a table's text that carries numbers.
    Only for an earlier run that kept no csv of its tables.  plain: the text is a printout
    with punctuation around its numbers ("0.95, 0.04", "modes95=1"), which is taken away first."""
    rows, sec, sub = {}, "", ""
    for ln in (text or "").splitlines():
        raw = ln.strip()
        if raw.startswith("##"):
            sec, sub = raw, ""
            continue
        if plain:
            raw = re.sub(r"[,{}:=]", " ", raw)
        toks = raw.replace("&", " ").replace("\\\\", " ").replace("$\\pm$", " ").split()
        if not toks:
            continue
        isnum = [bool(NUMTOK.fullmatch(t)) or t in ("--", "---") for t in toks]
        if not any(isnum):
            if len(toks) == 1 and not toks[0].startswith(("\\", "%", "#")):
                sub = toks[0]
            continue
        k = isnum.index(True)
        label = " ".join(toks[:k]) if k else toks[0]
        cells = [t for t, n in zip(toks[k if k else 1:], isnum[k if k else 1:]) if n]
        if not cells:
            continue
        key, n = (sec, sub, label), 1
        while key in rows:
            n += 1
            key = (sec, sub, f"{label} #{n}")
        rows[key] = cells
    return rows


def text_numbers_differ(new, old, plain=False):
    """Compare the numbers two versions of a table's text have in common: (numbers compared,
    numbers that differ, largest change, examples, rows only the old one has, rows only this
    one has: their numbers are not compared).  A row whose number of cells changed differs."""
    a = table_cells(Path(new).read_text(errors="replace"), plain)
    b = table_cells(Path(old).read_text(errors="replace"), plain)
    shared = diff = 0
    worst, examples = 0.0, []
    for key in a:
        if key not in b:
            continue
        if len(a[key]) != len(b[key]):
            diff += 1
            if len(examples) < 4:
                examples.append(f"{' / '.join(k for k in key if k)}: {len(b[key])} numbers -> {len(a[key])}")
        for i, (x, y) in enumerate(zip(a[key], b[key])):
            if not (NUMTOK.fullmatch(x) and NUMTOK.fullmatch(y)):
                continue
            shared += 1
            if x != y:
                diff += 1
                worst = max(worst, abs(float(x) - float(y)))
                if len(examples) < 4:
                    examples.append(f"{' / '.join(k for k in key if k)} [col {i + 1}]: {y} -> {x}")
    return shared, diff, worst, examples, len([k for k in b if k not in a]), [k for k in a if k not in b]


def collect(out, record):
    """Copy every item and its numbers into `out`; returns one entry per item."""
    writer = {f: st for st in record.get("steps", []) for f in st.get("writes", [])}
    report = []
    for idx, it in enumerate(items(), 1):
        src = Path(paths.PAPER) / it.path
        flat = it.path.replace("/", "_")
        r = {"idx": idx, "item": it, "flat": flat, "dest": None, "numbers": None,
             "verdict": "-", "detail": []}
        step = writer.get(str(src))
        if src.is_file():                          # the runner removes a step's files before it runs
            r["dest"] = out / f"{idx:03d}_{flat}"
            shutil.copy2(src, r["dest"])
            r["status"] = f"collected from {src}"
            if it.kind == "figure":
                sub, name = Path(it.path).parent.name, Path(it.path).stem
                npz = Path(paths.paper_data(f"{sub}/{name}"))
                folder, numbers = "figure_data", f"{sub}_{name}.npz"
            else:
                npz = src.with_suffix(".csv")
                folder, numbers = "table_data", npz.name
            if npz.is_file():
                (out / folder).mkdir(exist_ok=True)
                r["numbers"] = out / folder / numbers
                shutil.copy2(npz, r["numbers"])
        elif step and step.get("rc", 0) != 0:
            r["status"] = f"NOTHING: step {step['sid']} failed"
        else:
            r["status"] = "NOT FOUND"
        report.append(r)
    return report


def compare(report, baseline):
    """Set each entry's verdict against the earlier run; returns that run's items no longer made."""
    if not baseline.is_dir():
        for r in report:
            r["verdict"] = "MISSING" if r["dest"] is None else "new"
        return []
    earlier = {}
    for q in sorted(baseline.glob("[0-9][0-9][0-9]_*")):
        earlier.setdefault(ITEM.sub("", q.name), q)
    data = {}
    for q in sorted(baseline.glob("*_data/*")):
        data.setdefault(f"{q.parent.name}/{q.name}", q)
    for r in report:
        if r["dest"] is None:
            r["verdict"] = "MISSING"
            continue
        old = earlier.get(r["flat"])
        if old is None:
            r["verdict"], r["detail"] = "NOT COMPARED", ["the earlier run has no such item"]
            continue
        mine = r["numbers"] and f"{r['numbers'].parent.name}/{r['numbers'].name}"
        old_numbers = data.get(mine) if mine else None
        if r["item"].kind == "figure":
            same_picture = picture_hash(old) == picture_hash(r["dest"])
            if (r["numbers"] is None) != (old_numbers is None):
                r["verdict"] = "DIFFERENT" if not same_picture else "NOT COMPARED"
                r["detail"] = [("this run" if r["numbers"] is None else "the earlier run")
                               + " saved no numbers: only the pictures compared, "
                               + ("they differ" if not same_picture else "they are the same")]
                continue
            try:
                diff = (numbers_differ(r["numbers"], old_numbers)
                        if r["numbers"] is not None else None)
            except Exception as e:                      # noqa: BLE001 - unreadable numbers are a difference
                diff = f"could not compare the numbers: {type(e).__name__}: {e}"
            r["verdict"] = "same" if diff is None and same_picture else "DIFFERENT"
            if r["verdict"] != "same":
                r["detail"] = [diff or "numbers identical; the drawing changed"]
        elif r["numbers"] is not None and old_numbers is not None:
            n, diffs = cells_differ(r["numbers"], old_numbers)
            r["verdict"] = "same" if not diffs else "DIFFERENT"
            r["detail"] = [f"all {n} cells identical"] if not diffs else diffs[:4]
        else:
            shared, diff, worst, ex, gone, unpaired = text_numbers_differ(r["dest"], old,
                                                                          r["item"].kind == "numbers")
            if shared == 0:
                r["verdict"] = "NO SHARED NUMBERS"
            elif diff == 0:
                r["verdict"] = "same" if not unpaired else "PARTLY COMPARED"
                r["detail"] = [f"all {shared} shared numbers identical"
                               + (f" ({gone} earlier rows no longer made)" if gone else "")]
                if unpaired:
                    r["detail"].append(f"{len(unpaired)} rows of this run not in the earlier one, not compared: "
                                       + "; ".join(" / ".join(k for k in key if k) for key in unpaired[:3]))
            else:
                r["verdict"] = "DIFFERENT"
                r["detail"] = [f"{diff} of {shared} shared numbers differ, largest change {worst:.4g}"] + ex
    mine = {r["flat"] for r in report}
    return sorted(q.name for name, q in earlier.items() if name not in mine)


def write_report(out, report, gone, baseline, record):
    count = lambda v: sum(1 for r in report if r["verdict"] == v)
    look = [r for r in report if r["verdict"] in ("DIFFERENT", "MISSING", "NO SHARED NUMBERS",
                                                  "NOT COMPARED", "PARTLY COMPARED")]
    steps = record.get("steps", [])
    failed = [st for st in steps if st.get("rc", 0) != 0]
    t0, t1 = record.get("started"), record.get("finished")
    L = ["paper run", "=" * 70, ""]
    if t0:
        L += [f"started  : {datetime.fromtimestamp(t0):%Y-%m-%d %H:%M}",
              f"finished : {datetime.fromtimestamp(t1):%Y-%m-%d %H:%M}   ({human(t1 - t0)})"]
    L += [f"tree     : {Path.cwd()}", *record.get("notes", []),
          f"compared with : {baseline if baseline.is_dir() else 'nothing (no --baseline)'}", "",
          f"paper items collected : {sum(1 for r in report if r['dest'] is not None)} of {len(report)}",
          f"  same as before      : {count('same')}",
          f"  new                 : {count('new')}",
          f"  DIFFERENT           : {count('DIFFERENT') + count('NO SHARED NUMBERS')}",
          f"  MISSING             : {count('MISSING')}",
          f"  NOT COMPARED        : {count('NOT COMPARED') + count('PARTLY COMPARED')}",
          f"steps failed          : {len(failed)}", "",
          "Figures are compared by their saved numbers (figure_data) and their picture",
          "without the creation time; tables by every cell (table_data), or, against a",
          "run without table_data, by the numbers of the rows both runs have.  NOT",
          "COMPARED: the earlier run has nothing to compare an item (or some of its",
          "rows) with; that is not a pass.", ""]
    if gone:
        L += [f"REMOVED: {len(gone)} items of {baseline} not made by this run (not in the paper)", "-" * 70]
        L += [f"  {g}" for g in gone] + [""]
    if look:
        L += ["THESE ARE THE ONES TO LOOK AT", "-" * 70]
        for r in look:
            L.append(f"  {r['idx']:>3}  {r['verdict']:<18} {r['item'].label:<10} {r['item'].path}   [{r['status']}]")
            L += [f"         {d}" for d in r["detail"]]
        L.append("")
    if failed:
        L += ["FAILED STEPS", "-" * 70]
        L += [f"  {st['sid']}   {st.get('err', '')}   log: {st.get('log', '')}" for st in failed] + [""]
    L += ["ITEMS, in paper order", "-" * 70, f"{'#':<5}{'paper':<11}{'kind':<9}{'vs before':<19}file"]
    for r in report:
        L.append(f"{r['idx']:<5}{r['item'].label:<11}{r['item'].kind:<9}{r['verdict']:<19}{r['item'].path}")
        L += [f"{'':<44}{d}" for d in r["detail"]]
    timed = [st for st in steps if st.get("seconds", 0) >= 30] + record.get("groups", [])
    if timed:
        L += ["", "TIME", "-" * 70]
        for st in sorted(timed, key=lambda st: -st["seconds"]):
            L.append(f"  {human(st['seconds']):>9}  {st['sid']}")
        L.append("  (steps under 30 s not listed; jobs in a group overlap, so their times add up")
        L.append("   to more than the group's wall time)")
    L += ["", "Not made here: Fig. 1 (the strategy), Table I (the injection settings) and Fig. 12",
          "(the sub-jet drawing)."]
    (out / "REPORT.txt").write_text("\n".join(L) + "\n")
    return count, look


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="where the items go (emptied first)")
    ap.add_argument("--baseline", default=None, help="an earlier run's folder of items to compare with")
    ap.add_argument("--run", default=None, help="the runner's record of the steps (json)")
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    record = json.loads(Path(a.run).read_text()) if a.run and Path(a.run).is_file() else {}
    baseline = Path(a.baseline) if a.baseline else out / ".no_baseline"
    report = collect(out, record)
    gone = compare(report, baseline)
    count, look = write_report(out, report, gone, baseline, record)
    print(f"collected {sum(1 for r in report if r['dest'] is not None)} of {len(report)} -> {out}")
    print(f"same {count('same')}   new {count('new')}   "
          f"DIFFERENT {count('DIFFERENT') + count('NO SHARED NUMBERS')}   MISSING {count('MISSING')}   "
          f"NOT COMPARED {count('NOT COMPARED') + count('PARTLY COMPARED')}")
    print(f"report: {out / 'REPORT.txt'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
