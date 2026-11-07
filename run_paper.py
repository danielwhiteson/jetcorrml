#!/usr/bin/env python3
"""
run_paper.py — make every figure and table of the paper, in this tree.

Needs (README, Requirements): any python3 3.8 or later to start this file (it uses
only the standard library), uv on the PATH (it builds the environment of the
programs from uv.lock the first time: Python 3.14 and the packages; internet
once), and the two samples.  --dry-run checks all of it in seconds.

    python3 run_paper.py --dry-run      # check the requirements, list the steps, run nothing
    python3 run_paper.py                # the real thing (about 1.5 hours)
    python3 run_paper.py --collect-only # gather and compare again, run nothing
    python3 run_paper.py --force a,b    # re-run the named steps
    python3 run_paper.py --only a,b     # run these steps and no others (a partial run)
    python3 run_paper.py --jobs 4       # how many independent jobs run at once (default 6)
    python3 run_paper.py --baseline DIR # compare every item with an earlier run's paper_check
    python3 run_paper.py --mini         # the reduced run (settings.MINI): checks the code, not physics

The inputs are the dijet sample (--raw, default inputs/raw: jj*.pq) and the
large-R sample (--large-r-raw, default inputs/large_r: wgamma*.pq).  They are
linked into data/ and only read; one sample per tree.

The steps are jetcorrml.pipeline's, each with the files it reads and writes;
every choice the results depend on is in src/jetcorrml/settings.py, which the
programs read themselves.  A step runs when its files to read exist; its files
to write are removed first, and it has failed if it does not write them all.
A step that ran before is skipped unless its command changed, a file it writes
is missing, or a file it reads is newer.  A failure does not stop the run.

Steps of one group run side by side: each job is its own process with its own
seed and its usual thread settings, so the numbers do not depend on how many run
at once.  Each step's log is written as it runs: paper_check_logs/<step>.log.

At the end jetcorrml.paper.check gathers the paper's items into paper_check/
(REPORT.txt first) and compares them with --baseline.
"""

import argparse
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path

if sys.version_info < (3, 8):
    sys.exit(f"run_paper.py needs python3 3.8 or later (this is {sys.version.split()[0]}); "
             "the programs themselves run in the environment uv builds")
TREE = Path(__file__).resolve().parent
sys.path.insert(0, str(TREE / "src"))
from jetcorrml import paths, settings  # noqa: E402  (standard library only: the runner needs no environment)
from jetcorrml.pipeline import UV, Pipeline  # noqa: E402

OUT = TREE / "paper_check"                    # --out; logs go to <out>_logs
RAW = TREE / "inputs" / "raw"                 # --raw: the dijet sample
LARGE_R_RAW = TREE / "inputs" / "large_r"     # --large-r-raw: the large-R sample
JOBS = 6
MARK = TREE / ".done"                         # a marker per step that ran: when, and its commands
ALWAYS = {"figures"}                          # quick, and must follow whatever was remade before it
PRINT_LOCK = threading.Lock()


def human(sec):
    return str(timedelta(seconds=int(sec)))


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--collect-only", action="store_true")
    ap.add_argument("--force", default="", help="comma list of step ids to re-run")
    ap.add_argument("--only", default="",
                    help="comma list of step ids: run these and no others (a partial run). Everything is "
                         "still collected and compared; what the other steps make is reported as MISSING.")
    ap.add_argument("--jobs", type=int, default=JOBS)
    ap.add_argument("--out", default=str(OUT), help="where the collected results go")
    ap.add_argument("--baseline", default=None, help="an earlier run's paper_check to compare with")
    ap.add_argument("--prefix", default=settings.SAMPLE_PREFIX,
                    help=f"prefix of the raw dijet files, e.g. jj for jj*.pq (default {settings.SAMPLE_PREFIX})")
    ap.add_argument("--raw", default=None,
                    help="the directory of the dijet sample (<prefix>*.pq); default: the one this tree "
                         "is linked to, else inputs/raw")
    ap.add_argument("--large-r-raw", default=None,
                    help=f"the directory of the large-R sample ({paths.LARGE_R_PREFIX}*.pq); default: the "
                         f"one this tree is linked to, else inputs/large_r")
    ap.add_argument("--step-timeout-h", type=float, default=10.0)
    ap.add_argument("--mini", action="store_true",
                    help="the reduced run (jetcorrml.settings.MINI): every step and code path, few epochs and "
                         "retrainings; its numbers mean nothing.  For checking the code, in a tree of its own.")
    return ap.parse_args()


class Log:
    def __init__(self, logs, quiet):
        self.file = None if quiet else logs / "RUN.log"

    def __call__(self, msg):
        line = f"[{datetime.now():%H:%M:%S}] {msg}"
        with PRINT_LOCK:
            print(line, flush=True)
            if self.file:
                with open(self.file, "a") as fh:
                    fh.write(line + "\n")


def sample_of(directory, pattern):
    """A record of a sample: its directory, how many files, and a checksum of their names,
    sizes and times."""
    files = sorted(Path(directory).resolve().glob(pattern))
    h = hashlib.md5()
    for q in files:
        st = q.stat()
        h.update(f"{q.name}:{st.st_size}:{st.st_mtime_ns}".encode())
    return {"dir": str(Path(directory).resolve()), "pattern": pattern, "files": len(files), "md5": h.hexdigest()}


def samples(args):
    """(name, its folder, the link in data/, the pattern of its files) for the two samples."""
    for name, given, link, default, pattern in (
            ("dijet", args.raw, paths.RAW, RAW, f"{args.prefix}*.pq"),
            ("large-R", args.large_r_raw, paths.LARGE_R_RAW, LARGE_R_RAW, f"{paths.LARGE_R_PREFIX}*.pq")):
        dst = TREE / link
        src = Path(given).expanduser().resolve() if given else (dst.resolve() if dst.is_symlink() else default)
        yield name, src, dst, pattern


def requirements(args, say):
    """Check and print what the run needs (README, Requirements); True if all of it is there."""
    ok = True
    say(f"python3 : {sys.version.split()[0]}  ({sys.executable})  ok: 3.8 or later is enough")
    uv = shutil.which("uv")
    if uv:
        try:
            version = subprocess.run(["uv", "--version"], capture_output=True, text=True, timeout=60).stdout.strip()
        except Exception as e:                            # noqa: BLE001
            version = f"(could not ask its version: {e})"
        say(f"uv      : {uv}  {version}  ok: it builds the environment from uv.lock")
        want = (TREE / ".python-version").read_text().strip()
        try:
            found = subprocess.run(["uv", "python", "find"], cwd=TREE, capture_output=True, text=True,
                                   timeout=60).stdout.strip()
        except Exception:                                 # noqa: BLE001
            found = ""
        say(f"Python {want} : " + (f"{found}  ok: the programs run on it (.python-version)" if found else
            "not installed yet: uv downloads it the first time (internet needed)"))
    else:
        ok = False
        say("uv      : NOT FOUND on the PATH. Install it (https://docs.astral.sh/uv/) and open a new terminal.")
    for name, src, _, pattern in samples(args):
        n = len(list(src.glob(pattern)))
        ok = ok and n > 0
        say(f"{name + ' sample':<15}: {src}  " + (f"{n} files {pattern}" if n else
                                                   f"NOT FOUND: no {pattern} files there (see --raw, --large-r-raw)"))
    if args.baseline:
        b = Path(args.baseline).expanduser()
        found = (b / "REPORT.txt").is_file()
        ok = ok and found
        say(f"baseline       : {b}  " + ("ok" if found else "NOT FOUND (no REPORT.txt there)"))
    return ok


def link_inputs(args, say, partial):
    """Link the two samples into data/ (read-only use); stop if this tree ran on others."""
    notes, raw_dirs = [], []
    for name, src, dst, pattern in samples(args):
        link = dst.relative_to(TREE)
        if not any(src.glob(pattern)):
            if partial:
                say(f"NOTE: no {name} sample ({src}/{pattern}); a partial run goes on without it")
                continue
            sys.exit(f"no {name} sample: {src} holds no {pattern} (see --raw, --large-r-raw, --prefix)")
        if dst.is_symlink() and dst.resolve() != src:
            sys.exit(f"{dst} points at {dst.resolve()}, not {src}: one sample per tree. Use a fresh copy of the code.")
        if not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.symlink_to(src)
            say(f"linked {link} -> {src}")
        rec, now = MARK / f"sample_{name.lower().replace('-', '_')}.json", sample_of(src, pattern)
        if rec.is_file() and json.loads(rec.read_text()) != now:
            was = json.loads(rec.read_text())
            sys.exit(f"the {name} sample changed since this tree ran on it:\n  then: {was}\n  now : {now}\n"
                     f"Use a fresh copy of the code.")
        rec.write_text(json.dumps(now, sort_keys=True) + "\n")
        notes.append(f"{name} sample : {src}   ({now['files']} files {pattern}, set {now['md5'][:8]})")
        say(notes[-1])
        raw_dirs.append(src)
    return notes, raw_dirs


def check_environment(say):
    """The programs' environment (jetcorrml.environment): this tree's code, the Python of
    .python-version, and every package at its uv.lock version.  Stops the run if not."""
    say("checking the Python environment (the first time, uv builds it: a few minutes)")
    try:
        r = subprocess.run(UV + ["-m", "jetcorrml.environment"],
                           cwd=TREE, capture_output=True, text=True, timeout=3600)
    except Exception as e:                                # noqa: BLE001
        sys.exit(f"could not check the environment: {type(e).__name__}: {e}")
    out = {}
    for x in r.stdout.splitlines():
        key, _, rest = x.partition(" ")
        out.setdefault(key, []).append(rest)
    if r.returncode != 0 or "IMPORTED" not in out or "PYTHON" not in out:
        sys.exit("could not check the environment:\n" + r.stdout[-2000:] + r.stderr[-3000:])
    where = Path(out["IMPORTED"][-1]).resolve()
    if TREE not in where.parents:
        sys.exit(f"WRONG CODE WOULD RUN: `uv run python` in {TREE} imports\n  {where}\n"
                 f"It must import this tree's src/.  Nothing was run.  (Delete {TREE}/.venv and run again.)")
    say("code check: imports come from this tree")
    python, want = out["PYTHON"][-1], (TREE / ".python-version").read_text().strip()
    if python.split(".")[:2] != want.split(".")[:2]:
        sys.exit(f"WRONG PYTHON: the programs would run on Python {python}, not {want} (.python-version).  "
                 f"Nothing was run.  (Delete {TREE}/.venv and run again.)")
    packages = [x.split() for x in out.get("PACKAGE", [])]
    bad = [f"  {n}: {have} installed, uv.lock has {' '.join(w)}" for n, have, *w, verdict in packages
           if verdict != "ok"]
    if bad:
        sys.exit("WRONG PACKAGES: the environment is not the one of uv.lock.  Nothing was run.\n"
                 + "\n".join(bad) + f"\n(Delete {TREE}/.venv and run again.)")
    key = {n: have for n, have, *_ in packages}
    line = (f"environment : Python {python}, " + ", ".join(f"{n} {key[n]}" for n in
            ("numpy", "scipy", "torch", "scikit-learn", "pandas", "pyarrow", "awkward", "matplotlib") if n in key)
            + f"; all {len(packages)} packages as in uv.lock")
    say(line)
    return line


def fingerprint(raw_dirs):
    h = hashlib.md5()
    for d in raw_dirs:
        for q in sorted(x for x in d.rglob("*") if x.is_file()):
            st = q.stat()
            h.update(f"{q}:{st.st_mtime_ns}:{st.st_size}".encode())
    return h.hexdigest()


class Runner:
    """Runs the steps: one at a time, or a group's side by side."""

    def __init__(self, args, logs, say):
        self.args, self.logs, self.say = args, logs, say
        self.forced = {x for x in args.force.split(",") if x}
        self.env = dict(os.environ, PYTHONUNBUFFERED="1")
        self.record = []

    def _wait(self, p):
        """the return code of process p, killed (with its children) after the step time-out"""
        try:
            return p.wait(timeout=self.args.step_timeout_h * 3600), ""
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGKILL)
            p.wait()
            return 124, f"timed out after {self.args.step_timeout_h}h"

    def _done(self, st, cmdline):
        """ran before with the same commands, wrote everything, and nothing it reads changed since"""
        marker = MARK / st.sid
        if st.sid in ALWAYS or st.sid in self.forced or not marker.is_file():
            return False
        if marker.read_text().split("\n", 1)[-1] != cmdline:
            return False
        if not all((TREE / f).exists() for f in st.writes):
            return False
        t = marker.stat().st_mtime
        return all((TREE / f).stat().st_mtime <= t for f in st.reads if (TREE / f).exists())

    def _execute(self, st, log, env):
        """run the step's commands; (return code, error)"""
        if st.together:          # all at once, each into its own log; the step's log gathers them at the end
            parts = [self.logs / f"{st.sid}.{i}.log" for i in range(len(st.cmds))]
            procs = []
            for c, part in zip(st.cmds, parts):
                with open(part, "w") as fp:
                    fp.write(f"$ cd {TREE}\n$ {' '.join(c.argv)}\n\n")
                    fp.flush()
                    procs.append(subprocess.Popen(c.argv, cwd=TREE, stdout=fp, stderr=subprocess.STDOUT,
                                                  env=env, start_new_session=True))
            codes = [self._wait(p) for p in procs]
            with open(log, "w") as fo:
                for part in parts:
                    fo.write(part.read_text(errors="replace") + "\n")
                    part.unlink()
            bad = [(i, rc, e) for i, (rc, e) in enumerate(codes) if rc != 0]
            return (0, "") if not bad else (bad[0][1], f"command {bad[0][0] + 1} of {len(codes)} failed {bad[0][2]}".strip())
        with open(log, "w") as fo:
            for i, c in enumerate(st.cmds):
                fo.write(f"$ cd {TREE}\n$ {' '.join(c.argv)}\n\n")
                fo.flush()
                rc, err = self._wait(subprocess.Popen(c.argv, cwd=TREE, stdout=fo, stderr=subprocess.STDOUT,
                                                      env=env, start_new_session=True))
                if rc != 0:
                    return rc, (f"command {i + 1} of {len(st.cmds)} failed {err}".strip()
                                if len(st.cmds) > 1 else err)
        return 0, ""

    def step(self, st):
        log = self.logs / f"{st.sid}.log"
        cmdline = "\n".join(" ".join(c.argv) for c in st.cmds)
        entry = {"sid": st.sid, "writes": st.writes, "log": str(log), "rc": 0, "err": "", "seconds": 0.0}
        if self._done(st, cmdline):
            self.say(f"  {st.sid}: already done, skipping")
            return entry
        missing = [f for f in st.reads if not (TREE / f).exists()]
        if missing:
            entry.update(rc=2, err=f"not run: {missing[0]} is missing"
                         + (f" (and {len(missing) - 1} more)" if len(missing) > 1 else ""))
            for f in st.writes:                   # nothing an earlier run left may pass for this run's
                (TREE / f).unlink(missing_ok=True)
            (MARK / st.sid).unlink(missing_ok=True)
            self.say(f"  FAILED {st.sid}   {entry['err']}")
            return entry
        self.say(f"  start  {st.sid}" + (f"   {st.note}" if st.note else ""))
        t0 = time.time()
        for f in st.writes:                       # whatever is there afterwards, this run wrote
            (TREE / f).unlink(missing_ok=True)
        try:
            rc, err = self._execute(st, log, dict(self.env, **(st.env or {})))
        except Exception as e:                    # noqa: BLE001
            rc, err = 1, f"{type(e).__name__}: {e}"
        if rc == 0:
            unwritten = [f for f in st.writes if not (TREE / f).exists()]
            if unwritten:
                rc, err = 3, (f"did not write {unwritten[0]}"
                              + (f" (and {len(unwritten) - 1} more)" if len(unwritten) > 1 else ""))
        entry.update(rc=rc, err=err, seconds=time.time() - t0)
        if rc == 0:
            (MARK / st.sid).write_text(datetime.now().isoformat() + "\n" + cmdline)
            self.say(f"  ok     {st.sid}   in {human(entry['seconds'])}")
        else:
            (MARK / st.sid).unlink(missing_ok=True)
            self.say(f"  FAILED {st.sid}   rc={rc} {err} after {human(entry['seconds'])} -> {log}")
        return entry

    def run(self, steps):
        """Every step in order; the steps of one group side by side.  Returns the record."""
        jobs, groups, i = max(1, self.args.jobs), [], 0
        while i < len(steps):
            group = steps[i].group
            j = i + 1
            while group is not None and j < len(steps) and steps[j].group == group:
                j += 1
            batch = steps[i:j]
            if group is None:
                self.record.append(self.step(batch[0]))
            else:
                self.say(f"--- {len(batch)} independent jobs [{group}], {min(jobs, len(batch))} at a time")
                t0, left = time.time(), len(batch)
                with ThreadPoolExecutor(max_workers=jobs) as pool:
                    for f in as_completed([pool.submit(self.step, st) for st in batch]):
                        self.record.append(f.result())
                        left -= 1
                        if left and left % 5 == 0:
                            self.say(f"         [{group}] {left} of {len(batch)} still to finish")
                groups.append({"sid": f"[{group}] wall time", "seconds": time.time() - t0})
                self.say(f"--- [{group}] done in {human(time.time() - t0)}")
            i = j
        return groups


def dry_run(steps, jobs, say):
    say(f"tree: {TREE}")
    say(f"--- {len(steps)} steps; jobs in the same [group] run side by side, {jobs} at a time")
    seen = {}
    for st in steps:                            # jobs of one kind: show the first, count the rest
        key = (st.group, re.sub(r"\d+", "", st.note)) if st.group else st.sid
        seen.setdefault(key, []).append(st)
    for same in seen.values():
        st = same[0]
        more = f"   (+{len(same) - 1} more like it)" if len(same) > 1 else ""
        cmds = f"   ({len(st.cmds)} commands{', together' if st.together else ''})" if len(st.cmds) > 1 else ""
        say(f"  {st.sid:<36} {('[' + st.group + ']') if st.group else '':<15}"
            f"{' '.join(st.cmds[0].argv[:6])}" + cmds + ("  # " + st.note if st.note else "") + more)
    say("dry run: nothing executed")


def main():
    args = parse_args()
    out = Path(args.out).expanduser()
    logs = out.with_name(out.name + "_logs")
    say = Log(logs, quiet=args.dry_run)
    if not args.dry_run:
        logs.mkdir(parents=True, exist_ok=True)
    only = [x for x in args.only.split(",") if x]
    pipeline = Pipeline(args.mini)

    def steps():
        S = pipeline.steps(args.jobs, args.prefix)
        unknown = [x for x in only if x not in {st.sid for st in S}]
        if unknown:
            sys.exit(f"--only: no step named {', '.join(unknown)} (see --dry-run for the names)")
        return [st for st in S if not only or st.sid in only]

    if args.dry_run:
        ready = requirements(args, say)
        dry_run(steps(), args.jobs, say)
        say("ready to run" if ready else "NOT READY: see the lines marked NOT above")
        return 0 if ready else 1

    t0 = time.time()
    say("=" * 66)
    say(f"paper run, started {datetime.now():%Y-%m-%d %H:%M}")
    say(f"tree    : {TREE}")
    if only:
        say(f"PARTIAL : only {', '.join(only)}")
    say(f"results : {out}")
    say(f"logs    : {logs}   (each step writes its log while it runs)")
    say("=" * 66)
    if not requirements(args, say) and not (args.collect_only or only):
        sys.exit("not run: something it needs is missing (the lines marked NOT above; README, Requirements)")
    MARK.mkdir(exist_ok=True)

    notes = [f"PARTIAL RUN  : only {', '.join(only)}; the items of other steps are MISSING below"] if only else []
    groups, record, raw_ok = [], [], None
    if not args.collect_only:
        sample_notes, raw_dirs = link_inputs(args, say, partial=bool(only))
        notes += sample_notes
        notes.append(check_environment(say))
        fp = fingerprint(raw_dirs)
        say("fingerprinted the raw inputs (they must not change)")
        todo = steps()                                  # the raw inputs are linked now
        for s in {x for x in args.force.split(",") if x}:
            (MARK / s).unlink(missing_ok=True)
        say(f"--- {len(todo)} steps, up to {args.jobs} side by side")
        runner = Runner(args, logs, say)
        groups = runner.run(todo)
        record = runner.record
        raw_ok = fingerprint(raw_dirs) == fp
        say("raw inputs unchanged: " + ("YES" if raw_ok else "NO — SOMETHING WROTE THERE"))
        notes.append("raw inputs unchanged : " + ("YES" if raw_ok else "NO -- SOMETHING WROTE THERE"))
    run_file = logs / "run.json"
    if record:
        run_file.write_text(json.dumps({"started": t0, "finished": time.time(), "notes": notes,
                                        "steps": record, "groups": groups}, indent=1) + "\n")
    say("")
    say(f"--- collecting into {out}")
    cmd = UV + ["-m", "jetcorrml.paper.check", "--out", str(out), "--run", str(run_file)]
    if args.baseline:
        cmd += ["--baseline", str(Path(args.baseline).expanduser().resolve())]
    rc = subprocess.run(cmd, cwd=TREE).returncode
    failed = [e["sid"] for e in record if e["rc"] != 0]
    say("=" * 66)
    if failed:
        say(f"{len(failed)} step(s) failed: {', '.join(failed[:6])}")
    if rc != 0:
        say(f"COLLECTING FAILED (rc={rc}): see the lines above")
    say(f"total time {human(time.time() - t0)}")
    say(f"report: {out / 'REPORT.txt'}")
    say("=" * 66)
    return 0 if rc == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
