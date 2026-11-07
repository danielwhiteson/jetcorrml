"""
Make the paper's figures: one style, from saved numbers.

Run from the top of the pipeline folder, after the pipeline has run:

    uv run python scripts/figures.py              # compute the numbers, then draw
    uv run python scripts/figures.py --draw-only  # redraw from the saved numbers (seconds)
    uv run python scripts/figures.py --list       # the figures it makes
    uv run python scripts/figures.py --only insitu/ band   # only figures whose folder/name contains one of these

--mini reads the outputs of the reduced run (run_paper.py --mini); --prefix
names the raw dijet files the run read (<prefix>*.pq, default the paper's).

Nothing is trained or applied here.  "Compute" reads the files the pipeline
has written and saves the numbers that are drawn:

    output/paper/data/<dir>/<name>.npz         the numbers of one figure
    output/paper/figures/<dir>/<name>.pdf      the figure, <dir> as in the paper's figures/

The style (font, text size, colours, the rules) is src/jetcorrml/paper/style.py.
"""

import argparse
import sys
import time

from jetcorrml import paths, settings
from jetcorrml.paper import inputs, style
from jetcorrml.paper.registry import figures, function


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--draw-only", action="store_true", help="redraw from the saved numbers")
    ap.add_argument("--only", nargs="+", default=[], help="only figures whose folder/name contains one of these")
    ap.add_argument("--list", action="store_true", help="list the figures and stop")
    ap.add_argument("--mini", action="store_true", help="the outputs of the reduced run of run_paper.py --mini")
    ap.add_argument("--prefix", default=settings.SAMPLE_PREFIX, help="the raw dijet files are <prefix>*.pq")
    args = ap.parse_args()
    inputs.RUN, inputs.PREFIX = settings.get(args.mini), args.prefix

    figs = [f for f in figures() if not args.only or any(s in f"{f.subdir}/{f.name}" for s in args.only)]
    if args.list:
        for f in figs:
            print(f"figures/{f.subdir}/{f.name}.pdf")
        return 0
    if not figs:
        print("no figure of that name; see --list")
        return 1

    style.apply()
    failed = []
    t0 = time.time()
    for f in figs:
        t = time.time()
        try:
            if not args.draw_only:
                function(f.compute)(f.key, **f.args)
            style.apply()                           # a pipeline script read by compute may have set its own
            out = function(f.draw)(f.key, f.subdir, **f.dargs)
            print(f"  {f.key:<32} {time.time() - t:5.1f} s   {out}", flush=True)
        except Exception as e:                      # one figure failing must not stop the others
            failed.append(f.key)
            print(f"  {f.key:<32} FAILED: {type(e).__name__}: {e}", flush=True)
    print(f"{len(figs) - len(failed)} of {len(figs)} figures made in {time.time() - t0:.0f} s; "
          f"numbers in {paths.PAPER}/data, figures in {paths.PAPER}/figures")
    if failed:
        print("FAILED: " + ", ".join(failed))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
