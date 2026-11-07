"""
The run's steps, in order: each a list of commands with the files they read and
write.  run_paper.py runs them; this module only describes them (standard library
only, so the runner needs no environment).

A step's commands run one after the other (or all at once if ``together``).
Steps of one ``group`` are independent of each other and run side by side.  A
step runs only if the files it reads exist, and has failed if it does not write
the files it declares.
"""

from collections import namedtuple

from jetcorrml import paths, settings
from jetcorrml.paper.registry import figures, items

UV = ["uv", "run", "--locked", "python"]   # --locked: stop if uv.lock does not match pyproject.toml, never re-resolve

Cmd = namedtuple("Cmd", "argv reads writes")


class Step(namedtuple("Step", "sid cmds note group together env", defaults=("", None, False, None))):
    @property
    def writes(self):
        return sorted({f for c in self.cmds for f in c.writes})

    @property
    def reads(self):
        """what the step reads that it does not write itself"""
        return sorted({f for c in self.cmds for f in c.reads} - set(self.writes))


class Pipeline:
    """The commands of one run: the paper's (settings.PAPER) or the reduced one (settings.MINI)."""

    def __init__(self, mini=False):
        self.run = settings.get(mini)
        self.mini = ["--mini"] if mini else []
        self.insitu = self.run.insitu
        self.ds = self.insitu.dataset

    # ── small-R chain ─────────────────────────────────────────────────────────
    def inject(self, raw, ds):
        out = paths.injected(ds)
        return Cmd(UV + ["-m", "steps.inject", "-i", *raw, "-o", out, "--dataset", ds], raw, [out])

    def pileup(self, method, ds):
        """the pile-up correction by `method` ("nn" or "parametric")"""
        src, out = paths.injected(ds), (paths.nn_pileup(ds) if method == "nn" else paths.parametric_pileup(ds))
        return Cmd(UV + ["-m", f"steps.pileup_{method}", "-i", src, "--output", out]
                   + (self.mini if method == "nn" else []), [src], [out])

    def absolute(self, method, ds, pileup="nn"):
        """the absolute scale by `method`, after the `pileup` ("nn" or "parametric") pile-up
        correction (the NN absolute scale only follows the NN pile-up correction)"""
        src = paths.nn_pileup(ds) if pileup == "nn" else paths.parametric_pileup(ds)
        if method == "nn":
            out, factor = paths.nn_absolute(ds), paths.nn_absolute_factor(ds)
            return Cmd(UV + ["-m", "steps.absolute_nn", "-i", src, "--output", out, "--factor-out", factor]
                       + self.mini, [src], [out, factor])
        out = paths.parametric_absolute(ds, pileup)
        return Cmd(UV + ["-m", "steps.absolute_parametric", "-i", src, "--output", out], [src], [out])

    # ── large-R chain ─────────────────────────────────────────────────────────
    def inject_large_r(self, raw):
        out = paths.large_r_injected()
        return Cmd(UV + ["-m", "steps.inject_large_r", "-i", *raw, "-o", out], raw, [out])

    def pileup_large_r(self):
        src, out = paths.large_r_injected(), paths.large_r_pileup()
        return Cmd(UV + ["-m", "steps.pileup_nn_large_r", "-i", src, "--output", out] + self.mini, [src], [out])

    def absolute_large_r(self, subjets):
        src = paths.large_r_pileup()
        out, model = paths.large_r_absolute(subjets), paths.large_r_model(subjets)
        return Cmd(UV + ["-m", "steps.absolute_nn_large_r", "-i", src, "--output", out, "--model-out", model]
                   + (["--use-subjet-eta"] if subjets else []) + self.mini, [src], [out, model])

    # ── divisions and in situ ─────────────────────────────────────────────────
    def divide(self, d, s):
        src = paths.nn_absolute(self.ds)
        mc, data = paths.division(self.ds, d, s, "MC"), paths.division(self.ds, d, s, "DATA")
        return Cmd(UV + ["-m", "steps.divide", "-i", src, "--output-mc", mc, "--output-data", data,
                         "--dist", d, "--division", str(s)], [src], [mc, data])

    def insitu_out(self, d, s, method, kind, k=None):
        """the output of pass k of `method` on division s; the last pass (or k None) is the result"""
        last = k is None or k >= self.insitu.passes(method)
        return paths.insitu(self.ds, d, s, method, kind, k=None if last else k)

    def pairs(self, d, s, method, k):
        """how pass k of `method` chooses its pairs (on the corrected jets of the pass before),
        and the file that needs"""
        if k == 1:
            return [], []
        sel = self.insitu_out(d, s, method, "DATA", k - 1)
        return ["--pair-selection", self.insitu.pair_rule(k), "--selection-pt", sel], [sel]

    def insitu_pass(self, method, d, s, k):
        """pass k of the in-situ correction by `method` on division s.  Passes before the last
        are kept: their corrected data jets are what the next pass chooses its pairs on."""
        mc, data = paths.division(self.ds, d, s, "MC"), paths.division(self.ds, d, s, "DATA")
        out = [self.insitu_out(d, s, method, kind, k) for kind in ("MC", "DATA")]
        how, sel = self.pairs(d, s, method, k)
        return Cmd(UV + ["-m", f"steps.insitu_{method}", "--mc", mc, "--data", data]
                   + (["--dist", d] if method == "nn" else [])
                   + ["--output-mc", out[0], "--output-data", out[1]] + how
                   + (self.mini if method == "nn" else []), [mc, data] + sel, out)

    def band(self, jobs, d):
        """the band's retrainings on division 0 of distortion d; for the band's own distortion
        also the band's numbers.  The retrainings choose their pairs like the nominal last pass,
        on the nominal pass before it."""
        band = self.run.band
        how, sel = self.pairs(d, 0, "nn", self.insitu.passes("nn"))
        mc, data = paths.division(self.ds, d, 0, "MC"), paths.division(self.ds, d, 0, "DATA")
        writes = [f for b in range(band.retrainings)
                  for f in (paths.retraining(self.ds, d, b, "MC"), paths.retraining(self.ds, d, b, "DATA"),
                            paths.retraining_grid(self.ds, d, b))]
        if d == band.dist:
            writes += [paths.paper_table("band_modes", "csv"), paths.paper_table("band_modes", "txt")]
        return Cmd(UV + ["-m", "steps.band", "--dist", d, "--jobs", str(jobs)]
                   + ([] if d == band.dist else ["--refits-only"]) + how + self.mini,
                   [mc, data] + sel, writes)

    # ── the paper's tables and figures ────────────────────────────────────────
    @staticmethod
    def table(name, reads, mini=()):
        return Cmd(UV + [f"scripts/table_{name}.py"] + list(mini), reads,
                   [paths.paper_table(name, "csv"), paths.paper_table(name, "tex")])

    def figures(self, prefix):
        """All figures, from whatever the steps above wrote (it reads too many files to list)."""
        writes = [f"{paths.PAPER}/{it.path}" for it in items() if it.kind == "figure"]
        writes += [paths.paper_data(f.key) for f in figures()]
        return Cmd(UV + ["scripts/figures.py", "--prefix", prefix] + self.mini, [], writes)

    # ── the run ───────────────────────────────────────────────────────────────
    def steps(self, jobs, prefix):
        """The run's steps, in order.  `prefix`: the raw dijet files are <prefix>*.pq."""
        S = []

        def A(sid, cmds, note="", group=None, together=False, env=None):
            S.append(Step(sid, cmds, note, group, together, env))

        run, insitu, band = self.run, self.insitu, self.run.band
        raw = paths.raw_dijet_files(prefix)
        scan = [d.name for d in settings.PILEUP_SCAN]                  # lo-pT, then hi-pT, in rho_muN order
        variants = [d.name for d in settings.VARIANTS]                  # non-linear, interaction
        density = settings.DENSITY.name

        A("inject", [self.inject(raw, ds) for ds in scan], "injection of pile-up and response")
        # the main NN pile-up trainings run all at once, with 4 threads each
        A("pileup_nn", [self.pileup("nn", ds) for ds in scan], "NN pile-up, 14 datasets at once",
          together=True, env={"OMP_NUM_THREADS": "4"})
        A("pileup_parametric", [self.pileup("parametric", ds) for ds in scan], "parametric pile-up")
        A("table_pileup", [self.table("pileup", [paths.nn_pileup(ds) for ds in scan]
                                      + [paths.parametric_pileup(ds) for ds in scan])], "Table II")

        # inputs of the jobs in the big group below
        for ds in variants:
            A(f"inject_{ds}", [self.inject(raw, ds)], f"{ds}: injection")
        A("inject_large_r", [self.inject_large_r(paths.raw_large_r_files())], "large-R jets: injection")
        A("pileup_nn_large_r", [self.pileup_large_r()], "large-R jets: NN pile-up")

        # the long trainings, side by side
        for ds in scan:
            A(f"absolute_nn_{ds}", [self.absolute("nn", ds)], "NN absolute scale", "train")
        for ds in variants:
            A(f"pileup_nn_{ds}", [self.pileup("nn", ds)], f"{ds}: NN pile-up", "train")
        A("density", [self.inject(raw, density), self.pileup("nn", density), self.pileup("parametric", density)],
          "forward pile-up density study", "train")
        A("absolute_nn_large_r", [self.absolute_large_r(subjets=False), self.absolute_large_r(subjets=True)],
          "large-R absolute scale, without and with sub-jets", "train")

        for ds in variants:
            A(f"pileup_parametric_{ds}", [self.pileup("parametric", ds)], f"{ds}: parametric pile-up")

        A("absolute_parametric", [self.absolute("parametric", ds, "nn") for ds in scan],
          "parametric absolute scale on NN pile-up: the chain of Table III")
        A("absolute_parametric_full",
          [self.absolute("parametric", settings.pileup_dataset(run.resolution_rho, regime).name, "parametric")
           for regime in settings.REGIMES],
          "parametric absolute scale on parametric pile-up: the full parametric chain of Fig. 11")
        A("table_absolute", [self.table("absolute", [paths.nn_absolute(ds) for ds in scan]
                                        + [paths.parametric_absolute(ds, "nn") for ds in scan])], "Table III")

        # the divisions of the sample: Table IV uses run.divisions for every distortion; the check
        # of the band (Fig. 19) uses band.divisions divisions of the band.dists distortions (NN only)
        extra = [(d, s) for s in range(band.divisions) if s not in run.divisions for d in band.dists]
        every = [(d, s) for s in run.divisions for d in run.dists]
        for d, s in every + extra:
            A(f"divide_{d}_{s}", [self.divide(d, s)], f"division {s} of the sample", "divide")
        pn, pp = insitu.passes("nn"), insitu.passes("parametric")
        for k in range(1, max(pn, pp) + 1):
            # pass k needs pass k-1: one group per pass
            grp = f"insitu_pass{k}"
            if k <= pn:
                for d, s in every + extra:
                    A(f"insitu_nn_{d}_{s}_pass{k}", [self.insitu_pass("nn", d, s, k)],
                      f"NN in situ, pass {k} of {pn} (division {s})", grp)
            if k <= pp:
                for d, s in every:
                    A(f"insitu_parametric_{d}_{s}_pass{k}", [self.insitu_pass("parametric", d, s, k)],
                      f"parametric in situ, pass {k} of {pp} (division {s})", grp)
        A("band", [self.band(jobs, band.dist)],
          f"{band.retrainings} retrainings ({band.dist}, division 0) and the numbers of Sec. VIII")
        for d in band.dists:
            if d != band.dist:
                A(f"band_{d}", [self.band(jobs, d)], f"{band.retrainings} retrainings ({d}, division 0)")

        A("table_insitu", [self.table("insitu", [f for d, s in every for m in ("nn", "parametric")
                                                 for f in (paths.division(self.ds, d, s, "MC"),
                                                           paths.division(self.ds, d, s, "DATA"),
                                                           self.insitu_out(d, s, m, "MC"),
                                                           self.insitu_out(d, s, m, "DATA"))],
                                      self.mini)],
          "Table IV (mean and spread over the divisions)")
        A("table_density", [self.table("density", [paths.injected(density), paths.nn_pileup(density),
                                                   paths.parametric_pileup(density)])], "Table V")
        # last: reads only files written above (no training)
        A("figures", [self.figures(prefix)], "the paper's figures, with their numbers (output/paper)")
        return S
