"""Every choice the paper's results depend on, defined once.

The runner, every step and the paper figures read their settings from here.
Two profiles: ``get()`` the paper run, ``get(mini=True)`` the reduced run of
``run_paper.py --mini`` (a check of the code: the same steps and code paths, few
epochs and retrainings; its numbers mean nothing).

Datasets are defined by their settings; their names are derived from those
and never parsed back.  Standard library only: the runner needs no environment.
"""

from __future__ import annotations   # the runner imports this with the system python3 (3.8 and later)

from dataclasses import dataclass, replace

# ─────────────────────────── the samples ───────────────────────────
SAMPLE_PREFIX = "jj"          # the dijet files are <prefix>*.pq (run_paper.py --prefix for another sample)
MATCH_DR = 0.3                # reco-gen matching distance of the dijet sample, as written by the production
LARGE_R_MATCH_DR = 0.4         # the same for the large-R sample
CENTRAL_ETA = 0.8             # |eta| of the central region: the reference of the in-situ step


# ─────────────────────────── injection (Sec. IV A, V A, VI, App. A-C) ───────────────────────────
@dataclass(frozen=True)
class PileupTerms:
    """alpha = (a0 + a1 ln pT)(1 + a2 |eta|) per (N_PV - 1), beta likewise (b) per mu'."""
    a0: float = 0.0
    a1: float = 0.0
    a2: float = 0.0
    b0: float = 0.0
    b1: float = 0.0
    b2: float = 0.0


@dataclass(frozen=True)
class AbsoluteDistortion:
    """R(E, |eta|) = r0 (1 - r1 exp(-E/pt0)) (1 - r2 max(|eta| - eta0, 0)^2)."""
    r0: float
    r1: float
    pt0: float
    r2: float
    eta0: float


@dataclass(frozen=True)
class NonLinearTerms:
    """gamma (N_PV - 1)^2 (App. B) and kappa (N_PV - 1)(mu - <mu>) (App. C), on with epsilon = 1."""
    epsilon: float = 0.0
    g0: float = 0.0
    g1: float = 0.0
    g2: float = 0.0
    kappa: float = 0.0


@dataclass(frozen=True)
class EtaDensity:
    """App. A: rho A (f - 1) added, f - 1 = depth * sigmoid((|eta| - etac) / width)."""
    depth: float = 0.0
    etac: float = 2.0
    width: float = 0.15


@dataclass(frozen=True)
class Injection:
    pileup: PileupTerms = PileupTerms()
    absolute: AbsoluteDistortion | None = None    # None: no absolute-scale distortion
    mu: tuple | None = None                       # (lo, hi, rho_muN) of the mock mu'; None: mu of the file
    nonlinear: NonLinearTerms = NonLinearTerms()
    eta_density: EtaDensity = EtaDensity()
    noise: float = 2.0                            # GeV: Gaussian noise on each jet's pile-up shift
    seed: int = 42


@dataclass(frozen=True)
class LargeRInjection:
    """Large-R jets: the absolute distortion with a step at the crack instead of the forward roll-off,
    applied to the two sub-jets and the rest of the jet separately."""
    absolute: AbsoluteDistortion
    eta_crack: float
    c_depth: float
    c_width: float
    pileup: PileupTerms
    mu: tuple                     # (lo, hi, rho_muN) of the mock mu'
    noise: float = 2.0
    seed: int = 42


ABSOLUTE = AbsoluteDistortion(r0=0.90, r1=0.65, pt0=35.0, r2=0.08, eta0=0.7)
PILEUP_LO = PileupTerms(a0=0.4, a1=0.08, a2=0.15, b0=0.15, b1=0.04, b2=0.10)      # Table I, lo-pT
PILEUP_HI = PileupTerms(a0=0.4, a1=0.40, a2=0.15, b0=0.15, b1=0.20, b2=0.10)      # Table I, hi-pT
NONLINEAR = NonLinearTerms(epsilon=1.0, g0=-0.0022, g1=-0.0010, g2=0.15, kappa=0.0)
INTERACTION = NonLinearTerms(epsilon=1.0, g0=0.0, g1=0.0, g2=0.0, kappa=0.010)
MU_RANGE = (1.0, 120.0)       # the uncorrelated part of mu' is uniform in this range

RHO_MUN = (0.0, 0.1, 0.3, 0.5, 0.7, 0.8, 1.0)       # the seven correlations of the pile-up scan
REGIMES = ("lo", "hi")


def rho_code(rho):
    """The two-digit code of rho_muN in dataset names: 0.5 -> "05"."""
    return f"{round(rho * 10):02d}"


@dataclass(frozen=True)
class Dataset:
    """One injected dijet dataset."""
    name: str
    injection: Injection
    rho: float | None = None      # rho_muN of the pile-up scan
    regime: str | None = None     # "lo" or "hi" (the pT dependence of the injected pile-up)
    variant: str | None = None    # "nonlinear" (App. B) or "interaction" (App. C)


def pileup_dataset(rho, regime="lo", variant=None):
    """A dataset of the pile-up scan ("lo_rho05"); variant "nonlinear" or "interaction" adds the
    term of App. B or C ("nonlinear_rho05")."""
    terms = {None: NonLinearTerms(), "nonlinear": NONLINEAR, "interaction": INTERACTION}[variant]
    return Dataset(name=f"{variant or regime}_rho{rho_code(rho)}",
                   injection=Injection(pileup=PILEUP_LO if regime == "lo" else PILEUP_HI,
                                       absolute=ABSOLUTE, mu=(*MU_RANGE, rho), nonlinear=terms),
                   rho=rho, regime=regime, variant=variant)


PILEUP_SCAN = [pileup_dataset(r, regime) for regime in REGIMES for r in RHO_MUN]    # the 14 datasets
VARIANT_RHO = 0.5             # rho_muN of the non-linear and interaction studies (Figs. 22, 23)
VARIANTS = [pileup_dataset(VARIANT_RHO, "lo", v) for v in ("nonlinear", "interaction")]
DENSITY = Dataset(name="density",                  # App. A: no absolute distortion, no injected dependence
                  injection=Injection(eta_density=EtaDensity(depth=0.5, etac=2.0, width=0.15)))
DATASETS = {d.name: d for d in PILEUP_SCAN + VARIANTS + [DENSITY]}

LARGE_R = LargeRInjection(absolute=AbsoluteDistortion(r0=0.90, r1=0.30, pt0=300.0, r2=0.0, eta0=0.7),
                         eta_crack=1.37, c_depth=0.45, c_width=0.05,
                         pileup=PILEUP_LO, mu=(*MU_RANGE, 0.0))
LARGE_R_DATASET = "large_r"

PILEUP_EXAMPLE = pileup_dataset(0.1).name         # the example dataset of the pile-up and absolute sections
INJECTION_EXAMPLE = pileup_dataset(0.5).name      # the injected sample of Fig. 5


# ─────────────────────────── trainings ───────────────────────────
@dataclass(frozen=True)
class Training:
    epochs: int
    batch_size: int
    lr: float
    val_frac: float
    lam: float = 0.0                  # pile-up: weight of the softplus penalty on negative corrections
    early_stop_patience: int = 0
    sampler_min_count: int = 1000     # pile-up: minimum occupancy of an (N_PV, mu) sampling cell


@dataclass(frozen=True)
class Network:
    n_hidden: int
    width: int
    dropout: float = 0.0


@dataclass(frozen=True)
class NNStep:
    seed: int
    training: Training
    model: Network
    eta_bins: tuple = ()
    pt_bins: tuple = ()


# the |eta| and pT bins of the pile-up step: the NN's binned loss terms and the parametric fit
PILEUP_ETA_BINS = (0.0, 0.3, 0.8, 1.2, 2.1, 2.8, 3.5)
PILEUP_PT_BINS = (20.0, 30.0, 45.0, 60.0, 80.0, 110.0, 160.0, 200.0)
LARGE_R_PILEUP_ETA_BINS = (0.0, 0.3, 0.8, 1.2, 1.8, 2.1, 2.5)
LARGE_R_PILEUP_PT_BINS = (100.0, 150.0, 200.0, 300.0, 450.0, 600.0, 800.0, 1000.0)
# the |eta| bins of the forward pile-up density study (Table V and Fig. 21): ten of 0.25 up to 2.5,
# so that a change over a short range of |eta| can be seen (the density steps up at |eta| = 2)
DENSITY_ETA_BINS = tuple(0.25 * i for i in range(11))
# |eta| slices of the absolute scale's inversion tables (both methods)
ABSOLUTE_ETA_EDGES = (0, 0.3, 0.8, 1.1, 1.37, 1.52, 2.1, 2.5, 3.2, 4.5)
RECO_PT_MIN = 20.0            # GeV: the reconstructed-pT threshold of the samples' jets
CLOSURE_E_MIN = 50.0          # GeV: the absolute scale is judged on jets with E_gen above this,
ABSOLUTE_E_BUFFER = 1.05      # and above 1.05 x the pT threshold at the jet's eta (where it stops biasing R)


@dataclass(frozen=True)
class Insitu:
    """The in-situ step (Sec. VII), both methods.

    Each method runs in passes.  Every pass corrects the original data; it takes only the
    choice of its dijet pairs from the pass before:
      pass 1    both jets above pair_pt_min in uncorrected data, pair pT = their average ("both")
      pass 2    the inner (smaller-|eta|) jet above the cut after the pass-1 correction,
                pair pT = that jet's pT ("inner")
      pass 3+   the inner jet as in pass 2, pair pT = the average of the two ("inner-avg": pass 2
                files the pair under the inner jet's pT, and forward jets came out 3-4% high)
    The parametric method stops after two passes: further passes made it worse on every distortion.
    """
    dataset: str                  # the dataset the divisions are made of
    seed: int                     # NN: network initialisation and held-out split
    epochs: int
    width: int = 64               # NN: width of the two hidden layers of each network
    lr: float = 5e-4
    weight_decay: float = 1e-4
    val_frac: float = 0.15        # NN: jets held out of the reference term and of the scaler fit
    nn_passes: int = 4
    parametric_passes: int = 2
    pair_pt_min: float = 25.0     # dijet selection of both methods: jet pT [GeV]
    pair_dphi_min: float = 2.5    # back-to-back [rad]
    pair_pt3_frac: float = 0.25   # third-jet veto pT3 / pT_avg
    rho_dists: tuple = ("rhoeta",)    # distortions whose NN scale also takes rho as an input

    def passes(self, method):
        """The passes of method "nn" or "parametric"."""
        return self.nn_passes if method == "nn" else self.parametric_passes

    @staticmethod
    def pair_rule(k):
        """how pass k chooses its pairs ("both" needs no earlier pass)"""
        return "both" if k == 1 else ("inner" if k == 2 else "inner-avg")


@dataclass(frozen=True)
class Distortion:
    """The data half's distortion f (Sec. VII A)."""
    form: str = "linear"
    d0: float = 0.0
    d1: float = 0.0
    d2: float = 0.0
    d3: float = 0.0


DISTORTIONS = {
    "nom": Distortion(),
    "linear": Distortion("linear", 0.1, 0.08, 0.03),
    "quad": Distortion("quadratic", 0.1, 0.08, 0.03),
    "threshold": Distortion("threshold", 0.1, 0.08, 0.03),
    "expo": Distortion("expo", 0.1, 0.08, 0.03),
    "rhoeta": Distortion("rhoeta", 0.1, 0.08, 0.03, 0.05),
}
SPLIT_FRACTION = 0.5          # the share of the events that becomes the simulation


@dataclass(frozen=True)
class Band:
    """The nuisance-parameter band (Sec. VIII) and its check against the divisions (Fig. 19)."""
    dist: str = "expo"            # the band of the paper
    retrainings: int = 30
    seed_base: int = 1000         # retraining b trains with seed seed_base + b
    divisions: int = 20           # divisions the band is checked against (the first are Table IV's)
    dists: tuple = ("nom", "expo", "quad", "rhoeta")    # distortions of that check
    keep: int = 3                 # leading parameters of the reduced model of the figures


@dataclass(frozen=True)
class Settings:
    pileup_nn: NNStep
    large_r_pileup_nn: NNStep
    absolute_nn: NNStep
    large_r_absolute_nn: NNStep
    insitu: Insitu
    band: Band
    divisions: tuple = (0, 1, 2, 3, 4)    # MC/data divisions: Table IV is their mean and spread
    dists: tuple = ("nom", "linear", "quad", "threshold", "expo", "rhoeta")
    figure_dists: tuple = ("expo", "threshold", "rhoeta")     # the distortions of Figs. 16, 17
    resolution_rho: float = 0.1           # rho_muN of the resolution figure (Fig. 11)


PAPER = Settings(
    pileup_nn=NNStep(seed=42, training=Training(epochs=150, batch_size=4096, lr=3.0e-4, val_frac=0.15,
                                                lam=0.1, early_stop_patience=50, sampler_min_count=1000),
                     model=Network(n_hidden=4, width=256),
                     eta_bins=PILEUP_ETA_BINS, pt_bins=PILEUP_PT_BINS),
    large_r_pileup_nn=NNStep(seed=42, training=Training(epochs=20, batch_size=4096, lr=1.0e-4, val_frac=0.15,
                                                       lam=0.1, early_stop_patience=50, sampler_min_count=1000),
                            model=Network(n_hidden=4, width=256),
                            eta_bins=LARGE_R_PILEUP_ETA_BINS, pt_bins=LARGE_R_PILEUP_PT_BINS),
    absolute_nn=NNStep(seed=42, training=Training(epochs=100, batch_size=4096, lr=2.0e-3, val_frac=0.15),
                       model=Network(n_hidden=9, width=384, dropout=0.0)),
    large_r_absolute_nn=NNStep(seed=42, training=Training(epochs=240, batch_size=4096, lr=1.0e-4, val_frac=0.15),
                              model=Network(n_hidden=5, width=192, dropout=0.05)),
    insitu=Insitu(dataset=pileup_dataset(0.5).name, seed=123, epochs=100),
    band=Band(),
)

MINI_EPOCHS = 3               # 2 can leave OneCycleLR with a warm-up of zero steps


def _mini(s):
    """The reduced run: 3 epochs everywhere, divisions 0 and 1 for Table IV and 3 for the band
    check (it needs three of each), 3 retrainings."""
    ep = lambda step: replace(step, training=replace(step.training, epochs=MINI_EPOCHS))
    return replace(s, pileup_nn=ep(s.pileup_nn), large_r_pileup_nn=ep(s.large_r_pileup_nn),
                   absolute_nn=ep(s.absolute_nn), large_r_absolute_nn=ep(s.large_r_absolute_nn),
                   insitu=replace(s.insitu, epochs=MINI_EPOCHS),
                   band=replace(s.band, retrainings=3, divisions=3), divisions=(0, 1))


MINI = _mini(PAPER)


def get(mini=False):
    """The settings of the paper run, or of the reduced run."""
    return MINI if mini else PAPER
