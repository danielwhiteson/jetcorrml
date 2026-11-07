"""
What the NN steps share: the response network of the absolute scale (small-R and
large-R alike), the input standardisation, and the sampling weights of the
absolute-scale training.
"""

import numpy as np
import torch
import torch.nn as nn


class ResponseNet(nn.Module):
    """
    Learns the median jet energy response R(E_true, eta).

    Input  : n_features expanded features
    Hidden : n_hidden layers of `width` neurons, SiLU activations, residual skip
             connections from layer 2 onward, dropout after each activation
    Output : softplus(raw) / ln(2): R = 1 at initialisation, always > 0
    """

    def __init__(self, n_features, n_hidden, width, dropout):
        super().__init__()
        layers = [nn.Linear(n_features, width), nn.SiLU(), nn.Dropout(dropout)]
        for _ in range(n_hidden - 1):
            layers.append(nn.Linear(width, width))
            layers.append(nn.SiLU())
            layers.append(nn.Dropout(dropout))
        self.trunk = nn.ModuleList(layers)
        self.head = nn.Linear(width, 1)
        # output 0 at initialisation: softplus(0) / ln 2 = 1
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        h = self.trunk[0](x)  # Linear
        h = self.trunk[1](h)  # SiLU
        h = self.trunk[2](h)  # Dropout
        for i in range(3, len(self.trunk), 3):
            h_res = h
            h = self.trunk[i](h)  # Linear
            h = self.trunk[i + 1](h)  # SiLU
            h = self.trunk[i + 2](h)  # Dropout
            h = h + h_res  # residual skip
        raw = self.head(h)
        return torch.nn.functional.softplus(raw) / 0.6931471805599453


class Normaliser:
    """Zero-mean, unit-variance standardisation fitted on training data."""

    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, X):
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0) + 1e-8
        return self

    def transform(self, X):
        return (X - self.mean_) / self.std_

    def fit_transform(self, X):
        return self.fit(X).transform(X)


def flat_sampling_weights(log_e_gen, abs_eta, n_e_bins=25, n_eta_bins=12):
    """
    Per-jet sampling probabilities that partially flatten the (log E, |eta|)
    distribution: w ~ 1/sqrt(count) rather than 1/count, which would let the few
    jets of sparse cells (often outliers, at high E and forward) dominate; the
    square root still boosts high-E jets about tenfold over low-E ones.
    """
    e_edges = np.linspace(log_e_gen.min() - 1e-6, log_e_gen.max() + 1e-6, n_e_bins + 1)
    eta_edges = np.linspace(abs_eta.min() - 1e-6, abs_eta.max() + 1e-6, n_eta_bins + 1)

    ie = np.clip(np.digitize(log_e_gen, e_edges) - 1, 0, n_e_bins - 1)
    ieta = np.clip(np.digitize(abs_eta, eta_edges) - 1, 0, n_eta_bins - 1)
    cell = ie * n_eta_bins + ieta

    counts = np.bincount(cell, minlength=n_e_bins * n_eta_bins).astype(np.float64)
    counts = np.maximum(counts, 1.0)

    w = 1.0 / np.sqrt(counts[cell])
    w /= w.sum()
    return w.astype(np.float64)
