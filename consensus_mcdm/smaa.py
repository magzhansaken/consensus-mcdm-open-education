"""SMAA-2 (Lahdelma & Salminen 2001): rank acceptability indices under weight (and value) uncertainty.

b_i^r = probability that alternative i obtains rank r when weights are drawn from a distribution
(uniform on the simplex = no preference information; or Dirichlet concentrated around a central
weight vector) and criterion values are drawn from their empirical (seed) distribution.
Central weight vector w_i^c and confidence factor p_i^c are computed for rank 1.
The aggregation rule is any of the crisp methods in consensus_mcdm.mcdm (default TOPSIS); the classic
SMAA-2 uses an additive utility, which corresponds to the SAW rule here.
"""
from __future__ import annotations
import numpy as np
from .mcdm import METHODS, saw


def sample_weights(rng, k, n, center=None, concentration=None):
    if center is None or concentration is None:
        return rng.dirichlet(np.ones(k), size=n)
    alpha = np.maximum(np.asarray(center) * concentration, 1e-3)
    return rng.dirichlet(alpha, size=n)


def smaa2(X_samples, types, method="TOPSIS", n_weights=10000, seed=0, center=None, concentration=None):
    """X_samples: array (S, m, k) of decision matrices (one per simulation seed) or (m, k).
    Returns rank acceptability (m x m), central weights for rank 1 (m x k), confidence factors (m)."""
    rng = np.random.default_rng(seed)
    X_samples = np.asarray(X_samples, dtype=float)
    if X_samples.ndim == 2:
        X_samples = X_samples[None, :, :]
    S, m, k = X_samples.shape
    fn = METHODS.get(method, saw) if method != "SAW" else saw
    W = sample_weights(rng, k, n_weights, center, concentration)
    acc = np.zeros((m, m))
    wsum = np.zeros((m, k)); wcount = np.zeros(m)
    for t in range(n_weights):
        X = X_samples[rng.integers(S)]
        r = fn(X, W[t], types)["rank"]
        acc[np.arange(m), r - 1] += 1
        win = np.argmax(r == 1)
        wsum[win] += W[t]; wcount[win] += 1
    acc /= n_weights
    central = np.where(wcount[:, None] > 0, wsum / np.maximum(wcount[:, None], 1), np.nan)
    # confidence factor: probability of rank 1 when the central weights are used (over value samples)
    conf = np.zeros(m)
    for i in range(m):
        if wcount[i] == 0:
            continue
        wins = 0
        for s in range(S):
            r = fn(X_samples[s], central[i], types)["rank"]
            wins += int(r[i] == 1)
        conf[i] = wins / S
    return {"acceptability": acc, "central_weights": central, "confidence": conf, "n_weights": n_weights, "method": method}


def holistic_acceptability(acc):
    """Lahdelma & Salminen's holistic index with linearly decreasing meta-weights alpha_r = (m-r)/(m-1)."""
    m = acc.shape[0]
    alpha = (m - np.arange(1, m + 1)) / (m - 1)
    return acc @ alpha
