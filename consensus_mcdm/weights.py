"""Objective criterion weights: Shannon entropy, CRITIC, and their combination; weight scenarios."""
from __future__ import annotations
import numpy as np


def _norm_benefit(X, types):
    """Min-max normalisation to [0,1] with 1 = best (used by CRITIC)."""
    X = np.asarray(X, dtype=float); types = np.asarray(types)
    lo, hi = X.min(axis=0), X.max(axis=0)
    span = np.where(hi > lo, hi - lo, 1.0)
    R = np.where(types > 0, (X - lo) / span, (hi - X) / span)
    return R


def entropy_weights(X, types=None, eps=1e-12):
    """Shannon entropy weights on sum-normalised values (Zeleny 1982; Wang & Lee 2009).
    Cost criteria are first turned into benefit by reciprocal so that 'more information' means
    the same thing for every column."""
    X = np.asarray(X, dtype=float)
    if types is not None:
        types = np.asarray(types)
        X = np.where(types > 0, X, 1.0 / np.maximum(X, eps))
    P = X / X.sum(axis=0)
    m = X.shape[0]
    E = -(P * np.log(np.maximum(P, eps))).sum(axis=0) / np.log(m)
    d = 1 - E
    if d.sum() <= 0:
        return np.ones(X.shape[1]) / X.shape[1]
    return d / d.sum()


def critic_weights(X, types):
    """CRITIC (Diakoulaki, Mavrotas & Papayannakis 1995): C_j = sigma_j * sum_k (1 - r_jk)."""
    R = _norm_benefit(X, types)
    sigma = R.std(axis=0, ddof=1)
    if R.shape[0] < 2:
        return np.ones(R.shape[1]) / R.shape[1]
    corr = np.corrcoef(R, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0)
    C = sigma * (1 - corr).sum(axis=1)
    if C.sum() <= 0:
        return np.ones(R.shape[1]) / R.shape[1]
    return C / C.sum()


def combined_weights(w_entropy, w_critic, mode="geometric"):
    we, wc = np.asarray(w_entropy), np.asarray(w_critic)
    if mode == "geometric":
        w = np.sqrt(we * wc)
    elif mode == "arithmetic":
        w = 0.5 * (we + wc)
    else:
        raise ValueError(mode)
    return w / w.sum()


# Scenario weights of Tashatov et al., IoT 2026 7(2) 45, Table 21 (columns labelled Scenario 1..6 in
# the article; the text calls them Scenarios 2-7). Order of criteria C1..C11 as in that article:
# TPS, Latency, Finalization, Power, Computing, Storage, Fault tolerance, Attack resistance,
# Decentralization, Manageability, Operational cost.
IOT45_SCENARIOS = {
    "S2 performance":   [0.05983744, 0.20195136, 0.1121952, 0.0560976, 0.10097568, 0.02991872, 0.1707315, 0.1707315, 0.0292683, 0.05268294, 0.01560976],
    "S3 resources":     [0.02991872, 0.10097568, 0.0560976, 0.1121952, 0.20195136, 0.05983744, 0.1707315, 0.1707315, 0.0292683, 0.05268294, 0.01560976],
    "S4 security":      [0.02591552, 0.08746488, 0.0485916, 0.0485916, 0.08746488, 0.02591552, 0.2957745, 0.2957745, 0.0253521, 0.04563378, 0.01352112],
    "S5 governance":    [0.03285712, 0.11089278, 0.0616071, 0.0616071, 0.11089278, 0.03285712, 0.1875, 0.1875, 0.0642858, 0.11571444, 0.03428576],
    "S6 security-high": [0.02, 0.0675, 0.0375, 0.0375, 0.0675, 0.02, 0.3423915, 0.3423915, 0.0195651, 0.03521718, 0.01043472],
    "S7 balanced-gov":  [0.02967744, 0.10016136, 0.0556452, 0.0556452, 0.10016136, 0.02967744, 0.169355, 0.169355, 0.0870966, 0.15677388, 0.04645152],
}
IOT45_TYPES = [1, -1, 1, -1, -1, -1, 1, 1, 1, 1, -1]


def scenario_set(X, types):
    """All weight vectors used in the paper."""
    we = entropy_weights(X, types)
    wc = critic_weights(X, types)
    k = len(types)
    S = {
        "Equal": np.ones(k) / k,
        "Entropy": we,
        "CRITIC": wc,
        "Entropy x CRITIC (base)": combined_weights(we, wc, "geometric"),
    }
    for name, w in IOT45_SCENARIOS.items():
        S["IoT45 " + name] = np.asarray(w) / np.sum(w)
    return S
