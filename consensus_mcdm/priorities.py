"""Institutional priorities versus objective weights (Section 4.12 of the article).

Entropy and CRITIC weights describe how strongly the criteria discriminate between the alternatives and how much they
overlap; they do not express what a university or a consortium values. This module asks how the recommendation
changes with the priorities of the decision-makers, using only ordinal preference information that a consortium can
state without numerical elicitation:
  group_orders      the 24 orders of the four criterion groups (performance, resources, security, architecture and
                    practice); for each order, SMAA-2 draws weights uniformly from the part of the simplex in which the
                    group weights follow that order (criteria share their group weight uniformly at random), and the
                    rank-order-centroid (ROC) weights of the order give a point ranking;
  security_sweep    the weight of the security group raised from 0 to 1 with the other groups kept in the proportions
                    of the base weights, to locate the switching point between the leaders.
"""
from __future__ import annotations
import itertools
import numpy as np
from .criteria import TYPES
from .mcdm import rank_all, METHODS

GROUPS = {"P": (0, 1, 2), "R": (3, 4, 5), "S": (6, 7), "A": (8, 9, 10)}
GROUP_NAMES = {"P": "performance", "R": "resources", "S": "security", "A": "architecture and practice"}


def roc(k):
    """Rank-order-centroid weights for k ranked items (Barron & Barrett 1996)."""
    return np.array([sum(1.0 / j for j in range(i, k + 1)) / k for i in range(1, k + 1)])


def weights_for_order(order, rng=None, roc_point=False):
    """Criterion weights whose group sums follow `order` (most important first)."""
    k = len(GROUPS)
    if roc_point:
        g = roc(k)
    else:
        g = np.sort(rng.dirichlet(np.ones(k)))[::-1]
    w = np.zeros(11)
    for gi, grp in enumerate(order):
        idx = GROUPS[grp]
        share = np.ones(len(idx)) / len(idx) if roc_point else rng.dirichlet(np.ones(len(idx)))
        w[list(idx)] = g[gi] * share
    return w


def _first(X, w, feas_idx):
    Xf = X[feas_idx]
    R = rank_all(Xf, w, TYPES)
    return {m: int(feas_idx[int(np.argmin(R[m]["rank"]))]) for m in R}


def group_orders(Xs, keys, feas=None, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    feas_idx = np.arange(len(keys)) if feas is None else np.where(feas)[0]
    Xm = Xs.mean(axis=0)
    out = {}
    for order in itertools.permutations("PRSA"):
        cnt = {m: {k: 0 for k in keys} for m in METHODS}
        for _ in range(n):
            w = weights_for_order(order, rng)
            X = Xs[rng.integers(Xs.shape[0])]
            for m, i in _first(X, w, feas_idx).items():
                cnt[m][keys[i]] += 1
        roc_first = {m: keys[i] for m, i in _first(Xm, weights_for_order(order, roc_point=True), feas_idx).items()}
        out[">".join(order)] = {"b1": {m: {k: v / n for k, v in d.items() if v} for m, d in cnt.items()}, "roc_first": roc_first}
    return out


def security_sweep(Xm, keys, base_w, feas=None, grid=np.linspace(0.0, 1.0, 41)):
    feas_idx = np.arange(len(keys)) if feas is None else np.where(feas)[0]
    s_idx = list(GROUPS["S"]); other = [j for j in range(11) if j not in s_idx]
    rows = []
    for g in grid:
        w = np.zeros(11)
        bs = base_w[s_idx] / base_w[s_idx].sum(); bo = base_w[other] / base_w[other].sum()
        w[s_idx] = g * bs; w[other] = (1 - g) * bo
        rows.append({"security_weight": float(g), "first": {m: keys[i] for m, i in _first(Xm, w, feas_idx).items()}})
    return rows
