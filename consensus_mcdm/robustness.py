"""Rank-robustness checks: rank correlations, weight sensitivity, leave-one-alternative-out."""
from __future__ import annotations
import itertools
import numpy as np
from scipy.stats import spearmanr, kendalltau
from .mcdm import METHODS


def ws_coefficient(r1, r2):
    """WS rank-similarity coefficient (Salabun & Urbaniak 2020): weights top ranks more."""
    r1 = np.asarray(r1, dtype=float); r2 = np.asarray(r2, dtype=float)
    return float(1 - np.sum(2.0 ** (-r1) * np.abs(r1 - r2) / np.maximum(np.abs(r1 - 1), np.abs(r1 - len(r1)))))


def rank_agreement(ranks: dict):
    names = list(ranks)
    out = []
    for a, b in itertools.combinations(names, 2):
        rho = spearmanr(ranks[a], ranks[b]).correlation
        tau = kendalltau(ranks[a], ranks[b]).correlation
        out.append({"A": a, "B": b, "spearman": float(rho), "kendall": float(tau), "WS": ws_coefficient(ranks[a], ranks[b])})
    return out


def leave_one_out(X, w, types, names, method="TOPSIS"):
    """Rank reversal test: remove each alternative in turn and check whether the relative order of
    the remaining alternatives changes (Belton & Gear 1983)."""
    fn = METHODS[method]
    base = fn(X, w, types)["rank"]
    m = len(names)
    rows = []
    for i in range(m):
        keep = [j for j in range(m) if j != i]
        r = fn(X[keep], w, types)["rank"]
        base_sub = base[keep]
        # relative order preserved?
        order_base = np.argsort(base_sub); order_new = np.argsort(r)
        reversal = not np.array_equal(order_base, order_new)
        rows.append({"removed": names[i], "reversal": bool(reversal), "new_rank": dict(zip([names[j] for j in keep], r.tolist()))})
    n_rev = sum(r["reversal"] for r in rows)
    return {"rows": rows, "reversals": n_rev, "share": n_rev / m}


def weight_perturbation(X, w, types, names, method="TOPSIS", deltas=(0.1, 0.2, 0.3, 0.5), rng=None, n_trials=200):
    """Random multiplicative perturbation of weights: share of trials in which the top alternative
    changes and mean Spearman with the base ranking."""
    rng = rng or np.random.default_rng(0)
    fn = METHODS[method]
    base = fn(X, w, types)["rank"]
    top = int(np.argmax(base == 1))
    rows = []
    for d in deltas:
        same_top = 0; rhos = []
        for _ in range(n_trials):
            wp = w * (1 + rng.uniform(-d, d, size=len(w)))
            wp = np.maximum(wp, 1e-6); wp /= wp.sum()
            r = fn(X, wp, types)["rank"]
            same_top += int(r[top] == 1)
            rhos.append(spearmanr(base, r).correlation)
        rows.append({"delta": d, "top_unchanged": same_top / n_trials, "mean_spearman": float(np.nanmean(rhos))})
    return rows


def single_weight_sweep(X, w, types, names, method="TOPSIS", grid=np.linspace(0.0, 0.5, 11)):
    """For each criterion, set its weight to g (others rescaled) and record the rank-1 alternative."""
    fn = METHODS[method]
    k = len(w)
    out = np.empty((k, len(grid)), dtype=object)
    for j in range(k):
        for gi, g in enumerate(grid):
            wp = w.copy()
            rest = 1 - wp[j]
            wp = wp * ((1 - g) / rest) if rest > 0 else wp
            wp[j] = g
            r = fn(X, wp, types)["rank"]
            out[j, gi] = names[int(np.argmax(r == 1))]
    return out
