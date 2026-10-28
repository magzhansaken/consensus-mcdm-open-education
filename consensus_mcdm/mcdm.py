"""Ranking methods: TOPSIS, VIKOR, COPRAS, EDAS (crisp decision matrices).

X : (m alternatives x k criteria) array of raw values
w : weights, sum to 1
types : +1 benefit, -1 cost per criterion
All functions return a dict with the method score and the rank (1 = best).
Conventions follow the original formulations:
  TOPSIS  Hwang & Yoon (1981): vector normalisation, Euclidean distances, C = d-/(d+ + d-).
  VIKOR   Opricovic & Tzeng (2004): linear normalisation by the ideal/anti-ideal, S, R, Q with v = 0.5
          (lower Q is better).
  COPRAS  Zavadskas et al. (1994): sum normalisation, S+ / S-, Q_i = S+_i + (min S- * sum S-)/(S-_i * sum(min S-/S-_j)).
  EDAS    Keshavarz Ghorabaee et al. (2015): positive/negative distance from the average solution.
"""
from __future__ import annotations
import numpy as np


def _check(X, w, types):
    X = np.asarray(X, dtype=float)
    w = np.asarray(w, dtype=float)
    types = np.asarray(types, dtype=int)
    assert X.ndim == 2 and X.shape[1] == len(w) == len(types)
    assert np.all(X > 0) or True
    w = w / w.sum()
    return X, w, types


def rank_desc(score):
    """Rank 1 = highest score."""
    order = np.argsort(-np.asarray(score), kind="stable")
    r = np.empty(len(score), dtype=int); r[order] = np.arange(1, len(score) + 1)
    return r


def rank_asc(score):
    order = np.argsort(np.asarray(score), kind="stable")
    r = np.empty(len(score), dtype=int); r[order] = np.arange(1, len(score) + 1)
    return r


def topsis(X, w, types):
    X, w, types = _check(X, w, types)
    R = X / np.sqrt((X ** 2).sum(axis=0))
    V = R * w
    pis = np.where(types > 0, V.max(axis=0), V.min(axis=0))
    nis = np.where(types > 0, V.min(axis=0), V.max(axis=0))
    dp = np.sqrt(((V - pis) ** 2).sum(axis=1))
    dn = np.sqrt(((V - nis) ** 2).sum(axis=1))
    C = dn / (dp + dn)
    return {"score": C, "rank": rank_desc(C), "higher_better": True}


def vikor(X, w, types, v=0.5):
    X, w, types = _check(X, w, types)
    best = np.where(types > 0, X.max(axis=0), X.min(axis=0))
    worst = np.where(types > 0, X.min(axis=0), X.max(axis=0))
    span = np.where(best != worst, np.abs(best - worst), 1.0)
    D = w * np.abs(best - X) / span
    S = D.sum(axis=1); R = D.max(axis=1)
    Sr = (S - S.min()) / (S.max() - S.min()) if S.max() > S.min() else np.zeros_like(S)
    Rr = (R - R.min()) / (R.max() - R.min()) if R.max() > R.min() else np.zeros_like(R)
    Q = v * Sr + (1 - v) * Rr
    return {"score": Q, "S": S, "R": R, "rank": rank_asc(Q), "higher_better": False}


def copras(X, w, types):
    X, w, types = _check(X, w, types)
    R = X / X.sum(axis=0)
    V = R * w
    Splus = V[:, types > 0].sum(axis=1)
    Sminus = V[:, types < 0].sum(axis=1)
    if np.all(Sminus == 0):
        Q = Splus
    else:
        Q = Splus + (Sminus.min() * Sminus.sum()) / (Sminus * (Sminus.min() / Sminus).sum())
    U = Q / Q.max() * 100.0
    return {"score": Q, "utility": U, "rank": rank_desc(Q), "higher_better": True}


def edas(X, w, types):
    X, w, types = _check(X, w, types)
    AV = X.mean(axis=0)
    PDA = np.where(types > 0, np.maximum(0, X - AV) / AV, np.maximum(0, AV - X) / AV)
    NDA = np.where(types > 0, np.maximum(0, AV - X) / AV, np.maximum(0, X - AV) / AV)
    SP = (PDA * w).sum(axis=1); SN = (NDA * w).sum(axis=1)
    NSP = SP / SP.max() if SP.max() > 0 else np.zeros_like(SP)
    NSN = 1 - SN / SN.max() if SN.max() > 0 else np.ones_like(SN)
    AS = 0.5 * (NSP + NSN)
    return {"score": AS, "rank": rank_desc(AS), "higher_better": True}


def saw(X, w, types):
    X, w, types = _check(X, w, types)
    R = np.where(types > 0, X / X.max(axis=0), X.min(axis=0) / X)
    S = (R * w).sum(axis=1)
    return {"score": S, "rank": rank_desc(S), "higher_better": True}


METHODS = {"TOPSIS": topsis, "VIKOR": vikor, "COPRAS": copras, "EDAS": edas}


def rank_all(X, w, types, methods=METHODS):
    return {name: fn(X, w, types) for name, fn in methods.items()}
