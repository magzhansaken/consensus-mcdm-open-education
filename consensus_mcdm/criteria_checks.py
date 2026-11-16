"""Checks of the criterion set and of the two documented ordinal criteria (Section 4.11 of the article).

  smaa_ordinal     SMAA-O treatment (Lahdelma, Miettinen & Salminen 2003): the levels 0-4 of C8 and C10 are replaced by
                   random cardinal values that preserve only their order, with weights re-estimated or drawn uniformly;
  perturb_scores   every documented score moved by -1, 0 or +1 (clipped to 0-4), exhaustively per criterion or jointly;
  scoring_rules    alternative rules for the two scores, the scores of the preceding framework, and their removal;
  leave_one_out    the ranking with each of the eleven criteria removed in turn;
  fault_proxies    alternative definitions of C7 (documented crash tolerance, a combination);
  scale_shift      the ordinal scales shifted (1-5) or stretched (0-10), to expose normalisation effects.
All functions screen first when a feasibility mask is given and re-estimate the entropy x CRITIC weights.
"""
from __future__ import annotations
import itertools
import numpy as np
from .criteria import TYPES, ATTACK, MANAGE
from .weights import scenario_set
from .mcdm import rank_all, METHODS

ORD = (7, 9)   # column indices of C8 attack resistance and C10 manageability
KEYS = ["raft", "smartbft", "ibft2", "qbft", "clique", "cometbft", "hotstuff"]

# property-level evidence for the documented scores (Table 5 of the article)
C8_PROPS = ("Byzantine safety, f < n/3", "deterministic finality", "proposer rotation", "built-in equivocation evidence")
C8_TABLE = {"raft": (0, 1, 0, 0), "smartbft": (1, 1, 1, 0), "ibft2": (1, 1, 1, 0), "qbft": (1, 1, 1, 0),
            "clique": (0, 0, 1, 0), "cometbft": (1, 1, 1, 1), "hotstuff": (1, 1, 1, 0)}
C10_PROPS = ("validator-set change without restart", "contract- or configuration-based validator selection",
             "metrics endpoint", "maintained production release line with load-testing tooling")
C10_TABLE = {"raft": (1, 1, 1, 1), "smartbft": (1, 1, 1, 1), "ibft2": (1, 0, 1, 1), "qbft": (1, 1, 1, 1),
             "clique": (1, 0, 1, 1), "cometbft": (1, 0, 1, 1), "hotstuff": (0, 0, 0, 0)}
# 1-10 family scores of the preceding framework (Tashatov et al. 2026, Table 15): attack resistance, manageability
IOT45_FAMILY = {"raft": "Raft", "smartbft": "PBFT", "ibft2": "PBFT", "qbft": "PBFT", "clique": "PoA", "cometbft": "Tendermint", "hotstuff": "PBFT"}
IOT45_C8 = {"Raft": 10, "PBFT": 7, "PoA": 9, "Tendermint": 8}
IOT45_C10 = {"Raft": 3, "PBFT": 7, "PoA": 4, "Tendermint": 7}


def check_tables():
    for k in KEYS:
        assert sum(C8_TABLE[k]) == ATTACK[k], k
        assert sum(C10_TABLE[k]) == MANAGE[k], k
    return True


def first(X, feas=None, types=TYPES, weights=None):
    idx = np.arange(X.shape[0]) if feas is None else np.where(feas)[0]
    if len(idx) == 1:
        return {m: int(idx[0]) for m in METHODS}
    Xf = X[idx]; t = np.asarray(types)
    w = scenario_set(Xf, t)["Entropy x CRITIC (base)"] if weights is None else weights
    R = rank_all(Xf, w, t)
    return {m: int(idx[int(np.argmin(R[m]["rank"]))]) for m in R}


def _tally(results, keys):
    out = {m: {k: 0 for k in keys} for m in METHODS}
    for fp in results:
        for m, i in fp.items():
            out[m][keys[i]] += 1
    n = max(len(results), 1)
    return {m: {k: v / n for k, v in d.items() if v} for m, d in out.items()}


def smaa_ordinal(Xs, keys, feas=None, n=10000, seed=0, uniform_weights=False):
    rng = np.random.default_rng(seed)
    res = []
    for _ in range(n):
        X = Xs[rng.integers(Xs.shape[0])].copy()
        for j in ORD:
            levels = np.sort(rng.uniform(0.0, 1.0, size=5))   # cardinal values of the levels 0..4, order preserved
            X[:, j] = levels[np.clip(np.round(X[:, j]).astype(int), 0, 4)] + 1e-9
        w = rng.dirichlet(np.ones(X.shape[1])) if uniform_weights else None
        res.append(first(X, feas, weights=w))
    return _tally(res, keys)


def perturb_scores(Xm, keys, feas=None, joint_draws=10000, seed=0):
    out = {}
    for j, name in ((7, "C8"), (9, "C10")):
        res = []
        for d in itertools.product((-1, 0, 1), repeat=len(keys)):
            X = Xm.copy(); X[:, j] = np.clip(X[:, j] + np.array(d), 0, 4)
            if np.ptp(X[:, j]) == 0:
                X[0, j] += 1e-6
            res.append(first(X, feas))
        out[name] = _tally(res, keys)
    rng = np.random.default_rng(seed); res = []
    for _ in range(joint_draws):
        X = Xm.copy()
        for j in ORD:
            X[:, j] = np.clip(X[:, j] + rng.integers(-1, 2, size=len(keys)), 0, 4)
        res.append(first(X, feas))
    out["C8 and C10 jointly"] = _tally(res, keys)
    return out


def _replace(Xm, keys, col, values):
    X = Xm.copy(); X[:, col] = [values[k] for k in keys]; return X


def scoring_rules(Xm, keys, feas=None):
    rules = {}
    w2 = {k: 2 * C8_TABLE[k][0] + sum(C8_TABLE[k][1:]) for k in keys}                     # BFT safety counted twice
    rules["C8 with Byzantine safety counted twice (0-5)"] = _replace(Xm, keys, 7, w2)
    rules["C8 = Byzantine safety only (0/1)"] = _replace(Xm, keys, 7, {k: C8_TABLE[k][0] + 1e-9 for k in keys})
    rules["C10 without the release-line property (0-3)"] = _replace(Xm, keys, 9, {k: sum(C10_TABLE[k][:3]) + 1e-9 for k in keys})
    X = _replace(Xm, keys, 7, {k: IOT45_C8[IOT45_FAMILY[k]] for k in keys})
    rules["C8 and C10 from the 1-10 family scores of [8]"] = _replace(X, keys, 9, {k: IOT45_C10[IOT45_FAMILY[k]] for k in keys})
    out = {name: {m: keys[i] for m, i in first(X, feas).items()} for name, X in rules.items()}
    for name, cols in (("without C8", [7]), ("without C10", [9]), ("without C8 and C10 (measured only)", [7, 9])):
        keep = [j for j in range(Xm.shape[1]) if j not in cols]
        out[name] = {m: keys[i] for m, i in first(Xm[:, keep], feas, types=TYPES[keep]).items()}
    return out


def leave_one_out(Xm, keys, names, feas=None):
    out = {}
    for j, nm in enumerate(names):
        keep = [i for i in range(Xm.shape[1]) if i != j]
        Xk = Xm[:, keep]
        if feas is not None:
            sub = Xk[np.where(feas)[0]]
            if np.any(np.ptp(sub, axis=0) == 0):   # a column constant on the feasible set carries no weight anyway
                pass
        out[nm] = {m: keys[i] for m, i in first(Xk, feas, types=TYPES[keep]).items()}
    return out


def fault_proxies(Xm, keys, n=7, feas=None):
    from .protocols import catalogue, quorum
    P = catalogue()
    tol = {k: 100.0 * quorum(P[k], n)["f_crash"] / n for k in keys}
    out = {}
    out["C7 = documented crash tolerance f/n"] = {m: keys[i] for m, i in first(_replace(Xm, keys, 6, tol), feas).items()}
    comb = {k: 0.5 * (Xm[keys.index(k), 6] + tol[k] / max(tol.values()) * 100.0) for k in keys}
    out["C7 = mean of simulated share and documented tolerance"] = {m: keys[i] for m, i in first(_replace(Xm, keys, 6, comb), feas).items()}
    return out, tol


def scale_shift(Xm, keys, feas=None):
    out = {}
    for name, f in (("ordinal scales 1-5 (shift +1)", lambda v: v + 1.0), ("ordinal scales 0-10 (x2.5)", lambda v: 2.5 * v + 1e-9),
                    ("ordinal scales 1-10 (2.25 v + 1)", lambda v: 2.25 * v + 1.0)):
        X = Xm.copy()
        for j in ORD:
            X[:, j] = f(X[:, j])
        out[name] = {m: keys[i] for m, i in first(X, feas).items()}
    return out
