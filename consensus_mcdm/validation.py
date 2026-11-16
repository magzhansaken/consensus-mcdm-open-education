"""Propagation of the calibration and hold-out errors of the simulation model into the ranking.

Three complementary analyses (Section 4.9 of the article):

1. Model corrections. The two regimes in which the hold-out campaigns show the largest errors are corrected
   inside the simulation and the whole pipeline is repeated:
     RTT   extra commit latency per second of round-trip time, set from the slope of the Fabric hold-out
           between 0 and 100 ms that the model misses (crash-fault: Raft; Byzantine: SmartBFT); the extra time
           is client-visible latency and does not occupy the consensus path, because the measured throughput
           changed by at most 5 % (Raft) at 100 ms;
     REC   every failure timer multiplied by a recovery factor F; F = 2 reproduces the HotStuff hold-out with one
           crashed node and F = 5 the hold-out with three crashed nodes (repeated leader timeouts).
2. Monte Carlo over model error. Every simulated criterion value is multiplied by a log-normal error factor
   whose scale is the agreement error of its platform family for that quantity, at a 'typical' level (the
   regime of the base scenario) and a 'pessimistic' level (the worst hold-out of the family). One factor is
   drawn per family and quantity, so that alternatives calibrated on the same anchor share their error. The
   capacity of the screen is perturbed with the throughput factor.
3. Break-even factors: the factor by which one criterion of the first-ranked alternative would have to be
   wrong before the first place changes.
"""
from __future__ import annotations
import json, os
import numpy as np
from .criteria import TYPES
from .weights import scenario_set
from .mcdm import rank_all, METHODS

FAMILY = {"raft": "fabric", "smartbft": "fabric", "ibft2": "besu", "qbft": "besu", "clique": "besu",
          "cometbft": "cometbft", "hotstuff": "hotstuff"}
# error scales (multiplicative, log-normal sigma) by family and quantity
SIGMA = {
    "typical": {
        "tps": {"fabric": 0.018, "besu": 0.034, "cometbft": 0.037, "hotstuff": 0.031},
        "lat": {"fabric": 0.274, "besu": 0.057, "cometbft": 0.127, "hotstuff": 0.027},
        "cpu": {"fabric": 0.16, "besu": 0.16, "cometbft": 0.16, "hotstuff": 0.16},
        "loss": {"fabric": 0.36, "besu": 0.36, "cometbft": 0.36, "hotstuff": 0.36},
    },
    "pessimistic": {
        "tps": {"fabric": 0.105, "besu": 0.134, "cometbft": 0.086, "hotstuff": 0.077},
        "lat": {"fabric": 0.434, "besu": 0.56, "cometbft": 0.576, "hotstuff": 0.535},
        "cpu": {"fabric": 0.83, "besu": 0.5, "cometbft": 0.5, "hotstuff": 0.5},
        "loss": {"fabric": 1.0, "besu": 1.0, "cometbft": 1.0, "hotstuff": 1.0},
    },
}
SIGMA_STORAGE = 0.05
QUANTITY = {0: "tps", 1: "lat", 2: "lat", 3: "cpu", 4: "cpu", 10: "cpu"}   # C7 via loss share, C6 storage; C8-C10 none


def rtt_correction(rows):
    """Extra latency per second of RTT from the Fabric RTT hold-out: measured slope minus simulated slope (0 -> 100 ms)."""
    out = {}
    for proto in ("raft", "smartbft"):
        r = {round(float(x["rtt_ms"])): x for x in rows if x["set"] == "holdout_rtt" and x["protocol"] == proto}
        meas = (float(r[100]["lat_meas"]) - float(r[0]["lat_meas"])) / 0.1
        sim = (float(r[100]["lat_sim"]) - float(r[0]["lat_sim"])) / 0.1
        out[proto] = meas - sim
    return {"crash": out["raft"], "bft": out["smartbft"]}


def perturb(Xs, keys, level, rng, capacity=None, stress=None):
    """One model-error draw applied to every seed matrix; returns the perturbed matrices and the feasible mask."""
    S = SIGMA[level]
    fam = sorted(set(FAMILY.values()))
    f = {q: {fm: float(np.exp(rng.normal(0.0, S[q][fm]))) for fm in fam} for q in S}
    f_st = float(np.exp(rng.normal(0.0, SIGMA_STORAGE)))
    X = Xs.copy()
    for i, k in enumerate(keys):
        fm = FAMILY[k]
        for j, q in QUANTITY.items():
            X[:, i, j] *= f[q][fm]
        X[:, i, 5] *= f_st
        loss = 100.0 - X[:, i, 6]
        X[:, i, 6] = 100.0 - np.minimum(loss * f["loss"][fm], 100.0)
    feas = None
    if capacity is not None:
        feas = np.array([capacity[k] * f["tps"][FAMILY[k]] >= stress for k in keys])
    return X, feas


def first_places(X, feas=None, types=TYPES):
    """First-ranked alternative per method (weights re-estimated on the feasible set); None if no alternative is feasible."""
    idx = np.arange(X.shape[0]) if feas is None else np.where(feas)[0]
    if len(idx) == 0:
        return {m: None for m in METHODS}
    if len(idx) == 1:
        return {m: int(idx[0]) for m in METHODS}
    Xf = X[idx]
    w = scenario_set(Xf, types)["Entropy x CRITIC (base)"]
    R = rank_all(Xf, w, types)
    return {m: int(idx[int(np.argmin(R[m]["rank"]))]) for m in R}


def monte_carlo(Xs, keys, level, n_draws=2000, seed=0, capacity=None, stress=None):
    rng = np.random.default_rng(seed)
    S = Xs.shape[0]
    counts = {m: {k: 0 for k in keys + ["none"]} for m in METHODS}
    feas_count = {k: 0 for k in keys}
    for _ in range(n_draws):
        X, feas = perturb(Xs, keys, level, rng, capacity, stress)
        Xm = X[rng.integers(S)]
        fp = first_places(Xm, feas)
        for m, i in fp.items():
            counts[m]["none" if i is None else keys[i]] += 1
        if feas is not None:
            for i, k in enumerate(keys):
                feas_count[k] += int(feas[i])
    return {"first_share": {m: {k: v / n_draws for k, v in d.items()} for m, d in counts.items()},
            "feasible_share": {k: v / n_draws for k, v in feas_count.items()} if capacity is not None else None,
            "n_draws": n_draws}


def break_even(Xm, keys, target, criterion, direction=+1, feas=None, lo=1.0, hi=50.0):
    """Smallest factor (>= 1) by which `criterion` of `target` must be multiplied (direction +1) or divided (-1) before
    `target` loses the first place under each method; inf if it does not within `hi`. For C7 the factor applies to the
    loss share 100 - C7, and the loss is first set to 1 % if it is zero."""
    ti = keys.index(target)
    out = {}
    for m in METHODS:
        def first(fac):
            X = Xm.copy()
            if criterion == 6:
                loss = max(100.0 - X[ti, 6], 1.0)
                X[ti, 6] = 100.0 - min(loss * fac, 100.0)
            else:
                X[ti, criterion] = X[ti, criterion] * (fac if direction > 0 else 1.0 / fac)
            return first_places(X, feas)[m]
        if first(lo) != ti:
            out[m] = 1.0
            continue
        if first(hi) == ti:
            out[m] = float("inf")
            continue
        a, b = lo, hi
        for _ in range(40):
            c = np.sqrt(a * b)
            if first(c) == ti:
                a = c
            else:
                b = c
        out[m] = float(b)
    return out
