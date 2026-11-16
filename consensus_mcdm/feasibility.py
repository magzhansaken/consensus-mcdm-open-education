"""Peak-load feasibility screen and the boundaries of the recommendation.

The compensatory ranking methods (TOPSIS, VIKOR, COPRAS, EDAS) let a good value on one criterion
offset a poor value on another. For an education platform the capacity to serve the busiest hour
is not such a criterion: a protocol that rejects half of the enrolment transactions is not an
acceptable alternative however good its other values are. The functions below therefore add a
non-compensatory (conjunctive) screen in front of the ranking:

  capacity(p)         largest offered load at which at least `target` of the transactions are
                      committed within the 30-s client timeout (mean over seeds), found on a load grid;
  screen(...)         an alternative is feasible if, in every seed of the base scenario, at least
                      `target` of the stress-level transactions commit within 30 s;
  rank_feasible(...)  ranking of the feasible subset with weights re-estimated on that subset and
                      with the base weights of the full set;
  provisioned(P)      the alternatives with the block-formation setting that maximises capacity within
                      the measured or published range of each platform (Fabric batch 300, Besu block
                      capacity 440 transactions per 2-s block, HotStuff 1-s proposal interval);
  boundary(...)       first-ranked feasible alternative as a function of the stress-hour load.
"""
from __future__ import annotations
import copy
import numpy as np
from .sim import simulate
from .workload import arrivals, WorkloadParams, hourly_rate, summary
from .criteria import _cfg, decision_matrices, Scenario, TYPES
from .weights import scenario_set
from .mcdm import rank_all, METHODS
from .smaa import smaa2

TARGET = 0.99          # share of transactions that must commit within the 30-s timeout
CAP_GRID = tuple(range(50, 301, 5))


def availability_curve(P, sc, app_tier, seeds=range(10), loads=CAP_GRID):
    """Share committed within 30 s and goodput against offered load (base scenario, constant load)."""
    cfg = _cfg(sc, app_tier)
    rows = []
    for k, p in P.items():
        for lam in loads:
            av, tps, lat = [], [], []
            for s in seeds:
                sd = 70_000 + 1000 * s + lam
                r = simulate(p, cfg, arrivals(lam, cfg.duration_s, np.random.default_rng(sd)), seed=sd)
                av.append(r.availability); tps.append(r.throughput_tps); lat.append(r.lat_p95_s)
            rows.append({"key": k, "offered": lam, "avail_mean": float(np.mean(av)), "avail_min": float(np.min(av)),
                         "tps_mean": float(np.mean(tps)), "p95_mean": float(np.nanmean(lat))})
    return rows


def capacities(rows, target=TARGET):
    """Largest grid load at which the mean committed share is >= target at that load and every lower one."""
    out = {}
    for k in dict.fromkeys(r["key"] for r in rows):
        rr = sorted([r for r in rows if r["key"] == k], key=lambda r: r["offered"])
        cap = 0
        for r in rr:
            if r["avail_mean"] >= target:
                cap = r["offered"]
            else:
                break
        out[k] = cap
    return out


def stress_availability(raws, keys):
    """Committed share at the stress level per seed, from the raw outputs of decision_matrices."""
    return {k: np.array([r["stress"]["availability"] for r in raws[k]]) for k in keys}


def screen(av_stress, target=TARGET):
    """Feasible if every seed reaches the target at the stress level."""
    return {k: bool(np.min(v) >= target) for k, v in av_stress.items()}


def rank_feasible(Xm, keys, feasible, base_w, Xs=None, smaa_draws=10000, types=TYPES):
    idx = [i for i, k in enumerate(keys) if feasible[k]]
    out = {"feasible": [keys[i] for i in idx]}
    if len(idx) == 0:
        return out
    if len(idx) == 1:
        out["first_own"] = {m: keys[idx[0]] for m in METHODS}
        out["first_base"] = dict(out["first_own"])
        return out
    Xf = Xm[idx]
    W = scenario_set(Xf, types)
    w_own = W["Entropy x CRITIC (base)"]
    R_own = rank_all(Xf, w_own, types)
    R_base = rank_all(Xf, base_w, types)
    R_eq = rank_all(Xf, W["Equal"], types)
    out["weights_own"] = w_own.tolist()
    out["ranks_own"] = {m: {keys[idx[i]]: int(R_own[m]["rank"][i]) for i in range(len(idx))} for m in R_own}
    out["ranks_base"] = {m: {keys[idx[i]]: int(R_base[m]["rank"][i]) for i in range(len(idx))} for m in R_base}
    out["ranks_equal"] = {m: {keys[idx[i]]: int(R_eq[m]["rank"][i]) for i in range(len(idx))} for m in R_eq}
    out["first_own"] = {m: keys[idx[int(np.argmin(R_own[m]["rank"]))]] for m in R_own}
    out["first_base"] = {m: keys[idx[int(np.argmin(R_base[m]["rank"]))]] for m in R_base}
    out["scores_own"] = {m: {keys[idx[i]]: float(R_own[m]["score"][i]) for i in range(len(idx))} for m in R_own}
    if Xs is not None:
        Xsf = Xs[:, idx, :]
        sm = {}
        for m in METHODS:
            a = smaa2(Xsf, types, method=m, n_weights=smaa_draws, seed=142)
            sm[m] = {keys[idx[i]]: float(a["acceptability"][i, 0]) for i in range(len(idx))}
        out["smaa_b1_uniform"] = sm
    return out


def provisioned(P, besu_block_tx=440, fabric_batch=300, hotstuff_interval_s=1.0):
    """Capacity-provisioned variants: each platform set to the block-formation value that raises its
    capacity, within the calibrated range (Fabric batch 300, measured in the anchor), a published range
    (Besu: 300-400 tx/s at a 2-s block period with a high block gas limit) or a documented setting
    (HotStuff proposal interval). Failure timers are unchanged."""
    Q = copy.deepcopy(P)
    for k in ("raft", "smartbft"):
        Q[k].B_max = fabric_batch
    for k in ("ibft2", "qbft"):
        Q[k].B_max = besu_block_tx
    Q["hotstuff"].T_b = hotstuff_interval_s
    return Q


def stress_workload(target_tps, base: WorkloadParams | None = None):
    """Workload whose stress-hour load equals target_tps (by scaling the enrolment burst W6)."""
    wl = copy.deepcopy(base or WorkloadParams())
    wl.registration_burst_tps = 0.0
    base_hour = hourly_rate(wl, "stress")[wl.registration_hour]
    wl.registration_burst_tps = max(0.0, target_tps - base_hour)
    return wl


def boundary(P, app_tier, levels, seeds=range(10), n=7, target=TARGET):
    """For each stress-hour load: feasible set and first-ranked feasible alternative per method."""
    rows = []
    for L in levels:
        wl = stress_workload(L)
        sc = Scenario(n=n, wl=wl)
        keys, Xm, Xs, hw, raws = decision_matrices(P, sc, app_tier, seeds=seeds)
        av = stress_availability(raws, keys)
        feas = screen(av, target)
        base_w = scenario_set(Xm, TYPES)["Entropy x CRITIC (base)"]
        rf = rank_feasible(Xm, keys, feas, base_w)
        R_all = rank_all(Xm, base_w, TYPES)
        rows.append({"stress_tps": float(summary(wl)["stress_tps"]), "feasible": rf["feasible"],
                     "first_own": rf.get("first_own", {}), "first_unscreened": {m: keys[int(np.argmin(R_all[m]["rank"]))] for m in R_all},
                     "avail_min": {k: float(v.min()) for k, v in av.items()}})
    return rows


def ablation_set(P, n=7):
    """The seven deployment ablations of the main analysis, applied to a given protocol configuration."""
    P_hs = copy.deepcopy(P); P_hs["hotstuff"].T_b = 1.0
    P_rc = copy.deepcopy(P)
    for k in ("ibft2", "qbft"):
        P_rc[k].view_change_s = 10.0
    return {"base": (Scenario(n=n), P), "n4": (Scenario(n=4), P), "n10": (Scenario(n=10), P),
            "rtt20ms": (Scenario(n=n, rtt_s=0.02), P), "no_app_tier": (Scenario(n=n, apply_app_tier=False), P),
            "operator_removes_dead": (Scenario(n=n, exclude_dead=True), P),
            "hotstuff_Tb1s": (Scenario(n=n), P_hs), "besu_roundchange_10s": (Scenario(n=n), P_rc)}
