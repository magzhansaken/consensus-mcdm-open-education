"""Diagnostics behind the decision matrix: load response of every protocol and the fault timeline.

load_sweep    : committed throughput (within the 30-s client timeout) and mean latency of committed
                transactions for offered loads 25..250 tx/s at the base scenario, S seeds per point.
fault_timeline: share of transactions committed within 30 s per 10-s arrival bin in the fault scenario
                of criterion C7 (proposer crash at 120 s, second node at 300 s, peak load).
"""
from __future__ import annotations
import numpy as np
from .sim import simulate
from .workload import summary, arrivals
from .criteria import _cfg

LOADS = tuple(range(25, 251, 25))


def load_sweep(P, sc, app_tier, seeds=range(10), loads=LOADS):
    rows = []
    cfg = _cfg(sc, app_tier)
    for k, p in P.items():
        for lam in loads:
            tps, lat = [], []
            for s in seeds:
                rng = np.random.default_rng(50_000 + 1000 * s + lam)
                r = simulate(p, cfg, arrivals(lam, cfg.duration_s, rng), seed=50_000 + 1000 * s + lam)
                tps.append(r.throughput_tps); lat.append(r.lat_mean_s)
            tps, lat = np.array(tps), np.array(lat)
            rows.append({"key": k, "name": p.name, "offered": lam, "tps_mean": float(tps.mean()), "tps_sd": float(tps.std(ddof=1)),
                         "lat_mean": float(np.nanmean(lat)), "lat_sd": float(np.nanstd(lat, ddof=1)), "n_seeds": len(tps)})
    return rows


def fault_timeline(P, sc, app_tier, seed=7, bin_s=10.0):
    loads = summary(sc.wl)
    cfg = _cfg(sc, app_tier, faults=[(120.0, 0), (300.0, 1)])
    rows = []
    for k, p in P.items():
        tr = {}
        rng = np.random.default_rng(seed)
        simulate(p, cfg, arrivals(loads["deadline_peak_tps"], cfg.duration_s, rng), seed=seed, trace=tr)
        arr, com = tr["arrival"], tr["commit"]
        ok = np.isfinite(com) & ((com - arr) <= cfg.timeout_s)
        edges = np.arange(0.0, cfg.duration_s + bin_s, bin_s)
        idx = np.digitize(arr, edges) - 1
        for b in range(len(edges) - 1):
            m = idx == b
            n = int(m.sum())
            rows.append({"key": k, "name": p.name, "t_start": float(edges[b]), "arrivals": n,
                         "share_ok": float(ok[m].mean()) if n else float("nan")})
    return rows
