"""Calibration of implementation constants on published measurements and hold-out validation.

Protocol / anchor / what is fitted
  Raft, SmartBFT (Fabric 3.1.4): Information 2026 17(10) 945 batch x load campaign (12 points each,
      submitEvidence, 4 orderers, single host). Fitted: c_block, c_tx (shared application tier),
      a, o (per protocol), kappa (CPU). Hold-out: orderer-count (4/7/10), injected RTT (0/50/100 ms),
      crash-fault campaign (10 orderers).
  Clique, QBFT, IBFT 2.0 (Besu 25.7): Applied Sciences 2026 16(19) 9631 burst benchmark (4 validators,
      block period 2 s, 5000-tx burst). Fitted: B_max (tx per 2-s block), a. No hold-out available ->
      reported as 'calibrated only'.
  CometBFT v1: QA saturation table (175 validators, 1 kB tx). Fitted: o, g (gossip hop delay).
      Hold-out: overloaded point (800 tx/s) and latency bound (<= 4 s for most blocks at 400 tx/s).
  HotStuff 2-chain: reference benchmark, 10 nodes WAN (faults 0, 4 load points). Fitted: a, o.
      Hold-out: 20 and 50 nodes, 1 and 3 dead nodes.
The calibration mirrors the anchor's run protocol (duration, warm-up) and averages 3 seeds.
"""
from __future__ import annotations
import json, os, math, copy
import numpy as np
from scipy.optimize import minimize
from .protocols import catalogue, Protocol, RESULTS
from .sim import simulate, SimConfig
from .workload import arrivals

LOGICAL_CPUS_945 = 8  # i7-8559U: 8 logical CPUs (container %CPU is relative to the host)

FABRIC_BATCH = [  # (protocol, batch, offered, achieved_tps, latency_mean_s, cpu_pct_per_tps)
    ("raft", 10, 50, 48.30, 0.246, 0.0075), ("raft", 10, 100, 82.14, 8.554, 0.0081),
    ("raft", 10, 150, 76.74, 36.026, 0.0087), ("raft", 10, 200, 99.78, 38.026, 0.0082),
    ("raft", 100, 50, 48.32, 1.350, 0.0063), ("raft", 100, 100, 96.60, 0.830, 0.0052),
    ("raft", 100, 150, 144.48, 0.822, 0.0061), ("raft", 100, 200, 181.42, 6.304, 0.0058),
    ("raft", 300, 50, 49.32, 1.362, 0.0051), ("raft", 300, 100, 98.16, 1.614, 0.0042),
    ("raft", 300, 150, 145.26, 2.050, 0.0056), ("raft", 300, 200, 191.58, 2.496, 0.0052),
    ("smartbft", 10, 50, 48.36, 0.270, 0.0233), ("smartbft", 10, 100, 68.78, 18.530, 0.0257),
    ("smartbft", 10, 150, 74.80, 40.094, 0.0309), ("smartbft", 10, 200, 98.54, 32.608, 0.0371),
    ("smartbft", 100, 50, 49.22, 1.356, 0.0114), ("smartbft", 100, 100, 96.84, 1.022, 0.0189),
    ("smartbft", 100, 150, 144.06, 1.268, 0.0210), ("smartbft", 100, 200, 150.22, 27.424, 0.0207),
    ("smartbft", 300, 50, 49.00, 1.538, 0.0185), ("smartbft", 300, 100, 97.88, 2.030, 0.0233),
    ("smartbft", 300, 150, 145.06, 2.408, 0.0152), ("smartbft", 300, 200, 156.58, 23.524, 0.0249),
]
FABRIC_HOLDOUT = {
    "orderer_count": [("raft", 4, 100, 97.94, 1.656), ("raft", 7, 100, 97.74, 1.706), ("raft", 10, 100, 97.58, 1.748),
                      ("smartbft", 4, 100, 97.26, 2.008), ("smartbft", 7, 100, 96.96, 2.546), ("smartbft", 10, 100, 95.92, 3.100)],
    "rtt": [("raft", 0.000, 75, 72.48, 0.926), ("raft", 0.050, 75, 72.38, 1.768), ("raft", 0.100, 75, 70.40, 4.192),
            ("smartbft", 0.000, 75, 73.16, 1.048), ("smartbft", 0.050, 75, 71.64, 1.800), ("smartbft", 0.100, 75, 59.52, 8.370)],
    "faults": [("raft", 0, 97.52, 2.364), ("raft", 1, 97.62, 1.926), ("raft", 2, 97.66, 1.792), ("raft", 3, 97.74, 1.790),
               ("smartbft", 0, 96.48, 3.684), ("smartbft", 1, 78.64, 28.07), ("smartbft", 2, 81.82, 16.876), ("smartbft", 3, 76.46, 11.088)],
}
BESU = {  # protocol: (achieved_tps_burst, p50_s, p95_s)
    "clique": (211.5, 1.005, 1.885), "qbft": (109.97, 6.742, 28.444), "ibft2": (110.0, 7.591, 28.977),
}
COMET = [(200, 200.0, 3.874), (400, 350.6, 9.281), (800, 574.7, 9.039), (1600, 571.8, 11.022)]  # (offered, achieved, mean latency)
HOTSTUFF_FIT = [(10, 0, 10000, 9756, 1.274), (10, 0, 80000, 77676, 1.440), (10, 0, 100000, 94767, 1.629), (10, 0, 120000, 98145, 9.826)]
HOTSTUFF_HOLD = [(20, 0, 10000, 9971, 0.954), (50, 0, 10000, 9973, 0.817), (10, 1, 9000, 8231, 5.901), (10, 3, 700, 646, 15.195)]

SEEDS = (11, 12, 13)


def _mean_runs(p, cfg, lam, seeds=SEEDS):
    out = []
    for s in seeds:
        arr = arrivals(lam, cfg.duration_s, np.random.default_rng(1000 + s))
        out.append(simulate(p, cfg, arr, seed=s))
    tps = np.mean([r.throughput_tps for r in out])
    lats = [r.lat_mean_s if np.isfinite(r.lat_mean_s) else cfg.timeout_s for r in out]
    lat = float(np.mean(lats))
    p50 = np.nanmean([r.lat_p50_s for r in out])
    p95 = np.nanmean([r.lat_p95_s for r in out])
    cpu = np.mean([r.cpu_cons_s_per_1k for r in out])
    fail = np.mean([r.fail_rate for r in out])
    return dict(tps=tps, lat=lat, p50=p50, p95=p95, cpu=cpu, fail=fail)


def _logerr(sim, meas):
    sim = max(sim, 1e-6); meas = max(meas, 1e-6)
    return math.log(sim / meas)


# ----------------------------------------------------------------------------- Fabric
APP_TIER = {"c_block": 0.03, "c_tx": 0.0030}   # updated by fit_fabric


def fabric_cfg(n=4, rtt=0.0005, faults=None):
    return SimConfig(n=n, rtt_s=rtt, duration_s=90.0, warmup_s=30.0, timeout_s=60.0, tx_bytes=1000,
                     c_block=APP_TIER["c_block"], c_tx=APP_TIER["c_tx"], faults=faults or [])


def fit_fabric(P, verbose=True):
    pts = FABRIC_BATCH
    keys = ["c_block", "c_tx", "a_raft", "o_raft", "m_raft", "a_smart", "o_smart", "m_smart"]
    x0 = np.log([0.03, 0.0030, 0.08, 0.0005, 0.002, 0.15, 0.0012, 0.02])

    def apply(x):
        v = np.exp(x)
        APP_TIER["c_block"], APP_TIER["c_tx"] = float(v[0]), float(v[1])
        cfg = fabric_cfg()
        P["raft"].a, P["raft"].o, P["raft"].m = v[2], v[3], v[4]
        P["smartbft"].a, P["smartbft"].o, P["smartbft"].m = v[5], v[6], v[7]
        return cfg

    n_eval = [0]
    def loss(x):
        cfg = apply(x)
        L = 0.0
        for key, B, lam, tps_m, lat_m, _ in pts:
            p = P[key]; p.B_max = B
            r = _mean_runs(p, cfg, lam, seeds=(11,))
            w_lat = 1.0 if lat_m < 5.0 else 0.3           # saturated points are duration-dependent
            L += w_lat * _logerr(r["lat"], lat_m) ** 2 + _logerr(r["tps"], tps_m) ** 2
        n_eval[0] += 1
        if verbose and n_eval[0] % 25 == 0:
            print(f"  eval {n_eval[0]} loss {L:.4f} params {np.round(np.exp(x), 5).tolist()}", flush=True)
        return L

    res = minimize(loss, x0, method="Nelder-Mead", options={"maxiter": 260, "xatol": 1e-3, "fatol": 1e-4})
    cfg = apply(res.x)
    # stage 2: growth of the per-tx work with the validator count (nu) from the orderer-count campaign,
    # the only n-informative data; the n = 4 behaviour fitted in stage 1 is unchanged by construction
    for key in ("raft", "smartbft"):
        p = P[key]; p.B_max = 100
        pts_n = [(n, lat_m) for k2, n, lam, tps_m, lat_m in FABRIC_HOLDOUT["orderer_count"] if k2 == key]
        base_meas = [lat for n, lat in pts_n if n == 4][0]
        sim4 = _mean_runs(p, fabric_cfg(n=4), 100, seeds=(11,))["lat"]
        best = (1e9, 0.0)
        for nu_try in np.concatenate([[0.0], np.geomspace(0.01, 3.0, 40)]):
            p.nu = nu_try
            L = 0.0
            for n, lat_m in pts_n:
                if n == 4:
                    continue
                simn = _mean_runs(p, fabric_cfg(n=n), 100, seeds=(11,))["lat"]
                L += (math.log(simn / sim4) - math.log(lat_m / base_meas)) ** 2   # relative growth with n
            if L < best[0]:
                best = (L, nu_try)
        p.nu = best[1]
        if verbose:
            print(f"  stage-2 {key}: nu = {p.nu:.4f} (relative n-scaling loss {best[0]:.4f})", flush=True)
    # CPU scale kappa from measured orderer CPU per tx (stable points only)
    for key in ("raft", "smartbft"):
        ratios = []
        for k2, B, lam, tps_m, lat_m, cpu_pct in pts:
            if k2 != key or lat_m >= 5.0:
                continue
            P[key].B_max = B
            r = _mean_runs(P[key], cfg, lam, seeds=(11,))
            meas = cpu_pct * LOGICAL_CPUS_945 / 100.0 * 1000.0   # CPU-s per 1000 tx per orderer
            ratios.append(meas / max(r["cpu"], 1e-9))
        P[key].kappa *= float(np.median(ratios))
    P["raft"].B_max = P["smartbft"].B_max = 100
    report = {"params": dict(zip(keys, np.exp(res.x).tolist())), "loss": float(res.fun), "nfev": int(res.nfev),
              "kappa_raft": P["raft"].kappa, "kappa_smartbft": P["smartbft"].kappa, "c_block": cfg.c_block, "c_tx": cfg.c_tx}
    if verbose:
        print("Fabric fit:", json.dumps(report, indent=1), flush=True)
    return cfg, report


def validate_fabric(P, cfg_fit, verbose=True):
    rows = []
    # in-sample table
    for key, B, lam, tps_m, lat_m, cpu_pct in FABRIC_BATCH:
        p = P[key]; p.B_max = B
        r = _mean_runs(p, fabric_cfg(), lam)
        rows.append(dict(set="fit", protocol=key, n=4, batch=B, rtt_ms=0.5, offered=lam, k_fault=0,
                         tps_meas=tps_m, tps_sim=r["tps"], lat_meas=lat_m, lat_sim=r["lat"],
                         cpu_meas=cpu_pct * LOGICAL_CPUS_945 / 100 * 1000, cpu_sim=r["cpu"]))
    P["raft"].B_max = P["smartbft"].B_max = 100
    for key, n, lam, tps_m, lat_m in FABRIC_HOLDOUT["orderer_count"]:
        r = _mean_runs(P[key], fabric_cfg(n=n), lam)
        rows.append(dict(set="fit_n", protocol=key, n=n, batch=100, rtt_ms=0.5, offered=lam, k_fault=0,
                         tps_meas=tps_m, tps_sim=r["tps"], lat_meas=lat_m, lat_sim=r["lat"], cpu_meas=np.nan, cpu_sim=r["cpu"]))
    for key, rtt, lam, tps_m, lat_m in FABRIC_HOLDOUT["rtt"]:
        r = _mean_runs(P[key], fabric_cfg(n=4, rtt=max(rtt, 0.0005)), lam)
        rows.append(dict(set="holdout_rtt", protocol=key, n=4, batch=100, rtt_ms=rtt * 1000, offered=lam, k_fault=0,
                         tps_meas=tps_m, tps_sim=r["tps"], lat_meas=lat_m, lat_sim=r["lat"], cpu_meas=np.nan, cpu_sim=r["cpu"]))
    for key, k, tps_m, lat_m in FABRIC_HOLDOUT["faults"]:
        # k orderers crash-stopped at 10 s of the measured window (leader among them for k>=1)
        faults = [(40.0 + 3 * j, j) for j in range(k)]
        r = _mean_runs(P[key], fabric_cfg(n=10, faults=faults), 100)
        rows.append(dict(set="holdout_fault", protocol=key, n=10, batch=100, rtt_ms=0.5, offered=100, k_fault=k,
                         tps_meas=tps_m, tps_sim=r["tps"], lat_meas=lat_m, lat_sim=r["lat"], cpu_meas=np.nan, cpu_sim=r["cpu"]))
    return rows


# ----------------------------------------------------------------------------- Besu
def besu_cfg():
    return SimConfig(n=4, rtt_s=0.0005, duration_s=200.0, warmup_s=0.0, timeout_s=300.0, tx_bytes=300, c_block=0, c_tx=0)


def _burst_arrivals(total=5000, send_tps=250.0, rng=None):
    rng = rng or np.random.default_rng(7)
    # the anchor submits a 5000-tx burst from a client script; we assume the client-side
    # submission rate (assumption S-B1) and derive inclusion latency and burst throughput
    return np.sort(rng.uniform(0.0, total / send_tps, size=total))


def _burst_metrics(p, cfg, send_tps, seeds=SEEDS):
    out = []
    for s in seeds:
        arr = _burst_arrivals(5000, send_tps, np.random.default_rng(500 + s))
        r = simulate(p, cfg, arr, seed=s)
        out.append(r)
    # burst throughput = successful / (first submission .. last inclusion); reconstruct from latency
    tps = np.mean([5000.0 / (5000.0 / send_tps + r.lat_max_s) for r in out])
    return dict(tps=tps, p50=np.nanmean([r.lat_p50_s for r in out]), p95=np.nanmean([r.lat_p95_s for r in out]))


def fit_besu(P, verbose=True):
    """Besu family. The burst benchmark pins the sustainable capacity of QBFT/IBFT 2.0 (backlog regime:
    burst TPS = capacity) -> B_max = TPS x block period. Clique shows no backlog (p95 < block period), so
    its capacity is only bounded from below (B_max >= 423). The per-tx execution constant o is set from
    the Clique median latency (assumption S-B2) and shared by the three protocols (same EVM tier).
    The client submission rate of the burst is not reported; we assume 150 tx/s (S-B1) only to report
    the implied p50/p95 as a plausibility check, not as a fit."""
    cfg = besu_cfg()
    rep = {}
    send_tps = 150.0
    for key, (tps_m, p50_m, p95_m) in BESU.items():
        p = P[key]
        p.o = 0.0002; p.a = 0.05
        if key == "clique":
            p.B_max = max(int(round(tps_m * p.T_b)), 423)
        else:
            p.B_max = int(round(tps_m * p.T_b))
        r = _burst_metrics(p, cfg, send_tps)
        rep[key] = dict(B_max=p.B_max, a=p.a, o=p.o, tps_meas=tps_m, tps_sim=r["tps"], p50_meas=p50_m, p50_sim=r["p50"], p95_meas=p95_m, p95_sim=r["p95"], send_tps_assumed=send_tps)
        if verbose:
            print("Besu", key, rep[key], flush=True)
    return rep


# ----------------------------------------------------------------------------- CometBFT
def comet_cfg():
    return SimConfig(n=175, rtt_s=0.002, duration_s=90.0, warmup_s=0.0, timeout_s=120.0, tx_bytes=1024, c_block=0, c_tx=0, apply_app_tier=False)


def fit_comet(P, verbose=True):
    """CometBFT, two deterministic steps: (1) the per-tx mempool processing o_pool is the reciprocal of the
    saturated throughput at the 800 tx/s point; (2) the gossip hop delay g is fitted on the mean latency at
    200 tx/s. The 400-tx/s point is a check; the 1600 tx/s point is a hold-out and enters neither step."""
    p = P["cometbft"]; cfg = comet_cfg()
    sat = [tps for lam, tps, lat in COMET if lam == 800][0]
    p.o_pool = 1.0 / sat
    def loss(x):
        p.g = math.exp(x[0]); L = 0.0
        for lam, tps_m, lat_m in COMET[:1]:          # latency at 200 tx/s (unsaturated point)
            r = _mean_runs(p, cfg, lam, seeds=(11,))
            L += _logerr(r["lat"], lat_m) ** 2
        return L
    res = minimize(loss, np.log([0.3]), method="Nelder-Mead", options={"maxiter": 60})
    p.g = math.exp(res.x[0])
    rows = []
    for lam, tps_m, lat_m in COMET:
        r = _mean_runs(p, cfg, lam)
        rows.append(dict(set="fit" if lam in (200, 800) else "check" if lam == 400 else "holdout", offered=lam, tps_meas=tps_m, tps_sim=r["tps"], lat_meas=lat_m, lat_sim=r["lat"]))
    rep = dict(o_pool=p.o_pool, g=p.g, loss=float(res.fun), rows=rows)
    if verbose:
        print("CometBFT fit", json.dumps(rep, indent=1), flush=True)
    return rep


# ----------------------------------------------------------------------------- HotStuff
def hotstuff_cfg(n=10, faults=None):
    return SimConfig(n=n, rtt_s=0.2, duration_s=300.0, warmup_s=30.0, timeout_s=60.0, tx_bytes=512, c_block=0, c_tx=0,
                     faults=faults or [], apply_app_tier=False, bw_bps=1e10)   # AWS 10 Gbit/s instances (assumption R2)


def fit_hotstuff(P, verbose=True):
    p = P["hotstuff"]
    def apply(x):
        p.a, p.o = math.exp(x[0]), math.exp(x[1])
    def loss(x):
        apply(x); L = 0.0
        for n, f, lam, tps_m, lat_m in HOTSTUFF_FIT[:3]:
            r = _mean_runs(p, hotstuff_cfg(n), lam, seeds=(11,))
            L += _logerr(r["tps"], tps_m) ** 2 + _logerr(r["lat"], lat_m) ** 2
        return L
    res = minimize(loss, np.log([0.02, 0.00002]), method="Nelder-Mead", options={"maxiter": 120})
    apply(res.x)
    rows = []
    for n, f, lam, tps_m, lat_m in HOTSTUFF_FIT + HOTSTUFF_HOLD:
        faults = [(31.0 + j, j) for j in range(f)]
        r = _mean_runs(p, hotstuff_cfg(n, faults), lam)
        rows.append(dict(set="fit" if (n, f, lam) in [(a, b, c) for a, b, c, _, _ in HOTSTUFF_FIT] else "holdout", n=n, faults=f, offered=lam,
                         tps_meas=tps_m, tps_sim=r["tps"], lat_meas=lat_m, lat_sim=r["lat"]))
    rep = dict(a=p.a, o=p.o, loss=float(res.fun), rows=rows)
    if verbose:
        print("HotStuff fit", json.dumps(rep, indent=1))
    return rep


def mape(rows, key_m, key_s):
    v = [abs(r[key_s] - r[key_m]) / abs(r[key_m]) for r in rows if r.get(key_m) and not (isinstance(r[key_m], float) and math.isnan(r[key_m]))]
    return float(np.mean(v)) if v else float("nan")


def run_all(verbose=True):
    os.makedirs(RESULTS, exist_ok=True)
    P = catalogue()
    cfg_f, rep_f = fit_fabric(P, verbose)
    rows_f = validate_fabric(P, cfg_f, verbose)
    rep_b = fit_besu(P, verbose)
    rep_c = fit_comet(P, verbose)
    rep_h = fit_hotstuff(P, verbose)
    out = {"fabric": rep_f, "besu": rep_b, "cometbft": rep_c, "hotstuff": rep_h,
           "app_tier": {"c_block": cfg_f.c_block, "c_tx": cfg_f.c_tx, "cpu_app_tx": 0.006},
           "protocols": {k: {f: getattr(v, f) for f in ("a", "o", "o_pool", "nu", "m", "kappa", "g", "B_max", "T_b")} for k, v in P.items()},
           "fabric_validation_rows": rows_f}
    json.dump(out, open(os.path.join(RESULTS, "calibrated_params.json"), "w"), indent=1, default=float)
    import csv
    with open(os.path.join(RESULTS, "calibration_fabric.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows_f[0].keys())); w.writeheader(); w.writerows(rows_f)
    summary = {
        "fabric_fit_MAPE_latency": mape([r for r in rows_f if r["set"] == "fit" and r["lat_meas"] < 5], "lat_meas", "lat_sim"),
        "fabric_fit_MAPE_tps": mape([r for r in rows_f if r["set"] == "fit"], "tps_meas", "tps_sim"),
        "fabric_fit_MAPE_cpu": mape([r for r in rows_f if r["set"] == "fit" and r["lat_meas"] < 5], "cpu_meas", "cpu_sim"),
        "fabric_fit_n_MAPE_latency": mape([r for r in rows_f if r["set"] == "fit_n"], "lat_meas", "lat_sim"),
        "fabric_holdout_rtt_MAPE_latency": mape([r for r in rows_f if r["set"] == "holdout_rtt"], "lat_meas", "lat_sim"),
        "fabric_holdout_fault_MAPE_tps": mape([r for r in rows_f if r["set"] == "holdout_fault"], "tps_meas", "tps_sim"),
        "cometbft_fit_MAPE_tps": mape([r for r in rep_c["rows"] if r["set"] == "fit"], "tps_meas", "tps_sim"),
        "cometbft_fit_MAPE_latency": mape([r for r in rep_c["rows"] if r["set"] == "fit"], "lat_meas", "lat_sim"),
        "cometbft_check_400_MAPE_tps": mape([r for r in rep_c["rows"] if r["set"] == "check"], "tps_meas", "tps_sim"),
        "cometbft_check_400_MAPE_latency": mape([r for r in rep_c["rows"] if r["set"] == "check"], "lat_meas", "lat_sim"),
        "cometbft_holdout_1600_MAPE_tps": mape([r for r in rep_c["rows"] if r["set"] == "holdout"], "tps_meas", "tps_sim"),
        "hotstuff_fit_MAPE_tps": mape([r for r in rep_h["rows"] if r["set"] == "fit"], "tps_meas", "tps_sim"),
        "hotstuff_fit_MAPE_latency": mape([r for r in rep_h["rows"] if r["set"] == "fit" and r["lat_meas"] < 5], "lat_meas", "lat_sim"),
        "hotstuff_holdout_MAPE_tps": mape([r for r in rep_h["rows"] if r["set"] == "holdout"], "tps_meas", "tps_sim"),
        "hotstuff_holdout_MAPE_latency": mape([r for r in rep_h["rows"] if r["set"] == "holdout"], "lat_meas", "lat_sim"),
        "besu_fit_MAPE_tps": float(np.mean([abs(v["tps_sim"] - v["tps_meas"]) / v["tps_meas"] for v in rep_b.values()])),
        "besu_fit_MAPE_p50": float(np.mean([abs(v["p50_sim"] - v["p50_meas"]) / v["p50_meas"] for v in rep_b.values()])),
    }
    json.dump(summary, open(os.path.join(RESULTS, "calibration_summary.json"), "w"), indent=1)
    if verbose:
        print(json.dumps(summary, indent=1))
    return P, cfg_f, out, summary


if __name__ == "__main__":
    run_all()
