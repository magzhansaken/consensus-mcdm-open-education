"""Decision matrix for the open-education scenario (11 criteria, aligned with the project design /
Tashatov et al. 2026 criterion set).

C1  Throughput (tx/s, benefit)          sustainable throughput: achieved tx/s at the stress load (185 tx/s)
                                        minus losses; = offered if nothing fails (simulation)
C2  Latency (s, cost)                   mean commit latency at the deadline-day peak (108 tx/s) (simulation)
C3  Finalization (s, cost)              mean time to finality at nominal load (56 tx/s); deterministic
                                        protocols: = latency; Clique: + (n/2+1) block periods (simulation)
C4  Power (Wh per 1000 tx, cost)        network CPU energy proxy: whole-network CPU-seconds per 1000 tx x
                                        P_core (assumption E1: 15 W per fully loaded vCPU) / 3600 (simulation)
C5  Computing (%, cost)                 peak per-node CPU utilisation at the stress load, 4-vCPU nodes (simulation)
C6  Storage (bytes/tx, cost)            ledger bytes appended per tx per node at nominal load (model, from
                                        documented header/signature sizes)
C7  Fault tolerance (%, benefit)        availability (share of tx committed within 30 s) in the fault scenario:
                                        leader crash at 120 s and a second node crash at 300 s of a 600-s run
                                        at peak load (simulation; both faults tolerated by every protocol at n=7)
C8  Attack resistance (0-4, benefit)    documented properties: BFT safety (f<n/3), deterministic finality,
                                        proposer rotation, built-in equivocation evidence (Table 2 sources)
C9  Decentralization (0-1, benefit)     1 - Herfindahl index of block-proposer shares (simulation)
C10 Manageability (0-4, benefit)        documented: dynamic validator-set change without restart, metrics
                                        endpoint, Caliper/benchmark adapter, production LTS release (Table 2)
C11 Operational cost (USD/month, cost)  n x vCPU-hours priced at a public list price (assumption E2:
                                        0.04 USD per vCPU-hour, 0.10 USD per GB-month), sized from C5 and C6
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from .protocols import Protocol, quorum
from .sim import simulate, SimConfig
from .workload import WorkloadParams, summary, arrivals

CRITERIA = ["C1 Throughput", "C2 Latency", "C3 Finalization", "C4 Power", "C5 Computing", "C6 Storage",
            "C7 Fault tolerance", "C8 Attack resistance", "C9 Decentralization", "C10 Manageability", "C11 Operational cost"]
TYPES = np.array([1, -1, -1, -1, -1, -1, 1, 1, 1, 1, -1])
UNITS = ["tx/s", "s", "s", "Wh/1000 tx", "% of 4 vCPU", "bytes/tx", "% committed", "score 0-4", "1-HHI", "score 0-4", "USD/month"]

P_CORE_W = 15.0          # E1
PRICE_VCPU_H = 0.04      # E2
PRICE_GB_MONTH = 0.10    # E2
HOURS_MONTH = 730.0

# documented ordinal scores (sources in Table 2 of the paper)
ATTACK = {"raft": 1, "smartbft": 3, "ibft2": 3, "qbft": 3, "clique": 1, "cometbft": 4, "hotstuff": 3}
MANAGE = {"raft": 4, "smartbft": 4, "ibft2": 3, "qbft": 4, "clique": 3, "cometbft": 3, "hotstuff": 0}
# IBFT 2.0: validator changes only by on-chain voting; QBFT adds contract-based validator selection (Besu docs);
# Clique: deprecated and removed in Besu 26.4; HotStuff reference: static committee, no metrics/tooling.


@dataclass
class Scenario:
    n: int = 7
    rtt_s: float = 0.002
    cores: int = 4
    duration_s: float = 600.0
    warmup_s: float = 60.0
    apply_app_tier: bool = True
    exclude_dead: bool = False      # operator removes a validator from rotation after its first detected failure
    wl: WorkloadParams = None

    def __post_init__(self):
        if self.wl is None:
            self.wl = WorkloadParams()


def _cfg(sc: Scenario, app_tier, faults=None):
    return SimConfig(n=sc.n, rtt_s=sc.rtt_s, duration_s=sc.duration_s, warmup_s=sc.warmup_s, timeout_s=30.0,
                     tx_bytes=sc.wl.tx_bytes, c_block=app_tier["c_block"], c_tx=app_tier["c_tx"], cpu_app_tx=app_tier["cpu_app_tx"],
                     cores=sc.cores, faults=faults or [], apply_app_tier=sc.apply_app_tier, exclude_dead=sc.exclude_dead)


def evaluate(p: Protocol, sc: Scenario, app_tier: dict, seed: int) -> dict:
    """One seed -> the 11 criterion values and the raw simulation outputs."""
    loads = summary(sc.wl)
    rng = np.random.default_rng(seed)
    out = {}
    # stress load: capacity and peak utilisation
    cfg = _cfg(sc, app_tier)
    r_stress = simulate(p, cfg, arrivals(loads["stress_tps"], cfg.duration_s, rng), seed)
    r_peak = simulate(p, cfg, arrivals(loads["deadline_peak_tps"], cfg.duration_s, rng), seed + 1)
    r_nom = simulate(p, cfg, arrivals(loads["nominal_tps"], cfg.duration_s, rng), seed + 2)
    # fault scenario A13: proposer crash at 120 s and a second crash at 300 s; with n = 4 only the first,
    # so that every alternative can tolerate the injected faults (BFT tolerates (n-1)//3 crashes)
    n_crash = min(2, (sc.n - 1) // 3)
    faults = [(120.0, 0), (300.0, 1)][:n_crash]
    r_fault = simulate(p, _cfg(sc, app_tier, faults), arrivals(loads["deadline_peak_tps"], cfg.duration_s, rng), seed + 3)
    # C1 sustainable throughput: committed tx/s at the stress load
    out["C1 Throughput"] = r_stress.throughput_tps
    out["C2 Latency"] = r_peak.lat_mean_s
    out["C3 Finalization"] = r_nom.finality_mean_s
    out["C4 Power"] = r_nom.cpu_network_s_per_1k * P_CORE_W / 3600.0
    out["C5 Computing"] = r_stress.cpu_util_peak_pct
    out["C6 Storage"] = r_nom.ledger_bytes_per_tx
    out["C7 Fault tolerance"] = 100.0 * r_fault.availability
    out["C8 Attack resistance"] = ATTACK[p.key]
    out["C9 Decentralization"] = 1.0 - r_nom.proposer_hhi
    out["C10 Manageability"] = MANAGE[p.key]
    # C11: monthly cost of n nodes sized to the peak utilisation (vCPU-hours) + ledger growth
    daily_tx = loads["daily_tx_normal"]
    vcpu_needed = max(1.0, np.ceil(sc.cores * r_stress.cpu_util_peak_pct / 100.0 / 0.7))  # 70 % headroom
    gb_month = daily_tx * 30 * r_nom.ledger_bytes_per_tx / 1e9
    out["C11 Operational cost"] = sc.n * (vcpu_needed * PRICE_VCPU_H * HOURS_MONTH + gb_month * PRICE_GB_MONTH)
    raw = {"stress": r_stress.as_dict(), "peak": r_peak.as_dict(), "nominal": r_nom.as_dict(), "fault": r_fault.as_dict()}
    return {"values": out, "raw": raw}


def decision_matrices(P: dict, sc: Scenario, app_tier: dict, seeds=range(30)):
    """Returns names, X_mean (m x k), X_samples (S x m x k), CI95 half-widths (m x k), raw per seed."""
    keys = list(P)
    S = len(list(seeds))
    X = np.zeros((S, len(keys), len(CRITERIA)))
    raws = {k: [] for k in keys}
    for si, seed in enumerate(seeds):
        for ai, k in enumerate(keys):
            e = evaluate(P[k], sc, app_tier, seed=1000 * seed + 7)
            X[si, ai] = [e["values"][c] for c in CRITERIA]
            raws[k].append(e["raw"])
    mean = X.mean(axis=0)
    sd = X.std(axis=0, ddof=1) if S > 1 else np.zeros_like(mean)
    from scipy.stats import t as tdist
    hw = tdist.ppf(0.975, S - 1) * sd / np.sqrt(S) if S > 1 else np.zeros_like(mean)
    return keys, mean, X, hw, raws
