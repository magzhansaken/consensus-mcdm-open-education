"""Transferability of the anchor measurements across benchmark environments (Section 4.10).

The anchors were measured on different hosts and networks (Table 7 of the article):
  Fabric    one laptop host (Intel i7-8559U, 4 cores/8 threads), Docker, 4 orderers, Caliper, about 1000-B transactions
  Besu      one workstation (Intel i9-14900KS), Docker, 4 validators, 5000-transaction burst, EVM transactions
  CometBFT  DigitalOcean, one region, 200 nodes (175 validators), kvstore application, 1024-B transactions
  HotStuff  AWS, geo-distributed, 10-50 nodes, empty application, 512-B transactions
The model replaces the environment-specific inputs that the scenario defines (round-trip time, bandwidth, node count,
transaction size, run protocol, arrival process, application tier) and keeps the constants that the anchors determine
(per-block and per-transaction processing a, o, m, o_pool; the gossip hop delay g; the Fabric-calibrated
application tier). This module perturbs exactly those carried-over constants:
  hw_<family>   multiplies a, o, m (and o_pool) of a platform family: host speed of its anchor;
  app           multiplies the shared application tier (c_block, c_tx, CPU per transaction);
  net_cometbft  multiplies the CometBFT gossip hop delay, calibrated on a 175-validator cloud network.
"""
from __future__ import annotations
import copy
import numpy as np

FAMILIES = {"fabric": ("raft", "smartbft"), "besu": ("ibft2", "qbft", "clique"), "cometbft": ("cometbft",), "hotstuff": ("hotstuff",)}
FACTORS = {  # name: (low, high, kind)
    "hw_fabric": (0.5, 2.0, "log"), "hw_besu": (0.5, 2.0, "log"), "hw_cometbft": (0.5, 2.0, "log"), "hw_hotstuff": (0.5, 2.0, "log"),
    "app": (0.5, 2.0, "log"), "net_cometbft": (0.1, 1.0, "log"),
}
ENVIRONMENTS = [
    ("Hyperledger Fabric 3.1.4 (Raft, SmartBFT)", "one laptop host, Intel i7-8559U (4 cores, 8 threads), Docker", "4 orderers on one host; injected RTT 0--100 ms", "about 1000 B, Caliper at a fixed rate, 90-s runs", "chaincode and CouchDB on peers"),
    ("Hyperledger Besu 25.7 (Clique, QBFT, IBFT 2.0)", "one workstation, Intel i9-14900KS, Docker", "4 validators on one host", "EVM transactions, 5000-transaction burst", "EVM execution"),
    ("CometBFT v1", "DigitalOcean cloud, one region", "200 nodes, 175 validators", "1024 B, 90-s runs at fixed rates", "key-value store"),
    ("2-chain HotStuff v0.3.0", "AWS instances, geo-distributed", "10, 20, 50 nodes; wide-area network", "512 B, fixed-rate clients", "none (batches only)"),
]


def apply(P, app_tier, f: dict):
    """Return (P', app_tier') with the carried-over constants multiplied by the factors in f (missing -> 1)."""
    Q = copy.deepcopy(P)
    for fam, keys in FAMILIES.items():
        s = f.get(f"hw_{fam}", 1.0)
        if s != 1.0:
            for k in keys:
                Q[k].a *= s; Q[k].o *= s; Q[k].m *= s; Q[k].o_pool *= s
    if f.get("net_cometbft", 1.0) != 1.0:
        Q["cometbft"].g *= f["net_cometbft"]
    app = dict(app_tier)
    s = f.get("app", 1.0)
    if s != 1.0:
        app = {k: (v * s if k in ("c_block", "c_tx", "cpu_app_tx") else v) for k, v in app.items()}
    return Q, app


def sample(n: int, rng: np.random.Generator) -> list[dict]:
    """Latin hypercube over FACTORS, log-uniform."""
    names = list(FACTORS)
    u = np.empty((n, len(names)))
    for j in range(len(names)):
        u[:, j] = (rng.permutation(n) + rng.uniform(size=n)) / n
    out = []
    for i in range(n):
        d = {}
        for j, nm in enumerate(names):
            lo, hi, _ = FACTORS[nm]
            d[nm] = float(np.exp(np.log(lo) + u[i, j] * (np.log(hi) - np.log(lo))))
        out.append(d)
    return out
