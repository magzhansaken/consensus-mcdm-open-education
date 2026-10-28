"""Protocol catalogue: structural (documented) properties and implementation constants.

Structural properties come from the protocol specifications / platform documentation cited
in the paper (Table 2). Implementation constants (a, o, m, kappa, ...) are *calibrated* by
consensus_mcdm/calibrate.py on the published measurements in data/anchors.csv; the values below are
only starting points and are overwritten by results/calibrated_params.json when present.

Notation (per block, per node):
  service time  s = a + o*B + m*M_node + R*RTT + bytes/BW
  CPU seconds   c = kappa * (a + o*B + m*M_node) + c_app*B (application tier, see sim.py)
where B = transactions in the block, M_node = signatures verified by a node per block.
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
import json, os

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


@dataclass
class Protocol:
    key: str
    name: str
    platform: str
    fault_model: str            # 'crash' | 'byzantine'
    batching: str               # 'size_timeout' | 'period'
    rounds: int                 # sequential communication rounds per block (RTT multiples)
    msg_pattern: str            # 'linear' | 'all2all' | 'gossip' | 'broadcast'
    leader: str                 # 'stable' | 'rotating'
    finality: str               # 'deterministic' | 'probabilistic'
    byz_safe: int               # 1 if safety holds with f<n/3 Byzantine validators
    evidence: int               # 1 if built-in equivocation evidence/accountability
    sig_bytes: int = 72         # bytes of one commit signature/seal
    header_bytes: int = 600     # block header + metadata (excl. signatures)
    rotation_every: int = 1     # blocks per leader before rotation (rotating leaders)
    envelope_bytes: int = 200   # platform overhead per tx on the wire/ledger (signatures, certificates)
    mempool_max: float = float('inf')  # transactions admitted to the pending pool before new ones are rejected
    # ---- defaults (platform documentation) ----
    T_b: float = 2.0            # batch timeout / block period (s)
    B_max: int = 100            # max tx per block (MaxMessageCount / block capacity)
    view_change_s: float = 10.0 # leader-failure detection + replacement time (s)
    # ---- calibrated implementation constants (overwritten by calibration) ----
    a: float = 0.05             # fixed per-block processing (s)
    o: float = 0.002            # per-tx processing inside the block critical path (s/tx)
    o_pool: float = 0.0         # per-tx admission/mempool processing on every node, off the critical path (s/tx)
    nu: float = 0.0             # relative growth of the per-tx work per additional validator beyond anchor_n
    m: float = 0.0003           # per verified signature (s)
    kappa: float = 0.5          # CPU-seconds per second of processing (node utilisation factor)
    g: float = 0.0              # per-hop gossip delay (s), gossip protocols only
    app_tier: bool = True       # application execution included in anchors? (False -> add c_app)
    anchor_rtt_s: float = 0.001 # RTT of the calibration testbed
    anchor_n: int = 4
    source: str = ""

    def to_dict(self):
        return asdict(self)


def catalogue() -> dict[str, Protocol]:
    P = {}
    P["raft"] = Protocol("raft", "Raft", "Hyperledger Fabric 3.1 (etcdraft)", "crash", "size_timeout",
                         rounds=1, msg_pattern="linear", leader="stable", finality="deterministic",
                         byz_safe=0, evidence=0, header_bytes=600, sig_bytes=300, rotation_every=10**9,
                         T_b=2.0, B_max=100, view_change_s=5.0, a=0.08, o=0.0035, m=0.0002, kappa=0.3, envelope_bytes=2500,
                         app_tier=True, source="Fabric docs (etcdraft: ElectionTick 10 x 500 ms); 945")
    P["smartbft"] = Protocol("smartbft", "SmartBFT", "Hyperledger Fabric 3.1 (BFT orderer)", "byzantine", "size_timeout",
                             rounds=3, msg_pattern="all2all", leader="rotating", finality="deterministic",
                             byz_safe=1, evidence=0, header_bytes=600, sig_bytes=300, rotation_every=3,
                             T_b=2.0, B_max=100, view_change_s=20.0, a=0.15, o=0.004, m=0.0003, kappa=0.5, envelope_bytes=2500,
                             app_tier=True, source="Fabric 3 BFT config (ViewChangeTimeout 20 s; DecisionsPerLeader 3 library default); 945")
    P["ibft2"] = Protocol("ibft2", "IBFT 2.0", "Hyperledger Besu 25.7", "byzantine", "period",
                          rounds=3, msg_pattern="all2all", leader="rotating", finality="deterministic",
                          byz_safe=1, evidence=0, header_bytes=550, sig_bytes=65, rotation_every=1,
                          T_b=2.0, B_max=220, view_change_s=4.0, a=0.05, o=0.004, m=0.0003, kappa=0.5, mempool_max=4096,
                          app_tier=True, source="Besu docs (blockperiodseconds 2 and requesttimeoutseconds 4 as in the QBFT/IBFT tutorial genesis; tx-pool-max-size 4096); Appl. Sci. 2026 9631")
    P["qbft"] = Protocol("qbft", "QBFT", "Hyperledger Besu 25.7", "byzantine", "period",
                         rounds=3, msg_pattern="all2all", leader="rotating", finality="deterministic",
                         byz_safe=1, evidence=0, header_bytes=550, sig_bytes=65, rotation_every=1,
                         T_b=2.0, B_max=220, view_change_s=4.0, a=0.05, o=0.004, m=0.0003, kappa=0.5, mempool_max=4096,
                         app_tier=True, source="Besu docs (blockperiodseconds 2, requesttimeoutseconds 4 as in the tutorial genesis; contract-based validator selection); Appl. Sci. 2026 9631")
    P["clique"] = Protocol("clique", "Clique (PoA)", "Hyperledger Besu 25.7", "crash", "period",
                           rounds=0, msg_pattern="broadcast", leader="rotating", finality="probabilistic",
                           byz_safe=0, evidence=0, header_bytes=550, sig_bytes=65, rotation_every=1,
                           T_b=2.0, B_max=423, view_change_s=1.0, a=0.03, o=0.002, m=0.0003, kappa=0.5, mempool_max=4096,
                           app_tier=True, source="Besu docs (Clique; removed in Besu 26.4); Appl. Sci. 2026 9631")
    P["cometbft"] = Protocol("cometbft", "CometBFT (Tendermint)", "CometBFT v1", "byzantine", "period",
                             rounds=3, msg_pattern="gossip", leader="rotating", finality="deterministic",
                             byz_safe=1, evidence=1, header_bytes=700, sig_bytes=72, rotation_every=1,
                             T_b=1.0, B_max=20000, view_change_s=3.0, a=0.05, o=0.00005, o_pool=0.0017, m=0.0002, kappa=0.5, g=0.3, envelope_bytes=100, mempool_max=5000,
                             app_tier=False, anchor_rtt_s=0.002, anchor_n=175,
                             source="CometBFT docs (timeout_commit 1 s, timeout_propose 3 s; evidence module); QA v1 report")
    P["hotstuff"] = Protocol("hotstuff", "HotStuff (2-chain)", "reference implementation (asonnino/hotstuff v0.3.0)", "byzantine", "period",
                             rounds=3, msg_pattern="linear", leader="rotating", finality="deterministic",
                             byz_safe=1, evidence=0, header_bytes=400, sig_bytes=64, rotation_every=1,
                             T_b=0.1, B_max=100000, view_change_s=5.0, a=0.05, o=0.00001, m=0.0002, kappa=0.5, envelope_bytes=100,
                             app_tier=False, anchor_rtt_s=0.2, anchor_n=10,
                             source="Yin et al. 2019 (HotStuff); benchmark settings in repo (timeout 5 s, batch 500 kB/100 ms); rounds = batch dissemination + 2-chain commit, pipeline depth 2")
    return P


def load_calibrated(P: dict[str, Protocol], path: str | None = None) -> dict[str, Protocol]:
    path = path or os.path.join(RESULTS, "calibrated_params.json")
    if os.path.exists(path):
        cal = json.load(open(path))
        for k, d in cal.get("protocols", {}).items():
            if k in P:
                for f, v in d.items():
                    if hasattr(P[k], f):
                        setattr(P[k], f, v)
    return P


def quorum(p: Protocol, n: int) -> dict:
    """Documented tolerance figures for n participants."""
    if p.fault_model == "byzantine":
        f = (n - 1) // 3
        return {"f_byz": f, "f_crash": f, "quorum": 2 * f + 1}
    f = (n - 1) // 2
    return {"f_byz": 0, "f_crash": f, "quorum": n // 2 + 1}   # majority quorum (also correct for even n)


def signatures_per_block(p: Protocol, n: int) -> int:
    """Commit signatures stored per block (ledger overhead)."""
    if p.key == "raft":
        return 1                       # orderer signature in block metadata
    if p.key == "clique":
        return 1                       # signer seal
    if p.fault_model == "byzantine":
        return 2 * ((n - 1) // 3) + 1  # quorum certificate (2f+1 signatures)
    return 1


def sigs_verified_per_node(p: Protocol, n: int, is_leader: bool) -> int:
    """Signatures a node verifies per block (drives m*M_node)."""
    if p.msg_pattern == "linear":        # Raft, HotStuff: leader collects, followers verify QC/entry
        return (n - 1) if is_leader else max(1, (2 * ((n - 1) // 3) + 1) if p.fault_model == "byzantine" else 1)
    if p.msg_pattern in ("all2all", "gossip"):  # 2 vote rounds all-to-all + proposal
        return 2 * (n - 1) + 1
    return 1                              # Clique: one seal
