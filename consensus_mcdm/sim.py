"""Discrete-event simulation of block formation, consensus and commit for leader-based protocols.

Model (per block k):
  cut/propose time  t_k      - 'size_timeout': block closes when B_max tx are pending or T_b s after
                               the first pending tx (Fabric block cutter, HotStuff mempool batch);
                             - 'period': a block is proposed every T_b s (Besu block period,
                               CometBFT timeout_commit), carrying up to B_max pending tx.
  consensus service s_k      = a + o*B_k + m*M_node + rounds*RTT_k + hops*g + bytes_k/BW
                               (+ view_change_s when the scheduled leader is dead)
                               executed by a single consensus pipeline (occupancy s_k, or s_k/rounds
                               for the pipelined 2-chain HotStuff);
  commit/application service = c_block + c_tx*B_k on the executing node (single FIFO server);
  latency(tx)                = commit_time - arrival; finality adds k_final*T_b for probabilistic
                               protocols (Clique).
Randomness: Poisson arrivals, log-normal RTT (CV 0.2), log-normal processing jitter (CV 0.1);
everything is driven by numpy Generator(seed) so runs are reproducible.
"""
from __future__ import annotations
import math
import numpy as np
from dataclasses import dataclass, field
from .protocols import Protocol, quorum, signatures_per_block, sigs_verified_per_node


@dataclass
class SimConfig:
    n: int = 7                       # consensus participants
    rtt_s: float = 0.002             # campus LAN round-trip time
    bw_bps: float = 1e9              # link bandwidth
    tx_bytes: int = 1000
    duration_s: float = 600.0
    warmup_s: float = 60.0
    timeout_s: float = 30.0          # client-side transaction timeout (Caliper-like)
    c_block: float = 0.02            # application tier: per-block commit cost (s)
    c_tx: float = 0.0035             # application tier: per-tx validation/commit (s/tx)
    cpu_app_tx: float = 0.006        # application tier CPU-seconds per tx per executing node
    cores: int = 4                   # vCPU per node (for utilisation %)
    faults: list = field(default_factory=list)  # [(t_fail_s, node_index), ...]
    apply_app_tier: bool = True      # add application tier to protocols whose anchors lack it
    exclude_dead: bool = False       # after a detected leader failure the dead node is removed from rotation (operator action)


@dataclass
class SimResult:
    throughput_tps: float
    offered_tps: float
    lat_mean_s: float
    lat_p50_s: float
    lat_p95_s: float
    lat_max_s: float
    fail_rate: float
    finality_mean_s: float
    blocks: int
    mean_block_tx: float
    cpu_cons_s_per_1k: float       # consensus-node CPU-seconds per 1000 tx (per node, averaged)
    cpu_app_s_per_1k: float        # executing-node CPU-seconds per 1000 tx (per node)
    cpu_network_s_per_1k: float    # whole network CPU-seconds per 1000 tx
    cpu_util_peak_pct: float       # peak per-node utilisation (%) of the busiest node
    ledger_bytes_per_tx: float     # bytes appended to each node's ledger per tx
    proposer_hhi: float            # Herfindahl index of proposer shares (1 = single leader)
    availability: float            # share of tx committed within timeout
    n_committed: int

    def as_dict(self):
        return self.__dict__.copy()


def _lognormal(rng, mean, cv, size=None):
    if mean <= 0:
        return np.zeros(size) if size else 0.0
    sigma2 = math.log(1 + cv * cv)
    mu = math.log(mean) - sigma2 / 2
    return rng.lognormal(mu, math.sqrt(sigma2), size)


def simulate(p: Protocol, cfg: SimConfig, arrivals: np.ndarray, seed: int = 0, trace: dict | None = None) -> SimResult:
    rng = np.random.default_rng(seed)
    n = cfg.n
    arr = np.asarray(arrivals, dtype=float)
    N = len(arr)
    if N == 0:
        raise ValueError("no arrivals")
    q = quorum(p, n)
    hops = max(1, math.ceil(math.log(n) / math.log(10))) if p.msg_pattern == "gossip" and n > 1 else 0  # fan-out 10 gossip
    pipelined = (p.key == "hotstuff")
    add_app = cfg.apply_app_tier and (not p.app_tier)
    # application tier (commit) applies to Fabric (anchors include peers) and, by assumption, to
    # protocols whose anchors have no execution layer; Besu anchors already include EVM execution.
    app_commit = (p.platform.startswith("Hyperledger Fabric")) or add_app
    c_block = cfg.c_block if app_commit else 0.0
    c_tx = cfg.c_tx if app_commit else 0.0

    # leader schedule
    dead = {}  # node -> time of death
    removed = set()  # nodes taken out of the proposer rotation by the operator
    faults = sorted(cfg.faults)
    leader = 0
    decisions_on_leader = 0

    # outputs
    commit_time = np.full(N, np.inf)
    final_time = np.full(N, np.inf)
    cpu_node = np.zeros(n)              # consensus CPU seconds per node
    cpu_app_total = 0.0                 # app-tier CPU seconds per executing node
    proposer_count = np.zeros(n)
    ledger_bytes = 0.0
    blocks = 0
    block_sizes = []

    i0 = 0
    cons_free = 0.0
    commit_free = 0.0
    last_propose = -np.inf
    t_cut_prev = 0.0
    sig_block = signatures_per_block(p, n)
    M_leader = sigs_verified_per_node(p, n, True)
    M_follower = sigs_verified_per_node(p, n, False)
    k_final = (n // 2 + 1) if p.finality == "probabilistic" else 0
    T_b = p.T_b
    B_max = p.B_max

    dropped = np.zeros(N, dtype=bool)
    cap = p.mempool_max
    # admission/mempool processing: a single FIFO server per node that makes a transaction eligible
    # for inclusion o_pool seconds after it (or its predecessor) is admitted; off the block critical path
    if p.o_pool > 0:
        elig = np.empty(N); t_e = 0.0
        max_wait = cap * p.o_pool if np.isfinite(cap) else np.inf   # pool holds at most cap admitted tx
        for i in range(N):
            start = max(arr[i], t_e)
            if start - arr[i] > max_wait:          # pool full on arrival -> rejected
                dropped[i] = True; elig[i] = t_e     # keep elig monotone; excluded through `dropped`
                continue
            t_e = start + p.o_pool
            elig[i] = t_e
    else:
        elig = arr

    def admit(i_from, t_now):
        """Admission control: transactions that arrive while the pending pool holds >= cap
        transactions are rejected (CometBFT mempool full, Besu txpool full)."""
        if not np.isfinite(cap):
            return
        i_arr = np.searchsorted(elig, t_now, side="right")
        backlog = i_arr - i_from
        if backlog > cap:
            dropped[i_from + int(cap):i_arr] = True

    while i0 < N:
        if dropped[i0]:
            i0 += 1
            continue
        # ---------- block formation ----------
        if p.batching == "size_timeout":
            t_first = elig[i0]
            j_full = i0 + B_max - 1
            t_full = elig[j_full] if j_full < N else np.inf
            t_cut = min(t_full, t_first + T_b)
            # all tx eligible by t_cut, at most B_max
            i1 = min(np.searchsorted(elig, t_cut, side="right"), i0 + B_max)
            propose_t = t_cut
        else:  # period
            if p.key == "clique":
                propose_t = max(last_propose + T_b, elig[i0]) if last_propose > -np.inf else elig[i0] + T_b * rng.uniform()
            elif p.key == "cometbft":
                propose_t = max(cons_free + T_b, elig[i0]) if blocks else elig[i0] + T_b * rng.uniform()
            elif p.key == "hotstuff":   # proposer-driven: next proposal as soon as the pipeline slot frees
                propose_t = max(cons_free, elig[i0] + T_b) if blocks else elig[i0] + T_b
            else:  # Besu IBFT2/QBFT: minimum block period after the previous block, after its commit
                propose_t = max(last_propose + T_b, cons_free, elig[i0]) if blocks else elig[i0] + T_b * rng.uniform()
            i1 = min(np.searchsorted(elig, propose_t, side="right"), i0 + B_max)
            if i1 <= i0:  # nothing pending at this tick: propose at the first tick after the next arrival
                k_ticks = math.ceil((elig[i0] - propose_t) / T_b) if T_b > 0 else 0
                propose_t = propose_t + max(k_ticks, 0) * T_b
                if propose_t < elig[i0]:
                    propose_t = elig[i0]
                i1 = min(np.searchsorted(elig, propose_t, side="right"), i0 + B_max)
                if i1 <= i0:
                    i1 = i0 + 1
        admit(i0, propose_t)
        idx = np.arange(i0, i1)
        idx = idx[~dropped[idx]]
        if len(idx) == 0:
            # everything pending was rejected; move past the dropped run
            while i0 < N and dropped[i0]:
                i0 += 1
            continue
        B = len(idx)
        last_propose = propose_t

        # ---------- leader liveness ----------
        extra = 0.0
        for (tf, node) in faults:
            if tf <= propose_t:
                dead[node] = tf
        alive = n - len(dead)
        if alive < q["quorum"]:
            # no quorum: the system halts; remaining tx time out
            break
        if leader in dead:
            # leader failure detected -> view change / election, then next alive leader
            extra = p.view_change_s * (1.0 + 0.1 * rng.standard_normal())
            extra = max(extra, 0.0)
            if cfg.exclude_dead:
                removed.add(leader)
            leader = _next_alive(leader, n, dead)
            decisions_on_leader = 0

        # ---------- consensus ----------
        rtt = _lognormal(rng, cfg.rtt_s, 0.2) if cfg.rtt_s > 0 else 0.0
        block_bytes = B * (cfg.tx_bytes + p.envelope_bytes) + p.header_bytes + sig_block * p.sig_bytes
        # link capacity only (no per-connection window cap: benchmarks use parallel streams/tuned windows)
        bw_eff = cfg.bw_bps
        o_eff = p.o * (1.0 + p.nu * max(n - p.anchor_n, 0))   # per-tx work growing with the validator count
        proc_leader = p.a + o_eff * B + p.m * M_leader
        proc_follower = p.a + o_eff * B + p.m * M_follower
        proc = proc_leader * _lognormal(rng, 1.0, 0.1)
        # block dissemination: leader fan-out for leader-broadcast protocols; ~2x for gossip and for
        # mempool-batch protocols (HotStuff reference: batches are disseminated by their producers,
        # proposals carry digests)
        fanout = 2 if (p.msg_pattern == "gossip" or p.key == "hotstuff") else max(1, n - 1)
        t_xfer = max(block_bytes * 8.0 * fanout / cfg.bw_bps, block_bytes * 8.0 / bw_eff)  # NIC cap vs per-connection cap
        s_cons = proc + p.rounds * rtt + hops * p.g + t_xfer
        start = max(propose_t, cons_free) + extra
        cons_done = start + s_cons
        cons_free = start + (s_cons / 2.0 if pipelined else s_cons)   # 2-chain pipelining: a new proposal every other round
        # CPU accounting (consensus side)
        cpu_node[leader] += p.kappa * (proc_leader + p.o_pool * B)
        for v in range(n):
            if v != leader and v not in dead:
                cpu_node[v] += p.kappa * (proc_follower + p.o_pool * B)

        # ---------- commit / application tier ----------
        s_commit = (c_block + c_tx * B) * _lognormal(rng, 1.0, 0.1) if (c_block or c_tx) else 0.0
        cstart = max(cons_done, commit_free)
        cdone = cstart + s_commit
        commit_free = cdone
        if cfg.apply_app_tier:
            # the portal's application logic is the same on every platform: one Fabric-calibrated
            # CPU cost per transaction per executing node (assumption A5), independent of the
            # platform-specific latency-side execution constants
            cpu_app_total += cfg.cpu_app_tx * B
        commit_time[idx] = cdone
        final_time[idx] = cdone + k_final * T_b

        proposer_count[leader] += 1
        ledger_bytes += block_bytes
        blocks += 1
        block_sizes.append(B)

        # leader rotation
        if p.leader == "rotating":
            decisions_on_leader += 1
            if decisions_on_leader >= p.rotation_every:
                leader = (leader + 1) % n   # round-robin; a dead successor is detected at the next block
                while leader in removed:
                    leader = (leader + 1) % n
                decisions_on_leader = 0
        i0 = int(idx[-1]) + 1

    # ---------- metrics ----------
    if trace is not None:   # optional per-transaction trace for diagnostic figures
        trace["arrival"] = arr.copy(); trace["commit"] = commit_time.copy(); trace["dropped"] = dropped.copy()
    lat = commit_time - arr
    measured = (arr >= cfg.warmup_s) & (arr <= cfg.duration_s)
    # a transaction counts as successful if it commits within the client timeout; throughput is the
    # number of transactions committed inside the measurement window (Caliper-style)
    ok = measured & np.isfinite(lat) & (lat <= cfg.timeout_s)
    in_window = measured & np.isfinite(commit_time) & (commit_time <= cfg.duration_s)
    n_meas = int(measured.sum())
    n_ok = int(ok.sum())
    span = max(cfg.duration_s - cfg.warmup_s, 1e-9)
    lat_ok = lat[ok]
    fin_ok = (final_time - arr)[ok]
    tps = int((ok & in_window).sum()) / span
    offered = n_meas / span
    tx_total = max(N, 1)
    per_node_cons = cpu_node.mean() / tx_total * 1000.0
    cpu_app_per_node = cpu_app_total / tx_total * 1000.0
    exec_nodes = n  # one executing node per organisation
    cons_nodes = n
    network = cons_nodes * per_node_cons + (exec_nodes * cpu_app_per_node if cfg.apply_app_tier else 0.0)
    busiest = (cpu_node.max() / cfg.duration_s + (cpu_app_total / cfg.duration_s if cfg.apply_app_tier else 0.0)) / cfg.cores * 100.0
    shares = proposer_count / max(proposer_count.sum(), 1)
    hhi = float((shares ** 2).sum())
    return SimResult(
        throughput_tps=tps, offered_tps=offered,
        lat_mean_s=float(lat_ok.mean()) if n_ok else float("nan"),
        lat_p50_s=float(np.percentile(lat_ok, 50)) if n_ok else float("nan"),
        lat_p95_s=float(np.percentile(lat_ok, 95)) if n_ok else float("nan"),
        lat_max_s=float(lat[measured & np.isfinite(lat)].max()) if n_meas else float("nan"),
        fail_rate=1.0 - n_ok / max(n_meas, 1),
        finality_mean_s=float(fin_ok.mean()) if n_ok else float("nan"),
        blocks=blocks, mean_block_tx=float(np.mean(block_sizes)) if block_sizes else 0.0,
        cpu_cons_s_per_1k=per_node_cons, cpu_app_s_per_1k=cpu_app_per_node if cfg.apply_app_tier else 0.0,
        cpu_network_s_per_1k=network, cpu_util_peak_pct=busiest,
        ledger_bytes_per_tx=ledger_bytes / tx_total, proposer_hhi=hhi,
        availability=n_ok / max(n_meas, 1), n_committed=n_ok,
    )


def _next_alive(leader, n, dead):
    for k in range(1, n + 1):
        cand = (leader + k) % n
        if cand not in dead:
            return cand
    return leader
