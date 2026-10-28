import numpy as np
from consensus_mcdm.protocols import catalogue, quorum, signatures_per_block
from consensus_mcdm.sim import simulate, SimConfig
from consensus_mcdm.workload import arrivals, WorkloadParams, hourly_rate, summary, arrivals_profile
from consensus_mcdm.smaa import smaa2, holistic_acceptability
from consensus_mcdm.robustness import ws_coefficient, leave_one_out, rank_agreement


def _run(key, lam=50, seed=1, **kw):
    P = catalogue(); p = P[key]
    cfg = SimConfig(n=4, rtt_s=0.001, duration_s=120, warmup_s=20, **kw)
    arr = arrivals(lam, cfg.duration_s, np.random.default_rng(seed))
    return simulate(p, cfg, arr, seed=seed)


def test_reproducible_with_seed():
    a = _run("raft", seed=3); b = _run("raft", seed=3)
    assert a.as_dict() == b.as_dict()


def test_throughput_never_exceeds_offered_and_latency_positive():
    for key in catalogue():
        r = _run(key, lam=30)
        assert r.throughput_tps <= r.offered_tps * 1.05
        assert r.lat_mean_s > 0 and r.lat_p95_s >= r.lat_p50_s


def test_stable_leader_has_hhi_one_and_rotation_lower():
    r = _run("raft"); assert np.isclose(r.proposer_hhi, 1.0)
    r2 = _run("ibft2"); assert r2.proposer_hhi < 0.5


def test_probabilistic_finality_longer_than_latency():
    r = _run("clique"); assert r.finality_mean_s > r.lat_mean_s + 1.0


def test_quorum_rules():
    P = catalogue()
    assert quorum(P["raft"], 7) == {"f_byz": 0, "f_crash": 3, "quorum": 4}
    assert quorum(P["qbft"], 7) == {"f_byz": 2, "f_crash": 2, "quorum": 5}
    assert quorum(P["raft"], 4)["quorum"] == 3 and quorum(P["raft"], 10)["quorum"] == 6   # majority for even n
    assert quorum(P["qbft"], 4)["quorum"] == 3 and quorum(P["qbft"], 10)["quorum"] == 7
    assert signatures_per_block(P["qbft"], 7) == 5 and signatures_per_block(P["raft"], 7) == 1


def test_leader_crash_adds_delay_then_recovers():
    P = catalogue(); p = P["raft"]
    cfg = SimConfig(n=4, rtt_s=0.001, duration_s=200, warmup_s=20, faults=[(60.0, 0)])
    arr = arrivals(40, cfg.duration_s, np.random.default_rng(5))
    r = simulate(p, cfg, arr, seed=5)
    assert r.lat_max_s > p.view_change_s * 0.8          # the election pause is visible
    assert r.availability > 0.9                           # but the system recovers


def test_quorum_loss_halts():
    P = catalogue(); p = P["qbft"]  # n=4 tolerates 1 Byzantine/crash; 2 crashes -> no quorum
    cfg = SimConfig(n=4, rtt_s=0.001, duration_s=120, warmup_s=10, faults=[(30.0, 0), (31.0, 1)])
    arr = arrivals(40, cfg.duration_s, np.random.default_rng(5))
    r = simulate(p, cfg, arr, seed=5)
    assert r.availability < 0.5


def test_workload_profile_volume_matches_rates():
    p = WorkloadParams(); h = hourly_rate(p, "normal")
    rng = np.random.default_rng(0)
    arr = arrivals_profile(h, rng, hours=[10, 11])
    expected = (h[10] + h[11]) * 3600
    assert abs(len(arr) - expected) / expected < 0.02
    s = summary(p); assert s["stress_tps"] > s["deadline_peak_tps"] > s["peak_tps"] > s["nominal_tps"]


def test_smaa_rows_and_columns_sum_to_one():
    rng = np.random.default_rng(1)
    X = rng.uniform(1, 10, size=(5, 4, 3))
    res = smaa2(X, [1, -1, 1], method="TOPSIS", n_weights=400, seed=2)
    acc = res["acceptability"]
    assert np.allclose(acc.sum(axis=1), 1.0) and np.allclose(acc.sum(axis=0), 1.0)
    assert np.all(holistic_acceptability(acc) <= 1.0 + 1e-9)


def test_ws_and_loo():
    assert np.isclose(ws_coefficient([1, 2, 3, 4], [1, 2, 3, 4]), 1.0)
    assert ws_coefficient([1, 2, 3, 4], [4, 3, 2, 1]) < 0.5
    X = np.array([[10, 1], [8, 2], [6, 3], [4, 4]], float)
    loo = leave_one_out(X, np.array([0.5, 0.5]), [1, -1], ["a", "b", "c", "d"])
    assert loo["reversals"] == 0
    ag = rank_agreement({"m1": [1, 2, 3, 4], "m2": [1, 2, 4, 3]})
    assert 0.7 < ag[0]["spearman"] < 1.0
