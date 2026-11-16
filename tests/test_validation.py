import numpy as np
from consensus_mcdm import validation as va
from consensus_mcdm.protocols import catalogue
from consensus_mcdm.sim import simulate, SimConfig
from consensus_mcdm.workload import arrivals


def _run(key, **kw):
    p = catalogue()[key]
    cfg = SimConfig(n=4, rtt_s=0.02, duration_s=120, warmup_s=20, **kw)
    return simulate(p, cfg, arrivals(40, cfg.duration_s, np.random.default_rng(3)), seed=3)


def test_default_corrections_leave_the_model_unchanged():
    assert _run("qbft").as_dict() == _run("qbft", recovery_factor=1.0, rtt_extra_bft=0.0, rtt_extra_crash=0.0).as_dict()


def test_rtt_correction_adds_latency_but_not_throughput_loss():
    a, b = _run("raft"), _run("raft", rtt_extra_crash=30.0)
    assert b.lat_mean_s > a.lat_mean_s + 0.4 and abs(b.throughput_tps - a.throughput_tps) < 1.0


def test_rtt_correction_from_holdout_rows():
    rows = [{"set": "holdout_rtt", "protocol": p, "rtt_ms": r, "lat_meas": lm, "lat_sim": ls}
            for p, r, lm, ls in [("raft", "0", 1.0, 1.2), ("raft", "100", 4.0, 1.3), ("smartbft", "0", 1.0, 1.5), ("smartbft", "100", 8.0, 1.8)]]
    c = va.rtt_correction(rows)
    assert abs(c["crash"] - 29.0) < 1e-9 and abs(c["bft"] - 67.0) < 1e-9


def test_perturbation_shares_factors_within_a_family_and_keeps_documented_columns():
    Xs = np.ones((2, 7, 11)) * 50.0
    keys = ["raft", "smartbft", "ibft2", "qbft", "clique", "cometbft", "hotstuff"]
    X, feas = va.perturb(Xs, keys, "pessimistic", np.random.default_rng(1), capacity={k: 200 for k in keys}, stress=185)
    assert np.allclose(X[:, 2, :], X[:, 3, :]) and np.allclose(X[:, 0, :], X[:, 1, :])
    assert np.allclose(X[:, :, 7:10], 50.0)
    assert feas.shape == (7,)
