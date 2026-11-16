import numpy as np
from consensus_mcdm import environment as env
from consensus_mcdm.protocols import catalogue

APP = {"c_block": 0.05, "c_tx": 0.005, "cpu_app_tx": 0.006}


def test_unit_factors_leave_everything_unchanged():
    P = catalogue()
    Q, app = env.apply(P, APP, {})
    assert app == APP and all(Q[k].to_dict() == P[k].to_dict() for k in P)


def test_host_factor_scales_only_its_family():
    P = catalogue()
    Q, _ = env.apply(P, APP, {"hw_besu": 2.0})
    for k in ("ibft2", "qbft", "clique"):
        assert np.isclose(Q[k].o, 2 * P[k].o) and np.isclose(Q[k].a, 2 * P[k].a)
    for k in ("raft", "smartbft", "cometbft", "hotstuff"):
        assert Q[k].to_dict() == P[k].to_dict()
    assert P["qbft"].B_max == Q["qbft"].B_max


def test_application_and_gossip_factors():
    P = catalogue()
    Q, app = env.apply(P, APP, {"app": 0.5, "net_cometbft": 0.1})
    assert np.isclose(app["c_tx"], 0.0025) and np.isclose(Q["cometbft"].g, 0.1 * P["cometbft"].g)


def test_sample_within_ranges():
    for d in env.sample(30, np.random.default_rng(1)):
        for k, (lo, hi, _) in env.FACTORS.items():
            assert lo <= d[k] <= hi
