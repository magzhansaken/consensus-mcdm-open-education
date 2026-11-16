import itertools
import numpy as np
from consensus_mcdm import priorities as pr


def test_roc_weights():
    w = pr.roc(4)
    assert np.isclose(w.sum(), 1.0) and np.allclose(w, [0.5208333, 0.2708333, 0.1458333, 0.0625], atol=1e-6)


def test_sampled_weights_follow_the_group_order():
    rng = np.random.default_rng(0)
    for order in itertools.permutations("PRSA"):
        for _ in range(20):
            w = pr.weights_for_order(order, rng)
            g = [w[list(pr.GROUPS[x])].sum() for x in order]
            assert np.isclose(w.sum(), 1.0) and all(g[i] >= g[i + 1] - 1e-12 for i in range(3))


def test_security_sweep_endpoints():
    from consensus_mcdm.criteria import TYPES
    X = np.random.default_rng(1).uniform(1, 2, size=(3, 11)); bw = np.ones(11) / 11
    rows = pr.security_sweep(X, ["a", "b", "c"], bw)
    assert rows[0]["security_weight"] == 0.0 and rows[-1]["security_weight"] == 1.0
