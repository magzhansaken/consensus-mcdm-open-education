import numpy as np
import pytest
from consensus_mcdm.mcdm import topsis, vikor, copras, edas, saw, rank_desc
from consensus_mcdm.iot45 import reproduce, X15, TYPES
from consensus_mcdm.weights import IOT45_SCENARIOS, entropy_weights, critic_weights, combined_weights


def test_iot45_scenarios_reproduce_exactly():
    """Oracle: Tables 22-23 of Tashatov et al. (IoT 2026 7(2) 45) for Scenarios 2-7."""
    out = reproduce()
    for name, v in out.items():
        assert v["maxdiff_C"] < 5e-6, name
        assert v["maxdiff_Q"] < 5e-5, name
        assert np.array_equal(v["rankC"], v["rankC_pub"]), name
        assert np.array_equal(v["rankQ"], v["rankQ_pub"]), name


def test_topsis_dominance():
    # alternative 0 dominates alternative 1 on every criterion -> must rank first
    X = np.array([[10, 1, 5], [8, 2, 4], [9, 1.5, 4.5]], float)
    r = topsis(X, [1 / 3] * 3, [1, -1, 1])["rank"]
    assert r[0] == 1 and r[1] == 3


def test_vikor_bounds_and_monotone():
    X = np.array([[10, 1], [5, 5], [1, 10]], float)
    v = vikor(X, [0.5, 0.5], [1, -1])
    assert v["score"].min() == 0.0 and v["score"].max() == 1.0
    assert v["rank"][0] == 1


def test_copras_edas_saw_agree_on_dominated_set():
    X = np.array([[10, 1, 10], [9, 2, 9], [8, 3, 8], [7, 4, 7]], float)
    w = np.array([0.4, 0.3, 0.3]); t = [1, -1, 1]
    for fn in (copras, edas, saw, topsis):
        assert list(fn(X, w, t)["rank"]) == [1, 2, 3, 4], fn.__name__


def test_copras_textbook_small():
    # two criteria (one benefit, one cost), hand-computed
    X = np.array([[4, 2], [2, 4]], float); w = np.array([0.5, 0.5]); t = [1, -1]
    q = copras(X, w, t)["score"]
    # S+ = 0.5*[4/6, 2/6] ; S- = 0.5*[2/6, 4/6]; Q_i = S+_i + (minS- * sumS-)/(S-_i * sum(minS-/S-_j))
    Sp = 0.5 * np.array([4 / 6, 2 / 6]); Sm = 0.5 * np.array([2 / 6, 4 / 6])
    Q = Sp + (Sm.min() * Sm.sum()) / (Sm * (Sm.min() / Sm).sum())
    assert np.allclose(q, Q)


def test_edas_average_solution_has_zero_distances():
    X = np.array([[2, 2], [4, 4], [3, 3]], float)  # row 2 is the average -> SP=SN=0 -> AS = 0.5*(0 + 1)
    r = edas(X, [0.5, 0.5], [1, -1])
    assert np.isclose(r["score"][2], 0.5)


def test_entropy_equal_columns_equal_weights():
    X = np.array([[1, 10], [2, 20], [3, 30]], float)
    w = entropy_weights(X, [1, 1])
    assert np.allclose(w, [0.5, 0.5])


def test_entropy_constant_column_gets_zero():
    X = np.array([[5, 1], [5, 2], [5, 9]], float)
    w = entropy_weights(X, [1, 1])
    assert w[0] < 1e-9 and np.isclose(w[1], 1.0)


def test_critic_penalises_correlated_columns():
    rng = np.random.default_rng(0)
    a = rng.uniform(1, 10, 20)
    X = np.c_[a, a * 2 + 0.01 * rng.standard_normal(20), rng.uniform(1, 10, 20)]
    w = critic_weights(X, [1, 1, 1])
    assert w[2] > w[0] and w[2] > w[1]      # the independent column is more informative


def test_combined_weights_sum_to_one():
    w = combined_weights([0.2, 0.3, 0.5], [0.5, 0.3, 0.2])
    assert np.isclose(w.sum(), 1.0)


def test_iot45_transfer_published_order_is_stable_on_common_families():
    """The published TOPSIS order of the four common families equals a recomputation on those families alone,
    so the comparison in Section 4.7 does not depend on how the published order is read."""
    import json, os
    import pandas as pd
    from consensus_mcdm.iot45 import transfer
    from consensus_mcdm.criteria import TYPES as T
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    keys = json.load(open(os.path.join(here, "results", "raw_sim.json")))["keys"]
    Xm = pd.read_csv(os.path.join(here, "results", "decision_matrix.csv")).iloc[:7, 1:].values.astype(float)
    rows, agree = transfer(Xm, keys, T, IOT45_SCENARIOS)
    assert len(rows) == 3 * 6 * 4 and len(agree) == 3 * 4
    assert all(r["published_TOPSIS"] == r["recomputed4_TOPSIS"] for r in rows)
