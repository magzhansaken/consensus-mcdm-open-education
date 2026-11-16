import numpy as np
from consensus_mcdm.protocols import catalogue
from consensus_mcdm import feasibility as fz
from consensus_mcdm.workload import summary, WorkloadParams


def test_capacity_is_last_load_before_first_failure():
    rows = [{"key": "a", "offered": L, "avail_mean": v} for L, v in [(50, 1.0), (55, 0.995), (60, 0.97), (65, 1.0)]]
    assert fz.capacities(rows) == {"a": 55}
    assert fz.capacities(rows, target=0.95) == {"a": 65}


def test_screen_requires_every_seed():
    av = {"a": np.array([1.0, 0.999, 0.991]), "b": np.array([1.0, 1.0, 0.98])}
    assert fz.screen(av) == {"a": True, "b": False}
    assert fz.screen(av, 0.95) == {"a": True, "b": True}


def test_stress_workload_hits_target_and_keeps_other_levels():
    base = summary(WorkloadParams())
    for L in (150.0, 185.0, 240.0):
        s = summary(fz.stress_workload(L))
        assert abs(s["stress_tps"] - L) < 1e-9
        assert abs(s["deadline_peak_tps"] - base["deadline_peak_tps"]) < 1e-9
        assert abs(s["nominal_tps"] - base["nominal_tps"]) < 1e-9


def test_provisioned_changes_only_block_formation():
    P = catalogue(); Q = fz.provisioned(P)
    assert Q["qbft"].B_max == 440 and Q["ibft2"].B_max == 440 and P["qbft"].B_max == 220
    assert Q["raft"].B_max == 300 and Q["hotstuff"].T_b == 1.0
    for k in P:
        assert Q[k].view_change_s == P[k].view_change_s and Q[k].rounds == P[k].rounds


def test_rank_feasible_ignores_infeasible_alternatives():
    X = np.array([[100.0, 1.0], [200.0, 2.0], [150.0, 1.5]]); keys = ["a", "b", "c"]
    out = fz.rank_feasible(X, keys, {"a": False, "b": True, "c": True}, np.array([0.5, 0.5]), types=np.array([1, -1]))
    assert out["feasible"] == ["b", "c"]
    assert set(out["first_own"].values()) <= {"b", "c"}
