import numpy as np
from consensus_mcdm.protocols import catalogue
from consensus_mcdm.criteria import decision_matrices, Scenario, TYPES
from consensus_mcdm.robustness import leader_wilcoxon

APP = {"c_block": 0.0, "c_tx": 0.0, "cpu_app_tx": 0.0}


def _small():
    P = catalogue()
    P = {k: P[k] for k in ("raft", "qbft", "cometbft")}
    sc = Scenario(duration_s=60.0, warmup_s=10.0)
    return decision_matrices(P, sc, APP, seeds=range(2))


def test_replicate_uses_common_random_numbers_across_alternatives():
    keys, Xm, Xs, hw, raws = _small()
    for s in range(2):
        offered = {k: (raws[k][s]["stress"]["offered_tps"], raws[k][s]["nominal"]["offered_tps"]) for k in keys}
        assert len(set(offered.values())) == 1
    # different replicates draw different arrival sequences
    k = keys[0]
    assert raws[k][0]["stress"]["offered_tps"] != raws[k][1]["stress"]["offered_tps"]


def test_leader_comparison_reports_paired_and_independent_tests():
    rng = np.random.default_rng(0)
    base = rng.uniform(1.0, 2.0, size=(3, len(TYPES)))
    Xs = np.array([base * rng.uniform(0.97, 1.03, size=base.shape) for _ in range(12)])
    Xs[:, 0, :] = np.where(np.array(TYPES) > 0, Xs[:, 0, :] * 1.3, Xs[:, 0, :] / 1.3)  # alternative A dominates
    w = np.full(len(TYPES), 1.0 / len(TYPES))
    out = leader_wilcoxon(Xs, w, np.array(TYPES), ["A", "B", "C"], "A", "B")
    for m, r in out.items():
        assert r["seeds_first_better"] == 12
        assert r["p_holm"] < 0.01 and r["p_mannwhitney_holm"] < 0.01
        rho = r["spearman_rho"]
        assert np.isnan(rho) or -1.0 <= rho <= 1.0  # constant scores (e.g. VIKOR Q = 0) give no correlation
