import numpy as np
from consensus_mcdm import criteria_checks as cc
from consensus_mcdm.criteria import ATTACK, MANAGE


def test_property_tables_sum_to_the_documented_scores():
    assert cc.check_tables()
    assert all(sum(cc.C8_TABLE[k]) == ATTACK[k] and sum(cc.C10_TABLE[k]) == MANAGE[k] for k in cc.KEYS)


def test_smaa_ordinal_keeps_a_dominating_alternative_first():
    from consensus_mcdm.criteria import TYPES
    X = np.where(TYPES > 0, 2.0, 1.0) * np.ones((7, 11))
    X[3] = np.where(TYPES > 0, 3.0, 0.5)          # alternative 3 dominates on every criterion
    X[:, 7] = [1, 2, 2, 4, 1, 3, 0]; X[:, 9] = [2, 2, 1, 4, 1, 3, 0]
    out = cc.smaa_ordinal(X[None, :, :], cc.KEYS, n=200, seed=3)
    assert all(d == {"qbft": 1.0} for d in out.values())


def test_first_with_single_feasible_alternative():
    X = np.ones((7, 11)); X[:, 0] = np.arange(1, 8)
    feas = np.zeros(7, dtype=bool); feas[4] = True
    assert set(cc.first(X, feas).values()) == {4}
