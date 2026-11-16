import json, os
import numpy as np
from consensus_mcdm import education as ed
from consensus_mcdm.workload import WorkloadParams, summary, hourly_rate

STATS = {"rows_per_active_learner_day_mean": 5.8655, "deadline_day_rows_ratio_median": 1.6933,
         "active_share_daily_mean": 0.20894, "registration_busiest_day_share_median": 0.04598}


def test_oulad_anchored_reproduces_the_deadline_surge():
    wl, basis = ed.oulad_anchored(STATS)
    n = hourly_rate(wl, "normal"); d = hourly_rate(wl, "deadline")
    iot = wl.iot_devices / wl.telemetry_interval_s
    surge = (d - iot).sum() / (n - iot).sum()
    assert abs(surge - STATS["deadline_day_rows_ratio_median"]) < 0.01
    assert wl.registration_burst_tps < 1.0 and wl.events_per_student_day == 5.87


def test_hourly_exponent_one_is_identical_and_others_keep_the_daily_total():
    base = WorkloadParams()
    assert np.array_equal(hourly_rate(base), hourly_rate(WorkloadParams(hourly_exponent=1.0)))
    for e in (0.5, 1.5):
        assert abs(hourly_rate(WorkloadParams(hourly_exponent=e)).sum() - hourly_rate(base).sum()) < 1e-9


def test_lhs_respects_ranges_and_is_stratified():
    rows = ed.lhs(50, np.random.default_rng(0))
    for name, (field, lo, hi, sc) in ed.RANGES.items():
        v = np.array([r[field] for r in rows])
        assert v.min() >= lo - 1e-9 and v.max() <= hi + 1e-9
        u = (np.log(v) - np.log(lo)) / (np.log(hi) - np.log(lo)) if sc == "log" else (v - lo) / (hi - lo)
        assert sorted(np.floor(u * 50).astype(int).tolist()) == list(range(50))


def test_education_matrix_replaces_only_c1_c2_c11():
    X = np.arange(1.0, 23.0).reshape(2, 11)
    ind = {"a": {"E2_enrolment_records_per_hour": 10.0, "E1_receipt_2s_pct": 50.0, "E6_cost_per_learner_year_usd": 0.1},
           "b": {"E2_enrolment_records_per_hour": 20.0, "E1_receipt_2s_pct": 0.0, "E6_cost_per_learner_year_usd": 0.2}}
    Xe = ed.education_matrix(X, ["a", "b"], ind)
    assert np.array_equal(Xe[:, 2:10], X[:, 2:10]) and Xe[1, 1] == 0.01 and Xe[0, 0] == 10.0
