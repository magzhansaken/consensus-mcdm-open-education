"""Education-facing service indicators and the empirical grounding of the synthetic workload.

Indicators (computed from the same simulation runs as the decision matrix):
  E1  submission receipt   share of deadline-hour transactions (assessment submissions) committed within
                           2 s and within 5 s of submission (peak level)
  E2  enrolment capacity   enrolment records per hour that can be added to the enrolment hour while 99 % of
                           all transactions still commit within 30 s: (capacity - base load of that hour) x 3600
  E3  enrolment completion share of enrolment-hour transactions committed within 30 s (stress level)
  E4  credential finality  time until an issued credential record is final (nominal level, criterion C3)
  E5  records delayed      deadline-hour records not confirmed within 30 s in the crash scenario of C7,
                           per 10-min window with two node crashes
  E6  cost per learner     operating cost per active learner and year (C11 x 12 / W1)

Workload grounding: OULAD rates (Kuzilek et al. 2017) are read from results/oulad_workload_stats.json,
which scripts/oulad_workload_stats.py produces from the public OULAD release.
"""
from __future__ import annotations
import copy, json, os
import numpy as np
from .workload import WorkloadParams, hourly_rate, summary, _HOURLY

THRESHOLDS = (2.0, 5.0)


def indicators(raws: dict, keys: list, Xm: np.ndarray, capacity: dict, wl: WorkloadParams) -> dict:
    base_enrol_hour = float(hourly_rate(wl, "stress")[wl.registration_hour] - wl.registration_burst_tps)
    out = {}
    for i, k in enumerate(keys):
        rs = raws[k]
        f = lambda lvl, fld: float(np.nanmean([r[lvl][fld] for r in rs]))
        fault_window = f("fault", "offered_tps") * 540.0
        out[k] = {
            "E1_receipt_2s_pct": 100.0 * f("peak", "share_le_2s"),
            "E1_receipt_5s_pct": 100.0 * f("peak", "share_le_5s"),
            "E1_p95_s": f("peak", "lat_p95_s"),
            "E2_enrolment_records_per_hour": max(0.0, (capacity[k] - base_enrol_hour) * 3600.0),
            "E3_enrolment_completion_pct": 100.0 * f("stress", "availability"),
            "E4_credential_finality_s": float(Xm[i, 2]),
            "E5_records_delayed_per_incident": f("fault", "fail_rate") * fault_window,
            "E6_cost_per_learner_year_usd": float(Xm[i, 10]) * 12.0 / wl.students_active,
        }
    return out


def education_matrix(Xm: np.ndarray, keys: list, ind: dict) -> np.ndarray:
    """Education-facing criterion set: C1 replaced by E2 (enrolment capacity, benefit), C2 by E1 (share of
    deadline-hour submissions confirmed within 2 s, benefit), C11 by E6 (a rescaling with no effect on the
    scale-invariant methods). All other criteria are unchanged; the types are those of TYPES with C2 as benefit."""
    Xe = Xm.copy()
    for i, k in enumerate(keys):
        Xe[i, 0] = ind[k]["E2_enrolment_records_per_hour"]
        Xe[i, 1] = max(ind[k]["E1_receipt_2s_pct"], 0.01)
        Xe[i, 10] = ind[k]["E6_cost_per_learner_year_usd"]
    return Xe


def oulad_anchored(stats: dict, base: WorkloadParams | None = None, tx_per_registration: float = 1.0) -> tuple[WorkloadParams, dict]:
    """Workload with the learner-side rates measured in OULAD, at the scale of the base workload.
    W2  = interaction records (learner x resource x day) per active learner-day;
    W5  = evening multiplier that reproduces the median deadline-day surge of interaction records, given that the
          hours 19-23 carry the evening share of the daily profile;
    W6  = busiest-hour registrations: module registrations per intake (W1 / daily active share) x share of the
          busiest registration day x peak hourly share of the profile / 3600 s x ledger records per registration."""
    wl = copy.deepcopy(base or WorkloadParams())
    evening = float(_HOURLY[list(wl.deadline_hours)].sum())
    surge = stats["deadline_day_rows_ratio_median"]
    wl.events_per_student_day = round(stats["rows_per_active_learner_day_mean"], 2)
    wl.deadline_multiplier = round(1.0 + (surge - 1.0) / evening, 2)
    regs = wl.students_active / stats["active_share_daily_mean"]
    wl.registration_burst_tps = round(regs * stats["registration_busiest_day_share_median"] * float(_HOURLY.max()) / 3600.0 * tx_per_registration, 2)
    basis = {"evening_share_of_day": evening, "deadline_day_surge": surge, "registrations_per_intake": regs,
             "base_W5_daily_surge": 1.0 + (2.5 - 1.0) * evening}
    return wl, basis


# ranges of the workload parameters for the sensitivity analysis (low, high, scale)
RANGES = {
    "W1 active learners": ("students_active", 40_000, 200_000, "log"),
    "W2 events per learner-day": ("events_per_student_day", 3.0, 24.0, "log"),
    "W3 IoT devices": ("iot_devices", 1_000, 20_000, "log"),
    "W5 deadline multiplier": ("deadline_multiplier", 1.5, 4.0, "lin"),
    "W6 enrolment burst (tx/s)": ("registration_burst_tps", 0.0, 240.0, "lin"),
    "W7 transaction size (B)": ("tx_bytes", 250, 4_000, "log"),
    "W8 hourly concentration": ("hourly_exponent", 0.5, 1.5, "lin"),
}


def set_param(wl: WorkloadParams, field: str, value):
    wl = copy.deepcopy(wl)
    cur = getattr(wl, field)
    setattr(wl, field, int(round(value)) if isinstance(cur, int) and not isinstance(cur, bool) else float(value))
    return wl


def lhs(n: int, rng: np.random.Generator) -> list[dict]:
    """Latin hypercube sample over RANGES (log-uniform where marked)."""
    names = list(RANGES)
    u = np.empty((n, len(names)))
    for j in range(len(names)):
        u[:, j] = (rng.permutation(n) + rng.uniform(size=n)) / n
    out = []
    for i in range(n):
        d = {}
        for j, nm in enumerate(names):
            field, lo, hi, sc = RANGES[nm]
            d[field] = float(np.exp(np.log(lo) + u[i, j] * (np.log(hi) - np.log(lo)))) if sc == "log" else float(lo + u[i, j] * (hi - lo))
        out.append(d)
    return out
