"""Synthetic transaction workload of an IoT-enabled open-university information system.

THIS PROFILE IS SYNTHETIC. It is generated from declared parameters (Table of assumptions,
rows W1-W7), not from measured portal logs. Its purpose is to give every protocol the same
offered load with a realistic daily shape (two activity peaks, an evening submission peak
and a constant IoT telemetry floor). The daily shape follows qualitative evidence on
learner activity in open-university VLE logs (Kuzilek et al., 2017, Sci. Data 4:170171),
but no OULAD records are used.

Transaction classes written to the ledger (one ledger transaction = one anchored record):
  * learning events  : submissions, graded attempts, attendance confirmations, credential ops
  * IoT telemetry    : one aggregated record per device per telemetry interval
Read-only verification queries do not go through consensus and are not counted.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, asdict

# Hourly weights of learner activity (24 values, sum = 1). Two peaks: 10-13 h and 19-23 h.
_HOURLY = np.array([
    0.004, 0.002, 0.001, 0.001, 0.001, 0.003,   # 00-05
    0.010, 0.025, 0.045, 0.060, 0.070, 0.070,   # 06-11
    0.060, 0.050, 0.045, 0.045, 0.045, 0.050,   # 12-17
    0.055, 0.065, 0.075, 0.080, 0.075, 0.063,   # 18-23
])
_HOURLY = _HOURLY / _HOURLY.sum()


@dataclass
class WorkloadParams:
    students_active: int = 100_000      # W1 active learners on a weekday (national open-university scale)
    events_per_student_day: float = 12.0 # W2 ledger-anchored learning events per active learner per day
    iot_devices: int = 5_000            # W3 campus IoT devices (classroom/env sensors, access control)
    telemetry_interval_s: float = 120.0 # W4 one aggregated telemetry record per device per 2 min
    deadline_multiplier: float = 2.5    # W5 evening-peak multiplier on an assignment-deadline day
    deadline_hours: tuple = (19, 20, 21, 22, 23)
    registration_burst_tps: float = 120.0 # W6 extra load during a 1-h enrolment window (stress day)
    registration_hour: int = 10
    tx_bytes: int = 1_000               # W7 ledger transaction size (payload + signatures)

    def to_dict(self):
        return asdict(self)


def hourly_rate(p: WorkloadParams, day: str = "normal") -> np.ndarray:
    """Offered load (tx/s) for each hour of the day. day in {'normal','deadline','stress'}."""
    learn_total = p.students_active * p.events_per_student_day
    learn = learn_total * _HOURLY / 3600.0                      # tx/s per hour
    iot = np.full(24, p.iot_devices / p.telemetry_interval_s)   # constant floor
    rate = learn + iot
    if day in ("deadline", "stress"):
        for h in p.deadline_hours:
            rate[h] = iot[h] + learn[h] * p.deadline_multiplier
    if day == "stress":
        rate[p.registration_hour] += p.registration_burst_tps
    return rate


def summary(p: WorkloadParams) -> dict:
    """Load levels used in the experiments (tx/s)."""
    n = hourly_rate(p, "normal")
    d = hourly_rate(p, "deadline")
    s = hourly_rate(p, "stress")
    return {
        "nominal_tps": float(n.mean()),                 # daily mean of a normal day
        "peak_tps": float(n.max()),                     # busiest hour of a normal day
        "deadline_peak_tps": float(d.max()),            # busiest hour of a deadline day
        "stress_tps": float(s.max()),                   # busiest hour of the stress day
        "iot_floor_tps": float(p.iot_devices / p.telemetry_interval_s),
        "daily_tx_normal": float(n.sum() * 3600),
        "daily_tx_deadline": float(d.sum() * 3600),
    }


def arrivals(rate_tps: float, duration_s: float, rng: np.random.Generator) -> np.ndarray:
    """Poisson arrival times for a constant offered load."""
    n_exp = rate_tps * duration_s
    n = rng.poisson(n_exp)
    return np.sort(rng.uniform(0.0, duration_s, size=n))


def arrivals_profile(hourly_tps: np.ndarray, rng: np.random.Generator, hours=None) -> np.ndarray:
    """Non-homogeneous Poisson arrivals over the given hours (default: all 24)."""
    hours = range(24) if hours is None else hours
    parts = []
    for h in hours:
        t0 = h * 3600.0
        n = rng.poisson(hourly_tps[h] * 3600.0)
        parts.append(t0 + np.sort(rng.uniform(0.0, 3600.0, size=n)))
    return np.concatenate(parts) if parts else np.array([])
