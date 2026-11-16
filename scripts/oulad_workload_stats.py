#!/usr/bin/env python3
"""Workload statistics from the Open University Learning Analytics Dataset (OULAD).

Kuzilek, J.; Hlosta, M.; Zdrahal, Z. Open University Learning Analytics dataset. Sci. Data 2017, 4, 170171.
Licence: CC BY 4.0. Source: https://archive.ics.uci.edu/dataset/349 (DOI 10.24432/C5KK69).

Usage:  python scripts/oulad_workload_stats.py --data <folder with the OULAD csv files> [--download]
Writes results/oulad_workload_stats.json. The OULAD files are not redistributed in this repository.
Statistics: daily share of registered learners who are active; interaction records (learner x resource x day)
per active learner-day; daily peak-to-mean ratio; interaction records on assessment due days relative to the
presentation mean; timing of submissions relative to the deadline; concentration of registrations.
"""
import argparse, io, os, sys, json, zipfile, urllib.request
import numpy as np, pandas as pd

URL = "https://archive.ics.uci.edu/static/public/349/open+university+learning+analytics+dataset.zip"
ap = argparse.ArgumentParser()
ap.add_argument("--data", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "oulad"))
ap.add_argument("--download", action="store_true", help="download and unpack OULAD from the UCI repository into --data")
args = ap.parse_args()
D = os.path.abspath(args.data)
if args.download:
    os.makedirs(D, exist_ok=True)
    raw = urllib.request.urlopen(URL, timeout=300).read()
    zf = zipfile.ZipFile(io.BytesIO(raw))
    inner = [n for n in zf.namelist() if n.endswith(".zip")]
    if inner:
        zf = zipfile.ZipFile(io.BytesIO(zf.read(inner[0])))
    zf.extractall(D)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "oulad_workload_stats.json")
reg = pd.read_csv(f"{D}/studentRegistration.csv", na_values="?")
info = pd.read_csv(f"{D}/studentInfo.csv", na_values="?")
asm = pd.read_csv(f"{D}/assessments.csv", na_values="?")
sa = pd.read_csv(f"{D}/studentAssessment.csv", na_values="?")
cou = pd.read_csv(f"{D}/courses.csv")
vle = pd.read_csv(f"{D}/vle.csv", na_values="?")
sv = pd.read_csv(f"{D}/studentVle.csv", dtype={"id_site": np.int64, "date": np.int32, "sum_click": np.int32, "id_student": np.int64})
out = {}
key = ["code_module", "code_presentation"]
out["n_presentations"] = int(cou.shape[0]); out["n_enrolments"] = int(info.shape[0])
# ---- daily activity: active learners and interaction records per presentation-day (course days >= 0)
sv0 = sv[sv.date >= 0]
g = sv0.groupby(key + ["date"])
day = pd.DataFrame({"active": g.id_student.nunique(), "rows": g.size(), "clicks": g.sum_click.sum()}).reset_index()
regd = reg.groupby(key).size().rename("registered").reset_index()
day = day.merge(regd, on=key).merge(cou, on=key)
day = day[day.date < day.module_presentation_length]
day["active_share"] = day.active / day.registered
day["rows_per_active"] = day.rows / day.active
day["clicks_per_active"] = day.clicks / day.active
out["active_share_daily_mean"] = float(day.active_share.mean())
out["active_share_daily_median"] = float(day.active_share.median())
out["rows_per_active_learner_day_mean"] = float((day.rows.sum() / day.active.sum()))
out["clicks_per_active_learner_day_mean"] = float(day.clicks.sum() / day.active.sum())
# weekday pattern (course day 0 is the official start; weekday unknown) -> peak-to-mean of daily rows within presentation
pm = day.groupby(key).apply(lambda x: x.rows.max() / x.rows.mean(), include_groups=False)
out["daily_rows_peak_to_mean_median"] = float(pm.median()); out["daily_rows_peak_to_mean_range"] = [float(pm.min()), float(pm.max())]
q = day.groupby(key).apply(lambda x: np.percentile(x.rows, 95) / x.rows.mean(), include_groups=False)
out["daily_rows_p95_to_mean_median"] = float(q.median())
# ---- deadline effect: interaction records on TMA/CMA due days vs the mean of the same presentation
asm_d = asm.dropna(subset=["date"]).copy(); asm_d["date"] = asm_d.date.astype(int)
asm_d = asm_d[asm_d.assessment_type != "Exam"]
dl = asm_d[key + ["date", "assessment_type"]].drop_duplicates()
day_mean = day.groupby(key).rows.mean().rename("mean_rows").reset_index()
dd = dl.merge(day, on=key + ["date"], how="inner").merge(day_mean, on=key)
dd["ratio"] = dd.rows / dd.mean_rows
out["deadline_day_rows_ratio_median"] = float(dd.ratio.median())
out["deadline_day_rows_ratio_iqr"] = [float(dd.ratio.quantile(0.25)), float(dd.ratio.quantile(0.75))]
for t in ("TMA", "CMA"):
    out[f"deadline_day_rows_ratio_median_{t}"] = float(dd[dd.assessment_type == t].ratio.median())
# day before deadline
dl1 = dl.copy(); dl1["date"] = dl1.date - 1
dd1 = dl1.merge(day, on=key + ["date"]).merge(day_mean, on=key)
out["deadline_eve_rows_ratio_median"] = float((dd1.rows / dd1.mean_rows).median())
# submissions relative to deadline
sa2 = sa.merge(asm[["id_assessment", "code_module", "code_presentation", "date", "assessment_type"]], on="id_assessment")
sa2 = sa2[(sa2.is_banked == 0) & sa2.date.notna() & (sa2.assessment_type != "Exam")]
rel = sa2.date_submitted - sa2.date
out["submissions"] = int(len(sa2))
out["share_submitted_on_or_day_before_deadline"] = float(((rel >= -1) & (rel <= 0)).mean())
out["share_submitted_on_deadline_day"] = float((rel == 0).mean())
out["share_submitted_within_3_days_before"] = float(((rel >= -3) & (rel <= 0)).mean())
out["submissions_per_enrolment"] = float(len(sa2) / info.shape[0])
# ---- registration concentration
r = reg.dropna(subset=["date_registration"])
cnt = r.groupby(key + ["date_registration"]).size().rename("n").reset_index()
tot = r.groupby(key).size().rename("tot").reset_index()
cnt = cnt.merge(tot, on=key)
mx = cnt.groupby(key).apply(lambda x: x.n.max() / x.tot.iloc[0], include_groups=False)
out["registration_busiest_day_share_median"] = float(mx.median()); out["registration_busiest_day_share_range"] = [float(mx.min()), float(mx.max())]
span = r.groupby(key).date_registration.agg(lambda x: np.percentile(x, 90) - np.percentile(x, 10))
out["registration_p10_p90_span_days_median"] = float(span.median())
out["share_registered_before_start"] = float((r.date_registration < 0).mean())
# withdrawals as unregistration events
out["unregistration_share"] = float(reg.date_unregistration.notna().mean())
json.dump(out, open(OUT, "w"), indent=1)
print(json.dumps(out, indent=1))
