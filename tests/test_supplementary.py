import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import make_supplementary as ms


def test_assumption_ids_unique():
    ids = [r[0] for r in ms.table_s1()]
    assert len(ids) == len(set(ids)), [i for i in ids if ids.count(i) > 1]


def test_table_s2_pooled_fabric_matches_summary():
    import json
    rows = ms.table_s2()
    pooled = [r for r in rows if r["platform"].endswith("pooled")][0]
    s = json.load(open(os.path.join(ms.RES, "calibration_summary.json")))
    assert abs(pooled["mape_latency"] - s["fabric_fit_MAPE_latency"]) < 1e-12
    assert pooled["points"] == 15
