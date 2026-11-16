#!/usr/bin/env python3
"""Checks of the criterion set and the ordinal criteria (Section 4.11).  Usage: python run_criteria.py"""
import json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from consensus_mcdm.protocols import RESULTS
from consensus_mcdm.criteria import CRITERIA
from consensus_mcdm import criteria_checks as cc


def main():
    t0 = time.time()
    cc.check_tables()
    fj = json.load(open(os.path.join(RESULTS, "feasibility.json")))
    keys = list(fj["default"]["matrix"])
    data = {"default": (np.load(os.path.join(RESULTS, "decision_matrix_samples.npy")), np.array([fj["default"]["feasible"][k] for k in keys])),
            "provisioned": (np.load(os.path.join(RESULTS, "decision_matrix_samples_provisioned.npy")), None)}
    out = {"c8_properties": cc.C8_PROPS, "c8_table": cc.C8_TABLE, "c10_properties": cc.C10_PROPS, "c10_table": cc.C10_TABLE}
    for label, (Xs, feas) in data.items():
        Xm = Xs.mean(axis=0)
        r = {}
        r["baseline"] = {m: keys[i] for m, i in cc.first(Xm, feas).items()}
        r["smaa_o_reestimated"] = cc.smaa_ordinal(Xs, keys, feas, n=10000, seed=1)
        r["smaa_o_uniform_weights"] = cc.smaa_ordinal(Xs, keys, feas, n=10000, seed=2, uniform_weights=True)
        r["perturb"] = cc.perturb_scores(Xm, keys, feas)
        r["rules"] = cc.scoring_rules(Xm, keys, feas)
        r["leave_one_out"] = cc.leave_one_out(Xm, keys, CRITERIA, feas)
        r["fault_proxies"], tol = cc.fault_proxies(Xm, keys, feas=feas)
        r["scale"] = cc.scale_shift(Xm, keys, feas)
        out[label] = r
        print(f"[{time.time()-t0:.0f}s] {label} done", flush=True)
    out["documented_crash_tolerance_pct"] = tol
    json.dump(out, open(os.path.join(RESULTS, "criteria_checks.json"), "w"), indent=1, default=float)
    print(f"[{time.time()-t0:.0f}s] all done")


if __name__ == "__main__":
    main()
