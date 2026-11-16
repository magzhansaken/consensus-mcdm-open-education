#!/usr/bin/env python3
"""Workload grounding and sensitivity, and education-facing service indicators.

Steps (all reuse results/calibrated_params.json):
  levels      load levels of the base and the OULAD-anchored workload
  oulad       full pipeline (30 seeds) under the OULAD-anchored workload: screen, ranking, SMAA-2
  indicators  education-facing indicators E1-E6 and ranking on the education-facing criterion set, base workload
  oat         one-at-a-time sensitivity of the load levels and of the screened first place to W1-W8
  lhs         Latin hypercube over W1-W8: share of workloads under which each protocol ranks first after the screen
Usage:  python run_workload.py [--steps levels,oulad,indicators,oat,lhs] [--lhs-n 200]
"""
import argparse, json, os, sys, time, copy
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from consensus_mcdm.protocols import catalogue, load_calibrated, RESULTS
from consensus_mcdm.criteria import decision_matrices, CRITERIA, UNITS, TYPES, Scenario
from consensus_mcdm.weights import scenario_set
from consensus_mcdm.mcdm import rank_all, METHODS
from consensus_mcdm.smaa import smaa2
from consensus_mcdm.workload import WorkloadParams, summary
from consensus_mcdm import feasibility as fz
from consensus_mcdm import education as ed
from consensus_mcdm import report

FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")


def screened_first(PP, wl, app, seeds):
    keys, Xm, Xs, hw, raws = decision_matrices(PP, Scenario(n=7, wl=wl), app, seeds=seeds)
    av = fz.stress_availability(raws, keys)
    feas = fz.screen(av)
    bw = scenario_set(Xm, TYPES)["Entropy x CRITIC (base)"]
    rf = fz.rank_feasible(Xm, keys, feas, bw)
    return {"feasible": rf["feasible"], "first": rf.get("first_own", {}), "levels": summary(wl)}, (keys, Xm, Xs, raws, feas)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", default="levels,oulad,indicators,oat,lhs")
    ap.add_argument("--lhs-n", type=int, default=200)
    ap.add_argument("--seeds", type=int, default=30)
    args = ap.parse_args()
    steps = set(args.steps.split(","))
    t0 = time.time()
    P = load_calibrated(catalogue())
    app = json.load(open(os.path.join(RESULTS, "calibrated_params.json")))["app_tier"]
    configs = (("default", P), ("provisioned", fz.provisioned(P)))
    stats = json.load(open(os.path.join(RESULTS, "oulad_workload_stats.json")))
    path = os.path.join(RESULTS, "workload_sensitivity.json")
    out = json.load(open(path)) if os.path.exists(path) else {}
    base = WorkloadParams()
    wl_o, basis = ed.oulad_anchored(stats, base)

    if "levels" in steps:
        out["oulad_stats"] = stats
        out["oulad_anchored_params"] = wl_o.to_dict(); out["oulad_basis"] = basis
        out["levels"] = {"base": summary(base), "oulad_anchored": summary(wl_o)}
        print(f"[{time.time()-t0:.0f}s] levels: base {summary(base)['stress_tps']:.1f}, OULAD-anchored {summary(wl_o)['stress_tps']:.1f} tx/s", flush=True)

    if "oulad" in steps:
        res = {}
        for label, PP in configs:
            r, (keys, Xm, Xs, raws, feas) = screened_first(PP, wl_o, app, range(args.seeds))
            bw = scenario_set(Xm, TYPES)["Entropy x CRITIC (base)"]
            R = rank_all(Xm, bw, TYPES)
            sm = {m: dict(zip(keys, smaa2(Xs, TYPES, method=m, n_weights=10000, seed=42)["acceptability"][:, 0].tolist())) for m in METHODS}
            r.update({"ranks": {m: dict(zip(keys, R[m]["rank"].tolist())) for m in R}, "smaa_b1_uniform": sm,
                      "stress_committed_pct": {k: float(100 * np.mean([x["stress"]["availability"] for x in raws[k]])) for k in keys},
                      "matrix": {k: Xm[i].tolist() for i, k in enumerate(keys)}})
            res[label] = r
            print(f"[{time.time()-t0:.0f}s] OULAD-anchored {label}: feasible {r['feasible']} first {r['first']}", flush=True)
        out["oulad_anchored"] = res

    if "indicators" in steps:
        feas_json = json.load(open(os.path.join(RESULTS, "feasibility.json")))
        res = {}
        for label, PP in configs:
            keys, Xm, Xs, hw, raws = decision_matrices(PP, Scenario(n=7), app, seeds=range(args.seeds))
            cap = feas_json[label]["capacity_99"]
            ind = ed.indicators(raws, keys, Xm, cap, base)
            av = fz.stress_availability(raws, keys); feas = fz.screen(av)
            Xe = ed.education_matrix(Xm, keys, ind)
            te = TYPES.copy(); te[1] = 1
            bw = scenario_set(Xe, te)["Entropy x CRITIC (base)"]
            Re = rank_all(Xe, bw, te)
            rf = fz.rank_feasible(Xe, keys, feas, bw, types=te)
            res[label] = {"indicators": ind, "ranks_unscreened": {m: dict(zip(keys, Re[m]["rank"].tolist())) for m in Re},
                          "first_unscreened": {m: keys[int(np.argmin(Re[m]["rank"]))] for m in Re},
                          "feasible": rf["feasible"], "first_screened": rf.get("first_own", {}), "weights": bw.tolist()}
            print(f"[{time.time()-t0:.0f}s] indicators {label}: education-facing first unscreened {res[label]['first_unscreened']} screened {res[label]['first_screened']}", flush=True)
        out["indicators"] = res

    if "oat" in steps:
        res = {}
        for nm, (field, lo, hi, sc) in ed.RANGES.items():
            res[nm] = {}
            for side, v in (("low", lo), ("high", hi)):
                wl = ed.set_param(base, field, v)
                row = {"value": v, "levels": summary(wl)}
                for label, PP in configs:
                    r, _ = screened_first(PP, wl, app, range(10))
                    row[label] = {"feasible": r["feasible"], "first": r["first"]}
                res[nm][side] = row
            print(f"[{time.time()-t0:.0f}s] OAT {nm} done", flush=True)
        out["oat"] = res

    if "lhs" in steps:
        rng = np.random.default_rng(2026)
        samples = ed.lhs(args.lhs_n, rng)
        rows = []
        for i, d in enumerate(samples):
            wl = copy.deepcopy(base)
            for f, v in d.items():
                wl = ed.set_param(wl, f, v)
            row = {"params": d, "levels": summary(wl)}
            for label, PP in configs:
                r, _ = screened_first(PP, wl, app, range(5))
                row[label] = {"feasible": r["feasible"], "first": r["first"]}
            rows.append(row)
            if (i + 1) % 10 == 0:
                print(f"[{time.time()-t0:.0f}s] LHS {i+1}/{len(samples)}", flush=True)
        out["lhs"] = rows

    json.dump(out, open(path, "w"), indent=1, default=float)
    if "lhs" in out and "oat" in out and hasattr(report, "fig_workload_sensitivity"):
        report.fig_workload_sensitivity(out, os.path.join(FIG, "fig_workload_sensitivity.png"))
    print(f"[{time.time()-t0:.0f}s] all done")


if __name__ == "__main__":
    main()
