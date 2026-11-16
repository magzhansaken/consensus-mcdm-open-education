#!/usr/bin/env python3
"""Transferability across benchmark environments: perturb the constants that the anchors carry into the scenario.

Steps:  oat   each factor of consensus_mcdm.environment.FACTORS at the ends of its range (10 seeds), documented and provisioned
        lhs   Latin hypercube over all factors jointly (5 seeds)
        harmonised  Besu constants doubled (workstation -> laptop-class host of the Fabric anchor), 30 seeds
Usage:  python run_environment.py [--steps oat,lhs,harmonised] [--n 120]
"""
import argparse, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from consensus_mcdm.protocols import catalogue, load_calibrated, RESULTS
from consensus_mcdm.criteria import decision_matrices, TYPES, Scenario
from consensus_mcdm.weights import scenario_set
from consensus_mcdm.mcdm import rank_all
from consensus_mcdm import feasibility as fz
from consensus_mcdm import environment as env


def evaluate(PP, app, seeds, smaa=False):
    keys, Xm, Xs, hw, raws = decision_matrices(PP, Scenario(n=7), app, seeds=seeds)
    feas = fz.screen(fz.stress_availability(raws, keys))
    bw = scenario_set(Xm, TYPES)["Entropy x CRITIC (base)"]
    R = rank_all(Xm, bw, TYPES)
    rf = fz.rank_feasible(Xm, keys, feas, bw, Xs=Xs if smaa else None)
    return {"feasible": rf["feasible"], "first_screened": rf.get("first_own", {}),
            "first_unscreened": {m: keys[int(np.argmin(R[m]["rank"]))] for m in R},
            "smaa_b1": rf.get("smaa_b1_uniform"), "matrix": {k: Xm[i].tolist() for i, k in enumerate(keys)}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", default="oat,lhs,harmonised")
    ap.add_argument("--n", type=int, default=120)
    args = ap.parse_args()
    steps = set(args.steps.split(","))
    t0 = time.time()
    P = load_calibrated(catalogue())
    app0 = json.load(open(os.path.join(RESULTS, "calibrated_params.json")))["app_tier"]
    path = os.path.join(RESULTS, "environment_sensitivity.json")
    out = json.load(open(path)) if os.path.exists(path) else {}
    out["factors"] = env.FACTORS
    configs = (("default", P), ("provisioned", fz.provisioned(P)))
    if "oat" in steps:
        res = {}
        for nm, (lo, hi, _) in env.FACTORS.items():
            res[nm] = {}
            for side, v in (("low", lo), ("high", hi)):
                row = {"value": v}
                for label, PP in configs:
                    Q, app = env.apply(PP, app0, {nm: v})
                    r = evaluate(Q, app, range(10))
                    row[label] = {k: r[k] for k in ("feasible", "first_screened", "first_unscreened")}
                    row[label]["qbft"] = r["matrix"]["qbft"]; row[label]["cometbft"] = r["matrix"]["cometbft"]
                res[nm][side] = row
            print(f"[{time.time()-t0:.0f}s] OAT {nm}: " + "; ".join(f"{s} {res[nm][s]['default']['first_screened']} / {res[nm][s]['provisioned']['first_screened']}" for s in ("low", "high")), flush=True)
        out["oat"] = res
    if "harmonised" in steps:
        res = {}
        for label, PP in configs:
            Q, app = env.apply(PP, app0, {"hw_besu": 2.0})
            res[label] = evaluate(Q, app, range(30), smaa=True)
            print(f"[{time.time()-t0:.0f}s] harmonised {label}: {res[label]['first_screened']}", flush=True)
        out["harmonised_besu_x2"] = res
    if "lhs" in steps:
        rows = []
        for i, f in enumerate(env.sample(args.n, np.random.default_rng(404))):
            row = {"factors": f}
            for label, PP in configs:
                Q, app = env.apply(PP, app0, f)
                r = evaluate(Q, app, range(5))
                row[label] = {k: r[k] for k in ("feasible", "first_screened", "first_unscreened")}
            rows.append(row)
            if (i + 1) % 10 == 0:
                print(f"[{time.time()-t0:.0f}s] LHS {i+1}/{args.n}", flush=True)
        out["lhs"] = rows
    json.dump(out, open(path, "w"), indent=1, default=float)
    print(f"[{time.time()-t0:.0f}s] all done")


if __name__ == "__main__":
    main()
