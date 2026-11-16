#!/usr/bin/env python3
"""Institutional priorities versus objective weights (Section 4.12).  Usage: python run_priorities.py [--draws 2000]"""
import argparse, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from consensus_mcdm.protocols import RESULTS
from consensus_mcdm.criteria import TYPES
from consensus_mcdm.weights import scenario_set
from consensus_mcdm import priorities as pr
from consensus_mcdm import report

FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--draws", type=int, default=2000); args = ap.parse_args()
    t0 = time.time()
    fj = json.load(open(os.path.join(RESULTS, "feasibility.json")))
    keys = list(fj["default"]["matrix"])
    data = {"default": (np.load(os.path.join(RESULTS, "decision_matrix_samples.npy")), np.array([fj["default"]["feasible"][k] for k in keys])),
            "provisioned": (np.load(os.path.join(RESULTS, "decision_matrix_samples_provisioned.npy")), None)}
    out = {"groups": pr.GROUPS, "roc_weights": pr.roc(4).tolist()}
    for label, (Xs, feas) in data.items():
        Xm = Xs.mean(axis=0)
        idx = np.arange(len(keys)) if feas is None else np.where(feas)[0]
        bw = np.zeros(11); bw[:] = scenario_set(Xm[idx], TYPES)["Entropy x CRITIC (base)"]
        out[label] = {"orders": pr.group_orders(Xs, keys, feas, n=args.draws, seed=5),
                      "security_sweep": pr.security_sweep(Xm, keys, bw, feas),
                      "objective_group_weights": {g: float(bw[list(ix)].sum()) for g, ix in pr.GROUPS.items()}}
        print(f"[{time.time()-t0:.0f}s] {label} done", flush=True)
    json.dump(out, open(os.path.join(RESULTS, "priorities.json"), "w"), indent=1, default=float)
    if hasattr(report, "fig_priorities"):
        report.fig_priorities(out, os.path.join(FIG, "fig_priorities.png"))
    print(f"[{time.time()-t0:.0f}s] all done")


if __name__ == "__main__":
    main()
