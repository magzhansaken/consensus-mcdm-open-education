#!/usr/bin/env python3
"""Peak-load feasibility screen, capacity-provisioned alternatives and the boundary of the recommendation.

Usage:  python run_feasibility.py [--seeds 30] [--boundary-seeds 10]
Reads results/calibrated_params.json; writes results/feasibility_*.{csv,json} and figures/fig_feasibility.{png,pdf}.
"""
import argparse, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from consensus_mcdm.protocols import catalogue, load_calibrated, RESULTS
from consensus_mcdm.criteria import decision_matrices, CRITERIA, UNITS, TYPES, Scenario
from consensus_mcdm.weights import scenario_set
from consensus_mcdm.mcdm import rank_all, METHODS
from consensus_mcdm.smaa import smaa2
from consensus_mcdm import feasibility as fz
from consensus_mcdm import report

FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--boundary-seeds", type=int, default=10)
    ap.add_argument("--steps", default="screen,boundary,besu,ablations,perturb", help="comma-separated subset of steps")
    args = ap.parse_args()
    steps = set(args.steps.split(","))
    t0 = time.time()
    P = load_calibrated(catalogue())
    cal = json.load(open(os.path.join(RESULTS, "calibrated_params.json")))
    app = cal["app_tier"]
    sc = Scenario(n=7)
    path = os.path.join(RESULTS, "feasibility.json")
    out = json.load(open(path)) if os.path.exists(path) else {}
    out["target"] = fz.TARGET
    for label, PP in (("default", P), ("provisioned", fz.provisioned(P))) if "screen" in steps else ():
        curve = fz.availability_curve(PP, sc, app, seeds=range(10))
        report.save_rows(curve, os.path.join(RESULTS, f"feasibility_curve_{label}.csv"))
        cap = fz.capacities(curve)
        cap95 = fz.capacities(curve, target=0.95)
        keys, Xm, Xs, hw, raws = decision_matrices(PP, sc, app, seeds=range(args.seeds))
        names = [PP[k].name for k in keys]
        if label == "provisioned":
            report.save_matrix(names, Xm, hw, CRITERIA, UNITS, TYPES, os.path.join(RESULTS, "decision_matrix_provisioned.csv"))
            np.save(os.path.join(RESULTS, "decision_matrix_samples_provisioned.npy"), Xs)
        av = fz.stress_availability(raws, keys)
        feas = fz.screen(av)
        W = scenario_set(Xm, TYPES)
        base_w = W["Entropy x CRITIC (base)"]
        R_all = rank_all(Xm, base_w, TYPES)
        rf = fz.rank_feasible(Xm, keys, feas, base_w, Xs=Xs)
        # sensitivity of the screen to the service target
        thr = {str(t): [k for k, v in fz.screen(av, t).items() if v] for t in (0.95, 0.99, 0.999)}
        # full-set SMAA (provisioned) for comparison with the base analysis
        sm_all = {}
        if label == "provisioned":
            for m in METHODS:
                a = smaa2(Xs, TYPES, method=m, n_weights=10000, seed=42)
                sm_all[m] = {keys[i]: float(a["acceptability"][i, 0]) for i in range(len(keys))}
        stress_stats = {k: {"avail_mean": float(av[k].mean()), "avail_min": float(av[k].min()),
                            "goodput": float(np.mean([r["stress"]["throughput_tps"] for r in raws[k]])),
                            "p95_s": float(np.nanmean([r["stress"]["lat_p95_s"] for r in raws[k]])),
                            "peak_p95_s": float(np.nanmean([r["peak"]["lat_p95_s"] for r in raws[k]])),
                            "nominal_lat_s": float(np.nanmean([r["nominal"]["lat_mean_s"] for r in raws[k]]))} for k in keys}
        out[label] = {"capacity_99": cap, "capacity_95": cap95, "stress": stress_stats, "feasible": feas,
                      "feasible_by_target": thr, "ranks_unscreened": {m: {keys[i]: int(R_all[m]["rank"][i]) for i in range(len(keys))} for m in R_all},
                      "first_unscreened": {m: keys[int(np.argmin(R_all[m]["rank"]))] for m in R_all},
                      "screened": rf, "weights_full": base_w.tolist(), "smaa_b1_uniform_full": sm_all,
                      "matrix": {keys[i]: Xm[i].tolist() for i in range(len(keys))},
                      "halfwidth": {keys[i]: hw[i].tolist() for i in range(len(keys))}}
        print(f"[{time.time()-t0:.0f}s] {label}: capacity {cap}; feasible {[k for k in keys if feas[k]]}; "
              f"first unscreened {out[label]['first_unscreened']}; first screened {rf.get('first_own')}", flush=True)
    # boundary of the recommendation against the stress-hour load
    levels = list(range(110, 261, 10))
    for label, PP in (("default", P), ("provisioned", fz.provisioned(P))) if "boundary" in steps else ():
        b = fz.boundary(PP, app, levels, seeds=range(args.boundary_seeds))
        out[f"boundary_{label}"] = b
        print(f"[{time.time()-t0:.0f}s] boundary {label} done", flush=True)
    # Besu block capacity needed by QBFT
    besu = []
    import copy
    for btx in (220, 275, 330, 385, 440, 550) if "besu" in steps else ():
        PP = copy.deepcopy(P)
        for k in ("ibft2", "qbft"):
            PP[k].B_max = btx
        k2, X2, Xs2, _, raws2 = decision_matrices(PP, sc, app, seeds=range(10))
        av2 = fz.stress_availability(raws2, k2)
        feas2 = fz.screen(av2)
        bw = scenario_set(X2, TYPES)["Entropy x CRITIC (base)"]
        rf2 = fz.rank_feasible(X2, k2, feas2, bw)
        qi = k2.index("qbft")
        besu.append({"block_tx": btx, "capacity_tps_nominal": btx / 2.0, "qbft_avail_min": float(av2["qbft"].min()),
                     "qbft_C1": float(X2[qi, 0]), "qbft_C2": float(X2[qi, 1]), "qbft_C5": float(X2[qi, 4]), "qbft_C7": float(X2[qi, 6]),
                     "qbft_C11": float(X2[qi, 10]), "feasible": [k for k in k2 if feas2[k]], "first_screened": rf2.get("first_own", {})})
    if besu:
        out["besu_block_capacity"] = besu
        print(f"[{time.time()-t0:.0f}s] Besu capacity sweep done", flush=True)
    # deployment ablations with the screen, default and provisioned configuration
    if "ablations" in steps:
        out["ablations_screened"] = {}
        for label, PP in (("default", P), ("provisioned", fz.provisioned(P))):
            rows = {}
            for ab, (sc2, P2) in fz.ablation_set(PP, n=7).items():
                k2, X2, Xs2, _, raws2 = decision_matrices(P2, sc2, app, seeds=range(10))
                av2 = fz.stress_availability(raws2, k2)
                feas2 = fz.screen(av2)
                bw = scenario_set(X2, TYPES)["Entropy x CRITIC (base)"]
                rf2 = fz.rank_feasible(X2, k2, feas2, bw)
                R2 = rank_all(X2, bw, TYPES)
                rows[ab] = {"feasible": rf2["feasible"], "first_screened": rf2.get("first_own", {}),
                            "first_unscreened": {m: k2[int(np.argmin(R2[m]["rank"]))] for m in R2},
                            "avail_min": {k: float(v.min()) for k, v in av2.items()}}
            out["ablations_screened"][label] = rows
            print(f"[{time.time()-t0:.0f}s] screened ablations {label} done", flush=True)
    # provisioned configuration: ranking under the ten weight vectors and the single-weight sweep (TOPSIS)
    if "screen" in steps or "weights" in steps:
        from consensus_mcdm.robustness import single_weight_sweep
        kk = list(out["provisioned"]["matrix"]); Xp = np.array([out["provisioned"]["matrix"][k] for k in kk])
        Wp = scenario_set(Xp, TYPES)
        out["provisioned_weight_vectors"] = {wn: {m: kk[int(np.argmin(r["rank"]))] for m, r in rank_all(Xp, w, TYPES).items()} for wn, w in Wp.items()}
        sw = single_weight_sweep(Xp, Wp["Entropy x CRITIC (base)"], TYPES, kk, "TOPSIS")
        out["provisioned_single_weight_sweep_topsis"] = {CRITERIA[j]: list(sw[j]) for j in range(len(CRITERIA))}
    # random weight perturbations of the screened rankings (+-10..50 %, 200 trials each)
    if "perturb" in steps:
        from consensus_mcdm.robustness import weight_perturbation
        res = {}
        for label, f in (("default", "decision_matrix_samples.npy"), ("provisioned", "decision_matrix_samples_provisioned.npy")):
            kk = list(out[label]["matrix"]); Xp = np.load(os.path.join(RESULTS, f)).mean(axis=0)
            idx = [i for i, k in enumerate(kk) if out[label]["feasible"][k]]
            Xf = Xp[idx]; wf = scenario_set(Xf, TYPES)["Entropy x CRITIC (base)"]
            res[label] = {m: weight_perturbation(Xf, wf, TYPES, [kk[i] for i in idx], m, rng=np.random.default_rng(7)) for m in METHODS}
        out["screened_weight_perturbation"] = res
    json.dump(out, open(path, "w"), indent=1, default=float)
    if hasattr(report, "fig_feasibility"):
        report.fig_feasibility(out, os.path.join(FIG, "fig_feasibility.png"))
    print(f"[{time.time()-t0:.0f}s] all done")


if __name__ == "__main__":
    main()
