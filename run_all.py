#!/usr/bin/env python3
"""Run the complete experiment: calibration -> decision matrices (30 seeds) -> weights -> rankings ->
SMAA-2 -> robustness -> IoT-45 scenario transfer -> ablations -> tables and figures.

Usage:  python run_all.py [--seeds 30] [--skip-calibration]
Outputs: results/*.csv|json|md, figures/*.png|pdf
"""
import argparse, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from consensus_mcdm.protocols import catalogue, load_calibrated, RESULTS, quorum
from consensus_mcdm import calibrate
from consensus_mcdm.criteria import decision_matrices, CRITERIA, TYPES, UNITS, Scenario
from consensus_mcdm.weights import scenario_set, entropy_weights, critic_weights, combined_weights, IOT45_SCENARIOS
from consensus_mcdm.mcdm import rank_all, METHODS
from consensus_mcdm.smaa import smaa2, holistic_acceptability
from consensus_mcdm.robustness import rank_agreement, leave_one_out, weight_perturbation, single_weight_sweep
from consensus_mcdm.iot45 import reproduce, transfer as iot45_transfer
from consensus_mcdm.workload import WorkloadParams, summary, hourly_rate
from consensus_mcdm import report

FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--skip-calibration", action="store_true")
    ap.add_argument("--n", type=int, default=7)
    args = ap.parse_args()
    os.makedirs(RESULTS, exist_ok=True); os.makedirs(FIG, exist_ok=True)
    t0 = time.time()

    # 1. calibration
    if args.skip_calibration and os.path.exists(os.path.join(RESULTS, "calibrated_params.json")):
        P = load_calibrated(catalogue())
        cal = json.load(open(os.path.join(RESULTS, "calibrated_params.json")))
    else:
        P, cfg_f, cal, summ = calibrate.run_all(verbose=True)
    app_tier = cal["app_tier"]
    print(f"[{time.time()-t0:.0f}s] calibration done")

    # 2. decision matrices over seeds (base scenario)
    sc = Scenario(n=args.n)
    keys, Xm, Xs, hw, raws = decision_matrices(P, sc, app_tier, seeds=range(args.seeds))
    names = [P[k].name for k in keys]
    report.save_matrix(names, Xm, hw, CRITERIA, UNITS, TYPES, os.path.join(RESULTS, "decision_matrix.csv"))
    np.save(os.path.join(RESULTS, "decision_matrix_samples.npy"), Xs)
    json.dump({"keys": keys, "names": names, "criteria": CRITERIA, "types": TYPES.tolist(), "raw": raws}, open(os.path.join(RESULTS, "raw_sim.json"), "w"), default=float)
    print(f"[{time.time()-t0:.0f}s] decision matrices done ({args.seeds} seeds)")
    # diagnostics behind the matrix: load response and fault timeline (Figure 4)
    from consensus_mcdm.diagnostics import load_sweep, fault_timeline
    sweep_rows = load_sweep(P, sc, app_tier, seeds=range(10))
    report.save_rows(sweep_rows, os.path.join(RESULTS, "load_sweep.csv"))
    tl_rows = fault_timeline(P, sc, app_tier)
    report.save_rows(tl_rows, os.path.join(RESULTS, "fault_timeline.csv"))
    _lv = summary(sc.wl)
    report.fig_load_response(sweep_rows, tl_rows, (_lv["nominal_tps"], _lv["deadline_peak_tps"], _lv["stress_tps"]), os.path.join(FIG, "fig_load_response.png"))
    print(f"[{time.time()-t0:.0f}s] load sweep and fault timeline done")

    # 3. weights and rankings
    W = scenario_set(Xm, TYPES)
    report.save_weights(W, CRITERIA, os.path.join(RESULTS, "weights.csv"))
    base_w = W["Entropy x CRITIC (base)"]
    R = rank_all(Xm, base_w, TYPES)
    report.save_rankings(names, R, os.path.join(RESULTS, "rankings_base.csv"))
    # rankings per scenario per method
    rows = []
    for wname, w in W.items():
        for mname, res in rank_all(Xm, w, TYPES).items():
            for i, nm in enumerate(names):
                rows.append({"scenario": wname, "method": mname, "alternative": nm, "score": float(res["score"][i]), "rank": int(res["rank"][i])})
    report.save_rows(rows, os.path.join(RESULTS, "rankings_all_scenarios.csv"))
    # per-seed rankings (value uncertainty)
    seed_rows = []
    for s in range(Xs.shape[0]):
        for mname, res in rank_all(Xs[s], base_w, TYPES).items():
            for i, nm in enumerate(names):
                seed_rows.append({"seed": s, "method": mname, "alternative": nm, "rank": int(res["rank"][i])})
    report.save_rows(seed_rows, os.path.join(RESULTS, "rankings_per_seed.csv"))
    print(f"[{time.time()-t0:.0f}s] rankings done")

    # 4. SMAA-2 (weight uncertainty: no information; and centred on base weights) x value uncertainty (seeds)
    smaa_out = {}
    for mname in METHODS:
        a = smaa2(Xs, TYPES, method=mname, n_weights=10000, seed=42)
        b = smaa2(Xs, TYPES, method=mname, n_weights=10000, seed=43, center=base_w, concentration=50.0)
        smaa_out[mname] = {"uniform": a, "centred": b}
        report.save_smaa(names, a, os.path.join(RESULTS, f"smaa_uniform_{mname}.csv"))
        report.save_smaa(names, b, os.path.join(RESULTS, f"smaa_centred_{mname}.csv"))
    print(f"[{time.time()-t0:.0f}s] SMAA done")

    # 5. robustness
    agree = rank_agreement({m: R[m]["rank"] for m in R})
    report.save_rows(agree, os.path.join(RESULTS, "method_agreement.csv"))
    loo = {m: leave_one_out(Xm, base_w, TYPES, names, m) for m in METHODS}
    json.dump(loo, open(os.path.join(RESULTS, "leave_one_out.json"), "w"), indent=1, default=str)
    pert = {m: weight_perturbation(Xm, base_w, TYPES, names, m, rng=np.random.default_rng(7)) for m in METHODS}
    json.dump(pert, open(os.path.join(RESULTS, "weight_perturbation.json"), "w"), indent=1)
    sweep = single_weight_sweep(Xm, base_w, TYPES, names, "TOPSIS")
    report.save_sweep(sweep, CRITERIA, os.path.join(RESULTS, "single_weight_sweep_topsis.csv"))
    print(f"[{time.time()-t0:.0f}s] robustness done")

    # 6. IoT-45: reproduction + transfer of scenario weights to our matrix (common family members)
    rep = reproduce()
    json.dump({k: {kk: (vv.tolist() if hasattr(vv, "tolist") else vv) for kk, vv in v.items()} for k, v in rep.items()},
              open(os.path.join(RESULTS, "iot45_reproduction.json"), "w"), indent=1)
    transfer, transfer_agree = iot45_transfer(Xm, keys, TYPES, IOT45_SCENARIOS, rep)
    report.save_rows(transfer, os.path.join(RESULTS, "iot45_transfer.csv"))
    report.save_rows(transfer_agree, os.path.join(RESULTS, "iot45_agreement.csv"))
    print(f"[{time.time()-t0:.0f}s] IoT45 done")

    # 7. ablations / sensitivity scenarios: n=4 and n=10, RTT 20 ms, no application tier for CometBFT/HotStuff
    abl = {}
    import copy
    P_hs = copy.deepcopy(P); P_hs["hotstuff"].T_b = 1.0   # HotStuff with a 1-s proposal interval instead of 100 ms
    P_rc = copy.deepcopy(P); P_rc["ibft2"].view_change_s = 10.0; P_rc["qbft"].view_change_s = 10.0   # Besu requesttimeoutseconds 10 (config-reference example) instead of 4
    for label, (sc2, P2) in {"n4": (Scenario(n=4), P), "n10": (Scenario(n=10), P), "rtt20ms": (Scenario(n=args.n, rtt_s=0.02), P),
                             "no_app_tier": (Scenario(n=args.n, apply_app_tier=False), P),
                             "operator_removes_dead": (Scenario(n=args.n, exclude_dead=True), P),
                             "hotstuff_Tb1s": (Scenario(n=args.n), P_hs),
                             "besu_roundchange_10s": (Scenario(n=args.n), P_rc)}.items():
        k2, Xm2, Xs2, hw2, _ = decision_matrices(P2, sc2, app_tier, seeds=range(10))
        w2 = scenario_set(Xm2, TYPES)["Entropy x CRITIC (base)"]
        R2 = rank_all(Xm2, w2, TYPES)
        abl[label] = {"matrix": Xm2.tolist(), "weights": w2.tolist(), "ranks": {m: R2[m]["rank"].tolist() for m in R2}}
        report.save_matrix(names, Xm2, hw2, CRITERIA, UNITS, TYPES, os.path.join(RESULTS, f"decision_matrix_{label}.csv"))
    json.dump(abl, open(os.path.join(RESULTS, "ablations.json"), "w"), indent=1)
    print(f"[{time.time()-t0:.0f}s] ablations done")

    # 8. figures
    report.figures(names, Xm, hw, CRITERIA, TYPES, W, R, smaa_out, agree, pert, sweep, rep, transfer, abl, cal, sc, FIG)
    # 9. summary
    report.fig_robustness(RESULTS, os.path.join(FIG, "fig_robustness.png"))
    report.summary_md(names, keys, Xm, hw, CRITERIA, UNITS, TYPES, W, R, smaa_out, agree, loo, pert, rep, transfer, abl, cal, sc, os.path.join(RESULTS, "SUMMARY.md"), n_seeds=args.seeds)
    print(f"[{time.time()-t0:.0f}s] all done")


if __name__ == "__main__":
    main()
