#!/usr/bin/env python3
"""Effect of the validation errors of the simulation model on the decision (Section 4.9).

Steps:  corrections  pipeline repeated with the RTT and failure-recovery corrections (documented and provisioned)
        montecarlo   model-error Monte Carlo at the typical and pessimistic level
        breakeven    error factors at which the first place changes
Usage:  python run_validation.py [--steps corrections,montecarlo,breakeven] [--seeds 30]
"""
import argparse, csv, json, os, sys, time, copy
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from consensus_mcdm.protocols import catalogue, load_calibrated, RESULTS
from consensus_mcdm.criteria import decision_matrices, CRITERIA, TYPES, Scenario
from consensus_mcdm.weights import scenario_set
from consensus_mcdm.mcdm import rank_all, METHODS
from consensus_mcdm.smaa import smaa2
from consensus_mcdm import feasibility as fz
from consensus_mcdm import validation as va
from consensus_mcdm import criteria as crit
from consensus_mcdm import report

FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")


class _Cfg:
    """Context: patch criteria._cfg so that every SimConfig carries the corrections."""
    def __init__(self, **kw):
        self.kw = kw; self.orig = crit._cfg

    def __enter__(self):
        orig, kw = self.orig, self.kw
        def patched(sc, app_tier, faults=None):
            c = orig(sc, app_tier, faults)
            for k, v in kw.items():
                setattr(c, k, v)
            return c
        crit._cfg = patched
        import consensus_mcdm.feasibility as f2
        f2._cfg = patched
        return self

    def __exit__(self, *a):
        crit._cfg = self.orig
        import consensus_mcdm.feasibility as f2
        f2._cfg = self.orig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", default="corrections,recovery,montecarlo,breakeven")
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--draws", type=int, default=10000)
    args = ap.parse_args()
    steps = set(args.steps.split(","))
    t0 = time.time()
    P = load_calibrated(catalogue())
    app = json.load(open(os.path.join(RESULTS, "calibrated_params.json")))["app_tier"]
    rows = list(csv.DictReader(open(os.path.join(RESULTS, "calibration_fabric.csv"))))
    c_rtt = va.rtt_correction(rows)
    feas_json = json.load(open(os.path.join(RESULTS, "feasibility.json")))
    path = os.path.join(RESULTS, "validation_effect.json")
    out = json.load(open(path)) if os.path.exists(path) else {}
    out["rtt_extra_s_per_s"] = c_rtt
    configs = (("default", P), ("provisioned", fz.provisioned(P)))
    corrections = {"none": {}, "rtt": {"rtt_extra_crash": c_rtt["crash"], "rtt_extra_bft": c_rtt["bft"]},
                   "rec2": {"recovery_factor": 2.0}, "rec5": {"recovery_factor": 5.0},
                   "rtt_rec5": {"rtt_extra_crash": c_rtt["crash"], "rtt_extra_bft": c_rtt["bft"], "recovery_factor": 5.0}}

    if "corrections" in steps:
        res = {}
        for label, PP in configs:
            res[label] = {}
            for cname, kw in corrections.items():
                for rtt in (0.002, 0.02):
                    seeds = range(args.seeds) if rtt == 0.002 else range(10)
                    with _Cfg(**kw):
                        keys, Xm, Xs, hw, raws = decision_matrices(PP, Scenario(n=7, rtt_s=rtt), app, seeds=seeds)
                    av = fz.stress_availability(raws, keys); feas = fz.screen(av)
                    bw = scenario_set(Xm, TYPES)["Entropy x CRITIC (base)"]
                    R = rank_all(Xm, bw, TYPES)
                    rf = fz.rank_feasible(Xm, keys, feas, bw, Xs=Xs if rtt == 0.002 else None)
                    tag = f"{cname}@{int(rtt*1000)}ms"
                    res[label][tag] = {"matrix": {k: Xm[i].tolist() for i, k in enumerate(keys)},
                                       "first_unscreened": {m: keys[int(np.argmin(R[m]["rank"]))] for m in R},
                                       "feasible": rf["feasible"], "first_screened": rf.get("first_own", {}),
                                       "smaa_b1_screened": rf.get("smaa_b1_uniform")}
                    print(f"[{time.time()-t0:.0f}s] {label} {tag}: unscreened {res[label][tag]['first_unscreened']} screened {res[label][tag]['first_screened']}", flush=True)
        out["corrections"] = res

    if "recovery" in steps:   # recovery factor at which provisioned QBFT loses the first place
        res = {}
        for F in (2.5, 3.0, 3.5, 4.0, 4.5):
            with _Cfg(recovery_factor=F):
                keys, Xm, Xs, hw, raws = decision_matrices(fz.provisioned(P), Scenario(n=7), app, seeds=range(10))
            feas = fz.screen(fz.stress_availability(raws, keys)); bw = scenario_set(Xm, TYPES)["Entropy x CRITIC (base)"]
            rf = fz.rank_feasible(Xm, keys, feas, bw)
            res[str(F)] = {"qbft_C7": float(Xm[keys.index("qbft"), 6]), "cometbft_C7": float(Xm[keys.index("cometbft"), 6]), "first": rf["first_own"]}
            print(f"[{time.time()-t0:.0f}s] recovery x{F}: {res[str(F)]}", flush=True)
        out["recovery_sweep_provisioned"] = res

    if "montecarlo" in steps or "breakeven" in steps:
        Xs_d = np.load(os.path.join(RESULTS, "decision_matrix_samples.npy"))
        Xs_p = np.load(os.path.join(RESULTS, "decision_matrix_samples_provisioned.npy"))
        keys = list(feas_json["default"]["matrix"])
        stress = 185.0

    if "montecarlo" in steps:
        res = {}
        for label, Xs in (("default", Xs_d), ("provisioned", Xs_p)):
            cap = feas_json[label]["capacity_99"]
            res[label] = {}
            for level in ("typical", "pessimistic"):
                a = va.monte_carlo(Xs, keys, level, n_draws=args.draws, seed=11)
                b = va.monte_carlo(Xs, keys, level, n_draws=args.draws, seed=12, capacity=cap, stress=stress)
                res[label][level] = {"unscreened": a, "screened": b}
                print(f"[{time.time()-t0:.0f}s] MC {label} {level}: unscreened QBFT {[round(a['first_share'][m]['qbft'],3) for m in METHODS]}"
                      f" screened first {[max(b['first_share'][m], key=b['first_share'][m].get) for m in METHODS]}", flush=True)
        out["montecarlo"] = res

    if "breakeven" in steps:
        Xd = Xs_d.mean(axis=0); Xp = Xs_p.mean(axis=0)
        feas_d = np.array([feas_json["default"]["feasible"][k] for k in keys])
        be = {}
        be["default_unscreened_qbft"] = {c: va.break_even(Xd, keys, "qbft", j, d) for c, j, d in
                                         (("C2 latency x", 1, +1), ("C3 finalization x", 2, +1), ("C7 loss x", 6, +1), ("C1 throughput /", 0, -1))}
        be["default_screened_cometbft"] = {c: va.break_even(Xd, keys, "cometbft", j, d, feas=feas_d) for c, j, d in
                                           (("C2 latency x", 1, +1), ("C3 finalization x", 2, +1), ("C7 loss x", 6, +1), ("C1 throughput /", 0, -1))}
        be["provisioned_qbft"] = {c: va.break_even(Xp, keys, "qbft", j, d) for c, j, d in
                                  (("C2 latency x", 1, +1), ("C3 finalization x", 2, +1), ("C7 loss x", 6, +1), ("C1 throughput /", 0, -1))}
        # CometBFT latency overestimated? factor by which CometBFT latency and finality would have to fall to overtake QBFT
        def comet_gain(X, feas=None):
            out_ = {}
            ci, qi = keys.index("cometbft"), keys.index("qbft")
            for m in METHODS:
                f_ok = float("inf")
                for fac in np.geomspace(1.0, 20.0, 200):
                    Y = X.copy(); Y[ci, 1] /= fac; Y[ci, 2] /= fac
                    if va.first_places(Y, feas)[m] == ci:
                        f_ok = float(fac); break
                out_[m] = f_ok
            return out_
        be["default_unscreened_cometbft_latency_divisor"] = comet_gain(Xd)
        be["provisioned_cometbft_latency_divisor"] = comet_gain(Xp)
        out["break_even"] = be
        print(f"[{time.time()-t0:.0f}s] break-even done", flush=True)

    json.dump(out, open(path, "w"), indent=1, default=float)
    print(f"[{time.time()-t0:.0f}s] all done")


if __name__ == "__main__":
    main()
