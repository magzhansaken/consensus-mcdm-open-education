"""Tables (CSV/Markdown) and figures for the paper."""
from __future__ import annotations
import csv, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .workload import hourly_rate, summary

plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 150, "savefig.dpi": 300})
PALETTE = ["#1f4e79", "#c0504d", "#4f81bd", "#9bbb59", "#8064a2", "#f79646", "#4bacc6"]


def save_rows(rows, path):
    if not rows:
        return
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)


def save_matrix(names, Xm, hw, criteria, units, types, path):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["alternative"] + [f"{c} [{u}] ({'+' if t > 0 else '-'})" for c, u, t in zip(criteria, units, types)])
        for i, nm in enumerate(names):
            w.writerow([nm] + [f"{Xm[i, j]:.6g}" for j in range(len(criteria))])
        w.writerow([])
        w.writerow(["CI95 half-width"])
        for i, nm in enumerate(names):
            w.writerow([nm] + [f"{hw[i, j]:.3g}" for j in range(len(criteria))])


def save_weights(W, criteria, path):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["scenario"] + criteria)
        for k, v in W.items():
            w.writerow([k] + [f"{x:.4f}" for x in v])


def save_rankings(names, R, path):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["alternative"] + [f"{m} score" for m in R] + [f"{m} rank" for m in R])
        for i, nm in enumerate(names):
            w.writerow([nm] + [f"{R[m]['score'][i]:.4f}" for m in R] + [int(R[m]["rank"][i]) for m in R])


def save_smaa(names, res, path):
    acc = res["acceptability"]
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["alternative"] + [f"b_rank{r}" for r in range(1, len(names) + 1)] + ["confidence"] + [f"central_w{j+1}" for j in range(res["central_weights"].shape[1])])
        for i, nm in enumerate(names):
            w.writerow([nm] + [f"{acc[i, r]:.4f}" for r in range(len(names))] + [f"{res['confidence'][i]:.3f}"] + [f"{x:.4f}" if np.isfinite(x) else "" for x in res["central_weights"][i]])


def save_sweep(sweep, criteria, path):
    grid = np.linspace(0.0, 0.5, sweep.shape[1])
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["criterion"] + [f"w={g:.2f}" for g in grid])
        for j, c in enumerate(criteria):
            w.writerow([c] + list(sweep[j]))


# ------------------------------------------------------------------------------------------ figures
def fig_workload(sc, path):
    p = sc.wl
    from .workload import summary as _summary
    s = _summary(p)
    fig, ax = plt.subplots(figsize=(6.0, 2.9))
    h = np.arange(24)
    styles = (("normal day", "normal", PALETTE[0], "-", 1.6), ("deadline day", "deadline", PALETTE[1], "--", 1.4),
              ("stress day (enrolment hour)", "stress", PALETTE[5], "-.", 1.2))
    for lab, day, col, ls, lw in styles:
        ax.step(h, hourly_rate(p, day), where="mid", label=lab, color=col, lw=lw, ls=ls)
    ax.axhline(s["iot_floor_tps"], ls=":", color="grey", lw=1, label="IoT telemetry floor")
    for lev, txt in ((s["nominal_tps"], "nominal %.1f tx/s" % s["nominal_tps"]), (s["deadline_peak_tps"], "peak %.1f tx/s" % s["deadline_peak_tps"]),
                     (s["stress_tps"], "stress %.1f tx/s" % s["stress_tps"])):
        ax.axhline(lev, color="#777777", lw=0.6, ls=(0, (1, 3)))
        ax.text(0.3, lev + 2.5, txt, fontsize=6.5, color="#444444", va="bottom")
    ax.set_xlabel("hour of day"); ax.set_ylabel("offered load (tx/s)"); ax.set_xlim(0, 23); ax.set_ylim(0, 200)
    ax.set_xticks(range(0, 24, 3))
    ax.legend(frameon=False, fontsize=6.5, ncol=2, loc="upper right")
    fig.tight_layout(); fig.savefig(path); fig.savefig(path.replace(".png", ".pdf")); plt.close(fig)


def fig_calibration(cal, path):
    rows = cal.get("fabric_validation_rows", [])
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.0))
    axes = axes.ravel()
    # (a) batch campaign latency: measured vs simulated
    ax = axes[0]
    for key, col, mk, lab in (("raft", PALETTE[0], "o", "Raft"), ("smartbft", PALETTE[1], "s", "SmartBFT")):
        r = [x for x in rows if x["set"] == "fit" and x["protocol"] == key and x["lat_meas"] < 5]
        ax.scatter([x["lat_meas"] for x in r], [x["lat_sim"] for x in r], s=18, color=col, marker=mk, label=lab)
    lim = [0, 3.5]; ax.plot(lim, lim, "k--", lw=0.8); ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("measured latency (s)"); ax.set_ylabel("simulated latency (s)")
    ax.set_title("(a) Fabric 3.1, batch size x load (fit)", fontsize=8); ax.legend(frameon=False, fontsize=7)
    # (b) hold-out RTT
    ax = axes[1]
    for key, col, mk, lab in (("raft", PALETTE[0], "o", "Raft"), ("smartbft", PALETTE[1], "s", "SmartBFT")):
        r = sorted([x for x in rows if x["set"] == "holdout_rtt" and x["protocol"] == key], key=lambda x: x["rtt_ms"])
        ax.plot([x["rtt_ms"] for x in r], [x["lat_meas"] for x in r], color=col, marker=mk, lw=1.2, label=f"{lab} measured")
        ax.plot([x["rtt_ms"] for x in r], [x["lat_sim"] for x in r], color=col, marker=mk, lw=1.2, ls="--", mfc="white", label=f"{lab} simulated")
    ax.set_xlabel("injected inter-orderer RTT (ms)"); ax.set_ylabel("latency (s)")
    ax.set_title("(b) Fabric 3.1, RTT campaign (hold-out)", fontsize=8); ax.legend(frameon=False, fontsize=6)
    # (c) CometBFT throughput
    ax = axes[2]
    cr = cal.get("cometbft", {}).get("rows", [])
    if cr:
        ax.plot([x["offered"] for x in cr], [x["tps_meas"] for x in cr], color=PALETTE[6], marker="^", lw=1.2, label="measured (QA v1, 200 nodes)")
        ax.plot([x["offered"] for x in cr], [x["tps_sim"] for x in cr], color=PALETTE[6], marker="^", lw=1.2, ls="--", mfc="white", label="simulated")
    ax.set_xlabel("offered load (tx/s)"); ax.set_ylabel("achieved throughput (tx/s)")
    ax.set_title("(c) CometBFT v1, 200 nodes (1600 tx/s: hold-out)", fontsize=8); ax.legend(frameon=False, fontsize=6, loc="lower right")
    # (d) HotStuff throughput and latency at 10 nodes
    ax = axes[3]
    hr = cal.get("hotstuff", {}).get("rows", [])
    if hr:
        h0 = sorted([x for x in hr if x["n"] == 10 and x["faults"] == 0], key=lambda x: x["offered"])
        ax.plot([x["offered"] / 1000 for x in h0], [x["tps_meas"] / 1000 for x in h0], color=PALETTE[4], marker="d", lw=1.2, label="throughput measured")
        ax.plot([x["offered"] / 1000 for x in h0], [x["tps_sim"] / 1000 for x in h0], color=PALETTE[4], marker="d", lw=1.2, ls="--", mfc="white", label="throughput simulated")
        ax2 = ax.twinx()
        if "lat_meas" in h0[0]:
            ax2.plot([x["offered"] / 1000 for x in h0], [x["lat_meas"] for x in h0], color=PALETTE[3], marker="o", lw=1.0, label="latency measured")
            ax2.plot([x["offered"] / 1000 for x in h0], [x["lat_sim"] for x in h0], color=PALETTE[3], marker="o", lw=1.0, ls="--", mfc="white", label="latency simulated")
        ax2.set_ylabel("latency (s)", color=PALETTE[3]); ax2.tick_params(axis="y", labelsize=7, colors=PALETTE[3])
        h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, frameon=False, fontsize=6, loc="upper left")
    ax.set_xlabel("offered load (thousand tx/s)"); ax.set_ylabel("throughput (thousand tx/s)")
    ax.set_title("(d) HotStuff 2-chain, 10 nodes (fit)", fontsize=8)
    fig.tight_layout(); fig.savefig(path); fig.savefig(path.replace(".png", ".pdf")); plt.close(fig)


def fig_matrix(names, Xm, hw, criteria, types, path):
    """Heat map of the decision matrix: colour = min-max normalised value in the benefit direction
    (1 = best among the alternatives), annotation = raw mean value."""
    Xm = np.asarray(Xm, dtype=float); k = len(criteria); m = len(names)
    lo, hi = Xm.min(axis=0), Xm.max(axis=0); span = np.where(hi > lo, hi - lo, 1.0)
    R = np.where(np.asarray(types) > 0, (Xm - lo) / span, (hi - Xm) / span)
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    im = ax.imshow(R, cmap="YlGnBu", vmin=0, vmax=1, aspect="auto")
    short = [c.split(" ")[0] for c in criteria]
    labels = {"C1": "C1\nthrough-\nput", "C2": "C2\nlatency", "C3": "C3\nfinali-\nzation", "C4": "C4\npower", "C5": "C5\ncomput-\ning", "C6": "C6\nstorage",
              "C7": "C7\nfault\ntolerance", "C8": "C8\nattack\nresistance", "C9": "C9\ndecentra-\nlization", "C10": "C10\nmanage-\nability", "C11": "C11\noperational\ncost"}
    ax.set_xticks(range(k)); ax.set_xticklabels([labels[c] for c in short], fontsize=6)
    ax.set_yticks(range(m)); ax.set_yticklabels([n.split(" (")[0] for n in names], fontsize=7)
    fmt = ["{:.0f}", "{:.2f}", "{:.2f}", "{:.3f}", "{:.1f}", "{:.0f}", "{:.0f}", "{:.0f}", "{:.2f}", "{:.0f}", "{:.0f}"]
    for i in range(m):
        for j in range(k):
            ax.text(j, i, fmt[j].format(Xm[i, j]), ha="center", va="center", fontsize=6, color="white" if R[i, j] > 0.6 else "black")
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02); cb.set_label("normalised value (1 = best)", fontsize=6.5); cb.ax.tick_params(labelsize=6)
    ax.set_xticks(np.arange(-0.5, k, 1), minor=True); ax.set_yticks(np.arange(-0.5, m, 1), minor=True)
    ax.grid(which="minor", color="white", lw=0.6); ax.tick_params(which="minor", length=0)
    fig.tight_layout(); fig.savefig(path); fig.savefig(path.replace(".png", ".pdf")); plt.close(fig)


SCEN_LABEL = {"Equal": "Equal", "Entropy": "Entropy", "CRITIC": "CRITIC", "Entropy x CRITIC (base)": "Base",
              "IoT45 S2 performance": "S2", "IoT45 S3 resources": "S3", "IoT45 S4 security": "S4",
              "IoT45 S5 governance": "S5", "IoT45 S6 security-high": "S6", "IoT45 S7 balanced-gov": "S7"}


def short_name(nm):
    return nm.split(" (")[0]


def fig_rank_heatmap(names, W, Xm, types, path):
    from .mcdm import rank_all, METHODS
    scen = list(W); meth = list(METHODS)
    fig, axes = plt.subplots(1, len(meth), figsize=(7.4, 2.6), sharey=True)
    for mi, m in enumerate(meth):
        M = np.zeros((len(names), len(scen)))
        for sj, s in enumerate(scen):
            M[:, sj] = rank_all(Xm, W[s], types)[m]["rank"]
        ax = axes[mi]
        ax.imshow(M, cmap="Blues_r", vmin=1, vmax=len(names) + 1.5, aspect="auto")
        for i in range(len(names)):
            for j in range(len(scen)):
                v = int(M[i, j])
                ax.text(j, i, v, ha="center", va="center", fontsize=6.5, color="white" if v <= 2 else "#111111",
                        fontweight="bold" if v == 1 else "normal")
        ax.set_title(m, fontsize=8)
        ax.set_xticks(range(len(scen))); ax.set_xticklabels([SCEN_LABEL.get(s, s) for s in scen], rotation=90, fontsize=6.5)
        ax.tick_params(length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
        if mi == 0:
            ax.set_yticks(range(len(names))); ax.set_yticklabels([short_name(n) for n in names], fontsize=7)
    fig.tight_layout(); fig.savefig(path); fig.savefig(path.replace(".png", ".pdf")); plt.close(fig)


def fig_smaa(names, smaa_out, path):
    meth = list(smaa_out)
    fig, axes = plt.subplots(1, len(meth), figsize=(7.4, 2.6), sharey=True)
    m = len(names)
    cmap = plt.get_cmap("viridis")
    for mi, mname in enumerate(meth):
        acc = smaa_out[mname]["uniform"]["acceptability"]
        ax = axes[mi]; left = np.zeros(m)
        for r in range(m):
            ax.barh(range(m), acc[:, r], left=left, color=cmap(r / max(m - 1, 1)), label=f"rank {r+1}")
            left += acc[:, r]
        ax.set_title(mname, fontsize=8); ax.set_xlim(0, 1); ax.set_xlabel("rank acceptability", fontsize=7)
        if mi == 0:
            ax.set_yticks(range(m)); ax.set_yticklabels(names, fontsize=6.5)
    axes[-1].legend(frameon=False, fontsize=5.5, loc="lower right")
    fig.tight_layout(); fig.savefig(path); fig.savefig(path.replace(".png", ".pdf")); plt.close(fig)


def fig_sensitivity(names, criteria, sweep, pert, path):
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.7), gridspec_kw={"width_ratios": [1.6, 1]})
    ax = axes[0]
    grid = np.linspace(0.0, 0.5, sweep.shape[1])
    name_idx = {n: i for i, n in enumerate(names)}
    M = np.array([[name_idx[sweep[j, g]] for g in range(len(grid))] for j in range(len(criteria))])
    cmap = plt.get_cmap("tab10", len(names))
    im = ax.imshow(M, cmap=cmap, vmin=-0.5, vmax=len(names) - 0.5, aspect="auto")
    ax.set_yticks(range(len(criteria))); ax.set_yticklabels([c.split(" ", 1)[1] for c in criteria], fontsize=6)
    ax.set_xticks(range(len(grid))); ax.set_xticklabels([f"{g:.2f}" for g in grid], fontsize=6, rotation=60)
    ax.set_xlabel("weight assigned to the criterion (others rescaled)", fontsize=7); ax.set_title("(a) rank-1 alternative, TOPSIS", fontsize=8)
    cb = fig.colorbar(im, ax=ax, ticks=range(len(names)), fraction=0.04); cb.ax.set_yticklabels(names, fontsize=6)
    ax = axes[1]
    for mi, (m, rows) in enumerate(pert.items()):
        ax.plot([r["delta"] for r in rows], [r["top_unchanged"] for r in rows], marker="o", lw=1.2, color=PALETTE[mi], label=m)
    ax.set_xlabel("relative weight perturbation ±δ", fontsize=7); ax.set_ylabel("share of trials with unchanged rank 1", fontsize=7)
    ax.set_ylim(0, 1.05); ax.set_title("(b) random perturbation (200 trials)", fontsize=8); ax.legend(frameon=False, fontsize=6)
    fig.tight_layout(); fig.savefig(path); fig.savefig(path.replace(".png", ".pdf")); plt.close(fig)


def fig_iot45(rep, transfer, path):
    """Figure: (a) recomputed against published TOPSIS C and VIKOR Q of the preceding framework, Scenarios 2-7;
    (b) TOPSIS rank of the four common families: published order and this study with each PBFT-family member."""
    from matplotlib.gridspec import GridSpec
    from .iot45 import PUB_C, PUB_Q, COMMON, FAMILY_MEMBERS
    fig = plt.figure(figsize=(7.4, 2.75))
    gs = GridSpec(1, 2, width_ratios=[1.0, 2.6], wspace=0.55, figure=fig)
    ax = fig.add_subplot(gs[0])
    pc = np.concatenate([PUB_C[s] for s in rep]); rc = np.concatenate([rep[s]["C"] for s in rep])
    pq = np.concatenate([PUB_Q[s] for s in rep]); rq = np.concatenate([rep[s]["Q"] for s in rep])
    ax.plot([0, 1], [0, 1], color="#9a9a9a", lw=0.8, ls="--", zorder=1)
    ax.scatter(pc, rc, s=16, marker="o", color=PALETTE[0], edgecolor="white", linewidth=0.5, label="TOPSIS $C$", zorder=3)
    ax.scatter(pq, rq, s=16, marker="s", color=PALETTE[1], edgecolor="white", linewidth=0.5, label="VIKOR $Q$", zorder=2)
    dmax = max(np.abs(pc - rc).max(), np.abs(pq - rq).max())
    ex = int(np.floor(np.log10(dmax))); mant = dmax / 10 ** ex
    ax.text(0.97, 0.05, f"max $|\\Delta|$ = {mant:.1f}$\\times10^{{{ex}}}$", transform=ax.transAxes, ha="right", va="bottom", fontsize=6.5)
    ax.set_xlim(-0.03, 1.03); ax.set_ylim(-0.03, 1.03); ax.set_aspect("equal")
    ax.set_xlabel("published value", fontsize=7); ax.set_ylabel("recomputed value", fontsize=7)
    ax.tick_params(labelsize=6.5); ax.legend(frameon=False, fontsize=6.5, loc="upper left", handletextpad=0.2)
    ax.set_title("(a) Scenarios 2–7, 72 values", fontsize=7.5)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    # (b) rank grid
    axb = fig.add_subplot(gs[1])
    scen = list(dict.fromkeys(r["scenario"] for r in transfer))
    members = FAMILY_MEMBERS["PBFT"]
    cols = []  # (scenario, source)
    for s in scen:
        cols.append((s, "published")); cols += [(s, m) for m in members]
    M = np.zeros((len(COMMON), len(cols)))
    for j, (s, src) in enumerate(cols):
        for i, f in enumerate(COMMON):
            if src == "published":
                r = next(r for r in transfer if r["scenario"] == s and r["family"] == f and r["pbft_member"] == members[0])
                M[i, j] = r["published_TOPSIS"]
            else:
                r = next(r for r in transfer if r["scenario"] == s and r["family"] == f and r["pbft_member"] == src)
                M[i, j] = r["ours_TOPSIS"]
    axb.imshow(M, cmap="Blues_r", vmin=1, vmax=5.5, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = int(M[i, j])
            axb.text(j, i, v, ha="center", va="center", fontsize=6.5, color="white" if v <= 2 else "#111111",
                     fontweight="bold" if v == 1 else "normal")
    k = 1 + len(members)
    for g in range(1, len(scen)):
        axb.axvline(g * k - 0.5, color="white", lw=3)
    short = {"published": "P", "smartbft": "S", "ibft2": "I", "qbft": "Q"}
    axb.set_xticks(range(len(cols))); axb.set_xticklabels([short[c[1]] for c in cols], fontsize=6.5)
    lab = {"S2 performance": "S2\nperformance", "S3 resources": "S3\nresources", "S4 security": "S4\nsecurity",
           "S5 governance": "S5\ngovernance", "S6 security-high": "S6\nsecurity, high", "S7 balanced-gov": "S7\nbalanced"}
    for g, s in enumerate(scen):
        axb.text(g * k + (k - 1) / 2, -0.62, lab.get(s, s), ha="center", va="bottom", fontsize=6.3, linespacing=1.0)
    rowlab = {"Tendermint": "Tendermint/CometBFT", "PBFT": "PBFT/member", "Raft": "Raft/Raft", "PoA": "PoA/Clique"}
    axb.set_yticks(range(len(COMMON))); axb.set_yticklabels([rowlab[f] for f in COMMON], fontsize=6.5)
    axb.tick_params(length=0)
    for sp in axb.spines.values():
        sp.set_visible(False)
    axb.set_title("(b) TOPSIS rank among the four common families", fontsize=7.5, pad=24)
    fig.subplots_adjust(left=0.07, right=0.995, bottom=0.15, top=0.80)
    fig.savefig(path); fig.savefig(path.replace(".png", ".pdf")); plt.close(fig)


def figures(names, Xm, hw, criteria, types, W, R, smaa_out, agree, pert, sweep, rep, transfer, abl, cal, sc, figdir):
    fig_workload(sc, os.path.join(figdir, "fig_workload.png"))
    fig_calibration(cal, os.path.join(figdir, "fig_calibration.png"))
    fig_matrix(names, Xm, hw, criteria, types, os.path.join(figdir, "fig_matrix.png"))
    fig_rank_heatmap(names, W, Xm, types, os.path.join(figdir, "fig_rank_heatmap.png"))
    fig_smaa(names, smaa_out, os.path.join(figdir, "fig_smaa.png"))
    fig_sensitivity(names, criteria, sweep, pert, os.path.join(figdir, "fig_sensitivity.png"))
    fig_iot45(rep, transfer, os.path.join(figdir, "fig_iot45.png"))


# ------------------------------------------------------------------------------------------ summary
def summary_md(names, keys, Xm, hw, criteria, units, types, W, R, smaa_out, agree, loo, pert, rep, transfer, abl, cal, sc, path, n_seeds=None):
    L = []
    L.append("# Experiment summary (auto-generated by run_all.py)\n")
    s = summary(sc.wl)
    L.append(f"Scenario: n = {sc.n}, RTT = {sc.rtt_s*1000:.0f} ms, {sc.cores} vCPU/node, loads nominal/peak/stress = "
             f"{s['nominal_tps']:.1f} / {s['deadline_peak_tps']:.1f} / {s['stress_tps']:.1f} tx/s, seeds = {n_seeds if n_seeds is not None else 'see run'}.\n")
    L.append("## Decision matrix (mean over seeds; CI95 half-width)\n")
    L.append("| Alternative | " + " | ".join(f"{c} [{u}]" for c, u in zip(criteria, units)) + " |")
    L.append("|---|" + "---|" * len(criteria))
    for i, nm in enumerate(names):
        L.append(f"| {nm} | " + " | ".join(f"{Xm[i,j]:.3g} ± {hw[i,j]:.2g}" for j in range(len(criteria))) + " |")
    L.append("\n## Weights\n")
    L.append("| Scenario | " + " | ".join(c.split(" ")[0] for c in criteria) + " |")
    L.append("|---|" + "---|" * len(criteria))
    for k, v in W.items():
        L.append(f"| {k} | " + " | ".join(f"{x:.3f}" for x in v) + " |")
    L.append("\n## Rankings with the base weights (Entropy x CRITIC)\n")
    L.append("| Alternative | " + " | ".join(f"{m} score (rank)" for m in R) + " |")
    L.append("|---|" + "---|" * len(R))
    for i, nm in enumerate(names):
        L.append(f"| {nm} | " + " | ".join(f"{R[m]['score'][i]:.4f} ({int(R[m]['rank'][i])})" for m in R) + " |")
    L.append("\n## Method agreement (base weights)\n")
    L.append("| A | B | Spearman | Kendall | WS |\n|---|---|---|---|---|")
    for a in agree:
        L.append(f"| {a['A']} | {a['B']} | {a['spearman']:.3f} | {a['kendall']:.3f} | {a['WS']:.3f} |")
    L.append("\n## SMAA-2 rank-1 acceptability (uniform weights x seed matrices, 10 000 draws)\n")
    L.append("| Alternative | " + " | ".join(smaa_out) + " | holistic (TOPSIS) |\n|---|" + "---|" * (len(smaa_out) + 1))
    from .smaa import holistic_acceptability
    hol = holistic_acceptability(smaa_out["TOPSIS"]["uniform"]["acceptability"])
    for i, nm in enumerate(names):
        L.append(f"| {nm} | " + " | ".join(f"{smaa_out[m]['uniform']['acceptability'][i,0]:.3f}" for m in smaa_out) + f" | {hol[i]:.3f} |")
    L.append("\n## Leave-one-out rank reversals\n")
    for m, v in loo.items():
        L.append(f"- {m}: {v['reversals']} reversals out of {len(names)} removals")
    L.append("\n## Weight perturbation (share of trials with unchanged rank 1)\n")
    for m, rows in pert.items():
        L.append(f"- {m}: " + ", ".join(f"±{r['delta']:.0%}: {r['top_unchanged']:.2f} (ρ̄={r['mean_spearman']:.3f})" for r in rows))
    L.append("\n## IoT-45 reproduction (Scenarios 2-7, max |Δ| TOPSIS C / VIKOR Q)\n")
    for k, v in rep.items():
        L.append(f"- {k}: ΔC = {v['maxdiff_C']:.5f}, ΔQ = {v['maxdiff_Q']:.5f}; ranks identical: {np.array_equal(v['rankC'], v['rankC_pub']) and np.array_equal(v['rankQ'], v['rankQ_pub'])}")
    L.append("\n## Calibration summary\n")
    L.append("```\n" + json.dumps({k: v for k, v in cal.items() if k in ("fabric", "besu", "cometbft", "hotstuff", "app_tier")}, indent=1, default=float)[:6000] + "\n```")
    L.append("\n## Ablations (rank 1 by TOPSIS / VIKOR / COPRAS / EDAS)\n")
    for lab, v in abl.items():
        L.append(f"- {lab}: " + ", ".join(f"{m}: {names[int(np.argmin(v['ranks'][m]))]}" for m in v["ranks"]))
    open(path, "w").write("\n".join(L))


def fig_load_response(sweep, timeline, levels, path):
    """Figure: (a) committed throughput and (b) mean latency against offered load; (c) fault timeline."""
    order = [("raft", "Raft", PALETTE[0], "o"), ("smartbft", "SmartBFT", PALETTE[1], "s"), ("qbft", "IBFT 2.0 / QBFT", PALETTE[2], "^"),
             ("clique", "Clique", PALETTE[3], "v"), ("cometbft", "CometBFT", PALETTE[4], "D"), ("hotstuff", "HotStuff", PALETTE[5], "P")]
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.7))
    for key, lab, col, mk in order:
        rr = sorted([r for r in sweep if r["key"] == key], key=lambda r: r["offered"])
        x = [r["offered"] for r in rr]
        axes[0].plot(x, [r["tps_mean"] for r in rr], color=col, marker=mk, ms=3, lw=1.1, label=lab)
        axes[1].plot(x, [r["lat_mean"] for r in rr], color=col, marker=mk, ms=3, lw=1.1, label=lab)
        tl = sorted([r for r in timeline if r["key"] == key], key=lambda r: r["t_start"])
        axes[2].step([r["t_start"] for r in tl], [100 * r["share_ok"] for r in tl], where="post", color=col, lw=1.0, label=lab)
    for ax in axes[:2]:
        for lev in levels:
            ax.axvline(lev, color="#999999", lw=0.6, ls=":")
        ax.set_xlabel("offered load (tx/s)")
    axes[0].plot([0, 250], [0, 250], color="#bbbbbb", lw=0.6, ls="--")
    axes[0].set_ylabel("committed within 30 s (tx/s)"); axes[0].set_xlim(0, 255); axes[0].set_ylim(0, 255)
    axes[0].set_title("(a) throughput", fontsize=8)
    axes[1].set_yscale("log"); axes[1].set_ylabel("mean latency (s)"); axes[1].set_xlim(0, 255); axes[1].set_title("(b) latency", fontsize=8)
    for t in (120, 300):
        axes[2].axvline(t, color="#999999", lw=0.6, ls=":")
    axes[2].set_xlabel("arrival time (s)"); axes[2].set_ylabel("committed within 30 s (%)"); axes[2].set_ylim(-3, 103); axes[2].set_xlim(0, 600)
    axes[2].set_title("(c) fault scenario, 108 tx/s", fontsize=8)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=6, frameon=False, fontsize=6.5, bbox_to_anchor=(0.5, -0.01))
    for ax in axes:
        ax.tick_params(labelsize=7)
    fig.tight_layout(rect=(0, 0.07, 1, 1)); fig.savefig(path); fig.savefig(path.replace(".png", ".pdf")); plt.close(fig)


PROTO_COLOR = {"Raft": "#1f4e79", "SmartBFT": "#c0504d", "IBFT 2.0": "#9dc3e6", "QBFT": "#4f81bd", "Clique": "#9bbb59",
               "CometBFT": "#8064a2", "HotStuff": "#f79646"}


def fig_robustness(results_dir, path):
    """Figure: (a) SMAA-2 rank acceptability, TOPSIS, uniform weights; (b) first-rank acceptability by method,
    uniform and centred weights; (c) first-ranked alternative in single-weight sweeps (TOPSIS);
    (d) share of perturbation trials with unchanged first place."""
    import csv, json, os
    from matplotlib.patches import Patch
    meth = ["TOPSIS", "VIKOR", "COPRAS", "EDAS"]
    def smaa(kind, m):
        rows = list(csv.DictReader(open(os.path.join(results_dir, f"smaa_{kind}_{m}.csv"))))
        nm = [short_name(r["alternative"]) for r in rows]
        A = np.array([[float(r[f"b_rank{k}"]) for k in range(1, len(rows) + 1)] for r in rows])
        return nm, A
    fig = plt.figure(figsize=(7.4, 5.2))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.15], hspace=0.55, wspace=0.42)
    # (a)
    ax = fig.add_subplot(gs[0, 0])
    nm, A = smaa("uniform", "TOPSIS")
    order = list(np.argsort(-(A @ np.linspace(1, 0, A.shape[1]))))[::-1]
    cmap = plt.get_cmap("Blues_r")
    left = np.zeros(len(order))
    for r in range(A.shape[1]):
        vals = A[order, r]
        ax.barh(range(len(order)), vals, left=left, color=cmap(r / (A.shape[1] + 1.0)), edgecolor="white", lw=0.4, label=f"rank {r + 1}")
        left += vals
    ax.set_yticks(range(len(order))); ax.set_yticklabels([nm[i] for i in order], fontsize=7)
    ax.set_xlim(0, 1); ax.set_xlabel("rank acceptability $b^r_i$", fontsize=7.5); ax.tick_params(labelsize=7)
    ax.set_title("(a) SMAA-2, TOPSIS, uniform weights", fontsize=8)
    ax.legend(ncol=7, fontsize=5.2, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.07), handlelength=1.0, columnspacing=0.6)
    # (b)
    ax = fig.add_subplot(gs[0, 1])
    show = ["QBFT", "CometBFT", "Clique", "Raft", "HotStuff"]
    x = np.arange(len(meth)); wbar = 0.15
    for k, a in enumerate(show):
        u = [smaa("uniform", m)[1][smaa("uniform", m)[0].index(a), 0] for m in meth]
        c = [smaa("centred", m)[1][smaa("centred", m)[0].index(a), 0] for m in meth]
        ax.bar(x + (k - 2) * wbar, u, width=wbar, color=PROTO_COLOR[a], label=a)
        ax.scatter(x + (k - 2) * wbar, c, marker="_", s=60, color="black", lw=1.2, zorder=3)
    ax.set_xticks(x); ax.set_xticklabels(meth, fontsize=7); ax.set_ylim(0, 1.12); ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0]); ax.tick_params(labelsize=7)
    ax.set_ylabel("first-rank acceptability $b^1_i$", fontsize=7.5)
    ax.set_title("(b) $b^1_i$: bars uniform, ticks centred", fontsize=8)
    ax.legend(fontsize=5.8, frameon=False, ncol=5, loc="upper center", handlelength=0.9, columnspacing=0.7)
    # (c)
    ax = fig.add_subplot(gs[1, 0])
    rows = list(csv.DictReader(open(os.path.join(results_dir, "single_weight_sweep_topsis.csv"))))
    grid = [c for c in rows[0] if c.startswith("w=")]
    alts = list(PROTO_COLOR)
    M = np.array([[alts.index(short_name(r[g])) for g in grid] for r in rows])
    from matplotlib.colors import ListedColormap, BoundaryNorm
    cm = ListedColormap([PROTO_COLOR[a] for a in alts]); norm = BoundaryNorm(np.arange(-0.5, len(alts) + 0.5), cm.N)
    ax.imshow(M, cmap=cm, norm=norm, aspect="auto")
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r["criterion"].split(" ", 1)[0] for r in rows], fontsize=7)
    ax.set_xticks(range(0, len(grid), 2)); ax.set_xticklabels([grid[i][2:] for i in range(0, len(grid), 2)], fontsize=7)
    ax.set_xlabel("weight of the criterion (others rescaled)", fontsize=7.5)
    ax.set_title("(c) first place in single-weight sweeps, TOPSIS", fontsize=8)
    used = sorted(set(M.ravel()))
    ax.legend(handles=[Patch(color=PROTO_COLOR[alts[i]], label=alts[i]) for i in used], fontsize=6, frameon=False,
              loc="upper center", bbox_to_anchor=(0.5, -0.27), ncol=len(used), handlelength=1.0, columnspacing=0.8)
    # (d)
    ax = fig.add_subplot(gs[1, 1])
    pert = json.load(open(os.path.join(results_dir, "weight_perturbation.json")))
    for m, col, mk in zip(meth, ["#1f4e79", "#c0504d", "#4f81bd", "#9bbb59"], ["o", "s", "^", "D"]):
        ax.plot([r["delta"] for r in pert[m]], [r["top_unchanged"] for r in pert[m]], marker=mk, ms=4, lw=1.2, color=col, label=m)
    ax.set_xlabel("relative weight perturbation $\\pm\\delta$", fontsize=7.5); ax.set_ylabel("share with unchanged first place", fontsize=7.5)
    ax.set_ylim(0, 1.05); ax.set_xticks([0.1, 0.2, 0.3, 0.5]); ax.tick_params(labelsize=7)
    ax.set_title("(d) random weight perturbations, 200 trials", fontsize=8)
    ax.legend(fontsize=6.5, frameon=False, loc="lower left")
    fig.savefig(path, bbox_inches="tight"); fig.savefig(path.replace(".png", ".pdf"), bbox_inches="tight"); plt.close(fig)
