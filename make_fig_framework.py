#!/usr/bin/env python3
"""Figure 1: overview of the decision-support framework (static schematic, six stages + output)."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.size": 7.5, "font.family": "DejaVu Sans"})

W, H = 100.0, 64.0
fig, ax = plt.subplots(figsize=(7.3, 4.7))
ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
BLUE, ORANGE, GREY = "#1f4e79", "#b35c00", "#444444"

def box(x, y, w, h, title, lines, fc="#eef3f8", ec=BLUE, sub=None):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.0", fc=fc, ec=ec, lw=1.0))
    ax.text(x + w / 2, y + h - 1.6, title, ha="center", va="top", fontsize=8.0, fontweight="bold", color=ec)
    if sub:
        ax.text(x + w / 2, y + h - 5.2, sub, ha="center", va="top", fontsize=6.0, color=ec, style="italic")
    y0 = y + h - (8.2 if sub else 5.4)
    for i, ln in enumerate(lines):
        ax.text(x + 1.3, y0 - 2.95 * i, ln, ha="left", va="top", fontsize=6.3, color="#111111")

def arrow(x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=8, lw=0.9, color=GREY, shrinkA=0, shrinkB=0))

bw, bh, gap = 30.5, 26.0, 3.0
x1, x2, x3 = 1.0, 1.0 + bw + gap, 1.0 + 2 * (bw + gap)
yt, yb = H - 1.0 - bh, H - 1.0 - 2 * bh - 3.5

box(x1, yt, bw, bh, "1  Inputs", [
    "Published benchmark campaigns with", "open licences: Fabric 3.1, Besu 25.7,", "CometBFT v1, 2-chain HotStuff",
    "Protocol specifications: rounds, quorum,", "finality, rotation, failure timers", "Synthetic open-university workload"], sub="Sections 3.3, 3.4")
box(x2, yt, bw, bh, "2  Calibrated simulation", [
    "Discrete-event model: block formation,", "consensus rounds, signature processing,", "transfer, application-tier commit",
    "Calibration on anchor campaigns,", "validation on withheld campaigns", "Scenario θ; S = 30 seeds per alternative"], sub="Sections 3.4, 3.5")
box(x3, yt, bw, bh, "3  Decision matrix", [
    "7 protocols × 11 criteria", "C1 throughput, C2 latency, C3 finality",
    "C4 power, C5 computing, C6 storage", "C7 fault tolerance, C8 attack resistance",
    "C9 decentralisation, C10 manageability,", "C11 cost; means with 95 % CI (Eq. 1)"], sub="Sections 3.3, 4.2")
box(x1, yb, bw, bh, "4  Objective weights", [
    "Entropy weights (dispersion)", "CRITIC weights (contrast, conflict)",
    "Combined: normalised geometric mean", "Weight scenarios: equal, group", "emphasis, six published scenarios"], sub="Sections 3.6, 4.3")
box(x2, yb, bw, bh, "5  Ranking", [
    "TOPSIS, VIKOR, COPRAS, EDAS", "Scores and ranks per method (Eq. 2)",
    "Agreement: Spearman, Kendall, WS", "Per-seed rankings: share of seeds", "with the same first place"], sub="Sections 3.6, 4.4")
box(x3, yb, bw, bh, "6  Robustness and comparison", [
    "SMAA-2: 10 000 weights × 30 seed", "matrices → rank acceptability (Eq. 3)",
    "Leave-one-out rank reversal, weight", "perturbations, single-weight sweeps",
    "Six deployment ablations; published", "scenarios reproduced, weights transferred"], sub="Sections 3.7, 4.5–4.7")

# output strip
oy, oh = 0.8, yb - 0.8 - 3.0
ax.add_patch(FancyBboxPatch((x1, oy), 3 * bw + 2 * gap, oh, boxstyle="round,pad=0.3,rounding_size=1.0", fc="#fff4e5", ec=ORANGE, lw=1.0))
ax.text(x1 + 1.5, oy + oh / 2, "Output", ha="left", va="center", fontsize=8.0, fontweight="bold", color=ORANGE)
ax.text(x1 + 11.5, oy + oh / 2,
        "ranked shortlist with rank-acceptability indices; conditions that change the first place; "
        "one table of assumptions; open code and data",
        ha="left", va="center", fontsize=6.3, color="#111111")

# arrows: 1->2->3, 3 down to 4 (wrap), 4->5->6, 6 -> output
arrow(x1 + bw + 0.3, yt + bh / 2, x2 - 0.3, yt + bh / 2)
arrow(x2 + bw + 0.3, yt + bh / 2, x3 - 0.3, yt + bh / 2)
# wrap arrow from box 3 to box 4: go down the right side, then left along the middle gap, then down into box 4
ym = yt - 1.75
ax.plot([x3 + bw / 2, x3 + bw / 2], [yt - 0.3, ym], color=GREY, lw=0.9)
ax.plot([x3 + bw / 2, x1 + bw / 2], [ym, ym], color=GREY, lw=0.9)
arrow(x1 + bw / 2, ym, x1 + bw / 2, yb + bh + 0.3)
arrow(x1 + bw + 0.3, yb + bh / 2, x2 - 0.3, yb + bh / 2)
arrow(x2 + bw + 0.3, yb + bh / 2, x3 - 0.3, yb + bh / 2)
arrow(x3 + bw / 2, yb - 0.3, x3 + bw / 2, oy + oh + 0.3)
fig.subplots_adjust(left=0.005, right=0.995, top=0.995, bottom=0.005)
fig.savefig(os.path.join(FIG, "fig_framework.png"), dpi=300); fig.savefig(os.path.join(FIG, "fig_framework.pdf"))
print("saved")
