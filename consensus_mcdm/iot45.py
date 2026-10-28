"""Reproduction of the scenario rankings (Scenarios 2-7) of Tashatov et al., IoT 2026, 7(2), 45
(DOI 10.3390/iot7020045) on the inputs printed in that article, and transfer of their scenario
weights to the simulated decision matrix of this study.

Inputs transcribed from the article: Table 15 (6 alternatives x 11 criteria, 1-10 scale),
Table 14 (criterion types), Table 21 (scenario weights). Published outputs used as the test
oracle: Tables 22-23 (TOPSIS closeness C and VIKOR Q per scenario).
"""
from __future__ import annotations
import numpy as np
from .mcdm import topsis, vikor
from .weights import IOT45_SCENARIOS, IOT45_TYPES

ALTS = ["PoS", "DPoS", "Raft", "PBFT", "PoA", "Tendermint"]
X15 = np.array([[7, 7, 8, 8, 7, 8, 8, 7, 7, 7, 7],
                [9, 9, 8, 9, 8, 7, 7, 8, 8, 6, 8],
                [10, 10, 9, 10, 9, 8, 6, 10, 9, 3, 9],
                [8, 7, 10, 7, 8, 9, 9, 7, 8, 7, 6],
                [9, 9, 9, 10, 8, 7, 6, 9, 9, 4, 9],
                [8, 8, 10, 8, 8, 9, 9, 8, 8, 7, 7]], dtype=float)
TYPES = np.array(IOT45_TYPES)
PUB_C = {  # Tables 22: TOPSIS C per scenario (S2..S7)
    "S2 performance":   [0.56469, 0.38072, 0.37835, 0.62829, 0.36876, 0.65049],
    "S3 resources":     [0.55856, 0.41189, 0.37691, 0.58426, 0.37064, 0.62671],
    "S4 security":      [0.44490, 0.35591, 0.45883, 0.53868, 0.36907, 0.63512],
    "S5 governance":    [0.57936, 0.47626, 0.37108, 0.62737, 0.35151, 0.69194],
    "S6 security-high": [0.42418, 0.34426, 0.47009, 0.52871, 0.36979, 0.63218],
    "S7 balanced-gov":  [0.63577, 0.54075, 0.32855, 0.67614, 0.33705, 0.73493],
}
PUB_Q = {  # Tables 23: VIKOR Q per scenario
    "S2 performance":   [0.54593, 0.55120, 1.0, 0.32288, 0.72784, 0.04720],
    "S3 resources":     [0.39416, 0.30187, 1.0, 0.32288, 0.66270, 0.02796],
    "S4 security":      [0.82477, 0.48826, 0.98212, 0.58734, 1.0, 0.0],
    "S5 governance":    [0.68815, 0.36521, 1.0, 0.5, 0.90494, 0.0095],
    "S6 security-high": [0.87052, 0.49132, 0.92151, 0.62979, 1.0, 0.0],
    "S7 balanced-gov":  [0.68766, 0.34363, 1.0, 0.5, 0.90083, 0.01527],
}


def reproduce():
    out = {}
    for name, w in IOT45_SCENARIOS.items():
        w = np.asarray(w) / np.sum(w)
        t = topsis(X15, w, TYPES); v = vikor(X15, w, TYPES)
        out[name] = {"C": t["score"], "rankC": t["rank"], "Q": v["score"], "rankQ": v["rank"],
                     "maxdiff_C": float(np.abs(t["score"] - PUB_C[name]).max()),
                     "maxdiff_Q": float(np.abs(v["score"] - PUB_Q[name]).max()),
                     "rankC_pub": topsis_rank_from(PUB_C[name]), "rankQ_pub": vikor_rank_from(PUB_Q[name])}
    return out


def topsis_rank_from(C):
    C = np.asarray(C); o = np.argsort(-C); r = np.empty(len(C), int); r[o] = np.arange(1, len(C) + 1); return r


def vikor_rank_from(Q):
    Q = np.asarray(Q); o = np.argsort(Q); r = np.empty(len(Q), int); r[o] = np.arange(1, len(Q) + 1); return r


# mapping between the alternatives of the two studies (closest family member)
FAMILY_MAP = {"Raft": "raft", "PBFT": "smartbft", "PoA": "clique", "Tendermint": "cometbft"}

# every member of a family present in this study; the PBFT family has three descendants here
FAMILY_MEMBERS = {"Tendermint": ["cometbft"], "PBFT": ["smartbft", "ibft2", "qbft"], "Raft": ["raft"], "PoA": ["clique"]}
COMMON = ["Tendermint", "PBFT", "Raft", "PoA"]


def _order(r):
    """Ranks of a subset re-numbered 1..k in the same relative order."""
    r = np.asarray(r); o = np.argsort(r, kind="stable"); out = np.empty(len(r), int); out[o] = np.arange(1, len(r) + 1)
    return out


def transfer(Xm, keys, types, scenarios, rep=None):
    """Apply the published scenario weights to the measured matrix restricted to the four common families,
    once for each member of the PBFT family, and compare with the published order of the same families.
    Returns (rows, agreement)."""
    from scipy.stats import spearmanr, kendalltau
    from .mcdm import rank_all
    from .robustness import ws_coefficient
    rep = rep or reproduce()
    ia = [ALTS.index(f) for f in COMMON]
    rows, agree = [], []
    for member in FAMILY_MEMBERS["PBFT"]:
        mp = {f: (member if f == "PBFT" else FAMILY_MEMBERS[f][0]) for f in COMMON}
        idx = [keys.index(mp[f]) for f in COMMON]
        stats = {m: [] for m in ("TOPSIS", "VIKOR", "COPRAS", "EDAS")}
        for sname, w in scenarios.items():
            w = np.asarray(w) / np.sum(w)
            pubT = _order(np.asarray(rep[sname]["rankC_pub"])[ia]); pubV = _order(np.asarray(rep[sname]["rankQ_pub"])[ia])
            rec = rank_all(X15[ia], w, TYPES)          # published matrix restricted to the same four families
            ours = rank_all(Xm[idx], w, types)
            for j, f in enumerate(COMMON):
                rows.append({"pbft_member": member, "scenario": sname, "family": f, "our_key": mp[f],
                             "published_TOPSIS": int(pubT[j]), "published_VIKOR": int(pubV[j]),
                             "recomputed4_TOPSIS": int(rec["TOPSIS"]["rank"][j]), "recomputed4_VIKOR": int(rec["VIKOR"]["rank"][j]),
                             **{f"ours_{m}": int(ours[m]["rank"][j]) for m in stats}})
            for m in stats:
                pub = pubV if m == "VIKOR" else pubT
                b = np.asarray(ours[m]["rank"])
                stats[m].append((spearmanr(pub, b)[0], kendalltau(pub, b)[0], ws_coefficient(pub, b),
                                 int(np.argmin(pub) == np.argmin(b))))
        for m, v in stats.items():
            v = np.asarray(v, float)
            agree.append({"pbft_member": member, "method": m, "published_reference": "VIKOR" if m == "VIKOR" else "TOPSIS",
                          "spearman_mean": round(float(v[:, 0].mean()), 3), "spearman_min": round(float(v[:, 0].min()), 3),
                          "spearman_max": round(float(v[:, 0].max()), 3), "kendall_mean": round(float(v[:, 1].mean()), 3),
                          "WS_mean": round(float(v[:, 2].mean()), 3), "same_first_of_6": int(v[:, 3].sum())})
    return rows, agree
