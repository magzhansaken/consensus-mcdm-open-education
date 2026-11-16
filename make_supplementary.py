#!/usr/bin/env python3
"""Supplementary Materials of the article: Table S1 (assumptions) and Table S2 (agreement between model and anchors).

Table S1 is generated from ASSUMPTIONS.md, Table S2 from the calibration outputs of the last run
(results/calibration_fabric.csv and results/calibrated_params.json). The script writes
  results/table_S2.csv                     agreement set by set (machine-readable)
  supplementary/supplementary.tex          LaTeX source of the Supplementary Materials (Tables S1 and S2)
and, if pdflatex is available, supplementary/supplementary.pdf.
Usage:  python make_supplementary.py [--no-pdf]
"""
import argparse, csv, json, os, re, shutil, subprocess, sys
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(ROOT, "results")
OUT = os.path.join(ROOT, "supplementary")

UNI = {"×": r"$\times$", "≥": r"$\geq$", "≤": r"$\leq$", "–": "--", "—": "---", "≈": r"$\approx$", "→": r"$\rightarrow$",
       "->": r"$\rightarrow$", ">=": r"$\geq$", "<=": r"$\leq$", "ν": r"$\nu$", "κ": r"$\kappa$", "µ": r"$\mu$",
       "Ö": r"\"O", "ö": r"\"o", "é": r"\'e", "ü": r"\"u", "…": r"\ldots{}", "·": r"$\cdot$"}


def tex(s: str) -> str:
    s = s.replace("\\", r"\textbackslash{}")
    s = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", s)
    parts = re.split(r"(`[^`]+`)", s)
    out = []
    for p in parts:
        if p.startswith("`") and p.endswith("`"):
            q = p[1:-1].replace("_", r"\_").replace("%", r"\%")
            out.append(r"\texttt{" + q + "}")
            continue
        for a, b in (("&", r"\&"), ("%", r"\%"), ("#", r"\#"), ("_", r"\_"), ("$", r"\$")):
            p = p.replace(a, b)
        p = re.sub(r"(?<![\w\\])~", r"\\textasciitilde{}", p)
        for a, b in UNI.items():
            p = p.replace(a, b)
        p = re.sub(r"(\d) (\d{3})", r"\1\\,\2", p)   # thin space as thousands separator
        out.append(p)
    return "".join(out)


def table_s1():
    rows = []
    for line in open(os.path.join(ROOT, "ASSUMPTIONS.md"), encoding="utf-8"):
        if not line.startswith("| ") or line.startswith("| ID") or line.startswith("|---"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split(" | ")]
        if len(cells) != 5:
            raise ValueError(f"malformed row in ASSUMPTIONS.md: {line}")
        rows.append(cells)
    return rows


def mape(pairs):
    pairs = [(m, s) for m, s in pairs if m]
    return float(np.mean([abs(s - m) / m for m, s in pairs])) if pairs else None


def table_s2():
    fab = list(csv.DictReader(open(os.path.join(RES, "calibration_fabric.csv"))))
    for r in fab:
        for k in ("tps_meas", "tps_sim", "lat_meas", "lat_sim", "rtt_ms", "k_fault"):
            r[k] = float(r[k])
    cal = json.load(open(os.path.join(RES, "calibrated_params.json")))
    out = []

    def add(platform, setname, role, pts, n, tps, lat, note=""):
        out.append({"platform": platform, "set": setname, "role": role, "points": n,
                    "mape_throughput": tps, "mape_latency": lat, "note": note})

    names = {"raft": "Raft", "smartbft": "SmartBFT"}
    for proto in ("raft", "smartbft"):
        F = [r for r in fab if r["protocol"] == proto]
        st = [r for r in F if r["set"] == "fit" and r["lat_meas"] < 5]
        sat = [r for r in F if r["set"] == "fit" and r["lat_meas"] >= 5]
        P = f"Fabric 3.1.4, {names[proto]}"
        add(P, "batch size × load, stable points", "fit", st, len(st),
            mape([(r["tps_meas"], r["tps_sim"]) for r in st]), mape([(r["lat_meas"], r["lat_sim"]) for r in st]))
        add(P, "batch size × load, saturated points (latency >= 5 s)", "fit", sat, len(sat),
            mape([(r["tps_meas"], r["tps_sim"]) for r in sat]), mape([(r["lat_meas"], r["lat_sim"]) for r in sat]),
            "measured latency depends on the run length")
        fn = [r for r in F if r["set"] == "fit_n"]
        add(P, "number of orderers (4, 7, 10)", "fit (stage 2)", fn, len(fn),
            mape([(r["tps_meas"], r["tps_sim"]) for r in fn]), mape([(r["lat_meas"], r["lat_sim"]) for r in fn]))
        for rtt in sorted({r["rtt_ms"] for r in F if r["set"] == "holdout_rtt"}):
            h = [r for r in F if r["set"] == "holdout_rtt" and r["rtt_ms"] == rtt]
            ratio = np.mean([r["lat_meas"] / r["lat_sim"] for r in h])
            add(P, f"injected round-trip time {rtt:g} ms", "hold-out", h, len(h),
                mape([(r["tps_meas"], r["tps_sim"]) for r in h]), mape([(r["lat_meas"], r["lat_sim"]) for r in h]),
                f"measured / simulated latency {ratio:.1f}")
        for lab, sel in (("crash campaign, no crash (baseline)", lambda r: r["k_fault"] == 0),
                         ("crash campaign, 1-3 crashed orderers", lambda r: r["k_fault"] > 0)):
            hf = [r for r in F if r["set"] == "holdout_fault" and sel(r)]
            add(P, lab, "hold-out", hf, len(hf),
                mape([(r["tps_meas"], r["tps_sim"]) for r in hf]), mape([(r["lat_meas"], r["lat_sim"]) for r in hf]))

    st = [r for r in fab if r["set"] == "fit" and r["lat_meas"] < 5]
    add("Fabric 3.1.4, Raft and SmartBFT pooled", "batch size × load, stable points", "fit", st, len(st),
        mape([(r["tps_meas"], r["tps_sim"]) for r in st]), mape([(r["lat_meas"], r["lat_sim"]) for r in st]),
        "values quoted in Section 4.1")

    bnames = {"qbft": "QBFT", "ibft2": "IBFT 2.0", "clique": "Clique"}
    for k in ("qbft", "ibft2", "clique"):
        b = cal["besu"][k]
        tps = None if k == "clique" else abs(b["tps_sim"] - b["tps_meas"]) / b["tps_meas"]
        note = (f"p95 error {abs(b['p95_sim'] - b['p95_meas']) / b['p95_meas']:.0%}"
                + ("; throughput not fitted (no backlog, capacity is a lower bound)" if k == "clique" else ""))
        add(f"Besu 25.7, {bnames[k]}", "5000-transaction burst", "fit", None, 1, tps,
            abs(b["p50_sim"] - b["p50_meas"]) / b["p50_meas"], "latency = median; " + note)

    C = cal["cometbft"]["rows"]
    for s, role, label in (("fit", "fit", "offered 200 and 800 tx/s"), ("check", "check", "offered 400 tx/s"),
                           ("holdout", "hold-out", "offered 1600 tx/s")):
        R = [r for r in C if r["set"] == s]
        add("CometBFT v1 (175 validators)", label, role, R, len(R),
            mape([(r["tps_meas"], r["tps_sim"]) for r in R]), mape([(r["lat_meas"], r["lat_sim"]) for r in R]),
            "partially saturated run" if s == "check" else "")

    H = cal["hotstuff"]["rows"]
    groups = (("10 nodes, unsaturated (10k-100k tx/s)", "fit", [r for r in H if r["set"] == "fit" and r["lat_meas"] < 5], ""),
              ("10 nodes, saturated (120k tx/s)", "fit", [r for r in H if r["set"] == "fit" and r["lat_meas"] >= 5], "latency not fitted"),
              ("20 and 50 nodes", "hold-out", [r for r in H if r["set"] == "holdout" and r["faults"] == 0], ""),
              ("10 nodes, 1 and 3 crashed", "hold-out", [r for r in H if r["set"] == "holdout" and r["faults"] > 0],
               "repeated leader timeouts not modelled"))
    for label, role, R, note in groups:
        add("2-chain HotStuff", label, role, R, len(R),
            mape([(r["tps_meas"], r["tps_sim"]) for r in R]), mape([(r["lat_meas"], r["lat_sim"]) for r in R]), note)
    return out


def pct(v):
    return "--" if v is None else f"{100 * v:.1f}"


def write_tex(s1, s2):
    os.makedirs(OUT, exist_ok=True)
    L = [r"""\documentclass[a4paper,10pt]{article}
\usepackage[margin=1.8cm,landscape]{geometry}
\usepackage[T1]{fontenc}
\usepackage{lmodern,longtable,booktabs,array}
\usepackage[hidelinks]{hyperref}
\renewcommand{\arraystretch}{1.15}
\setlength{\LTcapwidth}{\textwidth}
\renewcommand{\tablename}{Table}
\renewcommand{\thetable}{S\arabic{table}}
\begin{document}
\begin{center}
{\large\bfseries Supplementary Materials}\\[4pt]
A Measurement-Anchored Multi-Criteria Decision-Support Framework for Selecting a Blockchain Consensus Protocol\\
in IoT-Enabled Open Education\\[4pt]
\small Baikonys, Zhuzbayev, Alkhanova, Adilova, Sabitova, Tashatov and Sarsenbay
\end{center}
\noindent Both tables are generated by \texttt{make\_supplementary.py} of the repository
\url{https://github.com/magzhansaken/consensus-mcdm-open-education} (release v1.1.0): Table~S1 from
\texttt{ASSUMPTIONS.md} and Table~S2 from the calibration outputs \texttt{results/calibration\_fabric.csv} and
\texttt{results/calibrated\_params.json}.

\footnotesize
\begin{longtable}{>{\raggedright\arraybackslash}p{0.9cm} >{\raggedright\arraybackslash}p{3.2cm} >{\raggedright\arraybackslash}p{8.6cm} >{\raggedright\arraybackslash}p{2.6cm} >{\raggedright\arraybackslash}p{7.2cm}}
\caption{Assumptions of the framework. Every number that is not a published measurement comes from one of these rows.
Kind: \emph{assumption} = declared choice with its sensitivity range; \emph{documented} = specification or configuration;
\emph{calibrated} = fitted on the anchors; \emph{method} and \emph{protocol} = settings of the analysis; rows F1, A7b,
W1--W8, W-O and E1--E6 were added or extended in the revision.}\label{tab:S1}\\
\toprule ID & Item & Value & Kind & Basis\\ \midrule \endfirsthead
\toprule ID & Item & Value & Kind & Basis\\ \midrule \endhead
\bottomrule \endfoot
"""]
    for r in s1:
        L.append(" & ".join(tex(c) for c in r) + r"\\" + "\n")
    L.append(r"""\end{longtable}

\begin{longtable}{>{\raggedright\arraybackslash}p{4.4cm} >{\raggedright\arraybackslash}p{5.6cm} >{\raggedright\arraybackslash}p{1.7cm} >{\centering\arraybackslash}p{1.1cm} >{\centering\arraybackslash}p{1.7cm} >{\centering\arraybackslash}p{1.7cm} >{\raggedright\arraybackslash}p{6.0cm}}
\caption{Agreement between model and anchor campaigns: mean absolute percentage error (MAPE, \%) of throughput and
latency by set. Fit = used to estimate the constants; check and hold-out = not used in the fit. Latency is the mean
latency except for Besu (median of the burst). Simulated values are means of three seeds per point.}\label{tab:S2}\\
\toprule Platform and protocol & Set & Role & Points & MAPE throughput & MAPE latency & Note\\ \midrule \endfirsthead
\toprule Platform and protocol & Set & Role & Points & MAPE throughput & MAPE latency & Note\\ \midrule \endhead
\bottomrule \endfoot
""")
    for r in s2:
        L.append(" & ".join([tex(r["platform"]), tex(r["set"]), tex(r["role"]), str(r["points"]),
                             pct(r["mape_throughput"]), pct(r["mape_latency"]), tex(r["note"])]) + r"\\" + "\n")
    L.append(r"""\end{longtable}
\end{document}
""")
    path = os.path.join(OUT, "supplementary.tex")
    open(path, "w", encoding="utf-8").write("".join(L))
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-pdf", action="store_true")
    a = ap.parse_args()
    s1, s2 = table_s1(), table_s2()
    with open(os.path.join(RES, "table_S2.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(s2[0]))
        w.writeheader(); w.writerows(s2)
    path = write_tex(s1, s2)
    print(f"Table S1: {len(s1)} rows; Table S2: {len(s2)} rows -> {path}")
    if not a.no_pdf and shutil.which("pdflatex"):
        for _ in range(2):
            subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "supplementary.tex"], cwd=OUT,
                           check=True, stdout=subprocess.DEVNULL)
        for ext in ("aux", "log", "out"):
            p = os.path.join(OUT, f"supplementary.{ext}")
            if os.path.exists(p):
                os.remove(p)
        print("PDF:", os.path.join(OUT, "supplementary.pdf"))


if __name__ == "__main__":
    sys.exit(main())
