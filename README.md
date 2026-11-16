# consensus-mcdm-open-education

Code and data for a measurement-anchored multi-criteria selection of a blockchain consensus protocol
for an IoT-enabled open-education platform: a calibrated discrete-event model of seven permissioned consensus
protocols (Raft, SmartBFT, IBFT 2.0, QBFT, Clique, CometBFT, HotStuff), a non-compensatory peak-load screen,
objective weighting, four ranking methods (TOPSIS, VIKOR, COPRAS, EDAS), SMAA-2 and a set of robustness analyses.

Release `v1.1.0` accompanies the revised manuscript (first revision); `v1.0` corresponds to the version submitted
on 28 October 2026.

## Repository layout

| Path | Content |
|---|---|
| `consensus_mcdm/` | Python package (see the module table below) |
| `data/` | Published anchor measurements used for calibration and validation, with sources and licences (`PROVENANCE.md`, `anchors.csv`) |
| `ASSUMPTIONS.md` | Every value that is not a published measurement, with its kind and basis (source of Table S1) |
| `run_all.py` | Single entry point: calibration, main pipeline and all analyses of the revision; writes `results/` and `figures/` |
| `run_feasibility.py`, `run_workload.py`, `run_validation.py`, `run_environment.py`, `run_criteria.py`, `run_priorities.py` | The analyses added in the revision; `run_all.py` calls them in this order, each can be run alone after the main pipeline |
| `scripts/oulad_workload_stats.py` | Recomputes the OULAD workload statistics (`results/oulad_workload_stats.json`) from the public OULAD release |
| `make_fig_framework.py` | Schematic of the pipeline (Figure 1) |
| `make_supplementary.py` | Tables S1 and S2 of the Supplementary Materials, written to `supplementary/` (called by `run_all.py`) |
| `tests/` | Unit tests (`pytest`, 47 tests) |
| `results/` | Outputs of the last full run (CSV, JSON, NPY, Markdown digest `SUMMARY.md`) |
| `figures/` | Figures of the last full run (PNG and PDF) |
| `supplementary/` | Supplementary Materials of the article (LaTeX source and PDF) |

| Module | Content |
|---|---|
| `workload.py` | Synthetic open-university workload W1–W8 and its load levels |
| `protocols.py` | Protocol catalogue: documented structure and calibrated constants |
| `sim.py` | Discrete-event model of block formation, consensus, commit, faults; optional validation-error corrections |
| `calibrate.py` | Fitting on the anchor campaigns and hold-out validation |
| `criteria.py` | The eleven criteria and the decision matrix with 95 % confidence intervals |
| `weights.py`, `mcdm.py`, `smaa.py`, `robustness.py` | Entropy, CRITIC, ranking methods, SMAA-2, rank reversal, perturbations, sweeps |
| `iot45.py` | Exact reproduction of the published scenario rankings of Tashatov et al. (2026) and transfer of their weights |
| `feasibility.py` | Peak-load screen (99 % committed within 30 s), capacities, provisioned configuration, boundary of the recommendation |
| `education.py` | Education-facing indicators E1–E6, OULAD-anchored workload, workload ranges and Latin hypercube |
| `validation.py` | Propagation of the validation errors: model corrections, Monte Carlo over model error, break-even factors |
| `environment.py` | Transferability of the anchor constants across benchmark environments |
| `criteria_checks.py` | Property tables of the documented scores, SMAA-O, score perturbations, alternative rules, leave-one-criterion-out |
| `priorities.py` | Institutional priority orders of the criterion groups (SMAA-2 with ordinal weight constraints, ROC weights) |
| `report.py`, `diagnostics.py` | Tables, figures, load sweep and fault timeline |

## Requirements

Python 3.11 or newer and the packages in `requirements.txt` (NumPy, SciPy, Matplotlib, pytest; pandas only for
`scripts/oulad_workload_stats.py`). Everything runs on a standard CPU; no GPU is needed.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Running

```bash
python -m pytest -q tests                           # 47 unit tests
python run_all.py --seeds 30                        # full run: calibration, main pipeline, all revision analyses
python run_all.py --seeds 30 --skip-calibration     # reuse results/calibrated_params.json
python run_all.py --seeds 30 --skip-revision        # main pipeline only (version 1.0.0 outputs)
python scripts/oulad_workload_stats.py --download   # optional: recompute the OULAD statistics (downloads OULAD from UCI)
```

On two CPU cores a full run takes about 1 h: the calibration about 16.5 min, the main pipeline about 1.2 min and the
revision analyses about 40 min, most of it in the two Latin-hypercube samples (`run_workload.py`, about 17 min;
`run_environment.py`, about 10 min). A from-scratch run of release v1.1.0 reproduces every file in `results/`
bit for bit (the random generators are seeded).

## Where each part of the article is produced

| Article | Script | Output |
|---|---|---|
| Figure 1 | `make_fig_framework.py` | `fig_framework` |
| Tables 4 (coverage) and 5 (documented properties) | `consensus_mcdm/criteria_checks.py` (property tables, asserted against the scores by the tests) | — |
| Table 7 (benchmark environments) | `consensus_mcdm/environment.py` (`ENVIRONMENTS`) | — |
| Table 8 (workload grounding), Figure 2 | `scripts/oulad_workload_stats.py`, `run_all.py` | `oulad_workload_stats.json`, `fig_workload` |
| Section 4.1, Figure 3 | `run_all.py` (calibration) | `calibration_summary.json`, `calibrated_params.json`, `calibration_fabric.csv`, `fig_calibration` |
| Tables S1 and S2 | `make_supplementary.py` | `supplementary/supplementary.pdf`, `table_S2.csv` |
| Sections 4.2–4.6, Tables 9–13, Figures 4–6 | `run_all.py` | `decision_matrix*.csv`, `load_sweep.csv`, `fault_timeline.csv`, `weights.csv`, `rankings_*.csv`, `smaa_*.csv`, `leave_one_out.json`, `weight_perturbation.json`, `wilcoxon_leaders.json`, `ablations.json`, `fig_load_response`, `fig_rank_heatmap`, `fig_robustness` |
| Section 4.7, Table 14, Figure 7 | `run_feasibility.py` | `feasibility.json`, `feasibility_curve_*.csv`, `decision_matrix_provisioned.csv`, `fig_feasibility` |
| Section 4.8, Tables 15–16, Figure 8 | `run_workload.py` | `workload_sensitivity.json`, `fig_workload_sensitivity` |
| Section 4.9, Table 17 | `run_validation.py` | `validation_effect.json` |
| Section 4.10, Table 18 | `run_environment.py` | `environment_sensitivity.json` |
| Section 4.11, Table 19 | `run_criteria.py` | `criteria_checks.json` |
| Section 4.12, Figure 9 | `run_priorities.py` | `priorities.json`, `fig_priorities` |
| Section 4.13, Figure 10 | `run_all.py` | `iot45_*.json`, `iot45_*.csv`, `fig_iot45` (published as `fig_comparison`) |
| Tables 20–21 (boundaries, threats) | summarise the outputs above | — |

Table and figure numbers refer to the revised manuscript; `results/SUMMARY.md` gives a digest of the main pipeline.

## Data and licences

The anchor measurements are taken from published benchmark campaigns; each file and row lists its source
and licence in `data/PROVENANCE.md` and `data/anchors.csv`. The open-university workload is synthetic and is
generated from the declared parameters W1–W8 in `ASSUMPTIONS.md`; no learner records are used. The OULAD statistics
that ground the workload (Kuzilek et al. 2017, CC BY 4.0) are computed by `scripts/oulad_workload_stats.py`; the
OULAD files themselves are not redistributed.

## Citation

See `CITATION.cff`. The bibliographic record of the associated article will be added after publication.

## Licence

The code is released under the MIT Licence (see `LICENSE`). Third-party data remain under their own licences
as listed in `data/PROVENANCE.md`.
