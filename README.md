# consensus-mcdm-open-education

Code and data for a measurement-anchored multi-criteria selection of a blockchain consensus protocol
for an open-education platform: a calibrated discrete-event model of seven permissioned consensus
protocols, objective weighting, four ranking methods and a robustness analysis.

## Repository layout

| Path | Content |
|---|---|
| `consensus_mcdm/` | Python package: workload generator, protocol definitions, simulation model, calibration, criteria, weights, ranking methods, SMAA-2, robustness tests, diagnostics, reproduction of published scenario rankings, report and figure helpers |
| `data/` | Published anchor measurements used for calibration and validation, with sources and licences (`PROVENANCE.md`, `anchors.csv`) |
| `ASSUMPTIONS.md` | Every value that is not a published measurement, with its kind and basis (rows A, E, W, L, R, B, M, O, D) |
| `run_all.py` | Single entry point that runs the whole pipeline and writes `results/` and `figures/` |
| `make_fig_framework.py` | Schematic of the pipeline |
| `tests/` | Unit tests (`pytest`) |
| `results/` | Outputs of the last full run (CSV, JSON, Markdown digest `SUMMARY.md`) |
| `figures/` | Figures of the last full run (PNG and PDF) |

## Requirements

Python 3.11 or newer and the packages in `requirements.txt` (NumPy, SciPy, Matplotlib, pytest).
Everything runs on a standard CPU; no GPU is needed.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Running

```bash
python -m pytest -q tests                          # 21 unit tests
python run_all.py --seeds 30 --skip-calibration    # reuse results/calibrated_params.json (about 2 min on 2 cores)
python run_all.py --seeds 30                       # full run including calibration (about 15 min on 2 cores)
```

Main outputs:

| File | Content |
|---|---|
| `results/calibration_summary.json`, `results/calibrated_params.json` | Fitted constants and fit/hold-out errors per anchor set |
| `results/decision_matrix.csv`, `results/decision_matrix_samples.npy` | Mean decision matrix with 95% confidence half-widths; per-seed matrices |
| `results/weights.csv` | Entropy, CRITIC, combined and scenario weight vectors |
| `results/rankings_base.csv`, `results/rankings_all_scenarios.csv`, `results/rankings_per_seed.csv`, `results/method_agreement.csv` | Rankings by TOPSIS, VIKOR, COPRAS and EDAS and their agreement |
| `results/smaa_*.csv` | SMAA-2 rank-acceptability indices (uniform and centred weights) |
| `results/leave_one_out.json`, `results/weight_perturbation.json`, `results/single_weight_sweep_topsis.csv` | Rank-reversal, perturbation and single-weight sweep tests |
| `results/ablations.json`, `results/decision_matrix_<ablation>.csv` | Deployment ablations |
| `results/load_sweep.csv`, `results/fault_timeline.csv` | Load response and fault timeline |
| `results/iot45_reproduction.json`, `results/iot45_transfer.csv`, `results/iot45_agreement.csv` | Reproduction of published scenario rankings and transfer of their weights |
| `results/SUMMARY.md` | Human-readable digest of a run |

## Data and licences

The anchor measurements are taken from published benchmark campaigns; each file and row lists its source
and licence in `data/PROVENANCE.md` and `data/anchors.csv`. The open-university workload is synthetic and is
generated from the declared parameters W1–W7 in `ASSUMPTIONS.md`; no learner records are used.

## Citation

The citation of the associated article will be added here after publication.

## Licence

The code is released under the MIT Licence (see `LICENSE`). Third-party data remain under their own licences
as listed in `data/PROVENANCE.md`.
