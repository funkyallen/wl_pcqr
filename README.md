# WL-PCQR Well-Log Prediction Intervals

This repository contains the reproducible code, data-placement instructions, split manifest, and aggregate result tables for proximity-calibrated prediction intervals in well-log reservoir-property estimation.

Repository URL: `https://github.com/funkyallen/wl_pcqr`

Final method:

```text
WL-PCQR: Well-Log Proximity-Calibrated Conformalized Quantile Regression
```

The paper studies prediction intervals for `PHIF`, `SW`, and `VSH` under cross-well transfer, target-well calibration, within-well depth-block evaluation, and key-log missingness stress. It also includes a porosity-only external validation on the Mendeley Gulf of Mexico/Dutch North Sea dataset. The current WL-PCQR mainline uses a fixed uniform-to-proximity CQR calibration path selected by calibration-set Winkler score under a coverage constraint. The focus is a compact method story, not a model-component stack.

## Repository Layout

- `data/README.md`: public-data sources, expected local paths, and file hashes. Raw third-party datasets are not redistributed in this clean repository.
- `data/raw/`: local-only location for SPWLA PDDA 2021 train/test CSV files after download.
- `data/external/mendeley_gom_dutch_north_sea/`: local-only location for the external Mendeley Gulf of Mexico/Dutch North Sea porosity validation data after download.
- `data/splits/lowo_manifest.csv`: leave-one-well-out protocol manifest.
- `scripts/`: reproducible dataset audit, WL-PCQR experiment, summary, plotting, and runtime scripts.
- `analysis/`: compact paper-facing aggregate result tables. Per-depth prediction dumps and historical trial directories are excluded.

## Main Reproduction Commands

Before running experiments, download the public datasets listed in `data/README.md` and place them in the expected local paths.

Dataset audit:

```powershell
python scripts/inspect_dataset.py
python scripts/make_dataset_audit_figures.py
python scripts/make_split_manifest.py
```

Paper-facing expanded baseline suite:

```powershell
python scripts/run_baseline_comparison_suite.py --max-per-well 160 --blocks 5 --seeds 42 7 123
```

External Mendeley porosity validation:

```powershell
python scripts/run_mendeley_external_validation.py --max-per-well 160 --blocks 5
```

Runtime comparison:

```powershell
python scripts/benchmark_runtime_methods.py --target PHIF --seed 42 --max-per-well 160 --methods wl_pcqr global_rf_cqr extratrees_pcqr ngboost_cqr_signed hgb_cqr_signed lightgbm_cqr_signed catboost_cqr_signed
```

Hard-well depth-track figure:

```powershell
python scripts/plot_wlpcqr_hard_well_depth_track.py
```

Balanced sampling sensitivity:

```powershell
python scripts/run_maxperwell_sensitivity.py --max-per-well 320
python scripts/summarize_maxperwell_sensitivity.py
```

Paper draft:

```powershell
cd paper
pdflatex -disable-installer -interaction=nonstopmode -halt-on-error -output-directory=build main.tex
bibtex build\main
pdflatex -disable-installer -interaction=nonstopmode -halt-on-error -output-directory=build main.tex
pdflatex -disable-installer -interaction=nonstopmode -halt-on-error -output-directory=build main.tex
```

## Paper-Facing Results

The most important result files are:

- `analysis/baseline_comparison/baseline_comparison_summary.csv`
- `analysis/baseline_comparison/baseline_comparison_raw.csv`
- `analysis/final_results/missing_stress_delta.csv`
- `analysis/final_results/calibration_sensitivity_mean.csv`
- `analysis/runtime_benchmark/runtime_results.csv`
- `docs/FINAL_RESULTS.md`
- `docs/RUNTIME_BENCHMARK.md`

## Data and Code Availability

The primary SPWLA PDDA 2021 data are publicly available from the SPWLA PDDA SIG `Machine-Learning-Competition-2021` repository. The external porosity sanity-check dataset is available from Mendeley Data under DOI `10.17632/sdv629nbjr.1`. Raw third-party data files are intentionally not tracked here; this repository keeps scripts and aggregate result tables needed to audit and reproduce the reported analyses after users download the public datasets. Manuscript source, local documentation drafts, generated PDF builds, and final figure export folders are kept outside the public code repository.

## Method Positioning

WL-PCQR should be described as a well-log-specific CQR calibration-path protocol. It is not a new conformal prediction theory, not a new random forest algorithm, and not a deep architecture contribution. The statistical caveat is central: target-well calibration supports marginal target-well coverage only when the labeled calibration depths are representative of the deployment depths. In contiguous within-well depth-block diagnostics, WL-PCQR intentionally reduces to its uniform CQR path level.
