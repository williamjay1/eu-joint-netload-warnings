# ASMBI evaluation revision, version asmbi-evaluation-20261008.1

This package reproduces the **evaluation of archived forecast distributions** and the new decision, calibration-sensitivity, simulation, early-hour classifier and price-control analyses. It does not claim to have rerun the original net-load marginal fits. The original raw-data acquisition, expanding quarterly additive distribution fits and historical calibration workflow belong to the earlier release at https://doi.org/10.5281/zenodo.23168553. This revision has no newly minted DOI.

All numerical inputs are approved derivative tables in `datasets/`. The launcher resolves its root from its own location, so the package can be extracted on another drive or platform. It does not read the author's F-drive archive, download data, or write to raw-data stores. Analysis outputs are written under the extracted package's `results/`; verification writes only `audit/reproducibility_checks.json`. On the author's workstation the extracted working package belongs on D. Do not use the raw/archive directory as the working output location.

## Quick verification

Use Python 3.12 and the versions in `requirements_revision.txt`. From the package root:

```text
python -B scripts/reproduce_revision.py --verify
```

With no flags the launcher also verifies by default. Verification recomputes the original fixed-policy selections from the full 2022–2025 forecast context, checks all stored period-specific counts and forecast scores, validates issue clocks and maturity cutoffs, reconstructs the nine-path perturbation classification and ordinary rank certificates, checks both simulation fixtures and their Monte Carlo SEs, and checks the vector/raster figure artifacts. It does not retrain a model, repeat randomized CDF integrations, generate plots, or recover historical provider revisions. The JSON report supplies the tested versions, input checksums, failures if any, and a formal scientific-file manifest.

## Explicit recomputation

```text
python -B scripts/reproduce_revision.py --analyses
python -B scripts/reproduce_revision.py --task policies --task stability --verify
python -B scripts/reproduce_revision.py --task simulation-narrow --task simulation-wide --task simulation-contrasts
python -B scripts/reproduce_revision.py --task numerical
python -B scripts/reproduce_revision.py --analyses --dry-run
```

`--analyses` runs the two simulations, paired simulation contrasts, fixed-policy evaluation, forecast diagnostics, early/price classifier analysis, marginal perturbation analysis and numerical CDF checks, then verifies. Fixed policies use 2,000 bootstrap draws; the original main early/price stage uses 1,000, with its existing postprocessing stage using 2,000. The options `--policy-reps` and `--early-reps` expose these choices. Changing them changes Monte Carlo precision and means that the resulting intervals are no longer the identical archived calculation. The classifier stage refits its new logistic models; it does not fit the original regional net-load marginal distributions.

| Stage | Input and computation | Principal output |
|---|---|---|
| `policies` | Three triples × four archived probability streams × q90/q95; full forecast-only planning context | `results/policy_analysis/` |
| `forecast-diagnostics` | Archived event probabilities and regional threshold-CDF values | `results/forecast_diagnostics/` |
| `early-price` | New regularized logits, C selected in 2023, past-matured quarterly refits; conditional price-release assumption | `results/early_price/` |
| `stability` | Fixed-R Gaussian probabilities under ±0.01/±0.03 regional CDF shifts; baseline plus eight signed paths | `results/stability/` |
| `numerical` | Full q90 integration tightening plus 96 q90/q95 independent randomized reference cases | `results/numerical/` |
| `simulation-narrow` | 400 paired 120-day replicates per persistence/weather-mixture condition; amplitude .012, marginal shift −.035 | `results/simulations/` |
| `simulation-wide` | Corrected exploratory fixture; amplitude .08, marginal shift +.035 | `results/simulations/wide_boundary/` |
| `simulation-contrasts` | Paired raw replicate contrasts for the narrow fixture | `results/simulations/paired_contrasts.csv` |

The wide fixture's figure contrasts are computed directly from synchronized raw replicates, rather than borrowed from the narrow fixture. The two fixtures differ in both boundary shape and shift direction; they are not an isolated amplitude ablation. Simulation errors are Monte Carlo estimation errors, not uncertainty about European operating outcomes.

Figure regeneration is optional:

```text
python -B scripts/reproduce_revision.py --task figures-policy --task figures-stability
```

It requires the plotting packages and the supplied house style module `scripts/figstyle.py`. The actual analysis interpreter used numpy 2.5.1/pandas 3.0.5; the dedicated plotting interpreter used numpy 2.5.3/pandas 3.0.6, matplotlib 3.11.2 and SciencePlots 2.2.2. A merged environment is not claimed to have been independently tested. PDF/SVG/EPS are vector originals; PNG and LZW TIFF are rendered directly at 1,200 dpi. The 150-dpi versions are inspection/source previews. The delivered Word manuscript embeds the 1,200-dpi PNG originals.

## Inputs and scientific scope

The nine derivative tables are `DE_LU_FR_BE.parquet`, `DK1.parquet`, `DK2.parquet`, `marginal_predictions.parquet`, `DK1_marginal.parquet`, `DK2_marginal.parquet`, `weather_features.parquet`, `dk_prices_dev.parquet` and `dk_prices_test.parquet`. Their schemas, date ranges, byte counts and SHA-256 values are recorded by verification. The bidding-zone forecast streams cover 2022–2025; the marginal/weather archives additionally provide 2021 context. The full original raw weather and energy archives are not silently bundled as newly redistributable data. Refer to the earlier release, provider licensing and the input audit when reconstructing acquisition or original fits.

Daily forecasts are associated with D−1 12:00 UTC issue time for a complete UTC target day. Revised historical provider snapshots do not prove what exact vintage was visible then: this is a delayed historical replay. New classifier labels use interval end plus seven days before issue in the original triple and fifteen days in the Danish checks. C selection uses only 2023 validation labels matured before the first 2024 issue; imputation and standardisation remain inside the classifier pipeline fitted on the matured 2022 training rows for each candidate. Outcome-free resource counting may use all issued 2023 forecast vectors. The verifier checks the candidate-level cutoffs and reconstructs their training/validation counts, and inspects the actual pipeline and fit calls without refitting. Past matured test labels enter later quarterly refits; C remains frozen from the development selection. This is chronological adaptation, not a once-fitted untouched holdout.

Primary fixed policies carry forecast and planned-cooldown history through the full 2022–2025 context before scoring 2024–2025. The separate early/price module's secondary cooldown comparisons currently start at the test boundary; they are not described here as fully warm-started. All main classifier comparisons without cooldown retain their recorded evaluation. Actual provider price-publication vintages are unavailable: local auction-day eligibility is enforced, but same-day price availability remains an explicit assumption.

The ±0.01/±0.03 CDF shifts are transparent sensitivity scales, not validated error radii or hourly confidence intervals. Nine finite paths include the original centre; eight corner paths alone need not reproduce an interior ranking. Only the ordinary, stateless gate receives sufficient continuous-box threshold/rank conditions. Stateful path invariance across the finite scenarios is weaker and does not certify arbitrary interior or time-varying perturbations.

## Formal-package exclusions and audit

Exclude `results/simulations/wide_boundary_initial/` and all Python bytecode/cache directories from the formal reproducibility package. That directory is a discarded debugging run, not an additional empirical result. Verification deliberately excludes it from the manifest and checks that the formal numerical ledger does not quote it. Do not copy the entire workspace blindly when packaging. No script in the analysis launcher modifies or deletes that debugging directory.

`audit/RESULT_LEDGER.csv.gz`, figure source CSVs, numerical CDF tables, mature-label logs and the final verification JSON make numeric claims traceable. Verification conditions on archived forecasts, fitted new classifiers, selected rules, and realized process boundaries. It verifies arithmetic and implementation consistency, not outage adequacy, staffing costs, optimal control, a fresh test sample, real-time deployment, or historical publication assumptions.

## Repository layout for version 1.1.0

* `scripts/` holds the revision pipeline. Entry point: `python scripts/reproduce_revision.py`, then `python scripts/verify_revision.py`.
* `scripts/legacy/` and `scripts/legacy_copula/` hold the modules the revision pipeline imports.
* `scripts_legacy/` and `results_legacy/` hold the earlier pipeline and its outputs, kept for provenance only.
* `results/` holds the generated outputs: the twelve result tables, the figure files (PDF, SVG and 1200 dpi PNG), and the analysis tables.
* `datasets/` holds the released analysis inputs; `audit/` holds the numerical ledger and the claim maps; `manuscript/` holds the manuscript sources.
* `audit/RESULT_LEDGER.csv.gz` is the gzip compressed numerical ledger. It carries more than two million cell records and is 165 MB uncompressed.
