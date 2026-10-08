# Cross-border joint high net load warnings

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23229890.svg)](https://doi.org/10.5281/zenodo.23229890)

Research code, derived evaluation data and externally hosted supplemental materials accompanying *Joint High Net Load Warnings and Review Allocation under Marginal Uncertainty in Europe*.

Author: **Junjie Zhang**, Shanghai International Studies University. [ORCID](https://orcid.org/0009-0004-8821-4018). Contact: junjiezhang2024@shisu.edu.cn.

The study evaluates coincident elevated net load in Germany–Luxembourg, France and Belgium, with local-weather transfers replacing Belgium by DK1 or DK2. The outcome is simultaneous exceedance of source-specific seasonal q90 levels in at least two areas; q95 is secondary. It is a monitoring indicator, not a probability of outage, reserve exhaustion or enacted dispatch.

## Reproduce the evaluated forecasts and decisions

Use Python 3.12. From the repository root:

```bash
python -m pip install -r requirements.txt
python -B reproduce.py
```

This command reads **105,192 hourly rows** in the three released Parquet tables, rather than simply trusting plot summaries. It recalculates complete-day Brier losses for 64 model/period/threshold combinations, replays all three forecast-only policies over the complete 2022–2025 forecast history, reconstructs realised event processes and checks 108 native/matched policy summaries for q90/q95 and three bridging definitions. The allocated plans and process IDs must match exactly. All nine daily sufficient-statistic tables are then rebuilt from these hourly data and used to reconstruct 18 matched coverage estimates and all 24 finite-family bootstrap/Holm contrasts. The program stops if the stored reference results fail its numerical checks.

Results are written to `generated/`. The delivered data and the code were actually executed together: the largest hourly-to-daily/policy difference was 2.22 × 10⁻¹⁶, and the largest finite-family inference difference was 1.11 × 10⁻¹⁶. These checks validate post-forecast evaluation. They do not refit forecasts, recover historical publication vintages or prove calibration.

For all seven figures and five core tables:

```bash
python -m pip install -r requirements-plotting.txt
python -B reproduce.py --figures
```

The renderers use the actual SciencePlots Nature style files at GitHub commit `b9b16959570bd2fbc9ff5118bacc423c3bddd592`, followed by 7 pt Arial and editable-text overrides. **Arial must already be installed**; fonts are not redistributed. Output plates are 180 mm wide, with editable PDF/SVG and native 1200 dpi PNG. The vectors have no intrinsic dpi. Geometry checks stop export on text overlap, line-through-text or clipping. Figure confidence intervals remain pointwise. New drawings require visual inspection even when geometry checks pass.

## Repository contents

- `supplemental_materials/`: the externally hosted supplemental document (Sections S1–S15), Fig. S1 and unchanged CSV table exports. Its [index](supplemental_materials/README.md) maps file names to the actual supplementary labels; the 26 `supplementary_table_*.csv` exports are not Tables S1–S26.
- `data/hourly/`: model-produced probabilities, joint event flags, issued-review plans and event-process IDs. No raw load, generation, weather grids, prices, personal data or credentials.
- `data/aggregates/`: evaluation sufficient statistics and fixed reference inference/sensitivity summaries.
- `data/reference/` and selected `results/`: original compact score, policy and contrast tables used for numerical readback and drawing.
- `data/plot_inputs/`: exact core-table cells; no full manuscript is distributed.
- `analysis/`: executable hourly and aggregate reconstruction.
- `scripts/`: original modelling, acquisition, evaluation and current drawing source, with clone-relative path adaptations documented in `docs/source_adaptations.json`.
- `config/`: recorded scientific settings; `docs/FULL_PIPELINE.md` explains the additional prerequisites of a fresh fit.
- `docs/`: data dictionary, source/measurement boundaries, figure captions and reconstruction map.

## Interpretation and research status

The original 2019–2021 observations were used for fitting, 2022–2023 for development and 2024–2025 for the final reanalysis. The 2024–2025 data had been inspected before this revision; they are not presented as an untouched new test. Danish transfers reuse German/French components and provide partial spatial transfer.

The M0–M3 factorial comparison fixes physical thresholds, samples, weather and correlation matrices while varying the additional conditional marginal map and selected t dependence. F1 denotes that additional map, not independently certified conditional calibration. The selected shared-correlation t candidate has ν = 3. The fixed review rules are P0 compulsory top two target hours, P1 probability gate 0.25 with up to two hours, and P2 gate 0.19 with risk-rise priority and six-hour cooldown. Outcome labels never choose review hours. Missing later labels affect scoring after allocation. The risk history and planned cooldown state are carried across the evaluation boundary.

Uniform retention to the common total number of selected hours gives exact expected exposure-standardised coverage; it is not a deployable controller guaranteeing an annual budget. A review slot is a selected delivery-hour window, not a measured hour of staff labour. Process clustering bridges 0, 1 or 2 fully observed inactive days and does not identify meteorological regimes by itself.

The later Holm analysis is an explicitly post hoc check within four six-contrast families. It does not adjust all supplementary searches. Hourly errors are not treated as independent observations; calendar-block inference and intact episode contributions are retained. Historical revised source observations with chosen lags support **delayed replay**, not proof of what every source made available in real time.

## Licensing and citation

Author-written software is under the MIT license; see `LICENSE`. Author-derived evaluation data have their own terms in `DATA_LICENSES.md`, with source attribution and retained third-party rights. The repository license is not a blanket relicense of provider data or Arial. SciencePlots style files retain their own MIT notice.

`CITATION.cff` describes the software; `.zenodo.json` records author-controlled archive metadata. The existing software/data version v1.0.0 is archived at [Zenodo, DOI 10.5281/zenodo.23100589](https://doi.org/10.5281/zenodo.23100589). That DOI identifies the existing code and derived-data deposit. The supplemental document and CSV exports added to the GitHub repository for the ASCE submission are hosted separately in [`supplemental_materials/`](supplemental_materials/README.md); they are not claimed to be included in that earlier Zenodo deposit. No new DOI or software version is assigned by this hosting update.

## Version 1.1.0: selective review revision

Release 1.1.0 accompanies *From Joint Probability Forecasts to Selective Review: Decision Value and
Calibration Sensitivity in Electricity Monitoring* (Applied Stochastic Models in Business and Industry).

It adds the revision analyses and their outputs:

* early-target against hourly-target model comparison under an identical feature set;
* daily-cap policy against process-cost contract, including synchronized cost curves and ablations;
* marginal-CDF perturbation of ordinary selection rules and full-state replay of stateful rules;
* known-mechanism simulations that separate gate crossings from capacity competition;
* Danish-pivotal endpoints and assumed-availability price controls.

The manuscript, the supporting information, the twelve result tables and the figures are generated from
`scripts/` by `scripts/reproduce_revision.py`; `results/` holds the generated outputs, `audit/` holds the
numerical ledger and the claim maps, and `datasets/` holds the released analysis inputs.

Earlier versions: 1.0.0 (10.5281/zenodo.23100589) and 1.0.1 (10.5281/zenodo.23168553).

## Version 1.1.0 archive

Version 1.1.0 is archived at DOI 10.5281/zenodo.23229890; the concept DOI 10.5281/zenodo.23100588 resolves to the newest version.
Version 1.0.1 remains at 10.5281/zenodo.23168553 and version 1.0.0 at 10.5281/zenodo.23100589.

## Scope of this deposit

This archive is the reproducibility package: analysis code, released analysis inputs, generated result
tables and figures, the numerical ledger and the parameter logs. It contains no journal submission files,
no cover letter and no duplicated table exports. The manuscript sources under `manuscript/` are included
only as documentation of the document build.
