# Cross-border joint high net load warnings

Research code and derived evaluation data accompanying *Joint high net load warnings and review allocation under marginal uncertainty: A cross-border European case study*.

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

`CITATION.cff` describes the software; `.zenodo.json` prepares author-controlled Zenodo metadata. No article DOI or repository DOI is asserted in these files. Version 1.0.0 is prepared; the author will connect the repository to Zenodo and create the tagged release. Cite the actual released version and its Zenodo DOI when that record exists.

