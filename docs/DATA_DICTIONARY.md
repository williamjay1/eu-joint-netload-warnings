# Released derived evaluation tables

Three tables, `data/hourly/DE_LU_FR_BE.parquet`, `DK1.parquet` and `DK2.parquet`, each contain 35,064 UTC target hours from 2022-01-01 through 2025-12-31. DK1 and DK2 name the replacement third area; German–Luxembourg and French components remain in both triples. These are forecast/evaluation derivatives, not raw electricity or meteorological observations.

| Field | Meaning |
| --- | --- |
| `target_time` (Parquet index) | Unique increasing UTC interval-start time. A day contains 24 UTC targets. |
| `issue_time` | Declared common issue for that target day, 12 UTC on the preceding day. It describes the replay schedule, not recovered source arrival. |
| `period` | Recorded forecast period; official evaluation groups are development (2022–2023) and reanalysis (2024–2025). |
| `y_q90`, `y_q95` | Binary joint event: at least two areas exceed their provider-specific seasonal level. Missing means a label or threshold cannot be established. These are not individual-provider MW series. |
| `M0_q*` | Original marginal stream, Gaussian dependence. |
| `M1_q*` | Additional conditional marginal map, same Gaussian correlation matrix. |
| `M2_q*` | Original marginal stream, selected shared-matrix t dependence. |
| `M3_q*` | Additional conditional marginal map, selected shared-matrix t dependence. |
| `Independence_q*` | Fitted independent-marginal reference event probability. |
| `Direct_logistic_q*` | Earlier direct logistic reference in the original triple; not fabricated for Danish triples. |
| `policy_p_q*` | Original selected Gaussian forecast used to allocate reviews, after requiring forecast-complete days. These are the exact probabilities supplied to the review rule, including NaNs. |
| `policy_event_q*` | Label stream restricted after allocation to complete scoring days. Unknown labels do not clear a forecast or cooldown state. |
| `P0_q*`, `P1_q*`, `P2_q*` | Boolean planned selections formed from issued probabilities across the complete stored history. P0 top two; P1 gate 0.25; P2 gate 0.19 plus rise priority and six-hour cooldown. |
| `event_process_id_q*_gap0/1/2` | Zero-based realised process ID on event hours only, for 2024–2025 scoring; missing otherwise. Fully observed inactive-day bridges are 0, 1 or 2 respectively. IDs are post-outcome annotations and never rule inputs. |

Probability columns are numerical values in [0,1] or NaN. Events are 0, 1 or NaN. Float values are exported without rounding; Parquet readback is checked exactly. Model comparison uses common complete days defined by all four probabilities and the event. Policy allocation uses all forecast-complete days, while later scoring excludes days with unavailable outcomes. Do not replace missing values with zero.

`data/aggregates/daily_loss_sufficient_statistics_*` contain full elapsed-calendar day positions, loss sums, scored-hour and event-hour counts, with empty positions retained. `daily_onset_sufficient_statistics_*_q*` contain intact episode contributions assigned to their onset day: process count, each policy's expected early coverage sum and paired differences. These support the original calendar-block inference. Released hourly tables allow all these sufficient statistics to be independently recalculated.

`data/aggregates/finite_family_inference.csv` is the fixed numerical reference for four post hoc six-contrast families. `data/reference/*_matrix_scores.csv` and selected result CSVs are original compact references. `data/plot_inputs/core_tables.json` holds every original table cell and caption; it contains no full manuscript.

See `DATA_LICENSES.md` for terms and required source attribution. Availability lags and source accounting boundaries are in `DATA_SOURCES.md`.
