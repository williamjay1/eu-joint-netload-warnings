# Claims, released inputs and executable checks

| Object | Released source | Check/run scope |
| --- | --- | --- |
| Four-model Brier comparisons, original and local Danish triples, q90/q95 and development/reanalysis | Three hourly Parquets, model probabilities and joint event flags | Hourly complete-day losses and 64 model/reference scores recomputed. Original matrix-score tables retained for audit. |
| Daily loss sufficient statistics | Hourly probabilities and event flags | Every daily sum/count recalculated on the full calendar and compared against recorded aggregates. Missing calendar positions are retained. |
| Native and matched review capacity, q90/q95, 0/1/2-day process bridge | Hourly policy risk, outcome stream and issued plan flags | Full-history P0/P1/P2 reallocation exactly matches stored flags; 108 native/matched summaries and process ID streams checked. Uniform retention gives exact expected exposure-standardised coverage. |
| Intact process/onset contributions | Hourly labels, replayed plans and fixed process definition | Six daily onset tables rebuilt from hourly events and scored review plans. Process contributions remain paired; episode boundaries are not reconnected by inference. |
| Four finite post hoc contrast families | Recomputed nine daily tables plus fixed expected inference CSV | All 24 model/policy contrasts, original pointwise intervals and within-family Holm values recomputed. This does not adjust every supplementary comparison. |
| Figures 1–2 | Fixed workflow and UTC schedule in source | Conceptual drawing; no empirical deployment or dispatch claim. |
| Figures 3–5 | Selected compact original/local policy, yearly and contrast CSVs | Exact plotted rows checked against compact historical numerical references; no model fitting. |
| Figure 6 | Cached 24-probability forecast-only quiet-day example | Same previously selected day, scores and planned slots. The package draws the case and does not represent it as independent validation. |
| Figure S1 | Original triple's released q90 issued-risk/plans/event stream | Same two post hoc directional cases, annotations and planned slots reconstructed. The outcomes do not choose warning hours. |
| Core Tables 1–5 | Released JSON of table cells and captions | All six CSV blocks (two panels in Table 4) regenerated from exact source strings, with measured typography and no value recalculation. |
| q95/block/label/macro-weather sensitivity summary | `data/aggregates/robustness_digest.csv` and selected fixed tables | Recorded sensitivity summaries supplied. A fresh full simulation/refit is not claimed by the release check. |
| Full forecast fitting from provider archives | Acquisition, preparation, threshold/marginal/dependence and transfer source plus recorded configs | Requires reacquiring provider archives, rebuilding intermediate provenance/timing/feature products and recording code/path adaptation. Not executed by the portable command. |

The manuscript itself is not included in this code/data repository. Data source boundaries and terms are documented in `DATA_SOURCES.md` and `DATA_LICENSES.md`. `docs/verification.json` records the actual tested reconstruction. Source-provider archival first publication and live operational availability are not verified by these checks.
