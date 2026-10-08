# Article build specification — ASMBI revision

Target: Applied Stochastic Models in Business and Industry, Wiley, Research Article, applied stochastic decision evaluation. Free Format editable manuscript, abstract <=250 words, <=5 keywords, short title <=70 characters. Expected 5,000–6,000 body words plus references; simple DOCX without decorative design. An accompanying self-contained Appendix S1 and reproducible analysis package are provided. This specification governs the manuscript; published ASCE artifacts remain unchanged.

## Evidence and claim spine
Numerical cell ledger: audit/RESULT_LEDGER.csv generated from result tables, source/row/field retained. Inputs audit/input_map.json; claim audit below. Completed policy, forecast, numerical, perturbation and mechanistic evidence is conditional on fixed archived forecast streams. Early/price analysis is an additional prespecified-in-revision computation and will enter only after executable outputs pass verification. No hypothetical result may enter prose.

Main finding: Allowing zero reviews removes much non-event work, but the choice among static gates, forecast-only adaptation and temporal ranking depends on process value and the cost of review. The sharp contrast is native workload versus early coverage, not Gaussian-versus-t model superiority.
Method contribution: A transparent decision evaluation links marginal CDF perturbations to gate crossings, capacity-dependent ranks and replayed cooldown state; it supplies sufficient conditions for ordinary selection stability and clearly weaker finite-scenario diagnostics for stateful rules. The bound and elementary cost gate are established probability/decision arguments, not new theorems.
Boundaries: No estimated staff savings, adequacy/outage label, optimal temporal policy, fresh 2024–25 holdout, certified hourly marginal error set, wholly independent Denmark replication or current-snapshot historical vintage.

## Questions and architecture
1. Where does selective review save resources and what detection is lost? Section3.1, Table1 and Figure1; removed-slot ledger and lost-process details.
2. Does a daily cap plus a mean workload target produce the same policy as a process-cost contract? Sections2.1/2.3 and3.2, Table1, Figure2; yearly drift, synchronized cost curves and ablations.
3. Can forecast calibration or copula complexity alter actual selections? Sections2.2/2.4 and3.3, Table2, Figure3; four-model factorial, conditional threshold position, gate/rank margins and numerical integration.
4. Which mechanisms and transfer boundaries persist? Sections2.6 and3.4–3.5, Figure4, Table3; ADEMP simulations, direct early model, price controls and Danish-specific processes.

Word plan: Introduction650; Methods1,700; Results1,800; Discussion700; Conclusion130; declarations250. References use consistent author-year style with original verified references plus decision and simulation foundations. Methods/Results are drafted before Introduction/Abstract. No long engineering adequacy exposition or new copula leaderboard. Each subsection begins with the result or evaluation question and develops connected prose.

Main figure plan (first referenced in relevant Results paragraph):
* Figure1: workload versus native early coverage and selected-slot decomposition. Both endpoints retain their units; no connecting line or fitted frontier implies an optimal prospective policy. Early-model points can enter after computation.
* Figure2: quarterly workload drift and complete process cost curves with no-review baseline; uncertainty from same synchronized draws. State whether bands are pointwise or simultaneous over a fixed curve grid, never globally across policies.
* Figure3: perturbation criticality, selected-hour certification and capacity-sensitive ranking changes. Ordinary continuous-box sufficient certificates distinguished from nine constant-shift scenarios including baseline; stateful rules are scenarios only.
* Figure4: known-probability simulation gate crossings and final selection changes under equal marginal perturbation magnitude, persistence and test-distribution shift; Monte Carlo uncertainty explicit. Retain both narrow-boundary and wide-boundary exploratory fixtures and document revisions.
Each figure uses 7pt or larger typography at <=183mm, TrueType PDF, editable SVG,1200dpi PNG plus internal150dpi preview; every exported figure is viewed and audited. Legends sit clear of data, text has no leader lines through other text.

Tables: Table1 primary native policies N/H/C and slots/day; Table2 four probability configurations plus native gate choices, score contrast with bootstrap intervals; Table3 original and Denmark (including pivotal subset) with targeted early and price comparisons if informative. Long quarterly tables, gap0/2 andq95 sensitivity, full cost grids, numerical references and parameter logs go to Appendix S1.

Equations: (1) hourly event and issued probability; (2) additive hour contract and gate; (3) process early-hour set and C; (4) process normalized utility; (5) at-least-two copula identity; (6) risk-rise ranking; (7) fixed-copula coupling bound; (8) gate/rank sufficient margins and certificate; (9) expected coverage under uniform retention; (10) known shared-shock simulation probability. Definitions accompany each; computations map respectively shared_eval,policy_analysis,legacy_copula,stability_analysis andmechanism_simulations.

## Claim strength audit
* Strong descriptive: Top2->gate subset832=19events+813non-events; early35->32; one missed process129eventhours. Exact fixed-stream identities.
* Moderate conditional: rolling lowers workload to358 but early24; CI of earlydifferenceversusgate excludes0 under fixed-policy calendarbootstrap. No general adaptation superiority.
* Descriptive/Pareto: originalpriority572/34 point-dominates gate618/32 for processbenefit, but yearly reversal and contrastinterval forbid universal superiority.
* Mechanistic: fixedcopula perturbations can change lists and require fullstate replay; empiricalhour-error radius unverified. Absolute interval coverage not claimed.
* Precise numerical: strictintegration convergence farbelow observedtinyBriercontrast; no selectionchanges across tighter numerical tolerance. This rules out testednumericalsetting as explanation, not statisticalequivalence or familyoptimality.
* Exploratory simulation: known DGP distinguishes average marginal error from decision consequence; outcomes/MCSE reported across scenarios; no proof of realoperatorvalue.
* Transfer: DanishcombinationsshareDE/FR; priceavailableassumption andlocalauction-day restriction; no livedeploymentclaim.

Sentence-level literature guidance: Introcompetitivejointforecast:Gioia2025,Hirsch2024; forecastvalue:Richardson2000,Ehm2016; selectioncontract:Franc2023; processfront/repeatvalue:Tatbul2018. Probabilitycalibration:Gneiting2007a/b,Ziegel2014,Dimitriadis2021; tproperties:Demarta2005; simulations:Morris2019. Data/timingcitations point to verifiedprovider official docs retainedfromoldrelease; Supplement supplies detailed originalforecastsettings. References and AIuse reflect actual workflow; no unsupportedcitationcountquota.

Delivery gate: all main numeric claims read directly from results; user attachment72requirements mapped to actual disposition; definition/code consistency verified; independent agent reads prose and outputs; DOCX rendered and inspected; figurebounds/font/overlap audited. No new public DOI is invented. Code bundle contains new scripts, input manifest and permitted derivative evaluation tables, and distinguishes evaluation reproduction from original raw-data refitting.
