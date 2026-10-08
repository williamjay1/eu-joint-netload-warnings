# Supporting tables

Each CSV identifies its panel and source file. Different panels use different fields; blank union-schema cells mean not applicable. Tables preserve complete source precision. Prior fixtures and analytic thinning are explicitly labeled previously executed.

## Table S1. Marginal calibration and conditional positions of seasonal thresholds

[Table_S1.csv](Table_S1.csv) — 264 rows.

- A_threshold_residuals: 168 rows.
- B_CDF_quantiles: 24 rows.
- C_seasonal_CDF_positions: 36 rows.
- D_2022_residual_scale: 36 rows.

## Table S2. Four-cell probability scores, paired contrasts and reliability support

[Table_S2.csv](Table_S2.csv) — 576 rows.

- A_scores: 48 rows.
- B_paired_contrasts: 48 rows.
- C_reliability_bins: 480 rows.

## Table S3. Deterministic convergence and independent randomized CDF checks

[Table_S3.csv](Table_S3.csv) — 108 rows.

- A_full_q90_convergence: 4 rows.
- B_independent_reference: 96 rows.
- C_plan_invariance: 8 rows.

## Table S4. Complete native policy evaluation and cost comparisons

[Table_S4.csv](Table_S4.csv) — 58,143 rows.

- A_native_counts: 312 rows.
- B_native_paired_contrasts: 432 rows.
- C_selection_categories: 312 rows.
- D_development_gates: 24 rows.
- E_cost_curves: 41,184 rows.
- F_paired_cost_contrasts: 15,840 rows.
- G_lost_early_processes: 3 rows.
- H_previous_native_and_thinning_gap_sensitivity: 36 rows.

## Table S5. Same-gate ranking and cooldown factorial ablation

[Table_S5.csv](Table_S5.csv) — 408 rows.

- A_native_ablation: 192 rows.
- B_paired_ablation: 216 rows.

## Table S6. Learned targets, mature chronological development and primary contrasts

[Table_S6.csv](Table_S6.csv) — 2,024 rows.

- A_target_support: 12 rows.
- B_regularization_validation: 42 rows.
- C_quarterly_refits: 112 rows.
- D_native_learned_policy_counts: 123 rows.
- E_target_specific_scores: 34 rows.
- F_primary_paired_contrasts: 63 rows.
- G_primary_paired_cost_contrasts: 1,638 rows.

## Table S7. Annual and quarterly workload, target-specific scores and calibration

[Table_S7.csv](Table_S7.csv) — 6,006 rows.

- A_native_year_quarter: 3,120 rows.
- B_2023_native_development: 1,872 rows.
- C_annual_learned_policies: 246 rows.
- D_quarterly_scores: 272 rows.
- E_quarterly_paired_scores: 224 rows.
- F_annual_calibration_bins: 272 rows.

## Table S8. Continuous-box sufficient certificates and finite-path summaries

[Table_S8.csv](Table_S8.csv) — 8 rows.

- A_continuous_box_certificates: 4 rows.
- B_finite_path_summary: 4 rows.

## Table S9. Complete constant-sign paths and shared-matrix copula list changes

[Table_S9.csv](Table_S9.csv) — 36 rows.

- A_eight_corner_paths: 32 rows.
- B_copula_list_changes: 4 rows.

## Table S10. Known-mechanism simulations, counterexample and Monte Carlo precision

[Table_S10.csv](Table_S10.csv) — 827 rows.

- narrow_means_MCSE: 72 rows.
- narrow_near_minus_far_paired: 198 rows.
- narrow_temporal_fixture: 2 rows.
- narrow_phase_fixture: 2 rows.
- wide_means_MCSE: 72 rows.
- wide_near_minus_far_paired: 198 rows.
- wide_temporal_fixture: 2 rows.
- wide_phase_fixture: 2 rows.
- previous_formal_Gaussian_t_metrics: 140 rows.
- previous_formal_Gaussian_t_paired: 23 rows.
- previous_formal_selected_df: 12 rows.
- previous_known_state_high: 52 rows.
- previous_known_state_lower: 52 rows.

## Table S11. Danish pivotal endpoints and assumed-availability price benchmarks

[Table_S11.csv](Table_S11.csv) — 688 rows.

- A_original_Danish_pivotal_endpoints: 416 rows.
- B_learned_pivotal_endpoints: 98 rows.
- C_price_and_control_native_counts: 102 rows.
- D_price_and_control_scores: 28 rows.
- E_paired_Danish_contrasts: 42 rows.
- F_price_exposure: 2 rows.

## Table S12. Information contracts and reproduction inventory

[Table_S12.csv](Table_S12.csv) — 63 rows.

- A_input_inventory: 9 rows.
- B_prediction_target_list_inventory: 39 rows.
- C_information_and_inference_contract: 15 rows.

## Interpretation

N/slots is review-window count; H/hits is selected event hours; C is once-per-process six-hour early coverage. All contrasts are candidate minus reference. Probability scores compare methods only within the same target. Costs are standardized reward ratios, not euros or hours of staff labor. Bootstrap intervals hold fitted probabilities and lists fixed. Sensitivity boxes are assumed scenarios, not statistical confidence regions.

Table S4 preserves complete original cost grids and previous gap0/gap1/gap2 uniform-thinning standardizations. Table S6 preserves primary learned-model paired cost grids; the additional 1,000-draw learned full-grid outputs remain in results/early_price/cost_curves.csv and paired_cost_contrasts.csv. Those secondary cooldown rules begin with an empty state at the start of2024 and are not compared as warm-start equivalents of original rules.

Table S10 contains new narrow/wide400-replication mechanisms, the previous500 Gaussian/t fixture and previous high/lower known-state independent-score screens. Monte Carlo uncertainty is labeled separately from empirical block-bootstrap uncertainty.
