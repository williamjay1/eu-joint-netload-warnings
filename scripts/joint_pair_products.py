"""Prespecified two-zone physical exceedances from unchanged fitted copulas.

Pair order is DE_LU--FR, DE_LU--BE, FR--BE. Threshold CDFs are used
unchanged, including exact zero/one boundaries. This module never fits a
model, inspects labels or changes the existing three-zone event integrals.
"""
from __future__ import annotations

import numpy as np
from dependence import (IntegrationSettings, _elliptical_cdf, _matrix,
                        _ranks_for_forecasts, _unit_matrix)

PAIRS = ((0, 1), (0, 2), (1, 2))
PAIR_NAMES = ("DE_LU_FR", "DE_LU_BE", "FR_BE")


def _result(probabilities, errors=None, points=None, orders=None, corrected=None,
            method="analytic_independent_pairs"):
    values = np.asarray(probabilities, dtype=float)
    if values.ndim != 2 or values.shape[1] != 3 or not np.isfinite(values).all():
        raise ValueError("Finite n-by-three prespecified pair probabilities required")
    n = len(values)
    return {"pairs": values, "pair_names": PAIR_NAMES,
            "numerical_error": np.zeros(n) if errors is None else errors,
            "integration_maxpts": np.zeros(n, dtype=int) if points is None else points,
            "integration_order": np.zeros(n, dtype=int) if orders is None else orders,
            "roundoff_corrected": np.zeros(n, dtype=bool) if corrected is None else corrected,
            "numerical_method": method}


def independent_pair_probabilities(V):
    v = _unit_matrix(V)
    return _result(np.column_stack([(1-v[:, i])*(1-v[:, j]) for i, j in PAIRS]))


def checkerboard_pair_probabilities(V, template=None, ranks=None, seed=2701):
    v, r = _ranks_for_forecasts(V, template, ranks, seed)
    # The within-cell exceedance fraction has the same shared marginals as
    # checkerboard_event_probabilities. This clipping locates a rank cell;
    # it does not clip or perturb the requested threshold CDFs.
    a = np.clip(r-r.shape[1]*v[:, None, :], 0, 1)
    return _result(np.column_stack([(a[:, :, i]*a[:, :, j]).mean(axis=1) for i, j in PAIRS]),
                   method="analytic_checkerboard_pairs")


def analog_pair_probabilities(model, V, X, issue_times):
    v, x = _unit_matrix(V), _matrix(X, "X")
    times = np.asarray(issue_times, dtype="datetime64[ns]")
    if len(x) != len(v) or times.shape != (len(v),) or np.any(np.isnat(times)):
        raise ValueError("V, X and issue_times must align")
    if np.any(times <= model.latest_available_time_):
        raise ValueError("Analog history must precede every forecast issue")
    _, indices = model.tree_.query((x-model.mean_)/model.scale_, k=model.n_analogs)
    return checkerboard_pair_probabilities(v, template=model.U_[indices], seed=model.seed)


def elliptical_pair_probabilities(V, R, df=np.inf, settings=None):
    v = _unit_matrix(V)
    settings = settings or IntegrationSettings()
    if not (df == np.inf or np.isfinite(df) and df > 0):
        raise ValueError("Positive finite df or positive infinity required")
    r = np.asarray(R, float)
    if r.shape == (3, 3):
        r = np.broadcast_to(r, (len(v), 3, 3))
    if r.shape != (len(v), 3, 3) or not np.isfinite(r).all():
        raise ValueError("R must have shape (3,3) or (n,3,3)")
    if (not np.allclose(r, r.transpose(0, 2, 1), atol=1e-12, rtol=0) or
            not np.allclose(np.diagonal(r, axis1=1, axis2=2), 1, atol=1e-12, rtol=0) or
            np.any(np.linalg.eigvalsh(r) <= 0)):
        raise ValueError("R must be a symmetric positive definite correlation matrix")
    pair_cdf = np.empty((len(v), 3)); errors = np.zeros(len(v))
    points = np.zeros(len(v), dtype=int); orders = np.zeros(len(v), dtype=int)
    interior = np.all((v > 0)&(v < 1), axis=1)
    fast_mask = interior if settings.backend == "elliptical_path" else np.zeros(len(v), bool)
    if fast_mask.any():
        from elliptical_event_fast import fast_four_cdfs
        positions = np.flatnonzero(fast_mask)
        # Reuse the validated four-CDF numerical API, in bounded batches.
        # Its first three columns are C_12, C_13, C_23 in PAIRS order.
        for start in range(0, len(positions), 512):
            selected = positions[start:start+512]
            four, error, order = fast_four_cdfs(v[selected], r[selected], df,
                                               tolerance=settings.tolerance/10)
            pair_cdf[selected] = four[:, :3]
            errors[selected], orders[selected] = error, order
    for row in np.flatnonzero(~fast_mask):
        budget = settings.initial_maxpts
        base_seed = settings.seed+int(row)*1000003
        while True:
            estimates = []
            for replicate in (0, 49979687):
                estimates.append(np.array([_elliptical_cdf(v[row, list(pair)],
                    r[row][np.ix_(pair, pair)], df, budget,
                    base_seed+104729*k+replicate, settings) for k, pair in enumerate(PAIRS)]))
            first, second = estimates
            average = (first+second)/2
            discrepancy = float(np.max(np.abs(first-second)))
            low = np.array([max(0., v[row, i]+v[row, j]-1) for i, j in PAIRS])
            high = np.array([min(v[row, i], v[row, j]) for i, j in PAIRS])
            legal = np.all((average >= low-settings.tolerance)&(average <= high+settings.tolerance))
            if discrepancy <= settings.tolerance and legal:
                break
            if budget >= settings.max_maxpts:
                raise RuntimeError(f"Pair CDF unresolved at row {row}: discrepancy={discrepancy}, maxpts={budget}")
            budget = min(budget*4, settings.max_maxpts)
        pair_cdf[row], errors[row], points[row] = average, discrepancy, budget
    probability = np.column_stack([1-v[:, i]-v[:, j]+pair_cdf[:, k] for k, (i, j) in enumerate(PAIRS)])
    lower = np.column_stack([np.maximum(0., 1-v[:, i]-v[:, j]) for i, j in PAIRS])
    upper = np.column_stack([np.minimum(1-v[:, i], 1-v[:, j]) for i, j in PAIRS])
    correction = np.maximum(lower-probability, probability-upper)
    if np.any(correction > settings.tolerance):
        raise RuntimeError("Pair probability violates Frechet bounds beyond integration tolerance")
    corrected = np.any(correction > 0, axis=1)
    probability = np.minimum(np.maximum(probability, lower), upper)
    method = "elliptical_correlation_path_pairs" if fast_mask.all() else (
        "elliptical_pairs_with_exact_boundary_reduction" if fast_mask.any() else "seeded_replicated_pair_CDF")
    return _result(probability, errors, points, orders, corrected, method)


def fitted_model_pair_probabilities(name, model, V, X=None, issue_times=None, settings=None):
    if name == "C1_independent":
        return independent_pair_probabilities(V)
    if name == "C8_analog_checkerboard":
        return analog_pair_probabilities(model, V, X, issue_times)
    return elliptical_pair_probabilities(V, model.correlation_matrices(X, len(V)), model.df_, settings)


def pair_numerical_metadata(prediction):
    return {"integration_discrepancy_max": float(np.max(prediction["numerical_error"])),
            "integration_maxpts_max": int(np.max(prediction["integration_maxpts"])),
            "integration_order_max": int(np.max(prediction["integration_order"])),
            "numerical_method": prediction["numerical_method"],
            "numerical_roundoff_correction_count": int(prediction["roundoff_corrected"].sum())}
