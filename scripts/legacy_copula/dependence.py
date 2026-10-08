"""Three-zone copula components for prospective net-load forecasting.

Training inputs must be *forward out-of-fold, marginally calibrated* PITs.
This module cannot certify the chronology of caller-supplied arrays.  It never
fits a marginal model or silently re-ranks PITs supplied to parametric models.

All event methods accept V[i, z] = F_z(threshold_z) and return a dictionary of
arrays (ge2, all3, count_probabilities, numerical_error).  Events are strict
exceedances U_z > V_z.  The continuous copulas make threshold ties null events.
X contains caller-selected low-dimensional weather/calendar basis functions;
only training-set standardisation is performed here.
"""

from __future__ import annotations

import copy
import inspect
from dataclasses import dataclass, asdict
from typing import Optional, Sequence

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, gammaln, logit
from scipy.spatial import cKDTree
from scipy.stats import kendalltau, multivariate_normal, multivariate_t, norm, t


PARTIAL_SHRINK = 1.0 - 1e-6
_NORMAL_RNG_KEY = (
    "rng" if "rng" in inspect.signature(multivariate_normal.cdf).parameters
    else "random_state" if "random_state" in inspect.signature(multivariate_normal.cdf).parameters
    else None
)
_T_RNG_KEY = (
    "rng" if "rng" in inspect.signature(multivariate_t.cdf).parameters
    else "random_state" if "random_state" in inspect.signature(multivariate_t.cdf).parameters
    else None
)


def _matrix(value, name: str, columns: Optional[int] = None) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite two-dimensional array")
    if columns is not None and array.shape[1] != columns:
        raise ValueError(f"{name} must have {columns} columns")
    return array


def _unit_matrix(value, name: str = "V") -> np.ndarray:
    array = _matrix(value, name, 3)
    if np.any((array < 0) | (array > 1)):
        raise ValueError(f"{name} must lie in [0, 1]")
    return array


def _pit(value, clip: float):
    array = _unit_matrix(value, "U")
    if not 0 < clip < 0.01:
        raise ValueError("pit_clip must lie in (0, 0.01)")
    if len(array) < 8:
        raise ValueError("At least 8 training rows are required numerically")
    clipped = np.clip(array, clip, 1 - clip)
    return clipped, float(np.mean(clipped != array))


def correlation_from_eta(eta) -> np.ndarray:
    """Positive definite correlation matrices from three unconstrained values.

    Ordering is (rho_12, rho_13, partial rho_23|1).  A fixed 1e-6 shrink away
    from +/-1 avoids floating-point singularities, including extreme inputs.
    Returns shape (n, 3, 3), also for one eta vector.
    """
    values = _matrix(eta, "eta", 3)
    a, b, c = (PARTIAL_SHRINK * np.tanh(values)).T
    sa, sb, sc = np.sqrt(1 - a*a), np.sqrt(1 - b*b), np.sqrt(1 - c*c)
    factor = np.zeros((len(values), 3, 3))
    factor[:, 0, 0] = 1
    factor[:, 1, 0] = a
    factor[:, 1, 1] = sa
    factor[:, 2, 0] = b
    factor[:, 2, 1] = c * sb
    factor[:, 2, 2] = sb * sc
    return factor @ factor.transpose(0, 2, 1)


def _initial_eta(U: np.ndarray) -> np.ndarray:
    # Kendall conversion is only an initial value.  Projection is necessary:
    # transforming a sample tau matrix elementwise does not ensure PSD.
    corr = np.eye(3)
    for i in range(3):
        for j in range(i + 1, 3):
            tau = kendalltau(U[:, i], U[:, j]).statistic
            corr[i, j] = corr[j, i] = np.sin(np.pi * (tau if np.isfinite(tau) else 0) / 2)
    eigenvalues, eigenvectors = np.linalg.eigh(corr)
    corr = (eigenvectors * np.maximum(eigenvalues, 1e-4)) @ eigenvectors.T
    diagonal = np.sqrt(np.diag(corr))
    corr = corr / np.outer(diagonal, diagonal)
    a, b = corr[0, 1], corr[0, 2]
    c = (corr[1, 2] - a*b) / np.sqrt((1-a*a) * (1-b*b))
    return np.arctanh(np.clip(np.array([a, b, c]) / PARTIAL_SHRINK, -0.98, 0.98))


def _log_density_and_eta_gradient(eta, scores, df=np.inf):
    """Vectorised elliptical copula log density and analytic eta gradient."""
    th = np.tanh(eta)
    a, b, c = (PARTIAL_SHRINK * th).T
    da = 1-a*a
    db = 1-b*b
    dc = 1-c*c
    sa, sb, sc = np.sqrt(da), np.sqrt(db), np.sqrt(dc)
    z1, z2, z3 = scores.T
    y2 = (z2-a*z1)/sa
    w3 = (z3-b*z1)/sb
    y3 = (w3-c*y2)/sc
    quad = z1*z1 + y2*y2 + y3*y3
    logdet_half = np.log(sa) + np.log(sb) + np.log(sc)
    if np.isinf(df):
        ld = -logdet_half - 0.5*(quad - np.sum(scores*scores, axis=1))
        multiplier = np.full(len(scores), 0.5)
    else:
        joint = (gammaln((df+3)/2) - gammaln(df/2)
                 - 1.5*np.log(df*np.pi) - logdet_half
                 - (df+3)/2 * np.log1p(quad/df))
        marginal = (gammaln((df+1)/2) - gammaln(df/2)
                    - 0.5*np.log(df*np.pi)
                    - (df+1)/2 * np.log1p(scores*scores/df))
        ld = joint - np.sum(marginal, axis=1)
        multiplier = (df+3) / (2*(df+quad))
    dy2_da = -z1/sa + a*y2/da
    dw3_db = -z1/sb + b*w3/db
    dy3_dc = -y2/sc + c*y3/dc
    dq_da = 2*(y2-c*y3/sc)*dy2_da
    dq_db = 2*y3*dw3_db/sc
    dq_dc = 2*y3*dy3_dc
    gradient_partial = np.column_stack([
        a/da - multiplier*dq_da,
        b/db - multiplier*dq_db,
        c/dc - multiplier*dq_dc,
    ])
    return ld, gradient_partial * (PARTIAL_SHRINK * (1-th*th))


class _CorrelationModel:
    def __init__(self, dynamic=False, l2=0.01, maxiter=400, pit_clip=1e-6):
        if l2 < 0:
            raise ValueError("l2 must be nonnegative")
        self.dynamic = bool(dynamic)
        self.l2 = float(l2)
        self.maxiter = int(maxiter)
        self.pit_clip = float(pit_clip)

    def _design(self, X, n, fit=False):
        if not self.dynamic:
            return np.ones((n, 1))
        if X is None:
            raise ValueError("Dynamic dependence requires X")
        x = _matrix(X, "X")
        if len(x) != n:
            raise ValueError("X row count does not match U or V")
        if fit:
            self.feature_mean_ = x.mean(axis=0)
            self.feature_scale_ = x.std(axis=0)
            self.feature_scale_[self.feature_scale_ < 1e-12] = 1.0
        if x.shape[1] != len(self.feature_mean_):
            raise ValueError("X has a different feature count than during training")
        return np.column_stack([np.ones(n), (x-self.feature_mean_)/self.feature_scale_])

    def fit(self, U, X=None, df=np.inf):
        u, self.clipped_pit_fraction_ = _pit(U, self.pit_clip)
        design = self._design(X, len(u), fit=True)
        scores = norm.ppf(u) if np.isinf(df) else t.ppf(u, df)
        initial = np.zeros((design.shape[1], 3))
        initial[0] = _initial_eta(u)

        def objective(flat):
            coef = flat.reshape(design.shape[1], 3)
            ld, grad_eta = _log_density_and_eta_gradient(design @ coef, scores, df)
            penalty = 0.5*self.l2*np.sum(coef[1:]**2)
            gradient = -(design.T @ grad_eta)/len(u)
            gradient[1:] += self.l2*coef[1:]
            return -float(ld.mean()) + penalty, gradient.ravel()

        result = minimize(objective, initial.ravel(), method="L-BFGS-B", jac=True,
                          options={"maxiter": self.maxiter, "ftol": 1e-11,
                                   "gtol": 1e-6, "maxls": 40})
        self.fit_info_ = {
            "converged": bool(result.success), "message": str(result.message),
            "n_rows": len(u), "n_coefficients": int(result.x.size),
            "n_iterations": int(result.nit), "penalized_negative_log_likelihood": float(result.fun),
            "clipped_pit_fraction": self.clipped_pit_fraction_,
        }
        if not result.success or not np.all(np.isfinite(result.x)):
            raise RuntimeError(f"Dependence optimisation did not converge: {self.fit_info_}")
        self.coef_ = result.x.reshape(design.shape[1], 3)
        return self

    def matrices(self, X=None, n=None):
        if not hasattr(self, "coef_"):
            raise RuntimeError("Fit the correlation model first")
        if n is None:
            n = len(_matrix(X, "X")) if X is not None else 1
        return correlation_from_eta(self._design(X, n) @ self.coef_)

    def log_density(self, U, X=None, df=np.inf):
        u = np.clip(_unit_matrix(U, "U"), self.pit_clip, 1-self.pit_clip)
        scores = norm.ppf(u) if np.isinf(df) else t.ppf(u, df)
        return _log_density_and_eta_gradient(self._design(X, len(u)) @ self.coef_, scores, df)[0]


@dataclass(frozen=True)
class IntegrationSettings:
    """CDF controls; numerical discrepancy is a convergence proxy, not a bound."""
    tolerance: float = 1e-4
    initial_maxpts: int = 16384
    max_maxpts: int = 262144
    seed: int = 91723
    normal_abseps: float = 1e-7
    backend: str = "scipy_replicated"

    def __post_init__(self):
        if not 0 < self.tolerance < 0.01:
            raise ValueError("Integration tolerance must lie in (0, 0.01)")
        if self.initial_maxpts < 128 or self.max_maxpts < self.initial_maxpts:
            raise ValueError("Invalid integration sample limits")
        if self.backend not in ("scipy_replicated", "elliptical_path"):
            raise ValueError("Unknown elliptical probability integration backend")


def _elliptical_cdf(v, corr, df, maxpts, seed, settings):
    # Exact boundary reduction prevents inf/-inf numerical problems.
    if np.any(v <= 0):
        return 0.0
    keep = v < 1
    if not np.any(keep):
        return 1.0
    v = v[keep]
    if len(v) == 1:
        return float(v[0])
    corr = corr[np.ix_(keep, keep)]
    rng = np.random.default_rng(seed)
    if np.isinf(df):
        if _NORMAL_RNG_KEY is None:
            raise RuntimeError("This scipy normal CDF lacks a seedable RNG; upgrade scipy on D:")
        value = multivariate_normal.cdf(
            norm.ppf(v), mean=np.zeros(len(v)), cov=corr, maxpts=maxpts,
            abseps=settings.normal_abseps, releps=settings.normal_abseps,
            **{_NORMAL_RNG_KEY: rng})
    else:
        if _T_RNG_KEY is None:
            raise RuntimeError("This scipy t CDF lacks a seedable RNG; upgrade scipy on D:")
        value = multivariate_t.cdf(
            t.ppf(v, df), loc=np.zeros(len(v)), shape=corr, df=df,
            maxpts=maxpts, **{_T_RNG_KEY: rng})
    if not np.isfinite(value):
        raise RuntimeError("Nonfinite elliptical CDF result")
    return float(value)


def _four_counts(v, corr, df, maxpts, seed, settings):
    pair_cdf = 0.0
    for k, indices in enumerate(((0, 1), (0, 2), (1, 2))):
        index = np.asarray(indices)
        pair_cdf += _elliptical_cdf(v[index], corr[np.ix_(index, index)], df,
                                    maxpts, seed + 104729*k, settings)
    c123 = _elliptical_cdf(v, corr, df, maxpts, seed + 314159, settings)
    ge2 = 1-pair_cdf+2*c123
    all3 = 1-v.sum()+pair_cdf-c123
    return np.array([c123, 1-c123-ge2, ge2-all3, all3])


def _event_result(counts, numerical_error=None, maxpts=None, corrections=None):
    counts = np.asarray(counts, dtype=float)
    n = len(counts)
    return {
        "ge2": counts[:, 2] + counts[:, 3],
        "all3": counts[:, 3],
        "none": counts[:, 0],
        "exactly_one": counts[:, 1],
        "exactly_two": counts[:, 2],
        "count_probabilities": counts,
        "numerical_error": np.zeros(n) if numerical_error is None else numerical_error,
        "integration_maxpts": np.zeros(n, dtype=int) if maxpts is None else maxpts,
        "roundoff_corrected": np.zeros(n, dtype=bool) if corrections is None else corrections,
    }


def elliptical_event_probabilities(V, R, df=np.inf, settings=None):
    """Finite-threshold counts from four CDF integrals, with replicated checks.

    Replicate discrepancy is monitored for all four counts.  Unresolved errors
    raise, rather than remove observations. Tiny within-tolerance negative
    probabilities are projected to the probability simplex and flagged.
    No asymptotic tail probability replaces the finite-threshold calculation.
    """
    v = _unit_matrix(V)
    if not (df == np.inf or (np.isfinite(df) and df > 0)):
        raise ValueError("df must be positive finite or positive infinity")
    settings = settings or IntegrationSettings()
    corr = np.asarray(R, dtype=float)
    if corr.shape == (3, 3):
        corr = np.broadcast_to(corr, (len(v), 3, 3))
    if corr.shape != (len(v), 3, 3) or not np.all(np.isfinite(corr)):
        raise ValueError("R must have shape (3,3) or (n,3,3)")
    if not np.allclose(corr, corr.transpose(0, 2, 1), atol=1e-10, rtol=0):
        raise ValueError("R must be symmetric")
    if not np.allclose(np.diagonal(corr, axis1=1, axis2=2), 1, atol=1e-10, rtol=0):
        raise ValueError("R must have unit diagonal")
    if np.any(np.linalg.eigvalsh(corr) <= 0):
        raise ValueError("R must be positive definite")
    if settings.backend == "elliptical_path":
        from elliptical_event_fast import fast_event_probabilities
        # Exact CDF boundaries use the existing dimension-reduced evaluator;
        # no epsilon perturbation changes the requested event definition.
        interior = np.all((v > 0) & (v < 1), axis=1)
        if interior.all():
            return fast_event_probabilities(v, corr, df, settings.tolerance)
        counts = np.empty((len(v), 4)); errors = np.empty(len(v))
        points = np.zeros(len(v), dtype=int); orders = np.zeros(len(v), dtype=int)
        corrected = np.zeros(len(v), dtype=bool)
        if interior.any():
            fast = fast_event_probabilities(v[interior], corr[interior], df, settings.tolerance)
            counts[interior] = fast["count_probabilities"]; errors[interior] = fast["numerical_error"]
            orders[interior] = fast["integration_order"]; corrected[interior] = fast["roundoff_corrected"]
        from dataclasses import replace
        boundary = elliptical_event_probabilities(v[~interior], corr[~interior], df,
                                                  replace(settings, backend="scipy_replicated"))
        counts[~interior] = boundary["count_probabilities"]; errors[~interior] = boundary["numerical_error"]
        points[~interior] = boundary["integration_maxpts"]; corrected[~interior] = boundary["roundoff_corrected"]
        result = _event_result(counts, errors, points, corrected)
        result["integration_order"] = orders
        result["numerical_method"] = "elliptical_path_with_exact_boundary_reduction"
        return result
    counts = np.empty((len(v), 4))
    errors = np.empty(len(v))
    used_points = np.empty(len(v), dtype=int)
    corrected = np.zeros(len(v), dtype=bool)
    for row in range(len(v)):
        points = settings.initial_maxpts
        # One deterministic seed per row and replicate. A seed is not data.
        base_seed = settings.seed + row*1000003
        while True:
            first = _four_counts(v[row], corr[row], df, points, base_seed, settings)
            second = _four_counts(v[row], corr[row], df, points, base_seed+49979687, settings)
            average = (first+second)/2
            discrepancy = float(np.max(np.abs(first-second)))
            legal = np.min(average) >= -settings.tolerance and np.max(average) <= 1+settings.tolerance
            if discrepancy <= settings.tolerance and legal:
                break
            if points >= settings.max_maxpts:
                raise RuntimeError(
                    f"CDF integration unresolved at row {row}: discrepancy={discrepancy}, "
                    f"counts={average.tolist()}, maxpts={points}")
            points = min(points*4, settings.max_maxpts)
        corrected[row] = bool(np.any((average < 0) | (average > 1)))
        if corrected[row]:
            average = np.clip(average, 0, 1)
            average /= average.sum()
        counts[row], errors[row], used_points[row] = average, discrepancy, points
    return _event_result(counts, errors, used_points, corrected)


class IndependentCopula:
    def fit(self, U=None, X=None):
        if U is not None:
            _unit_matrix(U, "U")
        return self

    def predict_event_probabilities(self, V, X=None, **kwargs):
        v = _unit_matrix(V)
        a = 1-v
        none = np.prod(v, axis=1)
        one = (a[:, 0]*v[:, 1]*v[:, 2] + v[:, 0]*a[:, 1]*v[:, 2]
               + v[:, 0]*v[:, 1]*a[:, 2])
        all3 = np.prod(a, axis=1)
        ge2 = (a[:, 0]*a[:, 1]+a[:, 0]*a[:, 2]+a[:, 1]*a[:, 2]-2*all3)
        return _event_result(np.column_stack([none, one, ge2-all3, all3]))


class GaussianCopula:
    def __init__(self, dynamic=False, l2=0.01, maxiter=400, pit_clip=1e-6,
                 integration_settings=None):
        self._correlation = _CorrelationModel(dynamic, l2, maxiter, pit_clip)
        self.integration_settings = integration_settings or IntegrationSettings()
        self.df_ = np.inf

    def fit(self, U, X=None):
        self._correlation.fit(U, X, df=np.inf)
        self.fit_info_ = self._correlation.fit_info_
        return self

    def correlation_matrices(self, X=None, n=None):
        return self._correlation.matrices(X, n)

    def log_density(self, U, X=None):
        return self._correlation.log_density(U, X, np.inf)

    def predict_event_probabilities(self, V, X=None, settings=None):
        v = _unit_matrix(V)
        return elliptical_event_probabilities(v, self.correlation_matrices(X, len(v)),
                                               settings=settings or self.integration_settings)


class StaticGaussianCopula(GaussianCopula):
    def __init__(self, **kwargs):
        super().__init__(dynamic=False, **kwargs)


class DynamicGaussianCopula(GaussianCopula):
    def __init__(self, **kwargs):
        super().__init__(dynamic=True, **kwargs)


class StudentTCopula:
    """Static/dynamic t copula, or exact shared-R comparison with a Gaussian.

    Supply a fitted Gaussian via shared_model to hold its R(X) unchanged.
    Otherwise correlation coefficients are fitted for each candidate df.
    Training pseudo-likelihood chooses among df_candidates; for a development-
    selected df, pass a singleton. All selection uses only the supplied U.
    """
    def __init__(self, df_candidates: Sequence[float] = (3, 5, 8, 15, 30, np.inf),
                 shared_model: Optional[GaussianCopula] = None, dynamic=False,
                 l2=0.01, maxiter=400, pit_clip=1e-6, integration_settings=None):
        self.df_candidates = tuple(float(value) for value in df_candidates)
        if not self.df_candidates or any(value <= 0 or np.isnan(value) for value in self.df_candidates):
            raise ValueError("df_candidates must contain positive values or infinity")
        self.shared_model = shared_model
        self.dynamic = dynamic
        self.l2, self.maxiter, self.pit_clip = l2, maxiter, pit_clip
        self.integration_settings = integration_settings or IntegrationSettings()

    def fit(self, U, X=None):
        u, clipped = _pit(U, self.pit_clip)
        candidates = []
        best_score = -np.inf
        for df in self.df_candidates:
            if self.shared_model is not None:
                # Copy prevents a later fit of the Gaussian from changing C5.
                model = copy.deepcopy(self.shared_model._correlation)
                if not hasattr(model, "coef_"):
                    raise RuntimeError("shared_model must already be fitted")
            else:
                model = _CorrelationModel(self.dynamic, self.l2, self.maxiter, self.pit_clip)
                model.fit(u, X, df=df)
            score = float(model.log_density(u, X, df).mean())
            candidates.append({"df": None if np.isinf(df) else df,
                               "gaussian_limit": bool(np.isinf(df)),
                               "mean_training_log_copula_density": score})
            if score > best_score:
                self._correlation = model
                self.df_ = df
                best_score = score
        self.fit_info_ = {
            "mode": "shared_R" if self.shared_model is not None else "refitted_R",
            "n_rows": len(u), "df": None if np.isinf(self.df_) else self.df_,
            "gaussian_limit": bool(np.isinf(self.df_)), "candidates": candidates,
            "clipped_pit_fraction": clipped,
            "correlation_fit": self._correlation.fit_info_,
        }
        return self

    def correlation_matrices(self, X=None, n=None):
        return self._correlation.matrices(X, n)

    def log_density(self, U, X=None):
        return self._correlation.log_density(U, X, self.df_)

    def predict_event_probabilities(self, V, X=None, settings=None):
        v = _unit_matrix(V)
        return elliptical_event_probabilities(v, self.correlation_matrices(X, len(v)), self.df_,
                                               settings or self.integration_settings)


def ranks_from_template(template, seed=2701):
    """Ranks 1..M; exact ties broken independently by zone, reproducibly.

    Stable member-order tie breaking across all zones would invent concordance.
    Random tie breaking means unresolved within-tie dependence is not assumed.
    """
    values = np.asarray(template, dtype=float)
    single = values.ndim == 2
    if single:
        values = values[None, ...]
    if values.ndim != 3 or values.shape[-1] != 3 or values.shape[1] < 2:
        raise ValueError("template must have shape (M,3) or (n,M,3), M >= 2")
    if not np.all(np.isfinite(values)):
        raise ValueError("template contains nonfinite values")
    rng = np.random.default_rng(seed)
    ranks = np.empty(values.shape, dtype=int)
    for row in range(len(values)):
        for zone in range(3):
            order = np.lexsort((rng.random(values.shape[1]), values[row, :, zone]))
            ranks[row, order, zone] = np.arange(1, values.shape[1]+1)
    return ranks[0] if single else ranks


def _ranks_for_forecasts(V, template=None, ranks=None, seed=2701):
    v = _unit_matrix(V)
    if (template is None) == (ranks is None):
        raise ValueError("Provide exactly one of template or ranks")
    r = np.asarray(ranks_from_template(template, seed) if ranks is None else ranks)
    if r.ndim == 2:
        r = np.broadcast_to(r, (len(v), *r.shape))
    if r.ndim != 3 or r.shape[0] != len(v) or r.shape[-1] != 3 or r.shape[1] < 2:
        raise ValueError("Rank dimensions do not match forecasts")
    expected = np.arange(1, r.shape[1]+1)[None, :, None]
    if not np.array_equal(np.sort(r, axis=1), np.broadcast_to(expected, r.shape)):
        raise ValueError("Each rank column must be a permutation of 1..M")
    return v, r.astype(float, copy=False)


def checkerboard_event_probabilities(V, template=None, ranks=None, seed=2701):
    v, r = _ranks_for_forecasts(V, template, ranks, seed)
    a = np.clip(r-r.shape[1]*v[:, None, :], 0, 1)
    b = 1-a
    all3 = np.prod(a, axis=2).mean(axis=1)
    ge2 = (a[:, :, 0]*a[:, :, 1] + a[:, :, 0]*a[:, :, 2]
           + a[:, :, 1]*a[:, :, 2] - 2*np.prod(a, axis=2)).mean(axis=1)
    none = np.prod(b, axis=2).mean(axis=1)
    one = (a[:, :, 0]*b[:, :, 1]*b[:, :, 2]
           + b[:, :, 0]*a[:, :, 1]*b[:, :, 2]
           + b[:, :, 0]*b[:, :, 1]*a[:, :, 2]).mean(axis=1)
    return _event_result(np.column_stack([none, one, ge2-all3, all3]))


def standard_ecc_event_probabilities(V, template=None, ranks=None, seed=2701):
    """Discrete ECC-Q sensitivity using quantiles k/(M+1), not continuous F*."""
    v, r = _ranks_for_forecasts(V, template, ranks, seed)
    exceed_count = np.sum(r/(r.shape[1]+1) > v[:, None, :], axis=2)
    counts = np.column_stack([(exceed_count == k).mean(axis=1) for k in range(4)])
    return _event_result(counts)


class AnalogCheckerboard:
    """Weather-nearest historical PIT *ranks*, with exact shared marginals.

    For leakage protection every query issue_time must follow *all* stored
    observation_times. Refit on the proper historical subset for each OOF fold.
    Observation times should represent when the outcome was legitimately
    available, not merely its measurement time when those differ.
    """
    def __init__(self, n_analogs=50, seed=2701):
        self.n_analogs = int(n_analogs)
        self.seed = int(seed)

    def fit(self, U, X, observation_times):
        self.U_ = _unit_matrix(U, "U").copy()
        x = _matrix(X, "X")
        times = np.asarray(observation_times, dtype="datetime64[ns]")
        if len(x) != len(self.U_) or times.shape != (len(x),) or np.any(np.isnat(times)):
            raise ValueError("Historical U, X and observation_times must align")
        if not 2 <= self.n_analogs <= len(x):
            raise ValueError("n_analogs must be between 2 and history size")
        self.latest_available_time_ = times.max()
        self.mean_ = x.mean(axis=0)
        self.scale_ = x.std(axis=0)
        self.scale_[self.scale_ < 1e-12] = 1
        self.tree_ = cKDTree((x-self.mean_)/self.scale_)
        return self

    def predict_event_probabilities(self, V, X, issue_times):
        v = _unit_matrix(V)
        x = _matrix(X, "X")
        times = np.asarray(issue_times, dtype="datetime64[ns]")
        if len(x) != len(v) or times.shape != (len(v),) or np.any(np.isnat(times)):
            raise ValueError("V, X and issue_times must align")
        if np.any(times <= self.latest_available_time_):
            raise ValueError("Refit analog history using only observations available before every issue_time")
        _, indices = self.tree_.query((x-self.mean_)/self.scale_, k=self.n_analogs)
        return checkerboard_event_probabilities(v, template=self.U_[indices], seed=self.seed)


class EventLogisticCalibrator:
    """Intercept/slope calibration; caller must supply forward OOF probabilities.

    The slope penalty shrinks to the identity map, not to zero. This is an
    event-only forecast and is not a full joint-distribution recalibration.
    """
    def __init__(self, l2=0.001, probability_clip=1e-6, nonnegative_slope=False):
        self.l2 = float(l2)
        self.probability_clip = float(probability_clip)
        self.nonnegative_slope = bool(nonnegative_slope)
        if self.l2 < 0 or not 0 < self.probability_clip < 0.01:
            raise ValueError("Invalid calibrator regularisation or probability clip")

    def _feature(self, probabilities):
        p = np.asarray(probabilities, dtype=float)
        if p.ndim != 1 or np.any(~np.isfinite(p)) or np.any((p < 0) | (p > 1)):
            raise ValueError("Probabilities must be a finite vector in [0,1]")
        return logit(np.clip(p, self.probability_clip, 1-self.probability_clip))

    def fit(self, probabilities, outcomes):
        x = self._feature(probabilities)
        y = np.asarray(outcomes, dtype=float)
        if y.shape != x.shape or not np.all(np.isin(y, [0, 1])):
            raise ValueError("Outcomes must be aligned binary values")
        if len(np.unique(y)) < 2:
            raise ValueError("Event calibration requires both outcomes; do not invent missing events")
        design = np.column_stack([np.ones(len(x)), x])

        def objective(beta):
            eta = design @ beta
            loss = np.mean(np.logaddexp(0, eta)-y*eta) + 0.5*self.l2*(beta[1]-1)**2
            grad = design.T @ (expit(eta)-y)/len(y)
            grad[1] += self.l2*(beta[1]-1)
            return float(loss), grad

        result = minimize(objective, np.array([0.0, 1.0]), jac=True, method="L-BFGS-B",
                          bounds=[(None, None), (0, None)] if self.nonnegative_slope else None,
                          options={"maxiter": 400, "ftol": 1e-12, "gtol": 1e-8})
        if not result.success:
            raise RuntimeError(f"Event calibration failed: {result.message}")
        self.coef_ = result.x
        self.fit_info_ = {"converged": True, "n_rows": len(y), "n_events": int(y.sum()),
                          "intercept": float(result.x[0]), "slope": float(result.x[1]),
                          "probability_clip": self.probability_clip}
        return self

    def predict(self, probabilities):
        if not hasattr(self, "coef_"):
            raise RuntimeError("Fit the calibrator first")
        return expit(self.coef_[0]+self.coef_[1]*self._feature(probabilities))


def sample_elliptical_copula(R, df=np.inf, seed=42):
    """Generate explicitly synthetic PIT vectors, one per supplied matrix."""
    matrices = np.asarray(R, dtype=float)
    if matrices.ndim != 3 or matrices.shape[1:] != (3, 3):
        raise ValueError("R must have shape (n,3,3)")
    rng = np.random.default_rng(seed)
    normal_scores = np.einsum("nij,nj->ni", np.linalg.cholesky(matrices),
                              rng.standard_normal((len(matrices), 3)))
    if np.isinf(df):
        return norm.cdf(normal_scores)
    scale = np.sqrt(rng.chisquare(df, len(matrices))/df)
    return t.cdf(normal_scores/scale[:, None], df)


def api_metadata():
    return {"zones": 3, "normal_cdf_rng_keyword": _NORMAL_RNG_KEY,
            "student_cdf_rng_keyword": _T_RNG_KEY,
            "default_integration_settings": asdict(IntegrationSettings()),
            "input_contract": "Forward OOF calibrated PITs for fit; common marginal CDF thresholds for predict"}
