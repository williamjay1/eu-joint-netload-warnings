"""Past-only named basis for conditional copula correlation predictors.

No weather file decoding or empirical forecast validation occurs here.  The
caller supplies a named numeric DataFrame and case-level timing manifest.
``fit_transform`` is for the historical design used by a downstream fit;
``transform`` is for issued forecasts and requires issue_time >= fit_cutoff.
NaN imputation, when enabled, uses training medians; infinity is always rejected.
"""
from __future__ import annotations
from release_paths import work_path as _work_path, raw_path as _raw_path

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import SplineTransformer


SCRIPT_VERSION = "1.1-named-past-only-basis"


def utc_timestamp(value, name="timestamp"):
    stamp = pd.Timestamp(value)
    if pd.isna(stamp) or stamp.tzinfo is None:
        raise ValueError(f"{name} must be a timezone-aware timestamp")
    return stamp.tz_convert("UTC")


def utc_index(value, name="target_times"):
    index = pd.DatetimeIndex(value)
    if index.tz is None or index.hasnans or index.has_duplicates:
        raise ValueError(f"{name} must be unique, nonmissing, timezone-aware timestamps")
    index = index.tz_convert("UTC")
    if not index.is_monotonic_increasing:
        raise ValueError(f"{name} must be increasing")
    return index


def validate_timing_manifest(manifest, target_times, *, fit_cutoff=None,
                             mode="forecast", interval_hours=1.0,
                             require_initialization=False,
                             minimum_release_delay_hours=0.0,
                             strict_issue_clock=False):
    """Check supplied timing assertions, not their historical authenticity.

    Required columns: target_time, issue_time, available_time.  Optional
    initialization_time enables forecast-release-delay checks.  Fit mode also
    requires the complete target interval to end by fit_cutoff.  Forecast mode
    requires the model cutoff to precede every issue.  All times must have zones;
    naive times are rejected rather than silently interpreted as UTC.
    """
    if not isinstance(manifest, pd.DataFrame):
        raise TypeError("timing_manifest must be a DataFrame")
    target = utc_index(target_times)
    if len(manifest) != len(target):
        raise ValueError("Timing manifest and feature rows differ")
    required = ["target_time", "issue_time", "available_time"]
    if require_initialization:
        required.append("initialization_time")
    if any(name not in manifest for name in required):
        raise ValueError(f"Timing manifest requires {required}")
    if mode not in {"fit", "forecast"} or interval_hours <= 0 or minimum_release_delay_hours < 0:
        raise ValueError("Invalid timing policy")
    result = pd.DataFrame(index=target)
    names = required + (["initialization_time"] if "initialization_time" in manifest and
                         "initialization_time" not in required else [])
    for name in names:
        converted = [utc_timestamp(value, name) for value in manifest[name]]
        result[name] = pd.DatetimeIndex(converted)
    if not pd.DatetimeIndex(result.target_time).equals(target):
        raise ValueError("Manifest target_time order must exactly match feature rows")
    if (result.available_time > result.issue_time).any():
        raise ValueError("Features became available after their forecast issue")
    if (result.issue_time >= result.target_time).any():
        raise ValueError("Forecast issue must precede target interval start")
    if strict_issue_clock:
        expected_issue = target.normalize()-pd.Timedelta(hours=12)
        if not pd.DatetimeIndex(result.issue_time).equals(expected_issue):
            raise ValueError("Each target day must use its previous-day 12:00 UTC issue")
        if not target.equals(target.floor("h")):
            raise ValueError("Target intervals must start at whole UTC hours")
    if "initialization_time" in result:
        earliest = result.initialization_time + pd.Timedelta(hours=minimum_release_delay_hours)
        if (earliest > result.available_time).any():
            raise ValueError("Declared weather availability violates the release-delay policy")
    elif minimum_release_delay_hours:
        raise ValueError("A release-delay check requires initialization_time")
    if fit_cutoff is not None:
        cutoff = utc_timestamp(fit_cutoff, "fit_cutoff")
        if mode == "fit":
            if (result.target_time + pd.Timedelta(hours=interval_hours) > cutoff).any():
                raise ValueError("A training target interval extends beyond fit_cutoff")
        elif (result.issue_time < cutoff).any():
            raise ValueError("Model fit_cutoff is later than a forecast issue")
    elif mode == "fit":
        raise ValueError("Fit mode requires a declared fit_cutoff")
    return result


class ConditionalWeatherBasis(TransformerMixin, BaseEstimator):
    """Low-dimensional named linear/spline basis with training-only statistics.

    linear_columns and smooth_columns partition ALL input columns.  Each named
    interaction is a pair of original columns, not a hidden all-pairs expansion.
    Interactions multiply training-standardized inputs; smooth inputs are clipped
    to their training support before the product.  Cubic spline extrapolation is
    constant.  Constant smooth columns produce a named zero column.  Output
    centering/scaling is also learned only on training rows.

    A single basis may supply the same X to Gaussian and shared-R t comparators.
    The downstream copula is responsible for fitting only prior, calibrated PITs.
    """
    def __init__(self, linear_columns=(), smooth_columns=(), interactions=(),
                 n_knots=4, allow_missing=True, max_interactions=6,
                 minimum_release_delay_hours=48.0, require_initialization=True,
                 strict_issue_clock=True):
        self.linear_columns = linear_columns
        self.smooth_columns = smooth_columns
        self.interactions = interactions
        self.n_knots = n_knots
        self.allow_missing = allow_missing
        self.max_interactions = max_interactions
        self.minimum_release_delay_hours = minimum_release_delay_hours
        self.require_initialization = require_initialization
        self.strict_issue_clock = strict_issue_clock

    def _timing(self, manifest, target, *, mode, fit_cutoff):
        return validate_timing_manifest(
            manifest, target, fit_cutoff=fit_cutoff, mode=mode,
            require_initialization=self.require_initialization,
            minimum_release_delay_hours=self.minimum_release_delay_hours,
            strict_issue_clock=self.strict_issue_clock)

    def _configuration(self):
        linear, smooth = list(self.linear_columns), list(self.smooth_columns)
        columns = linear + smooth
        if not columns or any(not isinstance(c, str) or not c for c in columns):
            raise ValueError("Configure nonempty, named input columns")
        if len(columns) != len(set(columns)):
            raise ValueError("Linear/smooth columns must be distinct")
        if not isinstance(self.n_knots, int) or not 2 <= self.n_knots <= 8:
            raise ValueError("Use 2--8 low-dimensional spline knots")
        if not isinstance(self.max_interactions, int) or not 0 <= self.max_interactions <= 12:
            raise ValueError("max_interactions must be between 0 and 12")
        pairs = [tuple(pair) for pair in self.interactions]
        if len(pairs) > self.max_interactions:
            raise ValueError("Too many configured interactions")
        seen = set()
        for pair in pairs:
            if len(pair) != 2 or pair[0] == pair[1] or any(c not in columns for c in pair):
                raise ValueError("Each interaction must name two distinct configured inputs")
            key = tuple(sorted(pair))
            if key in seen:
                raise ValueError("Duplicate symmetric interaction")
            seen.add(key)
        return linear, smooth, columns, pairs

    @staticmethod
    def _input(X, columns):
        if not isinstance(X, pd.DataFrame) or X.columns.has_duplicates:
            raise TypeError("Use a DataFrame with unique named columns")
        if set(X.columns) != set(columns) or len(X.columns) != len(columns):
            raise ValueError(f"Exact feature schema required; expected {columns}")
        index = utc_index(X.index)
        try:
            values = X.loc[:, columns].to_numpy(dtype=float)
        except (TypeError, ValueError) as error:
            raise ValueError("Features must be numeric") from error
        if np.isinf(values).any():
            raise ValueError("Infinite features require an upstream audit")
        if len(values) == 0:
            raise ValueError("Empty feature table")
        return values, index

    def _impute(self, values):
        missing = np.isnan(values)
        if missing.any() and not self.allow_missing:
            raise ValueError("Missing features are disabled by this configuration")
        return np.where(missing, self.medians_[None, :], values)

    def _unscaled(self, values):
        raw = self._impute(values)
        blocks, names = [], []
        for name in self.linear_columns_:
            j = self.columns_.index(name)
            blocks.append(((raw[:, j]-self.input_means_[j])/self.input_scales_[j])[:, None])
            names.append("linear::" + name)
        for name in self.smooth_columns_:
            j = self.columns_.index(name)
            spline = self.splines_[name]
            if spline is None:
                blocks.append(np.zeros((len(raw), 1)))
                names.append("smooth::" + name + "::constant")
            else:
                values = spline.transform(raw[:, [j]])
                blocks.append(values)
                names.extend(f"smooth::{name}::b{k}" for k in range(values.shape[1]))
        for left, right in self.interactions_:
            product = np.ones(len(raw))
            for name in (left, right):
                j = self.columns_.index(name)
                v = raw[:, j]
                if name in self.smooth_columns_:
                    v = np.clip(v, self.input_min_[j], self.input_max_[j])
                product *= (v-self.input_means_[j])/self.input_scales_[j]
            blocks.append(product[:, None])
            names.append(f"interaction::{left}*{right}")
        return np.column_stack(blocks), names

    def fit(self, X, y=None, *, timing_manifest, fit_cutoff):
        linear, smooth, columns, pairs = self._configuration()
        values, index = self._input(X, columns)
        if len(values) < 8:
            raise ValueError("At least eight historical feature rows are required")
        self._timing(timing_manifest, index, fit_cutoff=fit_cutoff, mode="fit")
        if np.isnan(values).all(axis=0).any():
            raise ValueError("All-missing training columns have no estimable median")
        if not self.allow_missing and np.isnan(values).any():
            raise ValueError("Missing features are disabled by this configuration")
        self.columns_, self.linear_columns_, self.smooth_columns_ = columns, linear, smooth
        self.interactions_ = pairs
        self.fit_cutoff_ = utc_timestamp(fit_cutoff, "fit_cutoff")
        self.medians_ = np.nanmedian(values, axis=0)
        raw = self._impute(values)
        self.input_means_, self.input_scales_ = raw.mean(axis=0), raw.std(axis=0)
        self.input_scales_[self.input_scales_ < 1e-12] = 1.0
        self.input_min_, self.input_max_ = raw.min(axis=0), raw.max(axis=0)
        self.splines_, self.knots_ = {}, {}
        for name in smooth:
            j = columns.index(name)
            knots = np.unique(np.quantile(raw[:, j], np.linspace(0, 1, self.n_knots)))
            self.knots_[name] = knots.tolist()
            self.splines_[name] = None if len(knots) < 2 else SplineTransformer(
                degree=3, knots=knots[:, None], extrapolation="constant", include_bias=False
            ).fit(raw[:, [j]])
        design, names = self._unscaled(values)
        self.output_means_, self.output_scales_ = design.mean(axis=0), design.std(axis=0)
        self.output_scales_[self.output_scales_ < 1e-12] = 1.0
        self.output_names_ = names
        self.feature_names_in_, self.n_features_in_ = np.asarray(columns, object), len(columns)
        self.fit_metadata_ = {
            "script_version": SCRIPT_VERSION, "fit_cutoff": self.fit_cutoff_.isoformat(),
            "training_rows": len(index), "first_target": index[0].isoformat(),
            "last_target_interval_end": (index[-1]+pd.Timedelta(hours=1)).isoformat(),
            "input_columns": columns, "output_columns": names,
            "linear_columns": linear, "smooth_columns": smooth,
            "interactions": [list(pair) for pair in pairs], "knots": self.knots_,
            "training_medians": dict(zip(columns, self.medians_.tolist())),
            "training_missing_counts": dict(zip(columns, np.isnan(values).sum(axis=0).tolist())),
            "extrapolation": "constant_for_smooths_and_smooth_inputs_in_interactions",
            "fitting_data": "past_features_only_no_prediction_period_refit",
            "timing_scope": "supplied_manifest_checked_historical_arrival_not_independently_verified",
            "minimum_release_delay_hours": self.minimum_release_delay_hours,
            "require_initialization": self.require_initialization,
            "strict_issue_clock": self.strict_issue_clock,
        }
        return self

    def _transformed(self, X):
        if not hasattr(self, "output_names_"):
            raise RuntimeError("Fit the basis before transforming")
        values, index = self._input(X, self.columns_)
        design, names = self._unscaled(values)
        result = (design-self.output_means_)/self.output_scales_
        if not np.isfinite(result).all() or names != self.output_names_:
            raise RuntimeError("Nonfinite or inconsistent basis output")
        return pd.DataFrame(result, index=index, columns=names)

    def fit_transform(self, X, y=None, *, timing_manifest, fit_cutoff):
        self.fit(X, y, timing_manifest=timing_manifest, fit_cutoff=fit_cutoff)
        return self._transformed(X)

    def transform(self, X, *, timing_manifest):
        if not hasattr(self, "fit_cutoff_"):
            raise RuntimeError("Fit the basis before transforming")
        self._timing(timing_manifest, X.index, fit_cutoff=self.fit_cutoff_, mode="forecast")
        return self._transformed(X)

    def transform_member_array(self, values, *, columns, target_times, timing_manifest):
        """Transform (case,member,input) after one case-level timing check.

        The caller owns identity alignment. This method does not fit statistics
        on the ensemble members and keeps their supplied ordering unchanged.
        It is useful for applying a single mean model to every ECC member.
        """
        if not hasattr(self, "fit_cutoff_"):
            raise RuntimeError("Fit the basis before transforming")
        index = utc_index(target_times)
        if list(columns) != self.columns_:
            raise ValueError("Member-array columns must match the fitted order exactly")
        values = np.asarray(values, dtype=float)
        if (values.ndim != 3 or values.shape[0] != len(index) or
                values.shape[2] != len(self.columns_) or values.shape[1] < 1):
            raise ValueError("Expected (case,member,named_input) values")
        if np.isinf(values).any():
            raise ValueError("Infinite member features require an upstream audit")
        self._timing(timing_manifest, index, fit_cutoff=self.fit_cutoff_, mode="forecast")
        flat = values.reshape(-1, values.shape[2])
        design, names = self._unscaled(flat)
        transformed = (design-self.output_means_)/self.output_scales_
        if names != self.output_names_ or not np.isfinite(transformed).all():
            raise RuntimeError("Nonfinite or inconsistent member basis output")
        return transformed.reshape(values.shape[0], values.shape[1], -1)

    def get_feature_names_out(self, input_features=None):
        if not hasattr(self, "output_names_"):
            raise RuntimeError("Fit the basis before requesting names")
        if input_features is not None and list(input_features) != self.columns_:
            raise ValueError("input_features differ from fitted schema")
        return np.asarray(self.output_names_, object)


def synthetic_timing(index):
    """Synthetic metadata helper; never a statement of actual service arrival."""
    index = utc_index(index)
    return pd.DataFrame({"target_time": index, "issue_time": index.normalize()-pd.Timedelta(hours=12),
                         "initialization_time": index.normalize()-pd.Timedelta(days=3),
                         "available_time": index.normalize()-pd.Timedelta(days=1)})


def synthetic_self_test(output=None):
    rng = np.random.default_rng(20260929)
    index = pd.date_range("2021-01-01", periods=180, freq="h", tz="UTC")
    X = pd.DataFrame({"temperature": rng.normal(size=len(index)), "spread": rng.uniform(size=len(index)),
                      "calendar": np.sin(np.arange(len(index))/24)}, index=index)
    X.iloc[3, 0] = np.nan
    cutoff = index[-1]+pd.Timedelta(hours=1)
    basis = ConditionalWeatherBasis(linear_columns=("calendar",), smooth_columns=("temperature", "spread"),
                                    interactions=(("temperature", "spread"),))
    train = basis.fit_transform(X, timing_manifest=synthetic_timing(index), fit_cutoff=cutoff)
    future = pd.date_range("2021-02-01", periods=24, freq="h", tz="UTC")
    query = pd.DataFrame({"temperature": np.linspace(-20, 20, 24), "spread": .5,
                          "calendar": np.sin(np.arange(24)/24)}, index=future)
    before = json.dumps(basis.fit_metadata_, sort_keys=True)
    prediction = basis.transform(query, timing_manifest=synthetic_timing(future))
    clipped = query.copy()
    for name in basis.smooth_columns_:
        j = basis.columns_.index(name)
        clipped[name] = clipped[name].clip(basis.input_min_[j], basis.input_max_[j])
    np.testing.assert_allclose(prediction, basis.transform(clipped, timing_manifest=synthetic_timing(future)))
    assert before == json.dumps(basis.fit_metadata_, sort_keys=True)
    assert np.isfinite(train.to_numpy()).all()
    np.testing.assert_allclose(prediction, basis.transform(query[list(reversed(query.columns))],
                                                         timing_manifest=synthetic_timing(future)))
    checks = ["training_only_statistics_unchanged_by_prediction",
              "constant_smooth_and_interaction_extrapolation",
              "named_column_reordering_preserves_output", "training_median_imputation"]

    def rejected(call, name):
        try:
            call()
        except (ValueError, TypeError):
            checks.append(name)
        else:
            raise AssertionError(name)

    rejected(lambda: basis.transform(query.assign(extra=1), timing_manifest=synthetic_timing(future)),
             "extra_or_missing_schema_rejected")
    rejected(lambda: basis.transform(query.iloc[::-1], timing_manifest=synthetic_timing(future)),
             "unsorted_target_rows_rejected")
    rejected(lambda: basis.transform(query, timing_manifest=synthetic_timing(future).iloc[::-1]),
             "reordered_manifest_rows_rejected")
    bad = query.copy(); bad.iloc[0, 0] = np.inf
    rejected(lambda: basis.transform(bad, timing_manifest=synthetic_timing(future)), "infinity_rejected")
    timing = synthetic_timing(future); timing.loc[0, "available_time"] = future[0]
    rejected(lambda: basis.transform(query, timing_manifest=timing), "future_available_weather_rejected")
    rejected(lambda: basis.transform(X, timing_manifest=synthetic_timing(index)),
             "model_unavailable_at_historical_issue_rejected")
    rejected(lambda: ConditionalWeatherBasis(smooth_columns=tuple(X.columns)).fit(
        X, timing_manifest=synthetic_timing(index), fit_cutoff=index[-1]), "training_interval_after_cutoff_rejected")
    timing = synthetic_timing(future)
    timing["available_time"] = timing["initialization_time"] + pd.Timedelta(hours=47)
    rejected(lambda: basis.transform(query, timing_manifest=timing), "public_release_delay_violation_rejected")
    timing = synthetic_timing(future)
    timing["issue_time"] += pd.Timedelta(hours=1)
    rejected(lambda: basis.transform(query, timing_manifest=timing), "wrong_daily_issue_clock_rejected")
    member_values = np.stack([query[basis.columns_].to_numpy(), query[basis.columns_].to_numpy()+.1], axis=1)
    member_output = basis.transform_member_array(member_values, columns=basis.columns_, target_times=future,
                                                 timing_manifest=synthetic_timing(future))
    np.testing.assert_allclose(member_output[:, 0], prediction)
    np.testing.assert_allclose(member_output[:, ::-1], basis.transform_member_array(
        member_values[:, ::-1], columns=basis.columns_, target_times=future,
        timing_manifest=synthetic_timing(future)))
    checks.extend(["member_array_matches_named_dataframe", "member_array_permutation_equivariance"])
    report = {"status": "passed", "evidence_type": "SYNTHETIC_ONLY_NO_REAL_WEATHER",
              "created_utc": datetime.now(timezone.utc).isoformat(), "checks": checks,
              "training_rows": len(train), "output_features": len(train.columns),
              "feature_names": list(train.columns),
              "limitation": "Checks establish schema/timing/numerical behavior, not forecast skill or conditional calibration."}
    if output is not None:
        path = Path(output).resolve()
        if not path.resolve().is_relative_to(_work_path().resolve()) or not path.name.startswith("conditional_") or path.exists():
            raise ValueError("Choose a new D: conditional_* report path")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args()
    if not args.self_test:
        parser.error("Only explicit synthetic self-test is provided; real fitting uses the Python API")
    print(json.dumps(synthetic_self_test(args.output), indent=2))
