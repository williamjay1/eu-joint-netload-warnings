"""Past-only weather-to-net-load rank templates for checkerboard ECC.

Each zone's additive Ridge mean model is fitted to ONE ensemble-mean feature
vector and ONE net-load label per historical hour. The same fitted model is
then applied to each future weather member. The resulting member net-load
values supply only ranks; this module never changes calibrated marginal CDFs.
Actual TIGGE decoding, forecast skill and historic arrival are not verified here.
"""
from __future__ import annotations
from release_paths import work_path as _work_path, raw_path as _raw_path

import argparse
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from conditional_features import (ConditionalWeatherBasis, synthetic_timing,
                                  utc_index, utc_timestamp, validate_timing_manifest)

SCRIPT_VERSION = "1.0-ensemble-mean-fit-member-rank-template"
DEFAULT_ZONES = ("DE_LU", "FR", "BE")


def _quarter_start(value):
    origin = utc_timestamp(value, "quarter_start")
    if origin != origin.normalize() or origin.day != 1 or origin.month not in (1, 4, 7, 10):
        raise ValueError("quarter_start must be a UTC calendar-quarter origin at 00:00")
    return origin


def _configured_columns(weather, calendar, linear, smooth):
    columns = list(weather) + list(calendar)
    if (not columns or len(columns) != len(set(columns)) or
            any(not isinstance(c, str) or not c for c in columns)):
        raise ValueError("Configure distinct, named weather and calendar columns")
    if set(linear).intersection(smooth) or set(linear).union(smooth) != set(columns):
        raise ValueError("Linear and smooth columns must partition the complete feature schema")
    if len(linear)+len(smooth) != len(columns):
        raise ValueError("Duplicate named linear/smooth features")
    return columns


def _calendar_table(calendar, columns, target):
    if (not isinstance(calendar, pd.DataFrame) or calendar.columns.has_duplicates or
            set(calendar.columns) != set(columns)):
        raise ValueError("Calendar input must have the exact named schema")
    if not utc_index(calendar.index, "calendar index").equals(target):
        raise ValueError("Calendar target order must match weather exactly")
    values = calendar.loc[:, list(columns)].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Calendar values must be finite")
    return values


def _member_tables(weather_members, zones, weather_columns):
    """Align explicit member IDs; reject missing members or unnamed columns.

    Input is zone -> DataFrame, indexed by sorted UTC target hours, with a
    two-level column index named ('member_id','feature'). Every member must have
    exactly all configured weather features. The first zone defines output
    member order; other zones are aligned by identity rather than position.
    All zones/members are asserted to use the case-level manifest initialization.
    Upstream decoding must independently establish that assertion.
    """
    if not isinstance(weather_members, Mapping) or set(weather_members) != set(zones):
        raise ValueError("Weather mapping must contain exactly the configured zones")
    target, member_ids, arrays = None, None, {}
    for zone in zones:
        frame = weather_members[zone]
        if (not isinstance(frame, pd.DataFrame) or not isinstance(frame.columns, pd.MultiIndex) or
                frame.columns.nlevels != 2 or list(frame.columns.names) != ["member_id", "feature"] or
                frame.columns.has_duplicates):
            raise ValueError("Use unique ('member_id','feature') weather columns")
        index = utc_index(frame.index, zone+" target times")
        if target is None:
            target = index
        elif not index.equals(target):
            raise ValueError("All zone target times must align exactly")
        ids = list(dict.fromkeys(frame.columns.get_level_values("member_id")))
        if len(ids) < 2 or any(pd.isna(value) for value in ids):
            raise ValueError("At least two explicit, nonmissing weather members are required")
        if member_ids is None:
            member_ids = ids
        elif set(ids) != set(member_ids):
            raise ValueError("Cross-zone weather member IDs differ; positional coupling is forbidden")
        expected = pd.MultiIndex.from_product([member_ids, list(weather_columns)],
                                              names=["member_id", "feature"])
        if len(frame.columns) != len(expected) or set(frame.columns) != set(expected):
            raise ValueError("Each member must contain exactly the named weather features")
        values = frame.loc[:, expected].to_numpy(dtype=float).reshape(len(index), len(ids), -1)
        if not np.isfinite(values).all():
            raise ValueError("Weather members must be complete and finite; exclude cases upstream jointly")
        arrays[zone] = values
    return target, tuple(member_ids), arrays


@dataclass
class RankTemplateResult:
    values: np.ndarray
    target_times: pd.DatetimeIndex
    member_ids: tuple
    zones: tuple
    diagnostics: dict


class WeatherNetLoadRankTemplate:
    """A quarter-frozen additive mean model applied to identity-aligned members.

    Fit accepts a full dated table but selects usable past target intervals and
    labels BEFORE estimating ensemble means, knots, scaling or regression. Label
    availability is an explicit conservative timestamp per hour (covering all
    three series); it cannot precede the hour's interval end. Future labels are
    not validated for finite response values and never affect fitted parameters.
    Missing labels within usable history are omitted separately per zone and
    counted. Weather cases with a missing member must be excluded upstream from
    every comparator; this module does not create an unrecorded member subset.
    """
    def __init__(self, weather_columns, calendar_columns=(), *, linear_columns=(),
                 smooth_columns=(), zones=DEFAULT_ZONES, alpha=10.0, n_knots=4,
                 minimum_training_rows=168, release_delay_hours=48.0,
                 require_selected_initialization=True):
        self.weather_columns = tuple(weather_columns)
        self.calendar_columns = tuple(calendar_columns)
        self.linear_columns = tuple(linear_columns)
        self.smooth_columns = tuple(smooth_columns)
        self.zones = tuple(zones)
        self.alpha = alpha
        self.n_knots = n_knots
        self.minimum_training_rows = minimum_training_rows
        self.release_delay_hours = release_delay_hours
        self.require_selected_initialization = require_selected_initialization

    def _timing(self, manifest, target, *, mode="forecast", fit_cutoff=None):
        timing = validate_timing_manifest(
            manifest, target, mode=mode, fit_cutoff=fit_cutoff,
            require_initialization=True, minimum_release_delay_hours=self.release_delay_hours,
            strict_issue_clock=True)
        if self.require_selected_initialization:
            expected = target.normalize()-pd.Timedelta(days=3)
            if not pd.DatetimeIndex(timing.initialization_time).equals(expected):
                raise ValueError("Selected weather initialization must be 00:00 UTC on target day minus three")
        return timing

    def fit(self, weather_members, calendar, net_load, *, timing_manifest,
            label_available_times, fit_cutoff, quarter_start):
        columns = _configured_columns(self.weather_columns, self.calendar_columns,
                                      self.linear_columns, self.smooth_columns)
        if len(self.zones) != 3 or len(set(self.zones)) != 3:
            raise ValueError("This study's template requires exactly three distinct zones")
        if not np.isfinite(self.alpha) or self.alpha <= 0 or self.minimum_training_rows < 8:
            raise ValueError("Use positive Ridge regularization and at least eight training rows")
        origin = _quarter_start(quarter_start)
        cutoff = utc_timestamp(fit_cutoff, "fit_cutoff")
        if cutoff > origin-pd.Timedelta(hours=12):
            raise ValueError("Fit cutoff must precede the quarter's first warning issue")
        target, ids, arrays = _member_tables(weather_members, self.zones, self.weather_columns)
        calendar_values = _calendar_table(calendar, self.calendar_columns, target)
        timing = self._timing(timing_manifest, target)
        if (not isinstance(net_load, pd.DataFrame) or net_load.columns.has_duplicates or
                set(net_load.columns) != set(self.zones) or
                not utc_index(net_load.index, "net-load index").equals(target)):
            raise ValueError("Net-load labels must match the weather targets and three named zones")
        if not isinstance(label_available_times, pd.Series) or not utc_index(
                label_available_times.index, "label availability index").equals(target):
            raise ValueError("Supply a target-indexed Series of conservative label availability")
        availability = pd.DatetimeIndex([utc_timestamp(value, "label_available_time")
                                        for value in label_available_times])
        interval_end = target+pd.Timedelta(hours=1)
        if (availability < interval_end).any():
            raise ValueError("An hourly label cannot be available before its interval ends")
        past = (interval_end <= cutoff) & (availability <= cutoff) & (target < origin)
        if past.sum() < self.minimum_training_rows:
            raise ValueError("Insufficient past labels available at the declared cutoff")
        self.fit_cutoff_, self.quarter_start_ = cutoff, origin
        self.quarter_end_ = origin+pd.DateOffset(months=3)
        self.member_ids_, self.columns_ = ids, columns
        self.bases_, self.models_, records = {}, {}, {}
        for zone in self.zones:
            # One row per hour; no repetition of labels across ensemble members.
            response = net_load.loc[:, zone].to_numpy(dtype=float)
            if np.isinf(response[past]).any():
                raise ValueError("Infinite past net-load labels require an upstream audit")
            usable = past & np.isfinite(response)
            if usable.sum() < self.minimum_training_rows:
                raise ValueError("Insufficient usable past labels in "+zone)
            mean_weather = arrays[zone][usable].mean(axis=1)
            features = pd.DataFrame(np.column_stack([mean_weather, calendar_values[usable]]),
                                    index=target[usable], columns=columns)
            basis = ConditionalWeatherBasis(
                linear_columns=self.linear_columns, smooth_columns=self.smooth_columns,
                n_knots=self.n_knots, allow_missing=False, interactions=(),
                minimum_release_delay_hours=self.release_delay_hours,
                require_initialization=True, strict_issue_clock=True)
            design = basis.fit_transform(features, timing_manifest=timing.loc[usable].reset_index(drop=True),
                                         fit_cutoff=cutoff)
            model = Ridge(alpha=self.alpha, solver="cholesky").fit(design.to_numpy(), response[usable])
            self.bases_[zone], self.models_[zone] = basis, model
            records[zone] = {
                "hourly_training_rows": int(usable.sum()), "label_repetitions_per_hour": 1,
                "member_count_in_weather_mean": len(ids),
                "missing_past_labels_omitted": int((past & ~np.isfinite(response)).sum()),
                "last_training_interval_end": (target[usable][-1]+pd.Timedelta(hours=1)).isoformat(),
                "last_training_label_available": availability[usable].max().isoformat(),
                "basis": basis.fit_metadata_,
            }
        self.fit_metadata_ = {
            "script_version": SCRIPT_VERSION, "fit_cutoff": cutoff.isoformat(),
            "quarter_start": origin.isoformat(), "quarter_end": self.quarter_end_.isoformat(),
            "member_ids": list(ids), "zones": list(self.zones), "ridge_alpha": self.alpha,
            "weather_columns": list(self.weather_columns), "calendar_columns": list(self.calendar_columns),
            "total_input_hours": len(target), "past_available_hours": int(past.sum()),
            "future_or_unavailable_hours_excluded": int((~past).sum()), "zone_training": records,
            "rank_only": True, "marginal_cdfs_modified": False,
            "weather_mapping": "ensemble_mean_hourly_fit_same_model_applied_to_each_member",
            "actual_historical_arrival_verified": False,
            "initialization_identity_scope": "supplied_case_manifest_asserted_for_all_zones_and_members",
        }
        return self

    def predict(self, weather_members, calendar, *, timing_manifest):
        if not hasattr(self, "fit_cutoff_"):
            raise RuntimeError("Fit the rank-template model before predicting")
        target, ids, arrays = _member_tables(weather_members, self.zones, self.weather_columns)
        if set(ids) != set(self.member_ids_):
            raise ValueError("Forecast member cohort differs from the fitted cohort")
        if (target < self.quarter_start_).any() or (target >= self.quarter_end_).any():
            raise ValueError("Forecast targets must lie in the fitted policy's declared quarter")
        timing = self._timing(timing_manifest, target, fit_cutoff=self.fit_cutoff_)
        calendar_values = _calendar_table(calendar, self.calendar_columns, target)
        templates, zone_reports = [], {}
        for zone in self.zones:
            features = np.concatenate([
                arrays[zone], np.broadcast_to(calendar_values[:, None, :],
                                              (len(target), len(ids), len(self.calendar_columns)))], axis=2)
            basis = self.bases_[zone]
            feature_order = [self.columns_.index(name) for name in basis.columns_]
            design = basis.transform_member_array(features[:, :, feature_order], columns=basis.columns_, target_times=target,
                                                   timing_manifest=timing)
            values = self.models_[zone].predict(design.reshape(-1, design.shape[2])).reshape(len(target), len(ids))
            if not np.isfinite(values).all():
                raise RuntimeError("A member net-load template is nonfinite")
            spread = np.ptp(values, axis=1)
            # Numerical tolerance is relative to the case's predicted level; it
            # diagnoses ties, not a physical threshold for useful weather spread.
            tolerance = np.maximum(1e-9, 1e-10*np.maximum(np.abs(values).max(axis=1), 1.0))
            sorted_values = np.sort(values, axis=1)
            tie_pairs = np.sum(np.diff(sorted_values, axis=1) <= tolerance[:, None], axis=1)
            zero_spread = spread <= tolerance
            support_excursions = {}
            for name in self.weather_columns:
                j = basis.columns_.index(name)
                original_j = self.columns_.index(name)
                outside = ((features[:, :, original_j] < basis.input_min_[j]) |
                           (features[:, :, original_j] > basis.input_max_[j]))
                support_excursions[name] = {
                    "member_values_outside_training_mean_support": int(outside.sum()),
                    "member_fraction_outside_training_mean_support": float(outside.mean()),
                    "training_ensemble_mean_min": float(basis.input_min_[j]),
                    "training_ensemble_mean_max": float(basis.input_max_[j]),
                    "extrapolation": "constant" if name in basis.smooth_columns_ else "linear",
                }
            zone_reports[zone] = {
                "zero_spread_case_count": int(zero_spread.sum()),
                "zero_spread_case_fraction": float(zero_spread.mean()),
                "cases_with_ties": int((tie_pairs > 0).sum()),
                "adjacent_tie_pair_count": int(tie_pairs.sum()),
                "template_spread_min_mw": float(spread.min()),
                "template_spread_median_mw": float(np.median(spread)),
                "template_spread_max_mw": float(spread.max()),
                "zero_spread_target_times": [stamp.isoformat() for stamp in target[zero_spread]],
                "member_weather_support_excursions": support_excursions,
                "tie_tolerance": "max(1e-9 MW, 1e-10 * case_max_abs_predicted_MW)",
                "status": "DEGENERATE_MEMBER_RANKS" if zero_spread.any() else
                          ("TIES_REQUIRE_EXPLICIT_TIE_POLICY" if tie_pairs.any() else "FINITE_DISTINCT_RANKS"),
            }
            templates.append(values)
        diagnostics = {
            "cases": len(target), "members": len(ids), "zones": zone_reports,
            "has_degenerate_cases": any(row["zero_spread_case_count"] > 0 for row in zone_reports.values()),
            "has_tied_cases": any(row["cases_with_ties"] > 0 for row in zone_reports.values()),
            "fit_cutoff": self.fit_cutoff_.isoformat(), "quarter_start": self.quarter_start_.isoformat(),
            "rank_only": True, "marginal_cdfs_modified": False,
            "tie_handling": "not_resolved_here_checkerboard_caller_must_record_seed_or_repair_degenerate_template",
        }
        return RankTemplateResult(np.stack(templates, axis=2), target, ids, self.zones, diagnostics)


def synthetic_self_test(output=None):
    rng = np.random.default_rng(20260930)
    target = pd.date_range("2021-01-01", periods=24*105, freq="h", tz="UTC")
    member_ids = ("p01", "p02", "p03", "p04", "p05")
    weather_columns = ("temperature", "wind_speed", "solar_flux")
    calendar_columns = ("hour_sin", "hour_cos")
    weather = {}
    labels = pd.DataFrame(index=target)
    calendar = pd.DataFrame({"hour_sin": np.sin(2*np.pi*target.hour/24),
                             "hour_cos": np.cos(2*np.pi*target.hour/24)}, index=target)
    columns = pd.MultiIndex.from_product([member_ids, weather_columns], names=["member_id", "feature"])
    for k, zone in enumerate(DEFAULT_ZONES):
        mean = rng.normal(size=(len(target), 1, 3)) + k*.2
        ensemble = mean+rng.normal(scale=.3, size=(len(target), len(member_ids), 3))
        weather[zone] = pd.DataFrame(ensemble.reshape(len(target), -1), index=target, columns=columns)
        means = ensemble.mean(axis=1)
        labels[zone] = 30000+k*3000-700*means[:, 0]-500*means[:, 1]-300*means[:, 2]+250*calendar.hour_cos
        labels[zone] += rng.normal(scale=20, size=len(target))
    availability = pd.Series(target+pd.Timedelta(days=2), index=target)
    cutoff = pd.Timestamp("2021-03-25", tz="UTC")
    origin = pd.Timestamp("2021-04-01", tz="UTC")
    config = dict(weather_columns=weather_columns, calendar_columns=calendar_columns,
                  linear_columns=calendar_columns, smooth_columns=weather_columns,
                  minimum_training_rows=168)
    model = WeatherNetLoadRankTemplate(**config).fit(
        weather, calendar, labels, timing_manifest=synthetic_timing(target),
        label_available_times=availability, fit_cutoff=cutoff, quarter_start=origin)
    forecast = (target >= origin) & (target < origin+pd.Timedelta(days=2))
    future_weather = {zone: frame.loc[forecast] for zone, frame in weather.items()}
    future_calendar, future_target = calendar.loc[forecast], target[forecast]
    result = model.predict(future_weather, future_calendar, timing_manifest=synthetic_timing(future_target))
    assert result.values.shape == (48, 5, 3)
    assert not result.diagnostics["has_degenerate_cases"]
    checks = ["n_by_member_by_three_zone_shape", "one_label_per_hour_not_member_replicates",
              "past_interval_and_label_availability_cutoff", "quarter_issue_cutoff_checked"]
    for record in model.fit_metadata_["zone_training"].values():
        assert record["label_repetitions_per_hour"] == 1
        assert pd.Timestamp(record["last_training_label_available"]) <= cutoff
    altered = labels.copy()
    excluded = (availability > cutoff) | (target+pd.Timedelta(hours=1) > cutoff)
    altered.loc[excluded, :] = 1e9+rng.normal(size=(excluded.sum(), 3))
    second = WeatherNetLoadRankTemplate(**config).fit(
        weather, calendar, altered, timing_manifest=synthetic_timing(target),
        label_available_times=availability, fit_cutoff=cutoff, quarter_start=origin)
    np.testing.assert_allclose(result.values, second.predict(
        future_weather, future_calendar, timing_manifest=synthetic_timing(future_target)).values,
        rtol=0, atol=0)
    checks.append("changing_unavailable_future_labels_does_not_change_issued_predictions")
    permutation = np.array([3, 0, 4, 1, 2])
    permuted_cols = pd.MultiIndex.from_product([np.array(member_ids)[permutation], weather_columns],
                                              names=["member_id", "feature"])
    permuted = {zone: frame.loc[:, permuted_cols] for zone, frame in future_weather.items()}
    result_permuted = model.predict(permuted, future_calendar, timing_manifest=synthetic_timing(future_target))
    np.testing.assert_allclose(result.values[:, permutation], result_permuted.values, rtol=0, atol=0)
    checks.append("simultaneous_member_permutation_equivariance")
    from dependence import checkerboard_event_probabilities
    common_threshold_cdf = rng.uniform(.6, .99, size=(len(future_target), 3))
    original_cdf = common_threshold_cdf.copy()
    coupled = checkerboard_event_probabilities(common_threshold_cdf, template=result.values)
    coupled_permuted = checkerboard_event_probabilities(common_threshold_cdf, template=result_permuted.values)
    np.testing.assert_array_equal(common_threshold_cdf, original_cdf)
    for name in ("ge2", "all3", "count_probabilities"):
        np.testing.assert_allclose(coupled[name], coupled_permuted[name], rtol=0, atol=1e-15)
    checks.extend(["checkerboard_probability_invariant_to_joint_member_permutation",
                   "shared_threshold_cdfs_unchanged_by_rank_template_coupling"])
    reordered = dict(future_weather)
    reordered["FR"] = reordered["FR"].loc[:, permuted_cols]
    np.testing.assert_allclose(result.values, model.predict(
        reordered, future_calendar, timing_manifest=synthetic_timing(future_target)).values, rtol=0, atol=0)
    checks.append("cross_zone_column_order_aligned_by_member_identity")
    constant = {}
    for zone, frame in future_weather.items():
        values = frame.to_numpy().reshape(48, 5, 3)
        repeated = np.repeat(values.mean(axis=1, keepdims=True), 5, axis=1)
        constant[zone] = pd.DataFrame(repeated.reshape(48, -1), index=future_target, columns=columns)
    degenerate = model.predict(constant, future_calendar, timing_manifest=synthetic_timing(future_target))
    assert degenerate.diagnostics["has_degenerate_cases"]
    assert all(row["zero_spread_case_count"] == 48 for row in degenerate.diagnostics["zones"].values())
    checks.append("constant_member_weather_explicit_degenerate_rank_alarm")

    def rejected(call, name):
        try:
            call()
        except (ValueError, TypeError):
            checks.append(name)
        else:
            raise AssertionError(name)

    mismatched = dict(future_weather)
    wrong = future_weather["BE"].copy()
    wrong.columns = pd.MultiIndex.from_product([("p01", "p02", "p03", "p04", "wrong_id"), weather_columns],
                                               names=["member_id", "feature"])
    mismatched["BE"] = wrong
    rejected(lambda: model.predict(mismatched, future_calendar, timing_manifest=synthetic_timing(future_target)),
             "cross_zone_wrong_member_ids_rejected")
    unavailable = synthetic_timing(future_target)
    unavailable["available_time"] = future_target
    rejected(lambda: model.predict(future_weather, future_calendar, timing_manifest=unavailable),
             "future_available_weather_rejected")
    bad_init = synthetic_timing(future_target)
    bad_init["initialization_time"] -= pd.Timedelta(hours=6)
    rejected(lambda: model.predict(future_weather, future_calendar, timing_manifest=bad_init),
             "different_selected_initialization_rejected")
    rejected(lambda: WeatherNetLoadRankTemplate(**config).fit(
        weather, calendar, labels, timing_manifest=synthetic_timing(target), label_available_times=availability,
        fit_cutoff=origin, quarter_start=origin), "fit_after_first_quarter_issue_rejected")
    earlier = target < origin
    rejected(lambda: model.predict({zone: frame.loc[earlier] for zone, frame in weather.items()}, calendar.loc[earlier],
                                  timing_manifest=synthetic_timing(target[earlier])),
             "outside_declared_forecast_quarter_rejected")
    report = {
        "status": "passed", "evidence_type": "SYNTHETIC_ONLY_NO_REAL_WEATHER",
        "created_utc": datetime.now(timezone.utc).isoformat(), "checks": checks,
        "input_hours": len(target), "usable_training_hours": model.fit_metadata_["past_available_hours"],
        "forecast_hours": len(future_target), "forecast_members": len(member_ids),
        "diagnostics": result.diagnostics, "constant_member_diagnostics": degenerate.diagnostics,
        "limitation": "No real TIGGE decoding, actual member-ID provenance, empirical forecast skill or calibrated-margin comparison has run.",
    }
    if output is not None:
        path = Path(output).resolve()
        if not path.resolve().is_relative_to(_work_path().resolve()) or not path.name.startswith("weather_rank_") or path.exists():
            raise ValueError("Choose a new D: weather_rank_* report path")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args()
    if not args.self_test:
        parser.error("Only an explicit synthetic self-test is available; real fitting uses the Python API")
    print(json.dumps(synthetic_self_test(args.output), indent=2))
