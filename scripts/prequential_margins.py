"""Quarterly raw margins with quarterly (default) or daily same-F calibration.

Panel/features schema matches train_margins.py. Default quarterly mode fixes F
and G: raw fit [2019, cal_start), calibration [cal_start, cal_end), where
cal_end=origin-7 days and cal_start=cal_end-calibration_days. In optional daily
mode raw F keeps that quarterly schedule, while G uses each target day's issue
minus seven days and a rolling, raw-fit-disjoint calibration window. Outcomes
are attached only after all predictions have been formed. No outcome scores
are calculated here. Historical final/revised labels are NOT certified as
their historical publication vintages by this software.

Example development run:
  python -B prequential_margins.py --panel panel.parquet --features features.json \
      --destination ./results/prequential_dev \
      --start 2021-01-01 --end 2024-01-01 --family additive --calibration-days 180

For 2024--2025, --allow-test AND a frozen configuration matching settings,
feature specification and code hashes are required. See expected_frozen_settings.
Synthetic guards: python -B prequential_margins.py --self-test
"""

from __future__ import annotations
from release_paths import work_path as _work_path, raw_path as _raw_path

import argparse
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import pickle
import shutil
import time

import numpy as np
import pandas as pd
from scipy.stats import norm

from margins import (BetaPITCalibration, GaussianAdditiveMargin, QuantileMargin,
                     QUANTILES, quantile_ppf, raw_margin_cdf_sf, beta_cdf_sf,
                     calibrated_margin_quantiles, calibrated_margin_cdf_sf, COHERENT_CALIBRATION_VERSION)
from forecast_design import REGIONS, FixedSeasonalThreshold
from adaptive_thresholds import QuarterlyCalendarThresholds, REGULARIZATION_ALPHA
from adaptive_threshold_cache import quarter_thresholds


SCRIPT_VERSION = "1.5-quarterly-F-optional-past-only-daily-G"
CALIBRATION_WINDOWS = (30, 60, 90, 180, 365)
CALIBRATION_MODES = ("quarterly", "daily")
THRESHOLD_MODES = ("fixed", "adaptive_trend", "adaptive_calendar")
RAW_START = pd.Timestamp("2019-01-01", tz="UTC")
REFERENCE_END = pd.Timestamp("2022-01-01", tz="UTC")
TEST_START = pd.Timestamp("2024-01-01", tz="UTC")
TEST_END = pd.Timestamp("2026-01-01", tz="UTC")
TARGET_COLUMNS = ["net_load_"+zone for zone in REGIONS]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def utc_time(value):
    timestamp = pd.Timestamp(value)
    return timestamp.tz_localize("UTC") if timestamp.tzinfo is None else timestamp.tz_convert("UTC")


def _quarter_boundary(value):
    timestamp = utc_time(value)
    if timestamp.day != 1 or timestamp.month not in (1, 4, 7, 10) or timestamp != timestamp.normalize():
        raise ValueError("Prediction start/end must be UTC calendar-quarter boundaries")
    return timestamp


@dataclass(frozen=True)
class QuarterlyWindow:
    origin: pd.Timestamp
    prediction_end: pd.Timestamp
    calibration_start: pd.Timestamp
    calibration_end: pd.Timestamp

    def as_json(self):
        return {name: value.isoformat() for name, value in asdict(self).items()}


def quarter_schedule(start, end, calibration_days=180, embargo_days=7):
    begin, finish = _quarter_boundary(start), _quarter_boundary(end)
    if begin < pd.Timestamp("2021-01-01", tz="UTC") or finish > TEST_END or finish <= begin:
        raise ValueError("Supported prediction range is 2021--2025, with end exclusive")
    if calibration_days not in CALIBRATION_WINDOWS:
        raise ValueError(f"Calibration days must be a predeclared choice: {CALIBRATION_WINDOWS}")
    if embargo_days < 1:
        raise ValueError("A positive embargo is needed before the quarter's first forecast issue")
    windows = []
    origin = begin
    while origin < finish:
        cal_end = origin-pd.Timedelta(days=int(embargo_days))
        cal_start = cal_end-pd.Timedelta(days=int(calibration_days))
        if cal_start <= RAW_START:
            raise ValueError("No raw training history remains before calibration")
        first_issue = origin-pd.Timedelta(hours=12)
        if not cal_end <= first_issue:
            raise ValueError("Calibration observations extend beyond the first forecast issue")
        quarter_end = origin+pd.DateOffset(months=3)
        windows.append(QuarterlyWindow(origin, quarter_end, cal_start, cal_end))
        origin = quarter_end
    return windows


def expected_frozen_settings(features_path, family, calibration_days=180,
                             embargo_days=7, model_parameters=None,
                             minimum_raw_rows=1000, minimum_calibration_rows=100,
                             reference_min_samples=100, threshold_mode="fixed", calibration_mode="quarterly",
                             daily_calibration_days=None):
    """Return required prequential_margins section; this does not freeze anything.

    The caller's JSON must additionally set status="frozen", an ISO
    frozen_at_utc, and test_outcomes_inspected=false. A data-access/provisional
    protocol file is intentionally not sufficient to unlock test prediction.
    """
    folder = Path(__file__).resolve().parent
    if calibration_mode not in CALIBRATION_MODES:
        raise ValueError(f"calibration_mode must be one of {CALIBRATION_MODES}")
    if calibration_mode == "daily" and (embargo_days != 7 or minimum_calibration_rows < 100):
        raise ValueError("Daily G requires exactly seven-day embargo and at least 100 valid calibration rows")
    if daily_calibration_days is not None and (calibration_mode != "daily" or
            daily_calibration_days not in (30, 60) or daily_calibration_days > calibration_days):
        raise ValueError("Explicit daily_calibration_days needs daily mode and 30/60 days no greater than raw holdout")
    effective_days = (calibration_days if daily_calibration_days is None else daily_calibration_days) if calibration_mode == "daily" else None
    if threshold_mode not in THRESHOLD_MODES:
        raise ValueError(f"threshold_mode must be one of {THRESHOLD_MODES}")
    if threshold_mode != "fixed" and embargo_days != 7:
        raise ValueError("Adaptive threshold candidates retain the predeclared seven-day cutoff")
    return {
        "script_version": SCRIPT_VERSION,
        "family": family,
        "calibration_days": int(calibration_days),
        "calibration_mode": calibration_mode,
        "daily_calibration_days": daily_calibration_days,
        "raw_holdout_days": int(calibration_days),
        "effective_daily_window_days": effective_days,
        "daily_calibration_contract": None if calibration_mode == "quarterly" else {
            "raw_F_schedule": "quarterly_raw_end_origin_minus_calibration_days_minus_embargo_days",
            "issue_time": "target_UTC_day_minus_12_hours",
            "label_cutoff": "daily_issue_minus_7_days",
            "window_start_inclusive": "max_daily_cutoff_minus_effective_daily_window_days_raw_fit_end_exclusive",
            "label_qualification": "target_interval_end_at_or_before_daily_cutoff",
            "minimum_effective_rows": int(minimum_calibration_rows),
            "first_day_window_span_days": min(float(effective_days), float(calibration_days)-.5),
            "past_query_labels": "permitted_only_after_their_interval_end_satisfies_daily_cutoff",
            "raw_PIT_cache": "incremental_only_newly_available_selected_rows_no_future_raw_PIT",
            "saved_G": "one_calibrator_per_UTC_target_day_first_G_legacy_field_nonrepresentative"},
        "embargo_days": int(embargo_days),
        "raw_training_start": RAW_START.isoformat(),
        "raw_fit_excludes_calibration_window": True,
        "same_raw_model_for_calibration_and_prediction": True,
        "calibration_numerical_contract": COHERENT_CALIBRATION_VERSION,
        "query_probability_epsilon_clipping": False,
        "beta_MLE_input_clip_epsilon": 1e-8,
        "test_policy": "quarterly_past_only_parameters_strategy_locked" if calibration_mode == "quarterly" else
                       "quarterly_F_daily_G_past_only_parameters_strategy_locked",
        "model_parameters": model_parameters or {},
        "minimum_raw_rows": int(minimum_raw_rows),
        "minimum_calibration_rows": int(minimum_calibration_rows),
        "reference_min_samples": int(reference_min_samples),
        "threshold_mode": threshold_mode,
        "threshold_reference_end_exclusive": REFERENCE_END.isoformat() if threshold_mode == "fixed" else None,
        "threshold_reference_available_time": (REFERENCE_END+pd.Timedelta(days=int(embargo_days))).isoformat() if threshold_mode == "fixed" else None,
        "threshold_eligibility_clock": "target_UTC_day_minus_12_hours_at_or_after_reference_available_time" if threshold_mode == "fixed" else "quarter_frozen_threshold_fitted_from_labels_ending_origin_minus_7_days_available_at_first_daily_issue",
        "adaptive_threshold_settings": None if threshold_mode == "fixed" else {
            "trend": threshold_mode == "adaptive_trend", "history_calendar_years": 3,
            "minimum_observed_days": 365, "minimum_coverage": .8,
            "regularization_alpha": REGULARIZATION_ALPHA,
            "quantiles": [.9, .95], "quantile_ordering": "paired_prediction_rearrangement",
            "solver_time_limit_seconds_per_zone_quantile": 120., "solver_threads": 2,
            "weather_inputs_used": False},
        "feature_spec_sha256": sha256(features_path),
        "code_sha256": {name: sha256(folder/name) for name in
                        ("prequential_margins.py", "margins.py", "forecast_design.py", "adaptive_thresholds.py", "adaptive_threshold_cache.py")},
    }


def _check_test_access(windows, allow_test, frozen_config, expected):
    if not any(window.prediction_end > TEST_START for window in windows):
        return None
    if not allow_test:
        raise PermissionError("2024--2025 predictions are sealed: --allow-test is required")
    if frozen_config is None or not Path(frozen_config).is_file():
        raise PermissionError("Test prediction also requires an existing frozen configuration")
    document = json.loads(Path(frozen_config).read_text(encoding="utf-8-sig"))
    if document.get("status") != "frozen" or document.get("test_outcomes_inspected") is not False:
        raise PermissionError("Configuration is not a preinspection frozen protocol")
    if "frozen_at_utc" not in document:
        raise PermissionError("Frozen configuration lacks its audit timestamp")
    freeze_time = utc_time(document["frozen_at_utc"])
    if pd.isna(freeze_time) or freeze_time > pd.Timestamp.now(tz="UTC"):
        raise PermissionError("Invalid frozen configuration timestamp")
    actual = document.get("prequential_margins")
    if actual != expected:
        differing = sorted({key for key in expected if not isinstance(actual, dict) or actual.get(key) != expected[key]})
        raise PermissionError(f"Frozen settings/code do not match this run: {differing}")
    return {"path": str(Path(frozen_config).resolve()), "sha256": sha256(frozen_config),
            "frozen_at_utc": freeze_time.isoformat(),
            "verification": "declaration_and_hashes_checked_not_independent_proof_of_no_prior_inspection"}


def _load_features(features_path):
    specification = json.loads(Path(features_path).read_text(encoding="utf-8-sig"))
    features = specification.get("feature_columns")
    if not isinstance(features, list) or not features or len(set(features)) != len(features):
        raise ValueError("feature_columns must be a nonempty unique list")
    forbidden = ("net_load_", "pit_", "raw_pit_", "cdf_", "threshold_", "quantile_")
    if any(not isinstance(name, str) or name.startswith(forbidden) or
           name in {"event", "outcome", "target_time", "origin"} for name in features):
        raise ValueError("Response, retrospective PIT or outcome-derived columns cannot be predictors")
    return features, specification


def _load_panel(panel_path, features, end):
    frame = pd.read_parquet(panel_path)
    if "target_time" in frame.columns:
        frame = frame.set_index("target_time")
    frame.index = pd.DatetimeIndex(pd.to_datetime(frame.index, utc=True))
    if frame.index.has_duplicates:
        raise ValueError("Duplicate target timestamps")
    if (frame.index != frame.index.floor("h")).any():
        raise ValueError("Targets must be UTC hourly interval starts")
    # Do not inspect, score, or summarise later outcomes when running development.
    frame = frame.loc[(frame.index >= RAW_START) & (frame.index < end)].sort_index()
    required = list(dict.fromkeys(features+TARGET_COLUMNS))
    absent = [name for name in required if name not in frame.columns]
    if absent:
        raise ValueError(f"Missing panel columns: {absent}")
    frame = frame[required].apply(pd.to_numeric, errors="raise")
    if np.isinf(frame.to_numpy(dtype=float)).any():
        raise ValueError("Infinite panel values must be audited upstream")
    frame.index.name = "target_time"
    return frame


def _model(family, parameters):
    if family == "lightgbm":
        return QuantileMargin(**parameters)
    if family == "additive":
        return GaussianAdditiveMargin(**parameters)
    raise ValueError("family must be lightgbm or additive")


def _storage_check(destination, minimum_free_bytes=2*1024**3):
    output = Path(destination).resolve()
    if not output.is_relative_to(_work_path().resolve()):
        raise ValueError("Research outputs must remain under the configured work root")
    if not output.name.startswith("prequential_"):
        raise ValueError("Use a results directory whose name starts with prequential_")
    free = shutil.disk_usage(_work_path()).free
    if free < minimum_free_bytes:
        raise OSError(f"D: free space {free} is below {minimum_free_bytes} bytes; no alternate drive is used")
    if output.exists():
        raise FileExistsError("Choose a new prequential_ output directory; completed outputs are not overwritten")
    return output, free


def _fit_zone(fit, calibration, prediction, feature_columns, target_column,
              family, model_parameters, minimum_raw_rows, minimum_calibration_rows,
              thresholds, window, threshold_available_time):
    fit_valid = fit[feature_columns].notna().all(axis=1) & fit[target_column].notna()
    cal_valid = calibration[feature_columns].notna().all(axis=1) & calibration[target_column].notna()
    raw_data = fit.loc[fit_valid]
    cal_data = calibration.loc[cal_valid]
    if len(raw_data) < minimum_raw_rows or len(cal_data) < minimum_calibration_rows:
        raise ValueError(f"{target_column}: inadequate fit/calibration rows ({len(raw_data)}, {len(cal_data)})")
    if raw_data.index.max() >= window.calibration_start or cal_data.index.min() < window.calibration_start:
        raise AssertionError("Raw training overlaps calibration")
    if cal_data.index.max()+pd.Timedelta(hours=1) > window.calibration_end:
        raise AssertionError("Calibration label extends past its cutoff")
    raw_model = _model(family, model_parameters).fit(
        raw_data[feature_columns].to_numpy(), raw_data[target_column].to_numpy())
    raw_cal_pit, raw_cal_sf = raw_margin_cdf_sf(
        raw_model, cal_data[feature_columns].to_numpy(), cal_data[target_column].to_numpy())
    calibrator = BetaPITCalibration().fit(raw_cal_pit)
    zone = target_column.removeprefix("net_load_")
    output = pd.DataFrame(index=prediction.index)
    complete = prediction[feature_columns].notna().all(axis=1)
    good = prediction.loc[complete]
    X = good[feature_columns].to_numpy()
    raw_quantiles = raw_model.predict_quantiles(X) if len(good) and family == "lightgbm" else None
    available = good.index.normalize()-pd.Timedelta(hours=12) >= threshold_available_time
    # All predictions below use features and historical thresholds only.
    for q in (.9, .95):
        qname = f"q{int(q*100)}_{zone}"
        output["threshold_"+qname] = np.nan
        output["raw_cdf_"+qname] = np.nan
        output["cdf_"+qname] = np.nan
        if available.any():
            indices = good.index[available]
            cutoffs = thresholds[q].loc[indices, target_column].to_numpy()
            cached_quantiles = raw_quantiles[available] if raw_quantiles is not None else None
            raw, survival = raw_margin_cdf_sf(raw_model, X[available], cutoffs,
                                             raw_quantiles=cached_quantiles)
            output.loc[indices, "threshold_"+qname] = cutoffs
            output.loc[indices, "raw_cdf_"+qname] = raw
            output.loc[indices, "cdf_"+qname] = beta_cdf_sf(calibrator, raw, survival)[0]
    if len(good):
        quantiles = calibrated_margin_quantiles(raw_model, calibrator, X,
                                                 raw_quantiles=raw_quantiles)
        for j, q in enumerate(QUANTILES):
            output.loc[good.index, f"quantile_{q}_{zone}"] = quantiles[:, j]
    else:
        for q in QUANTILES:
            output[f"quantile_{q}_{zone}"] = np.nan
    # Observations are attached only after predictions; they never fit their own F/G.
    output[target_column] = prediction[target_column]
    output["raw_pit_"+zone] = np.nan
    output["pit_"+zone] = np.nan
    observed = complete & prediction[target_column].notna()
    if observed.any():
        rows = prediction.loc[observed]
        pit, survival = raw_margin_cdf_sf(raw_model, rows[feature_columns].to_numpy(),
                                          rows[target_column].to_numpy())
        output.loc[rows.index, "raw_pit_"+zone] = pit
        output.loc[rows.index, "pit_"+zone] = beta_cdf_sf(calibrator, pit, survival)[0]
    metadata = {
        "zone": zone, "fit_n": len(raw_data), "calibration_n": len(cal_data),
        "fit_observed_start": raw_data.index.min().isoformat(),
        "fit_last_target": raw_data.index.max().isoformat(),
        "calibration_observed_start": cal_data.index.min().isoformat(),
        "calibration_last_target": cal_data.index.max().isoformat(),
        "calibration_last_label_end": (cal_data.index.max()+pd.Timedelta(hours=1)).isoformat(),
        "calibrator": "two_parameter_monotone_beta_CDF",
        "calibration_numerical_contract": COHERENT_CALIBRATION_VERSION,
        "query_probability_epsilon_clipping": False,
        "beta_MLE_input_clip_epsilon": 1e-8,
        "beta_MLE_input_clip_fraction": float(np.mean((raw_cal_pit <= 1e-8) | (raw_cal_sf <= 1e-8))),
        "beta_a": float(calibrator.a_), "beta_b": float(calibrator.b_),
        "same_raw_model_for_calibration_and_prediction": True,
        "fit_calibration_overlap_rows": 0,
        "prediction_own_outcomes_used_for_fit": False,
        "prediction_complete_feature_rows": int(complete.sum()),
        "prediction_observed_label_rows": int(observed.sum()),
        "raw_calibration_pit_numerical_clip_fraction": float(np.mean((raw_cal_pit <= 1e-12) | (raw_cal_pit >= 1-1e-12))),
        "raw_calibration_pit_numerical_clip_fraction_interpretation": "compatibility diagnostic at legacy 1e-12 thresholds; raw query PIT values are not actually epsilon clipped",
    }
    return output, {"model": raw_model, "calibrator": calibrator,
                    "feature_columns": feature_columns}, metadata


def daily_calibration_window(day, raw_fit_end, calibration_days, embargo_days=7):
    day, raw_fit_end = pd.Timestamp(day), pd.Timestamp(raw_fit_end)
    if day.tzinfo is None or raw_fit_end.tzinfo is None or pd.isna(day) or pd.isna(raw_fit_end):
        raise ValueError("Daily G boundaries must be explicit timezone-aware timestamps")
    day, raw_fit_end = day.tz_convert("UTC"), raw_fit_end.tz_convert("UTC")
    if day != day.normalize() or embargo_days != 7 or calibration_days not in CALIBRATION_WINDOWS:
        raise ValueError("Daily G needs midnight UTC, seven-day embargo and declared calibration window")
    issue = day-pd.Timedelta(hours=12)
    cutoff = issue-pd.Timedelta(days=embargo_days)
    start = max(cutoff-pd.Timedelta(days=calibration_days), raw_fit_end)
    if start >= cutoff:
        raise ValueError("No disjoint daily calibration window remains")
    return day, issue, cutoff, start


def _fit_zone_daily(fit, history, prediction, feature_columns, target_column,
                    family, model_parameters, minimum_raw_rows, minimum_calibration_rows,
                    thresholds, window, threshold_available_time, calibration_days, embargo_days=7,
                    daily_calibration_days=None):
    """Fit quarterly F once and daily G using an incremental genuinely past PIT cache."""
    if minimum_calibration_rows < 100 or embargo_days != 7:
        raise ValueError("Daily G requires minimum 100 rows and exactly seven-day embargo")
    if daily_calibration_days is not None and (daily_calibration_days not in (30, 60) or daily_calibration_days > calibration_days):
        raise ValueError("Explicit daily G window must be 30/60 days no greater than raw holdout")
    effective_days = calibration_days if daily_calibration_days is None else daily_calibration_days
    fit_valid = fit[feature_columns].notna().all(axis=1)&fit[target_column].notna()
    raw_data = fit.loc[fit_valid]
    if len(raw_data) < minimum_raw_rows:
        raise ValueError(f"{target_column}: inadequate raw fit rows ({len(raw_data)})")
    raw_end = window.calibration_start
    if raw_data.index.max()+pd.Timedelta(hours=1) > raw_end:
        raise AssertionError("Raw training label interval extends beyond daily raw-fit boundary")
    raw_model = _model(family, model_parameters).fit(raw_data[feature_columns].to_numpy(),
                                                  raw_data[target_column].to_numpy())
    zone = target_column.removeprefix("net_load_")
    output = pd.DataFrame(index=prediction.index)
    complete = prediction[feature_columns].notna().all(axis=1)
    good = prediction.loc[complete]
    raw_quantiles = raw_model.predict_quantiles(good[feature_columns].to_numpy()) if len(good) and family == "lightgbm" else None
    rawq_frame = pd.DataFrame(raw_quantiles, index=good.index) if raw_quantiles is not None else None
    for q in (.9, .95):
        for prefix in ("threshold_", "raw_cdf_", "cdf_"):
            output[prefix+f"q{int(q*100)}_{zone}"] = np.nan
    for q in QUANTILES:
        output[f"quantile_{q}_{zone}"] = np.nan
    available = pd.Series(prediction.index.normalize()-pd.Timedelta(hours=12) >= threshold_available_time,
                          index=prediction.index)
    valid_history = history[feature_columns].notna().all(axis=1)&history[target_column].notna()
    cached_u = pd.Series(dtype=float)
    cached_sf = pd.Series(dtype=float)
    calibrators, records = {}, []
    for day in prediction.index.normalize().unique():
        day, issue, cutoff, start = daily_calibration_window(day, raw_end, effective_days, embargo_days)
        selected = valid_history&(history.index >= start)&(history.index+pd.Timedelta(hours=1) <= cutoff)
        cal_index = history.index[selected]
        if len(cal_index) < minimum_calibration_rows:
            raise ValueError(f"{target_column}/{day}: inadequate daily calibration rows ({len(cal_index)})")
        if cal_index.min() < raw_end or cal_index.max()+pd.Timedelta(hours=1) > cutoff:
            raise AssertionError("Daily calibration violated raw-fit or interval-end cutoff")
        new = cal_index[~cal_index.isin(cached_u.index)]
        if len(new):
            rows = history.loc[new]
            u, sf = raw_margin_cdf_sf(raw_model, rows[feature_columns].to_numpy(), rows[target_column].to_numpy())
            cached_u = pd.concat((cached_u, pd.Series(u, index=new)))
            cached_sf = pd.concat((cached_sf, pd.Series(sf, index=new)))
        u, sf = cached_u.loc[cal_index].to_numpy(), cached_sf.loc[cal_index].to_numpy()
        calibrator = BetaPITCalibration().fit(u)
        calibrators[day.isoformat()] = calibrator
        day_index = good.index[good.index.normalize() == day]
        X = good.loc[day_index, feature_columns].to_numpy()
        rawq = rawq_frame.loc[day_index].to_numpy() if rawq_frame is not None else None
        if len(day_index):
            quantiles = calibrated_margin_quantiles(raw_model, calibrator, X, raw_quantiles=rawq)
            for j, q in enumerate(QUANTILES):
                output.loc[day_index, f"quantile_{q}_{zone}"] = quantiles[:, j]
            eligible_index = day_index[available.loc[day_index].to_numpy()]
            if len(eligible_index):
                rq = rawq_frame.loc[eligible_index].to_numpy() if rawq_frame is not None else None
                for q in (.9, .95):
                    cutoff_values = thresholds[q].loc[eligible_index, target_column].to_numpy()
                    raw, survival = raw_margin_cdf_sf(raw_model, good.loc[eligible_index, feature_columns].to_numpy(),
                                                     cutoff_values, raw_quantiles=rq)
                    name = f"q{int(q*100)}_{zone}"
                    output.loc[eligible_index, "threshold_"+name] = cutoff_values
                    output.loc[eligible_index, "raw_cdf_"+name] = raw
                    output.loc[eligible_index, "cdf_"+name] = beta_cdf_sf(calibrator, raw, survival)[0]
        calendar_index = prediction.index[prediction.index.normalize() == day]
        observed = complete.loc[calendar_index]&prediction.loc[calendar_index, target_column].notna()
        record = {"zone": zone, "target_day": day.isoformat(), "issue_time": issue.isoformat(),
            "label_cutoff": cutoff.isoformat(), "window_start_inclusive": start.isoformat(),
            "raw_fit_end_exclusive": raw_end.isoformat(),
            "raw_holdout_days": int(calibration_days), "effective_daily_window_days": int(effective_days),
            "calendar_window_span_days": float((cutoff-start)/pd.Timedelta(days=1)),
            "calibration_rows": int(len(cal_index)), "first_calibration_target": cal_index.min().isoformat(),
            "last_calibration_target": cal_index.max().isoformat(),
            "last_calibration_label_end": (cal_index.max()+pd.Timedelta(hours=1)).isoformat(),
            "raw_fit_overlap_rows": 0, "raw_calibration_disjoint": True,
            "beta_a": float(calibrator.a_), "beta_b": float(calibrator.b_),
            "beta_MLE_input_clip_epsilon": 1e-8,
            "beta_MLE_input_clip_fraction": float(np.mean((u <= 1e-8)|(sf <= 1e-8))),
            "raw_PIT_cache_newly_available_rows": int(len(new)),
            "own_or_unavailable_labels_used": False, "legally_available_previous_query_labels_permitted": True,
            "prediction_calendar_rows": int(len(calendar_index)),
            "prediction_complete_feature_rows": int(len(day_index)),
            "prediction_observed_label_rows": int(observed.sum()),
            "prediction_target_start": calendar_index.min().isoformat(),
            "prediction_target_end_exclusive": (calendar_index.max()+pd.Timedelta(hours=1)).isoformat()}
        records.append(record)
    # Attach outcomes and issued PIT only after all daily predictive products.
    output[target_column] = prediction[target_column]
    output["raw_pit_"+zone] = np.nan
    output["pit_"+zone] = np.nan
    for day in prediction.index.normalize().unique():
        observed = complete&prediction[target_column].notna()&(prediction.index.normalize() == day)
        rows = prediction.loc[observed]
        if len(rows):
            rq = rawq_frame.loc[rows.index].to_numpy() if rawq_frame is not None else None
            u, sf = raw_margin_cdf_sf(raw_model, rows[feature_columns].to_numpy(), rows[target_column].to_numpy(), raw_quantiles=rq)
            output.loc[rows.index, "raw_pit_"+zone] = u
            output.loc[rows.index, "pit_"+zone] = beta_cdf_sf(calibrators[day.isoformat()], u, sf)[0]
    first = records[0]
    metadata = {"zone": zone, "calibration_mode": "daily", "fit_n": len(raw_data),
        "raw_holdout_days": int(calibration_days), "effective_daily_window_days": int(effective_days),
        "fit_observed_start": raw_data.index.min().isoformat(), "fit_last_target": raw_data.index.max().isoformat(),
        "calibration_n": first["calibration_rows"], "calibration_observed_start": first["first_calibration_target"],
        "calibration_last_target": first["last_calibration_target"],
        "calibration_last_label_end": first["last_calibration_label_end"],
        "beta_a": first["beta_a"], "beta_b": first["beta_b"],
        "legacy_beta_fields_scope": "first_daily_G_only_nonrepresentative_of_whole_quarter",
        "calibrator": "daily_two_parameter_monotone_beta_CDF", "daily_calibrator_count": len(calibrators),
        "daily_calibration": records, "calibration_numerical_contract": COHERENT_CALIBRATION_VERSION,
        "query_probability_epsilon_clipping": False, "beta_MLE_input_clip_epsilon": 1e-8,
        "same_raw_model_for_calibration_and_prediction": True, "fit_calibration_overlap_rows": 0,
        "prediction_own_outcomes_used_for_fit": False,
        "legally_available_previous_query_labels_permitted": True,
        "prediction_complete_feature_rows": int(complete.sum()),
        "prediction_observed_label_rows": int((complete&prediction[target_column].notna()).sum())}
    model = {"model": raw_model, "calibrator": calibrators[records[0]["target_day"]],
             "calibrator_scope": "first_daily_G_legacy_compatibility_nonrepresentative",
             "calibrators_by_day": calibrators, "daily_calibration_records": records,
             "calibration_mode": "daily", "feature_columns": feature_columns}
    return output, model, metadata


def run_prequential(panel_path, features_path, destination, start, end,
                    family="lightgbm", calibration_days=180, embargo_days=7,
                    model_parameters=None, allow_test=False, frozen_config=None,
                    minimum_raw_rows=1000, minimum_calibration_rows=100,
                    reference_min_samples=100, save_models=True, threshold_mode="fixed", calibration_mode="quarterly",
                    daily_calibration_days=None):
    windows = quarter_schedule(start, end, calibration_days, embargo_days)
    # Refuse sealed requests before loading any panel or feature data.
    if any(window.prediction_end > TEST_START for window in windows) and not allow_test:
        raise PermissionError("2024--2025 sealed: explicit --allow-test plus a frozen configuration required")
    if any(window.prediction_end > TEST_START for window in windows) and frozen_config is None:
        raise PermissionError("Missing frozen configuration for sealed predictions")
    model_parameters = model_parameters or {}
    features, specification = _load_features(features_path)
    expected = expected_frozen_settings(features_path, family, calibration_days, embargo_days,
                                         model_parameters, minimum_raw_rows,
                                         minimum_calibration_rows, reference_min_samples, threshold_mode, calibration_mode,
                                         daily_calibration_days)
    freeze = _check_test_access(windows, allow_test, frozen_config, expected)
    destination, free = _storage_check(destination)
    frame = _load_panel(panel_path, features, windows[-1].prediction_end)
    fixed_threshold_available_time = REFERENCE_END+pd.Timedelta(days=int(embargo_days))
    fixed_threshold_models = {}
    if threshold_mode == "fixed" and windows[-1].prediction_end > REFERENCE_END:
        reference = frame.loc[(frame.index >= RAW_START) & (frame.index < REFERENCE_END), TARGET_COLUMNS]
        for q in (.9, .95):
            fixed_threshold_models[q] = FixedSeasonalThreshold(q=q, min_samples=reference_min_samples).fit(reference)
    destination.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    manifest = {
        "status": "running", "script_version": SCRIPT_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "data_type": "SYNTHETIC_ONLY" if specification.get("synthetic") is True else "OBSERVATIONAL_INPUT_NOT_YET_EMPIRICALLY_VALIDATED",
        "settings": expected, "frozen_configuration": freeze,
        "prediction_start": windows[0].origin.isoformat(),
        "prediction_end_exclusive": windows[-1].prediction_end.isoformat(),
        "panel_sha256": sha256(panel_path), "D_free_bytes_before_run": free,
        "feature_columns": features, "quarters": [],
        "forward_PIT_contract": "F and G both fitted strictly before each forecast's target quarter" if calibration_mode == "quarterly" else
            "Quarterly raw F fixed; each target UTC day uses its own G fitted from interval-END-qualified labels before daily issue minus seven days; PIT and every predictive product share that exact F/G; legally available earlier query labels may calibrate later days",
        "calibration_mode": calibration_mode,
        "threshold_mode": threshold_mode,
        "threshold_reference_available_time": fixed_threshold_available_time.isoformat() if threshold_mode == "fixed" else None,
        "threshold_prediction_eligibility": expected["threshold_eligibility_clock"],
        "pre_availability_event_probability_policy": "NaN until the chosen threshold is available at the daily forecast issue; forward PIT remains valid",
        "limitations": [
            "Raw F excludes the latest calibration_days plus embargo_days of labels.",
            "Same-F calibration prevents model/calibrator refit mismatch but does not guarantee future or conditional calibration.",
            "Short calibration windows may transfer poorly between seasons; choose only on 2022--2023 development.",
            "Upstream feature publication timing and historical revised-label vintages require separate audits.",
            "2024--2025 past labels may update later quarterly F and, only in daily mode, later daily G under the frozen cutoff strategy.",
            "Threshold information timing uses declared cutoff/availability assumptions; historical observation revisions are not thereby certified as known.",
        ],
    }
    manifest_path = destination/"prequential_manifest.json"
    tables = []
    try:
        for window in windows:
            fit = frame.loc[(frame.index >= RAW_START) & (frame.index < window.calibration_start)]
            calibration = frame.loc[(frame.index >= window.calibration_start) & (frame.index < window.calibration_end)]
            expected_index = pd.date_range(window.origin, window.prediction_end, freq="h", inclusive="left")
            if threshold_mode == "fixed":
                threshold_available_time = fixed_threshold_available_time
                thresholds = {q: model.predict(expected_index) for q, model in fixed_threshold_models.items()}
                threshold_metadata = {"mode": "fixed", "reference_end_exclusive": REFERENCE_END.isoformat(),
                                      "available_at_issue": threshold_available_time.isoformat(),
                                      "predictive_outcomes_used": False}
            else:
                thresholds, threshold_fit, threshold_diagnostics, memo = quarter_thresholds(
                    frame[TARGET_COLUMNS], window.origin, trend=threshold_mode == "adaptive_trend",
                    zones=tuple(TARGET_COLUMNS), min_coverage=.8, solver_time_limit=120.)
                threshold_available_time = window.origin-pd.Timedelta(hours=12)
                threshold_metadata = {"mode": threshold_mode, "fit": threshold_fit, "memo": memo,
                                      "prediction": threshold_diagnostics,
                                      "available_at_issue": threshold_available_time.isoformat(),
                                      "availability_assumption": "Fit can use audited labels ending origin minus seven days before first daily issue; historical revised-label release vintages are not certified",
                                      "predictive_outcomes_used": False}
            prediction = frame.reindex(expected_index)
            prediction.index.name = "target_time"
            table = pd.DataFrame(index=expected_index)
            table.index.name = "target_time"
            table["origin"] = window.origin
            table["issue_time"] = table.index.normalize()-pd.Timedelta(hours=12)
            table["feature_complete"] = prediction[features].notna().all(axis=1)
            table["threshold_available"] = table.issue_time >= threshold_available_time
            if calibration_mode == "daily":
                table["calibration_mode"] = "daily"
                table["calibration_target_day"] = table.index.normalize()
            quarter = {**window.as_json(), "raw_fit_start": RAW_START.isoformat(),
                       "raw_fit_end_exclusive": window.calibration_start.isoformat(),
                       "first_forecast_issue": (window.origin-pd.Timedelta(hours=12)).isoformat(),
                       "calibration_mode": calibration_mode,
                       "calibration_schedule_window_interpretation": "actual_quarter_fixed_G_window" if calibration_mode == "quarterly" else
                            "raw_F_exclusion_schedule_only_actual_G_windows_recorded_per_day_per_zone",
                       "thresholds": threshold_metadata,
                       "zones": {}}
            label = window.origin.strftime("%Y%m%d")
            for target_column in TARGET_COLUMNS:
                if calibration_mode == "quarterly":
                    zone_output, model, metadata = _fit_zone(
                        fit, calibration, prediction, features, target_column, family,
                        model_parameters, minimum_raw_rows, minimum_calibration_rows,
                        thresholds, window, threshold_available_time)
                else:
                    zone_output, model, metadata = _fit_zone_daily(
                        fit, frame, prediction, features, target_column, family,
                        model_parameters, minimum_raw_rows, minimum_calibration_rows,
                        thresholds, window, threshold_available_time, calibration_days, embargo_days, daily_calibration_days)
                table = table.join(zone_output)
                zone = metadata["zone"]
                quarter["zones"][zone] = metadata
                if save_models:
                    model_path = destination/f"prequential_{label}_{family}_{zone}.pickle"
                    with model_path.open("wb") as stream:
                        pickle.dump({**model, "window": window.as_json(), "metadata": metadata}, stream)
                    metadata["model_file"] = str(model_path)
                    metadata["model_sha256"] = sha256(model_path)
            shard = destination/f"prequential_{label}_{family}_shared_margins.parquet"
            table.to_parquet(shard)
            quarter["prediction_file"] = str(shard)
            quarter["prediction_sha256"] = sha256(shard)
            quarter["prediction_rows"] = len(table)
            manifest["quarters"].append(quarter)
            tables.append(table)
            manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8")
            print(json.dumps({"stage": "quarter_completed", "origin": window.origin.isoformat(),
                              "family": family, "prediction_rows": len(table),
                              "data_type": manifest["data_type"]}), flush=True)
        combined = pd.concat(tables).sort_index()
        output = destination/f"prequential_{family}_shared_margins.parquet"
        combined.to_parquet(output)
        manifest.update(status="completed", elapsed_seconds=time.monotonic()-started,
                        prediction_path=str(output), prediction_sha256=sha256(output),
                        prediction_rows=len(combined))
    except Exception as error:
        manifest.update(status="failed", error_type=type(error).__name__, error=str(error),
                        elapsed_seconds=time.monotonic()-started)
        manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8")
        raise
    manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8")
    return manifest


def synthetic_self_test(destination=None):
    """Small real executions plus chronology tests; no empirical power-data claim."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    base = Path(destination or str(_work_path() / f"results/prequential_selftest_{stamp}"))
    base, free = _storage_check(base)
    base.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    checks = []
    schedule = quarter_schedule("2022-01-01", "2022-04-01", 180, 7)
    window = schedule[0]
    assert window.calibration_end == pd.Timestamp("2021-12-25", tz="UTC")
    assert window.calibration_start == window.calibration_end-pd.Timedelta(days=180)
    assert window.calibration_end < window.origin-pd.Timedelta(hours=12)
    checks.append("quarter_fit_calibration_and_first_issue_boundaries")
    for kwargs in ({}, {"allow_test": True}):
        try:
            run_prequential("NONEXISTENT_PANEL", "NONEXISTENT_FEATURES", base/"prequential_forbidden",
                            "2024-01-01", "2024-04-01", **kwargs)
            raise AssertionError("Sealed guard unexpectedly accepted an unauthorised run")
        except PermissionError:
            pass
    checks.append("sealed_test_denied_before_any_data_read")
    times = pd.date_range("2019-01-01", "2022-04-01", freq="h", inclusive="left", tz="UTC")
    rng = np.random.default_rng(20260929)
    hour = times.hour.to_numpy()/24
    season = (times.dayofyear.to_numpy()-1)/365.2425
    synthetic = pd.DataFrame({"target_time": times, "hour_sin": np.sin(2*np.pi*hour),
                              "hour_cos": np.cos(2*np.pi*hour), "season_sin": np.sin(2*np.pi*season),
                              "forecast_temperature_proxy": rng.normal(size=len(times))})
    for index, target in enumerate(TARGET_COLUMNS):
        synthetic[target] = (100+10*index+12*synthetic.hour_sin
                             +8*synthetic.season_sin+4*synthetic.forecast_temperature_proxy
                             +rng.normal(scale=4+index, size=len(times)))
    panel = base/"prequential_synthetic_panel.parquet"
    synthetic.to_parquet(panel, index=False)
    features = base/"prequential_synthetic_features.json"
    features.write_text(json.dumps({"synthetic": True, "feature_columns":
                                   ["hour_sin", "hour_cos", "season_sin", "forecast_temperature_proxy"]}), encoding="utf-8")
    expected = expected_frozen_settings(features, "additive")
    fake = base/"prequential_fake_frozen.json"
    fake.write_text(json.dumps({"status": "frozen", "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
                               "test_outcomes_inspected": False, "prequential_margins": {**expected, "calibration_days": 90}}), encoding="utf-8")
    try:
        _check_test_access(quarter_schedule("2024-01-01", "2024-04-01"), True, fake, expected)
        raise AssertionError("Configuration mismatch was not rejected")
    except PermissionError:
        pass
    fake.write_text(json.dumps({"status": "frozen", "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
                               "test_outcomes_inspected": False, "prequential_margins": expected}), encoding="utf-8")
    _check_test_access(quarter_schedule("2024-01-01", "2024-04-01"), True, fake, expected)
    checks.append("freeze_settings_and_code_hash_contract")
    run1 = run_prequential(panel, features, base/"prequential_additive",
                            "2021-10-01", "2022-04-01", family="additive",
                            model_parameters={"alpha": 10.0, "n_knots": 4}, save_models=False)
    table1 = pd.read_parquet(run1["prediction_path"])
    reference_available_time = REFERENCE_END+pd.Timedelta(days=7)
    first_eligible_target = reference_available_time.normalize()+pd.Timedelta(days=1)
    earlier = table1.index.normalize()-pd.Timedelta(hours=12) < reference_available_time
    assert table1.loc[earlier, ["pit_"+zone for zone in REGIONS]].notna().all().all()
    assert table1.loc[earlier, ["cdf_q90_"+zone for zone in REGIONS]].isna().all().all()
    assert table1.loc[~earlier, ["cdf_q90_"+zone for zone in REGIONS]].notna().all().all()
    assert table1.loc[first_eligible_target-pd.Timedelta(hours=1), "threshold_available"] == False
    assert table1.loc[first_eligible_target, "threshold_available"] == True
    checks.append("forward_PIT_valid_but_reference_CDF_requires_availability_at_daily_issue")
    for quarter in run1["quarters"]:
        for record in quarter["zones"].values():
            assert utc_time(record["fit_last_target"]) < utc_time(quarter["calibration_start"])
            assert utc_time(record["calibration_last_label_end"]) <= utc_time(quarter["calibration_end"])
            assert record["same_raw_model_for_calibration_and_prediction"] is True
            assert record["prediction_own_outcomes_used_for_fit"] is False
    checks.append("same_F_fit_calibration_prediction_metadata")
    # Corrupt ONLY prediction-quarter outcomes; probabilities must remain exact.
    changed = synthetic.copy()
    future = changed.target_time >= REFERENCE_END
    changed.loc[future, TARGET_COLUMNS] += 500
    changed.loc[changed.target_time == first_eligible_target, "net_load_FR"] = np.nan
    altered_panel = base/"prequential_synthetic_changed_future.parquet"
    changed.to_parquet(altered_panel, index=False)
    run2 = run_prequential(altered_panel, features, base/"prequential_future_changed",
                            "2022-01-01", "2022-04-01", family="additive",
                            model_parameters={"alpha": 10.0, "n_knots": 4}, save_models=False)
    table2 = pd.read_parquet(run2["prediction_path"])
    prediction_columns = [name for name in table2 if name.startswith(("cdf_", "raw_cdf_", "quantile_", "threshold_"))]
    np.testing.assert_allclose(table1.loc[table2.index, prediction_columns].to_numpy(dtype=float),
                               table2[prediction_columns].to_numpy(dtype=float),
                               rtol=0, atol=1e-12, equal_nan=True)
    assert not np.allclose(table1.loc[table2.index, "pit_FR"], table2.pit_FR)
    assert np.isnan(table2.loc[first_eligible_target, "pit_FR"])
    assert np.isfinite(table2.loc[first_eligible_target, "cdf_q90_FR"])
    checks.append("future_outcome_values_or_missingness_change_PIT_but_not_own_forecasts")
    lightgbm = run_prequential(panel, features, base/"prequential_lightgbm_smoke",
                                "2022-01-01", "2022-04-01", family="lightgbm",
                                model_parameters={"n_estimators": 5, "num_leaves": 7,
                                                  "min_child_samples": 50, "n_jobs": 1},
                                save_models=False)
    smoke = pd.read_parquet(lightgbm["prediction_path"])
    eligible = smoke.index.normalize()-pd.Timedelta(hours=12) >= reference_available_time
    assert smoke.loc[~eligible, ["cdf_q90_"+zone for zone in REGIONS]].isna().all().all()
    values = smoke.loc[eligible, ["cdf_q90_"+zone for zone in REGIONS]].to_numpy()
    assert np.isfinite(values).all() and ((values > 0) & (values < 1)).all()
    checks.append("both_additive_and_lightgbm_end_to_end_synthetic_smoke")
    for mode in ("adaptive_calendar", "adaptive_trend"):
        original_run = run_prequential(panel, features, base/f"prequential_{mode}_original",
                                       "2022-01-01", "2022-04-01", family="additive",
                                       model_parameters={"alpha": 10.0, "n_knots": 4},
                                       save_models=False, threshold_mode=mode)
        altered_run = run_prequential(altered_panel, features, base/f"prequential_{mode}_changed_future",
                                      "2022-01-01", "2022-04-01", family="additive",
                                      model_parameters={"alpha": 10.0, "n_knots": 4},
                                      save_models=False, threshold_mode=mode)
        original = pd.read_parquet(original_run["prediction_path"])
        altered = pd.read_parquet(altered_run["prediction_path"])
        columns = [name for name in original if name.startswith(("cdf_", "raw_cdf_", "quantile_", "threshold_"))]
        np.testing.assert_allclose(original[columns].to_numpy(dtype=float), altered[columns].to_numpy(dtype=float),
                                   rtol=0, atol=1e-12, equal_nan=True)
        assert original.threshold_available.all()
        for zone in REGIONS:
            assert np.isfinite(original[f"cdf_q90_{zone}"]).all()
            assert (original[f"threshold_q90_{zone}"] <= original[f"threshold_q95_{zone}"]).all()
        for quarter in original_run["quarters"]:
            fit = quarter["thresholds"]["fit"]
            assert utc_time(fit["history_end_exclusive"]) == utc_time(quarter["origin"])-pd.Timedelta(days=7)
            assert utc_time(fit["history_end_exclusive"]) < utc_time(quarter["first_forecast_issue"])
            assert fit["weather_inputs_used"] is False and quarter["thresholds"]["predictive_outcomes_used"] is False
            for zone_metadata in fit["zones"].values():
                assert utc_time(zone_metadata["last_label_end"]) <= utc_time(fit["history_end_exclusive"])
        assert np.isnan(altered.loc[first_eligible_target, "pit_FR"])
        assert np.isfinite(altered.loc[first_eligible_target, "cdf_q90_FR"])
        assert original_run["settings"]["threshold_mode"] == mode
        assert "adaptive_thresholds.py" in original_run["settings"]["code_sha256"]
        checks.append(mode+"_past_only_ordered_thresholds_and_future_outcomes_guard")
    report = {"status": "passed", "data_type": "SYNTHETIC_ONLY_NO_REAL_ENERGY_DATA",
              "created_utc": datetime.now(timezone.utc).isoformat(),
              "checks": checks, "n_checks": len(checks), "D_free_bytes_before_run": free,
              "script_sha256": sha256(Path(__file__)),
              "margin_script_sha256": sha256(Path(__file__).with_name("margins.py")),
              "forecast_design_sha256": sha256(Path(__file__).with_name("forecast_design.py")),
              "adaptive_thresholds_sha256": sha256(Path(__file__).with_name("adaptive_thresholds.py")),
              "synthetic_panel_sha256": sha256(panel), "synthetic_features_sha256": sha256(features),
              "synthetic_panel_rows": len(synthetic), "elapsed_seconds": time.monotonic()-started,
              "warning": "Computational and chronology checks only; calibration and skill on real data remain untested."}
    report_path = base/"prequential_selftest_report.json"
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"status": "passed", "n_checks": len(checks), "report": str(report_path),
                      "elapsed_seconds": report["elapsed_seconds"]}), flush=True)
    return report


def daily_calibration_self_test(destination=None):
    """Short SYNTHETIC executions; no observational panels or empirical scores."""
    from copy import deepcopy
    import ast
    import inspect
    from joint_forward_runner import _current_marginal_contract, _provenance_records, _validate_provenance
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    base, free = _storage_check(destination or str(_work_path(f"results/prequential_daily_selftest_{stamp}")))
    base.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    source_folder = Path(__file__).resolve().parent
    source_hashes = {name: sha256(source_folder/name) for name in
                     ("prequential_margins.py", "joint_forward_runner.py", "margins.py")}
    checks, family_records = [], []
    report_path = base/"prequential_daily_selftest_report.json"
    report = {"status": "running", "data_type": "SYNTHETIC_ONLY", "real_test_values_reads": 0,
              "source_sha256": source_hashes, "D_free_bytes_before_run": free}
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    try:
        rng = np.random.default_rng(2026093007)
        names = [f"x{i}" for i in range(6)]
        origin = pd.Timestamp("2021-01-01", tz="UTC")
        raw_end = origin-pd.Timedelta(days=67)
        finish = origin+pd.Timedelta(days=22)
        window = QuarterlyWindow(origin, finish, raw_end, origin-pd.Timedelta(days=7))
        fit_index = pd.date_range(raw_end-pd.Timedelta(hours=600), raw_end, freq="h", inclusive="left")
        X = rng.normal(size=(len(fit_index), 6))
        fit = pd.DataFrame(X, index=fit_index, columns=names)
        fit["net_load_FR"] = 1000+150*X[:, 0]+40*rng.normal(size=len(fit_index))
        index = pd.date_range(raw_end, finish, freq="h", inclusive="left")
        X = rng.normal(size=(len(index), 6))
        history = pd.DataFrame(X, index=index, columns=names)
        history["net_load_FR"] = 1000+150*X[:, 0]+80*rng.normal(size=len(index))
        history.loc[origin+pd.Timedelta(days=3, hours=4), "net_load_FR"] = np.nan
        history.loc[origin+pd.Timedelta(days=5, hours=3), "x0"] = np.nan
        query = history.loc[history.index >= origin].copy()
        calibration = history.loc[history.index < window.calibration_end]
        thresholds = {q: pd.DataFrame({"net_load_FR": np.full(len(query), value)}, index=query.index)
                      for q, value in ((.9, 1150.), (.95, 1250.))}
        feature_path = base/"prequential_synthetic_features.json"
        feature_path.write_text(json.dumps({"synthetic": True, "feature_columns": names}), encoding="utf-8")
        history.to_parquet(base/"prequential_synthetic_history.parquet")
        fit.to_parquet(base/"prequential_synthetic_raw_fit.parquet")
        parameters = {"additive": {"n_knots": 4, "mean_smooth_features": [0, 1],
                                  "mean_extrapolation": "constant", "scale_smooth_features": [0, 1]},
                      "lightgbm": {"n_estimators": 4, "num_leaves": 5, "min_child_samples": 20}}
        before_path = source_folder.parent/"results/daily_G_source_before_20260930T065020442Z/prequential_margins.py"
        if not before_path.is_file():
            raise FileNotFoundError("Original quarterly source snapshot required for this validation")
        old_tree = ast.parse(before_path.read_text(encoding="utf-8-sig"))
        old_fit = next(node for node in old_tree.body if isinstance(node, ast.FunctionDef) and node.name == "_fit_zone")
        new_fit = ast.parse(inspect.getsource(_fit_zone)).body[0]
        assert ast.dump(old_fit, include_attributes=False) == ast.dump(new_fit, include_attributes=False)
        assert inspect.signature(run_prequential).parameters["calibration_mode"].default == "quarterly"
        checks.append("existing_quarterly_fit_function_AST_identical_and_default_remains_quarterly")
        for family in ("additive", "lightgbm"):
            args = (names, "net_load_FR", family, parameters[family], 300, 100,
                    thresholds, window, origin-pd.Timedelta(hours=12))
            quarter, quarter_model, quarter_meta = _fit_zone(fit, calibration, query, *args)
            changed_query = query.copy()
            changed_query["net_load_FR"] = 1e6
            quarter_changed, _, _ = _fit_zone(fit, calibration, changed_query, *args)
            products = [column for column in quarter if column.startswith(("quantile_", "cdf_", "raw_cdf_", "threshold_"))]
            np.testing.assert_array_equal(quarter[products].to_numpy(), quarter_changed[products].to_numpy())
            daily, model, metadata = _fit_zone_daily(fit, history, query, *args, 60)
            np.testing.assert_array_equal(quarter.filter(regex="^threshold_").to_numpy(),
                                          daily.filter(regex="^threshold_").to_numpy())
            np.testing.assert_array_equal(quarter.filter(regex="^raw_cdf_").to_numpy(),
                                          daily.filter(regex="^raw_cdf_").to_numpy())
            day = origin+pd.Timedelta(days=10)
            _, _, cutoff, _ = daily_calibration_window(day, raw_end, 60)
            changed_history = history.copy()
            changed_history.loc[changed_history.index >= cutoff, "net_load_FR"] = np.nan
            changed, _, _ = _fit_zone_daily(fit, changed_history, changed_history.loc[query.index], *args, 60)
            day_rows = daily.index.normalize() == day
            np.testing.assert_array_equal(daily.loc[day_rows, products].to_numpy(), changed.loc[day_rows, products].to_numpy())
            # Exact cutoff interval starts are unavailable: their interval ends one hour later.
            assert cutoff in changed_history.index
            assert history.loc[cutoff, "net_load_FR"] != changed_history.loc[cutoff, "net_load_FR"]
            legal = history.copy()
            legal_rows = (legal.index >= origin)&(legal.index < origin+pd.Timedelta(days=5))
            legal.loc[legal_rows, "net_load_FR"] += 800
            legal_prediction, legal_model, _ = _fit_zone_daily(fit, legal, legal.loc[query.index], *args, 60)
            first_day = daily.index.normalize() == origin
            np.testing.assert_array_equal(daily.loc[first_day, products].to_numpy(), legal_prediction.loc[first_day, products].to_numpy())
            later = origin+pd.Timedelta(days=20)
            original_G = model["calibrators_by_day"][later.isoformat()]
            changed_G = legal_model["calibrators_by_day"][later.isoformat()]
            beta_delta = max(abs(original_G.a_-changed_G.a_), abs(original_G.b_-changed_G.b_))
            cdf_delta = float(np.nanmax(np.abs(daily.loc[daily.index.normalize() == later, "cdf_q90_FR"].to_numpy()-
                                                legal_prediction.loc[legal_prediction.index.normalize() == later, "cdf_q90_FR"].to_numpy())))
            assert beta_delta > 1e-5 and cdf_delta > 1e-5
            identity, quarterly_identity = 0., 0.
            for date in (origin, day, later):
                rows = query.loc[(query.index.normalize() == date)&query[names].notna().all(axis=1)]
                X = rows[names].to_numpy()
                G = model["calibrators_by_day"][date.isoformat()]
                for q in QUANTILES:
                    value = daily.loc[rows.index, f"quantile_{q}_FR"].to_numpy()
                    identity = max(identity, float(np.max(np.abs(calibrated_margin_cdf_sf(model["model"], G, X, value)[0]-q))))
                    value = quarter.loc[rows.index, f"quantile_{q}_FR"].to_numpy()
                    quarterly_identity = max(quarterly_identity, float(np.max(np.abs(
                        calibrated_margin_cdf_sf(quarter_model["model"], quarter_model["calibrator"], X, value)[0]-q))))
                observed = rows.loc[rows.net_load_FR.notna()]
                u, sf = raw_margin_cdf_sf(model["model"], observed[names].to_numpy(), observed.net_load_FR.to_numpy())
                np.testing.assert_array_equal(daily.loc[observed.index, "pit_FR"].to_numpy(), beta_cdf_sf(G, u, sf)[0])
            assert identity < 1e-9 and quarterly_identity < 1e-9
            assert metadata["daily_calibration"][0]["calendar_window_span_days"] == 59.5
            decoded = pickle.loads(pickle.dumps(model))
            assert len(decoded["calibrators_by_day"]) == 22
            assert decoded["calibrator_scope"] == "first_daily_G_legacy_compatibility_nonrepresentative"
            daily30, model30, metadata30 = _fit_zone_daily(fit, history, query, *args, 60, 7, 30)
            assert metadata30["raw_holdout_days"] == 60 and metadata30["effective_daily_window_days"] == 30
            assert metadata30["daily_calibration"][0]["calendar_window_span_days"] == 30.
            np.testing.assert_array_equal(daily.filter(regex="^raw_cdf_").to_numpy(), daily30.filter(regex="^raw_cdf_").to_numpy())
            np.testing.assert_array_equal(daily.filter(regex="^threshold_").to_numpy(), daily30.filter(regex="^threshold_").to_numpy())
            family_records.append({"family": family, "quarterly_own_full_query_mutation_prediction_bit_identity": True,
                "daily_own_future_unavailable_labels_mutation_same_day_prediction_bit_identity": True,
                "exact_cutoff_interval_start_is_unavailable": True, "legally_available_prior_query_labels_change_later_G": beta_delta,
                "legally_available_prior_query_labels_change_later_threshold_CDF": cdf_delta,
                "prior_query_mutation_cannot_change_first_day_products": True, "all17_quantile_CDF_max_abs_error": identity,
                "quarterly_all17_quantile_CDF_max_abs_error": quarterly_identity,
                "attached_PIT_uses_same_issued_day_G": True, "daily_G_pickle_count": len(decoded["calibrators_by_day"]),
                "default_daily_first_window_span_days": 59.5, "explicit_daily30_raw60_first_window_span_days": 30.,
                "daily30_raw60_keeps_raw_F_CDF_and_physical_threshold_bit_identity": True})
            for mode_daily_days, frame, meta in ((None, daily, metadata), (30, daily30, metadata30)):
                contract = expected_frozen_settings(feature_path, family, 60, model_parameters=parameters[family],
                    minimum_raw_rows=300, calibration_mode="daily", daily_calibration_days=mode_daily_days)
                shared = pd.DataFrame({"origin": origin, "feature_complete": query[names].notna().all(axis=1),
                    "calibration_mode": "daily", "calibration_target_day": query.index.normalize()}, index=query.index)
                zone_meta = {}
                for zone in REGIONS:
                    renamed = frame.rename(columns={column: column[:-2]+zone if column.endswith("FR") else column
                                                    for column in frame.columns})
                    shared = shared.join(renamed)
                    zone_meta[zone] = deepcopy(meta)
                    zone_meta[zone]["zone"] = zone
                    for item in zone_meta[zone]["daily_calibration"]:
                        item["zone"] = zone
                record = {**window.as_json(), "raw_fit_end_exclusive": raw_end.isoformat(),
                    "first_forecast_issue": (origin-pd.Timedelta(hours=12)).isoformat(), "calibration_mode": "daily",
                    "calibration_schedule_window_interpretation": "raw_F_exclusion_schedule_only_actual_G_windows_recorded_per_day_per_zone",
                    "zones": zone_meta, "prediction_rows": len(shared)}
                provenance = {"settings": contract, "calibration_mode": "daily", "quarters": [record]}
                _current_marginal_contract(provenance, feature_path)
                _provenance_records(provenance)
                _validate_provenance(shared, provenance)
                mutations = ("cutoff", "missing_day", "overlap", "label_end", "rows", "zone", "effective_window", "mode")
                for mutation in mutations:
                    bad = deepcopy(provenance)
                    item = bad["quarters"][0]["zones"]["BE"]["daily_calibration"][1]
                    if mutation == "cutoff": item["label_cutoff"] = (pd.Timestamp(item["label_cutoff"])+pd.Timedelta(hours=1)).isoformat()
                    elif mutation == "missing_day": bad["quarters"][0]["zones"]["BE"]["daily_calibration"].pop()
                    elif mutation == "overlap": item["raw_fit_overlap_rows"] = 1
                    elif mutation == "label_end": item["last_calibration_label_end"] = (pd.Timestamp(item["label_cutoff"])+pd.Timedelta(hours=1)).isoformat()
                    elif mutation == "rows": item["prediction_calendar_rows"] = 23
                    elif mutation == "zone": bad["quarters"][0]["zones"].pop("BE")
                    elif mutation == "effective_window": item["effective_daily_window_days"] = 29
                    elif mutation == "mode": bad["quarters"][0]["calibration_mode"] = "quarterly"
                    try:
                        _provenance_records(bad)
                        raise AssertionError("Corrupt daily provenance accepted: "+mutation)
                    except (ValueError, KeyError, TypeError): pass
                bad_shared = shared.iloc[:-1]
                try:
                    _validate_provenance(bad_shared, provenance)
                    raise AssertionError("Missing declared daily calendar row accepted")
                except ValueError: pass
                bad_shared = shared.copy()
                bad_shared.iloc[0, bad_shared.columns.get_loc("pit_BE")] = np.nan
                try:
                    _validate_provenance(bad_shared, provenance)
                    raise AssertionError("Daily PIT row coverage mismatch accepted")
                except ValueError: pass
                checks.append(f"{family}_daily_{mode_daily_days or 60}_all3zone_metadata_time_disjointness_and_actual_rows_10_corruption_guards")
        checks.append("two_families_same_F_G_quantile_threshold_PIT_chronology_and_explicit_daily30_raw60")
        panel_value_reads = 0
        original_loader = globals()["_load_panel"]
        def forbidden_loader(*args, **kwargs):
            nonlocal panel_value_reads
            panel_value_reads += 1
            raise AssertionError("A sealed probability/outcome value read occurred")
        globals()["_load_panel"] = forbidden_loader
        try:
            contract = expected_frozen_settings(feature_path, "additive", 60, calibration_mode="daily", daily_calibration_days=30)
            for mutation in ("missing_mode", "wrong_mode", "wrong_daily_window", "missing_contract"):
                bad = deepcopy(contract)
                if mutation == "missing_mode": bad.pop("calibration_mode")
                elif mutation == "wrong_mode": bad["calibration_mode"] = "quarterly"
                elif mutation == "wrong_daily_window": bad["effective_daily_window_days"] = 60
                elif mutation == "missing_contract": bad.pop("daily_calibration_contract")
                frozen = base/f"prequential_synthetic_rejected_{mutation}.json"
                frozen.write_text(json.dumps({"status": "frozen", "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
                    "test_outcomes_inspected": False, "prequential_margins": bad}), encoding="utf-8")
                try:
                    run_prequential("NONEXISTENT_SYNTHETIC_PANEL", feature_path, base/"prequential_never_created",
                        "2024-01-01", "2024-04-01", family="additive", calibration_days=60,
                        allow_test=True, frozen_config=frozen, calibration_mode="daily", daily_calibration_days=30)
                    raise AssertionError("Bad daily freeze accepted: "+mutation)
                except PermissionError: pass
                try:
                    _current_marginal_contract({"settings": bad}, feature_path)
                    raise AssertionError("Runner accepted bad upstream daily contract: "+mutation)
                except (PermissionError, ValueError): pass
            assert panel_value_reads == 0
        finally:
            globals()["_load_panel"] = original_loader
        checks.append("four_missing_wrong_daily_freeze_contracts_prequential_and_runner_rejected_before_0_panel_value_reads")
        for kwargs in ({"daily_calibration_days": 30}, {"calibration_mode": "daily", "calibration_days": 30, "daily_calibration_days": 60},
                       {"calibration_mode": "daily", "daily_calibration_days": 90}):
            try:
                expected_frozen_settings(feature_path, "additive", **kwargs)
                raise AssertionError("Invalid explicit daily window accepted")
            except ValueError: pass
        checks.append("explicit_daily_window_scope_and_no_larger_than_raw_holdout_guards")
        # Exercise the production public entry point and persisted provenance.
        # All observations here are generated by this test, including all of Q1.
        quarter_end = origin+pd.DateOffset(months=3)
        extension_index = pd.date_range(finish, quarter_end, freq="h", inclusive="left")
        extension_X = rng.normal(size=(len(extension_index), len(names)))
        extension = pd.DataFrame(extension_X, index=extension_index, columns=names)
        extension["net_load_FR"] = 1000+150*extension_X[:, 0]+80*rng.normal(size=len(extension_index))
        full_panel = pd.concat((fit, history, extension)).sort_index()
        for position, zone in enumerate(REGIONS):
            if zone != "FR": full_panel["net_load_"+zone] = full_panel.net_load_FR+10*position
        panel_path = base/"prequential_synthetic_fullquarter_panel.parquet"
        full_panel.to_parquet(panel_path)
        production = []
        for mode, supplied_days, label in (("quarterly", None, "quarterly_default"),
                                          ("daily", None, "daily60_raw60"), ("daily", 30, "daily30_raw60")):
            kwargs = {} if mode == "quarterly" else {"calibration_mode": mode, "daily_calibration_days": supplied_days}
            produced = run_prequential(panel_path, feature_path, base/("prequential_"+label),
                "2021-01-01", "2021-04-01", family="additive", calibration_days=60,
                model_parameters=parameters["additive"], minimum_raw_rows=300, **kwargs)
            assert produced["status"] == "completed" and produced["data_type"] == "SYNTHETIC_ONLY"
            table = pd.read_parquet(produced["prediction_path"])
            assert len(table) == 2160 and not table.threshold_available.any()
            assert table.filter(regex="^cdf_q").isna().all().all()
            _current_marginal_contract(produced, feature_path)
            _validate_provenance(table, produced)
            for zone in REGIONS:
                zone_metadata = produced["quarters"][0]["zones"][zone]
                with Path(zone_metadata["model_file"]).open("rb") as stream:
                    saved = pickle.load(stream)
                if mode == "daily":
                    assert len(saved["calibrators_by_day"]) == 90
                    assert saved["metadata"]["daily_calibrator_count"] == 90
                else:
                    assert "calibrators_by_day" not in saved
            production.append({"calibration_mode": mode, "daily_calibration_days": supplied_days,
                "status": produced["status"], "rows": len(table), "three_zone_persisted_models_validated": True,
                "manifest": str(base/("prequential_"+label)/"prequential_manifest.json"),
                "elapsed_seconds": produced["elapsed_seconds"]})
        checks.append("public_entry_point_default_quarterly_and_daily60_daily30_threezones_complete2160h_saved_G_and_runner_provenance")
        assert source_hashes == {name: sha256(source_folder/name) for name in source_hashes}
        report.update(status="passed", checks=checks, n_checks=len(checks), family_checks=family_records, production_runs=production,
            sealed_panel_value_reads=panel_value_reads, elapsed_seconds=time.monotonic()-started,
            source_before_quarterly_sha256=sha256(before_path),
            warning="Software chronology/coherence verification only; no observed calibration or forecasting improvement is established.")
    except Exception as error:
        report.update(status="failed", checks=checks, family_checks=family_records,
                      error_type=type(error).__name__, error=str(error), elapsed_seconds=time.monotonic()-started)
        report_path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
        raise
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"status": report["status"], "report": str(report_path), "elapsed_seconds": report["elapsed_seconds"]}), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel")
    parser.add_argument("--features")
    parser.add_argument("--destination")
    parser.add_argument("--start", default="2021-01-01")
    parser.add_argument("--end", default="2024-01-01", help="Exclusive quarter boundary")
    parser.add_argument("--family", choices=["lightgbm", "additive"], default="lightgbm")
    parser.add_argument("--calibration-days", type=int, choices=CALIBRATION_WINDOWS, default=180)
    parser.add_argument("--calibration-mode", choices=CALIBRATION_MODES, default="quarterly")
    parser.add_argument("--daily-calibration-days", type=int, choices=(30, 60), help="Optional daily G window; raw F still excludes --calibration-days")
    parser.add_argument("--embargo-days", type=int, default=7)
    parser.add_argument("--threshold-mode", choices=THRESHOLD_MODES, default="fixed")
    parser.add_argument("--model-parameters", help="JSON file with margin constructor parameters")
    parser.add_argument("--allow-test", action="store_true")
    parser.add_argument("--frozen-config")
    parser.add_argument("--no-save-models", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--daily-self-test", action="store_true", help="Short synthetic daily G chronology and provenance guards only")
    args = parser.parse_args()
    if args.daily_self_test:
        if args.self_test:
            parser.error("Choose either --self-test or --daily-self-test")
        daily_calibration_self_test(args.destination)
        return 0
    if args.self_test:
        synthetic_self_test(args.destination)
        return 0
    if not all((args.panel, args.features, args.destination)):
        parser.error("--panel, --features and --destination are required outside --self-test")
    parameters = json.loads(Path(args.model_parameters).read_text(encoding="utf-8-sig")) if args.model_parameters else {}
    result = run_prequential(args.panel, args.features, args.destination, args.start, args.end,
                             args.family, args.calibration_days, args.embargo_days, parameters,
                             args.allow_test, args.frozen_config, save_models=not args.no_save_models,
                             threshold_mode=args.threshold_mode, calibration_mode=args.calibration_mode,
                             daily_calibration_days=args.daily_calibration_days)
    print(json.dumps({"status": result["status"], "prediction_path": result["prediction_path"],
                      "prediction_rows": result["prediction_rows"], "elapsed_seconds": result["elapsed_seconds"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
