"""Quarterly joint forecasts from shared forward marginals under a frozen policy.

Input observations are used only after their declared availability cutoff. This
runner defaults to 2021--2023 input. 2024--2025 input requires explicit test
authorization and checked frozen settings, code, weather schema/specification,
and upstream forward-marginal provenance before any CLI label-column load.
Weather publication timing remains a supplied source-specific contract.
"""
from __future__ import annotations
from release_paths import work_path as _work_path, raw_path as _raw_path

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from conditional_features import ConditionalWeatherBasis, synthetic_timing, utc_index, utc_timestamp
from dependence import (AnalogCheckerboard, DynamicGaussianCopula, EventLogisticCalibrator, IndependentCopula,
                        IntegrationSettings, StaticGaussianCopula, StudentTCopula,
                        checkerboard_event_probabilities, correlation_from_eta, sample_elliptical_copula)
from weather_rank_template import WeatherNetLoadRankTemplate
from joint_pair_products import (PAIR_NAMES, checkerboard_pair_probabilities,
                                 fitted_model_pair_probabilities, pair_numerical_metadata)

SCRIPT_VERSION = "1.8-independent-C7-known-calendar"
ZONES = ("DE_LU", "FR", "BE")
TEST_START = pd.Timestamp("2024-01-01", tz="UTC")
PIT_START = pd.Timestamp("2021-01-01", tz="UTC")
HISTORY_END = pd.Timestamp("2026-01-01", tz="UTC")
CODE_FILES = ("joint_forward_runner.py", "conditional_features.py", "dependence.py",
              "weather_rank_template.py", "prequential_margins.py", "margins.py",
              "forecast_design.py", "adaptive_thresholds.py", "adaptive_threshold_cache.py",
              "elliptical_event_fast.py", "joint_pair_products.py")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return "inf" if value > 0 else "-inf" if value < 0 else None
    return value


def _canonical_hash(value):
    return hashlib.sha256(json.dumps(_json_safe(value), sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode("utf-8")).hexdigest()


def _code_hashes():
    folder = Path(__file__).resolve().parent
    return {name: sha256(folder/name) for name in CODE_FILES}


def _numerical_metadata(prediction, name, backend):
    method = prediction.get("numerical_method")
    if method is None:
        method = "analytic_independent" if name == "C1_independent" else "analytic_checkerboard" if name in (
            "C7_weather_checkerboard", "C8_analog_checkerboard") else backend
    return {"integration_discrepancy_max": float(np.max(prediction["numerical_error"])),
            "integration_maxpts_max": int(np.max(prediction["integration_maxpts"])),
            "integration_order_max": int(np.max(prediction.get("integration_order", np.zeros(len(prediction["ge2"]), dtype=int)))),
            "numerical_method": method,
            "numerical_roundoff_correction_count": int(prediction["roundoff_corrected"].sum())}


def weather_feature_schema(frame_or_path):
    """Logical Arrow feature schema; a path is inspected through metadata only."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    if isinstance(frame_or_path, pd.DataFrame):
        schema = pa.Schema.from_pandas(frame_or_path.iloc[:0], preserve_index=False)
    else:
        schema = pq.read_schema(frame_or_path)
    return [{"name": field.name, "type": str(field.type)} for field in schema
            if field.name != "target_time" and not field.name.startswith("__index_level_")]


def ecc_calendar_contract(settings, schema=None, feature_spec=None, *, require_spec=True):
    """Bind optional C7-only known calendar metadata without enlarging R(X)."""
    columns = list(settings.ecc_calendar_columns)
    if (len(columns) != len(set(columns)) or any(not isinstance(name, str) or not name for name in columns)
            or any(name.startswith(('net_load_', 'load_', 'lag_', 'pit_', 'raw_pit_', 'cdf_',
                                    'threshold_', 'event_', 'quantile_', 'observed_')) for name in columns)):
        raise PermissionError('ECC calendar requires distinct named known-calendar columns, without observed loads or lags')
    if schema is None:
        if feature_spec is not None:
            raise PermissionError('An independent ECC calendar specification requires its calendar schema')
        return {'mode': 'legacy_weather_feature_subset', 'columns': columns,
                'schema': None, 'schema_sha256': None, 'feature_spec_sha256': None,
                'eligibility_policy': 'same_common_query_mask; calendar_missing_is_hard_failure_requires_upstream_repair',
                'does_not_enter_R_or_C8_basis': True}
    names = [field['name'] for field in schema]
    if len(names) != len(set(names)) or set(names) != set(columns) or not columns:
        raise PermissionError('Independent ECC calendar schema must exactly match configured known-calendar columns')
    spec_sha = None
    if feature_spec is None:
        if require_spec:
            raise PermissionError('Frozen independent ECC calendar requires its own exact feature specification file')
    else:
        if not Path(feature_spec).is_file():
            raise PermissionError('Independent ECC calendar feature specification is missing')
        specification = json.loads(Path(feature_spec).read_text(encoding='utf-8-sig'))
        if specification.get('feature_columns') != columns:
            raise PermissionError('Independent ECC calendar specification must list exactly the configured ordered columns')
        spec_sha = sha256(feature_spec)
    return {'mode': 'independent_known_calendar', 'columns': columns,
            'schema': schema, 'schema_sha256': _canonical_hash(schema), 'feature_spec_sha256': spec_sha,
            'eligibility_policy': 'same_common_query_mask; calendar_missing_is_hard_failure_requires_upstream_repair',
            'does_not_enter_R_or_C8_basis': True}


def _current_marginal_contract(provenance, feature_spec):
    """Reconstruct upstream expected settings without loading observations."""
    from prequential_margins import expected_frozen_settings
    actual = provenance.get("settings") if isinstance(provenance, dict) else None
    if not isinstance(actual, dict) or feature_spec is None or not Path(feature_spec).is_file():
        raise PermissionError("Frozen test requires standard marginal manifest.settings and its feature-spec file")
    required = ("family", "calibration_days", "embargo_days", "model_parameters", "minimum_raw_rows",
                "minimum_calibration_rows", "reference_min_samples", "threshold_mode", "calibration_mode",
                "daily_calibration_days")
    if any(key not in actual for key in required):
        raise PermissionError("Incomplete upstream frozen prequential settings")
    expected = expected_frozen_settings(feature_spec, **{key: actual[key] for key in required})
    if _json_safe(actual) != _json_safe(expected):
        raise PermissionError("Upstream marginal strategy/specification/code does not match current frozen contract")
    return expected


def expected_joint_frozen_settings(settings, *, weather_schema, weather_feature_spec,
                                   marginal_provenance, marginal_feature_spec, qnames=("q90", "q95"),
                                   ecc_calendar_schema=None, ecc_calendar_feature_spec=None):
    """Construct a reviewable contract; this helper does not authorize or freeze a run."""
    settings.validate()
    if weather_feature_spec is None or not Path(weather_feature_spec).is_file():
        raise PermissionError("Frozen test requires the actual named weather-feature specification file")
    names = [field["name"] for field in weather_schema]
    if (len(names) != len(set(names)) or
            set(names) != set(settings.linear_columns+settings.smooth_columns)):
        raise PermissionError("Weather schema does not match configured named feature columns")
    calendar_contract = ecc_calendar_contract(settings, ecc_calendar_schema, ecc_calendar_feature_spec)
    marginal = _current_marginal_contract(marginal_provenance, marginal_feature_spec)
    try:
        _provenance_records(marginal_provenance)
    except (ValueError, KeyError, TypeError) as error:
        raise PermissionError("Invalid upstream quarterly provenance metadata") from error
    return {"script_version": SCRIPT_VERSION, "settings": _json_safe(asdict(settings)),
            "qnames": list(qnames), "zones": list(ZONES), "PIT_history_start": PIT_START.isoformat(),
            "maximum_input_end_exclusive": HISTORY_END.isoformat(),
            "test_policy": marginal["test_policy"],
            "marginal_calibration_mode": marginal["calibration_mode"],
            "marginal_daily_calibration_contract": marginal["daily_calibration_contract"],
            "weather_feature_schema": weather_schema, "weather_feature_schema_sha256": _canonical_hash(weather_schema),
            "weather_feature_spec_sha256": sha256(weather_feature_spec),
            "ecc_calendar_contract": calendar_contract,
            "marginal_settings_sha256": _canonical_hash(marginal),
            "marginal_feature_spec_sha256": sha256(marginal_feature_spec), "code_sha256": _code_hashes()}


def _frozen_document(path):
    if path is None or not Path(path).is_file():
        raise PermissionError("Test input requires an existing frozen configuration")
    document = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if document.get("status") != "frozen" or document.get("test_outcomes_inspected") is not False:
        raise PermissionError("Configuration is not a frozen preinspection audit declaration")
    try:
        stamp = utc_timestamp(document.get("frozen_at_utc"), "frozen_at_utc")
    except (ValueError, TypeError) as error:
        raise PermissionError("Missing or invalid timezone-aware freeze timestamp") from error
    if pd.isna(stamp) or stamp > pd.Timestamp.now(tz="UTC"):
        raise PermissionError("Freeze timestamp must be nonmissing and no later than the current clock")
    return document, stamp


def check_joint_test_access(*, required, allow_test, frozen_config, settings, weather_schema,
                            weather_feature_spec, marginal_provenance, marginal_feature_spec,
                            qnames=("q90", "q95"), ecc_calendar_schema=None, ecc_calendar_feature_spec=None):
    """Metadata-only authorization gate; CLI calls this before reading any labels.

    ``test_outcomes_inspected=false`` is an audit declaration at the freeze time.
    It is not a claim that authorized test execution never subsequently reads
    test observations, which later quarterly fits may legitimately use.
    """
    if not required:
        return None
    if not allow_test:
        raise PermissionError("2024--2025 input/prediction remains sealed: explicit --allow-test is required")
    document, stamp = _frozen_document(frozen_config)
    expected = expected_joint_frozen_settings(settings, weather_schema=weather_schema,
        weather_feature_spec=weather_feature_spec, marginal_provenance=marginal_provenance,
        marginal_feature_spec=marginal_feature_spec, qnames=qnames,
        ecc_calendar_schema=ecc_calendar_schema, ecc_calendar_feature_spec=ecc_calendar_feature_spec)
    actual = document.get("joint_forward_runner")
    if _json_safe(actual) != _json_safe(expected):
        different = sorted(key for key in expected if not isinstance(actual, dict) or
                           _json_safe(actual.get(key)) != _json_safe(expected[key]))
        raise PermissionError("Joint frozen strategy/settings/schema/spec/code mismatch: "+str(different))
    marginal = _current_marginal_contract(marginal_provenance, marginal_feature_spec)
    if _json_safe(document.get("prequential_margins")) != _json_safe(marginal):
        raise PermissionError("Joint frozen document lacks the exact shared prequential marginal contract")
    upstream = marginal_provenance.get("frozen_configuration")
    if not isinstance(upstream, dict) or not upstream.get("path") or not upstream.get("sha256"):
        raise PermissionError("Shared marginal provenance lacks its actual frozen configuration identity")
    upstream_document, upstream_stamp = _frozen_document(upstream["path"])
    if (sha256(upstream["path"]) != upstream["sha256"] or
            _json_safe(upstream_document.get("prequential_margins")) != _json_safe(marginal)):
        raise PermissionError("Shared marginal freeze file/hash/contract does not match its run provenance")
    if upstream.get("frozen_at_utc") != upstream_stamp.isoformat():
        raise PermissionError("Shared marginal provenance freeze timestamp disagrees with its frozen file")
    return {"path": str(Path(frozen_config).resolve()), "sha256": sha256(frozen_config),
            "frozen_at_utc": stamp.isoformat(),
            "label_access_contract": "CLI_checks_before_loading_labels_API_checks_before_label_use_caller_owns_initial_loading",
            "verification": "freeze_time_preinspection_declaration_and_hashes_checked_not_independent_proof",
            "later_test_observations_policy": "only_available_past_labels_may_update_later_quarters_under_locked_strategy" if marginal["calibration_mode"] == "quarterly" else
                "only_available_past_labels_may_update_quarterly_F_and_later_daily_G_under_locked_strategy",
            "upstream_frozen_configuration_sha256": upstream["sha256"]}


def _quarter(value):
    stamp = utc_timestamp(value)
    if stamp != stamp.normalize() or stamp.day != 1 or stamp.month not in (1, 4, 7, 10):
        raise ValueError("Use timezone-aware UTC calendar-quarter boundaries")
    return stamp


def _origins(index):
    return pd.DatetimeIndex([pd.Timestamp(year=value.year, month=1+3*((value.month-1)//3),
                                         day=1, tz="UTC") for value in index])


@dataclass(frozen=True)
class JointForwardSettings:
    history_years: int = 2
    embargo_days: int = 7
    minimum_pit_rows: int = 168
    l2: float = .01
    maxiter: int = 400
    df_candidates: tuple = (3, 5, 8, 15, 30, np.inf)
    linear_columns: tuple = ()
    smooth_columns: tuple = ()
    interactions: tuple = ()
    n_knots: int = 4
    release_delay_hours: float = 12.0
    n_analogs: int = 50
    checkerboard_seed: int = 2701
    include_ecc: bool = True
    ecc_members: int = 20
    ecc_weather_columns: tuple = ("t2m", "wind_speed", "marine_wind_speed", "dswrf")
    ecc_calendar_columns: tuple = ()
    ecc_linear_weather_columns: tuple = ()
    ecc_smooth_weather_columns: tuple = ("t2m", "wind_speed", "marine_wind_speed", "dswrf")
    ecc_ridge_alpha: float = 10.0
    threshold_available_time: str = "2022-01-08T00:00:00+00:00"
    include_event_baselines: bool = True
    include_pit_diagnostics: bool = True
    include_pair_products: bool = False
    pit_diagnostic_levels: tuple = (.9, .95)
    minimum_event_rows: int = 168
    minimum_each_event_class: int = 5
    event_logistic_C: float = 1.0
    integration_tolerance: float = 1e-4
    integration_backend: str = "scipy_replicated"
    integration_initial_maxpts: int = 16384
    integration_max_maxpts: int = 262144
    integration_seed: int = 91723

    def validate(self):
        if self.history_years < 1 or self.embargo_days < 1 or self.minimum_pit_rows < 8:
            raise ValueError("Invalid past-history, embargo or minimum PIT setting")
        if self.minimum_event_rows < 8 or self.minimum_each_event_class < 1 or self.event_logistic_C <= 0:
            raise ValueError("Invalid event-baseline training setting")
        if self.release_delay_hours < 0:
            raise ValueError("Release delay cannot be negative")
        if self.n_analogs < 2 or self.n_analogs > self.minimum_pit_rows or self.ecc_members < 2:
            raise ValueError("Invalid analog count or expected ECC member count")
        if self.ecc_ridge_alpha <= 0:
            raise ValueError("ECC Ridge alpha must be positive")
        if not isinstance(self.include_pair_products, bool):
            raise ValueError("Pair-product scope requires an explicit boolean flag")
        if (not isinstance(self.include_pit_diagnostics, bool) or
                not self.pit_diagnostic_levels or len(set(self.pit_diagnostic_levels)) != len(self.pit_diagnostic_levels) or
                any(level not in (.9, .95) for level in self.pit_diagnostic_levels)):
            raise ValueError("PIT diagnostics require distinct explicit levels .9/.95 and a boolean flag")
        utc_timestamp(self.threshold_available_time, "threshold_available_time")
        return IntegrationSettings(tolerance=self.integration_tolerance,
                                   backend=self.integration_backend,
                                   initial_maxpts=self.integration_initial_maxpts,
                                   max_maxpts=self.integration_max_maxpts,
                                   seed=self.integration_seed)


def _development_index(frame, name, test_authorized=False):
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(name+" must be a DataFrame")
    index = utc_index(frame.index, name+" index")
    limit = HISTORY_END if test_authorized else TEST_START
    if len(index) == 0 or index.min() < PIT_START or index.max() >= limit:
        raise ValueError(name+" is outside the authorized 2021--"+("2025" if test_authorized else "2023")+" input period")
    if not index.equals(index.floor("h")):
        raise ValueError("All case indices must be whole UTC hours")
    if frame.columns.has_duplicates:
        raise ValueError("Duplicate input columns")
    return index


def _daily_provenance(record, settings):
    """Check every declared daily G using metadata only, before label access."""
    from prequential_margins import daily_calibration_window
    origin = _quarter(record["origin"])
    finish = utc_timestamp(record["prediction_end"])
    raw_end = utc_timestamp(record["raw_fit_end_exclusive"])
    days = pd.date_range(origin, finish, freq="D", inclusive="left")
    cal_days, embargo = settings["calibration_days"], settings["embargo_days"]
    effective_days = settings["effective_daily_window_days"]
    supplied_days = settings["daily_calibration_days"]
    minimum = settings["minimum_calibration_rows"]
    if (finish != finish.normalize() or not origin < finish <= origin+pd.DateOffset(months=3) or
            cal_days not in (30, 60, 90, 180, 365) or embargo != 7 or minimum < 100 or
            settings["raw_holdout_days"] != cal_days or
            (supplied_days is not None and (supplied_days not in (30, 60) or supplied_days > cal_days)) or
            effective_days != (cal_days if supplied_days is None else supplied_days)):
        raise ValueError("Invalid daily marginal duration, embargo or minimum calibration rows")
    if (record.get("calibration_mode") != "daily" or
            record.get("calibration_schedule_window_interpretation") !=
            "raw_F_exclusion_schedule_only_actual_G_windows_recorded_per_day_per_zone" or
            raw_end != origin-pd.Timedelta(days=cal_days+embargo) or
            utc_timestamp(record["calibration_start"]) != raw_end or
            utc_timestamp(record["calibration_end"]) != origin-pd.Timedelta(days=embargo) or
            utc_timestamp(record["first_forecast_issue"]) != origin-pd.Timedelta(hours=12)):
        raise ValueError("Daily G must retain the original quarterly raw-F exclusion schedule")
    zones = record.get("zones")
    if not isinstance(zones, dict) or set(zones) != set(ZONES):
        raise ValueError("Daily G provenance must contain exactly all three zones")
    result = {}
    for zone in ZONES:
        meta = zones[zone]
        daily = meta.get("daily_calibration")
        if (meta.get("zone") != zone or meta.get("calibration_mode") != "daily" or
                meta.get("legacy_beta_fields_scope") != "first_daily_G_only_nonrepresentative_of_whole_quarter" or
                not isinstance(daily, list) or len(daily) != len(days) or
                meta.get("daily_calibrator_count") != len(days) or
                meta.get("raw_holdout_days") != cal_days or meta.get("effective_daily_window_days") != effective_days or
                meta.get("same_raw_model_for_calibration_and_prediction") is not True or
                meta.get("fit_calibration_overlap_rows") != 0 or
                meta.get("prediction_own_outcomes_used_for_fit") is not False or
                meta.get("legally_available_previous_query_labels_permitted") is not True):
            raise ValueError("Missing or inconsistent daily marginal zone/calibrator provenance")
        by_day, features, observed = {}, 0, 0
        for position, (day, item) in enumerate(zip(days, daily)):
            _, issue, cutoff, start = daily_calibration_window(day, raw_end, effective_days, embargo)
            exact_times = {"target_day": day, "issue_time": issue, "label_cutoff": cutoff,
                           "window_start_inclusive": start, "raw_fit_end_exclusive": raw_end,
                           "prediction_target_start": day,
                           "prediction_target_end_exclusive": day+pd.Timedelta(days=1)}
            if item.get("zone") != zone or any(utc_timestamp(item.get(key), key) != value
                                               for key, value in exact_times.items()):
                raise ValueError("Daily G day/issue/cutoff/window or target-row coverage mismatch")
            if item["target_day"] in by_day:
                raise ValueError("Duplicate daily G record")
            first = utc_timestamp(item["first_calibration_target"])
            last = utc_timestamp(item["last_calibration_target"])
            label_end = utc_timestamp(item["last_calibration_label_end"])
            span = float((cutoff-start)/pd.Timedelta(days=1))
            count = item["calibration_rows"]
            new = item["raw_PIT_cache_newly_available_rows"]
            calendar = item["prediction_calendar_rows"]
            complete = item["prediction_complete_feature_rows"]
            known = item["prediction_observed_label_rows"]
            counts = (count, new, calendar, complete, known)
            if (any(not isinstance(value, (int, np.integer)) or isinstance(value, bool) for value in counts) or
                    not start <= first <= last or not label_end == last+pd.Timedelta(hours=1) or
                    label_end > cutoff or first < raw_end or
                    not minimum <= count <= int((cutoff-start)/pd.Timedelta(hours=1)) or
                    count > int((last-first)/pd.Timedelta(hours=1))+1 or
                    abs(item["calendar_window_span_days"]-span) > 1e-12 or
                    item.get("raw_holdout_days") != cal_days or item.get("effective_daily_window_days") != effective_days or
                    (position == 0 and (span != min(effective_days, cal_days-.5) or new != count)) or
                    (position > 0 and not 0 <= new <= 24) or
                    item.get("raw_fit_overlap_rows") != 0 or item.get("raw_calibration_disjoint") is not True or
                    item.get("own_or_unavailable_labels_used") is not False or
                    item.get("legally_available_previous_query_labels_permitted") is not True or
                    not 0 <= known <= complete <= calendar == 24 or
                    item.get("beta_MLE_input_clip_epsilon") != 1e-8 or
                    not 0 <= item["beta_MLE_input_clip_fraction"] <= 1 or
                    not all(np.isfinite(item[key]) and item[key] > 0 for key in ("beta_a", "beta_b"))):
                raise ValueError("Daily G label interval, raw/calibration disjointness or row/parameter audit failed")
            by_day[day.isoformat()] = item
            features += complete
            observed += known
        first = daily[0]
        compatibility = {"calibration_n": "calibration_rows", "calibration_observed_start": "first_calibration_target",
                         "calibration_last_target": "last_calibration_target",
                         "calibration_last_label_end": "last_calibration_label_end", "beta_a": "beta_a", "beta_b": "beta_b"}
        if (any(meta.get(key) != first[value] for key, value in compatibility.items()) or
                meta.get("prediction_complete_feature_rows") != features or
                meta.get("prediction_observed_label_rows") != observed or
                utc_timestamp(meta["fit_last_target"])+pd.Timedelta(hours=1) > raw_end or
                meta.get("fit_n", 0) < settings["minimum_raw_rows"]):
            raise ValueError("Daily G first-calibrator compatibility fields or aggregate audit mismatch")
        result[zone] = by_day
    return result


def _provenance_records(provenance):
    """Validate quarterly F and mode-specific G metadata before any label load."""
    if not isinstance(provenance, dict) or not isinstance(provenance.get("quarters"), list):
        raise ValueError("Supply the forward marginal manifest with quarter fit/calibration windows")
    settings = provenance.get("settings", {})
    mode = settings.get("calibration_mode", "quarterly")
    if mode not in ("quarterly", "daily") or provenance.get("calibration_mode", mode) != mode:
        raise ValueError("Unknown or inconsistent upstream marginal calibration mode")
    if mode == "daily" and not isinstance(settings.get("daily_calibration_contract"), dict):
        raise ValueError("Daily G requires an explicit daily calibration contract")
    records = {}
    for record in provenance["quarters"]:
        origin = _quarter(record["origin"])
        if origin in records:
            raise ValueError("Duplicate marginal quarter in provenance")
        cal_start = utc_timestamp(record["calibration_start"])
        cal_end = utc_timestamp(record["calibration_end"])
        raw_end = utc_timestamp(record["raw_fit_end_exclusive"])
        if not raw_end <= cal_start < cal_end <= origin-pd.Timedelta(hours=12):
            raise ValueError("Marginal raw fit/calibration overlap or future fit detected")
        if record.get("calibration_mode", mode) != mode:
            raise ValueError("Marginal quarter mode disagrees with its upstream settings")
        if mode == "daily":
            _daily_provenance(record, settings)
        records[origin] = record
    if not records:
        raise ValueError("No upstream marginal quarterly provenance")
    return records


def _validate_provenance(shared, provenance):
    records = _provenance_records(provenance)
    mode = provenance.get("settings", {}).get("calibration_mode", "quarterly")
    target_origin = _origins(shared.index)
    if "origin" not in shared:
        raise ValueError("Shared table must record its marginal forecast origin")
    declared = pd.DatetimeIndex([utc_timestamp(value, "marginal origin") for value in shared.origin])
    if not declared.equals(target_origin):
        raise ValueError("Marginal origin does not match the target quarter")
    if mode == "daily":
        if not {"calibration_mode", "calibration_target_day", "feature_complete"}.issubset(shared.columns):
            raise ValueError("Daily shared table lacks its day-specific G and feature audit columns")
        day_column = pd.DatetimeIndex([utc_timestamp(value, "calibration_target_day")
                                      for value in shared.calibration_target_day])
        if (not shared.calibration_mode.eq("daily").all() or not day_column.equals(shared.index.normalize()) or
                shared.feature_complete.isna().any() or
                not shared.feature_complete.isin((True, False)).all()):
            raise ValueError("Daily shared-table mode/day/feature-completeness mismatch")
    summaries = []
    for origin in target_origin.unique():
        if origin not in records:
            raise ValueError("No marginal provenance for "+origin.isoformat())
        record = records[origin]
        cal_start = utc_timestamp(record["calibration_start"])
        cal_end = utc_timestamp(record["calibration_end"])
        raw_end = utc_timestamp(record["raw_fit_end_exclusive"])
        summary = {"origin": origin.isoformat(), "raw_fit_end_exclusive": raw_end.isoformat(),
                   "calibration_start": cal_start.isoformat(), "calibration_end": cal_end.isoformat()}
        if mode == "daily":
            daily = _daily_provenance(record, provenance["settings"])
            quarter = shared.loc[target_origin == origin]
            finish = utc_timestamp(record["prediction_end"])
            expected_index = pd.date_range(origin, finish, freq="h", inclusive="left")
            if not quarter.index.equals(expected_index) or record.get("prediction_rows", len(expected_index)) != len(expected_index):
                raise ValueError("Daily marginal shared table must retain every declared UTC calendar hour")
            for day in expected_index.normalize().unique():
                rows = quarter.loc[quarter.index.normalize() == day]
                for zone in ZONES:
                    item = daily[zone][day.isoformat()]
                    pit = rows["pit_"+zone].to_numpy(dtype=float)
                    labels = rows["net_load_"+zone].to_numpy(dtype=float)
                    actual_known = rows.feature_complete.to_numpy(dtype=bool)&np.isfinite(labels)
                    if (item["prediction_calendar_rows"] != len(rows) or
                            item["prediction_complete_feature_rows"] != int(rows.feature_complete.sum()) or
                            item["prediction_observed_label_rows"] != int(actual_known.sum()) or
                            not np.array_equal(np.isfinite(pit), actual_known)):
                        raise ValueError("Daily G per-zone/day records disagree with actual shared feature/label/PIT rows")
            summary.update(calibration_mode="daily", daily_calibrator_count_per_zone=len(daily[ZONES[0]]),
                           G_window_interpretation="actual_daily_windows_in_bound_upstream_manifest",
                           first_daily_G_cutoff=daily[ZONES[0]][origin.isoformat()]["label_cutoff"],
                           last_daily_G_cutoff=daily[ZONES[0]][(finish-pd.Timedelta(days=1)).isoformat()]["label_cutoff"],
                           daily_three_zone_time_disjointness_and_row_coverage_checked=True)
        summaries.append(summary)
    return summaries


def _training_event_mask(values, labels, minimum_rows, minimum_class):
    good = np.isfinite(values).all(axis=1) & np.isfinite(labels)
    observed = labels[good]
    counts = [int((observed == value).sum()) for value in (0, 1)]
    available = len(observed) >= minimum_rows and min(counts) >= minimum_class
    return good, {"available": bool(available), "training_rows": int(good.sum()),
                  "non_event_count": counts[0], "event_count": counts[1],
                  "status": "available" if available else "insufficient_past_forward_event_history"}


def _event_labels(shared, q):
    labels = shared[["net_load_"+zone for zone in ZONES]].to_numpy(dtype=float)
    threshold = shared[["threshold_"+q+"_"+zone for zone in ZONES]].to_numpy(dtype=float)
    complete = np.isfinite(labels).all(axis=1) & np.isfinite(threshold).all(axis=1)
    outcome = (np.sum(labels > threshold, axis=1) >= 2).astype(float)
    outcome[~complete] = np.nan
    return outcome


def _event_design(V, X):
    values = np.asarray(V, dtype=float)
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("Direct-event inputs must be finite shared CDFs in [0,1]")
    exceed = 1-values
    pair = np.column_stack([exceed[:, 0]*exceed[:, 1], exceed[:, 0]*exceed[:, 2],
                            exceed[:, 1]*exceed[:, 2]])
    return np.column_stack([exceed, pair, X])


def _prepare_member_weather(member_weather, index, timing, settings, test_authorized=False):
    """Accept actual GEFS zone-prefix wide rows or explicit zone-long rows.

    Both schemas retain target_time, member_id, initialization_time, issue_time.
    A missing member case is excluded from C7 and reported, never filled. The
    supplied case manifest is checked against every member's initialization and
    issue. It does not authenticate each historical public arrival.
    """
    if not isinstance(member_weather, pd.DataFrame):
        raise TypeError("member_weather must be a DataFrame")
    frame = member_weather.copy()
    if "target_time" not in frame:
        if frame.index.name != "target_time":
            raise ValueError("Member weather requires a target_time column or named index")
        frame = frame.reset_index()
    required = {"target_time", "member_id", "initialization_time", "issue_time"}
    if not required.issubset(frame.columns) or frame.columns.has_duplicates:
        raise ValueError("Member weather requires explicit target/member/initialization/issue columns")
    target = pd.DatetimeIndex(frame.target_time)
    if target.tz is None or target.hasnans:
        raise ValueError("Member weather target times must have explicit timezones")
    target = target.tz_convert("UTC")
    if len(target) and (target.min() < PIT_START or target.max() >= (HISTORY_END if test_authorized else TEST_START)):
        raise ValueError("Member weather is outside the authorized 2021 input onward period")
    frame["target_time"] = target
    frame = frame.loc[frame.target_time.isin(index)].copy()
    expected_ids = set(range(1, settings.ecc_members+1))
    if not set(frame.member_id.unique()).issubset(expected_ids):
        raise ValueError("Member IDs must explicitly identify the configured GEFS members 1..M")
    if frame.member_id.isna().any():
        raise ValueError("Missing member ID")
    for name in ("initialization_time", "issue_time", "available_time"):
        if name not in frame:
            continue
        values = pd.DatetimeIndex(frame[name])
        if values.tz is None or values.hasnans:
            raise ValueError("Member "+name+" must have explicit timezone")
        values = values.tz_convert("UTC")
        expected = pd.DatetimeIndex(timing[name].reindex(frame.target_time))
        if not values.equals(expected):
            raise ValueError("Member "+name+" disagrees with case-level forecast manifest")
    zones, fields = ZONES, list(settings.ecc_weather_columns)
    if not fields or len(fields) != len(set(fields)) or any(name.startswith(("net_load", "pit_", "cdf_", "event")) for name in fields):
        raise ValueError("ECC requires distinct named forecast weather fields only")
    frames, complete_zone = {}, []
    for zone in zones:
        if "zone" in frame:
            if not set(frame.zone.unique()).issubset(set(zones)):
                raise ValueError("Unexpected zone in member weather")
            values = frame.loc[frame.zone == zone]
            column_map = dict(zip(fields, fields))
        else:
            values = frame
            column_map = {zone+"_"+field: field for field in fields}
        if not set(column_map).issubset(values.columns) or values.duplicated(["target_time", "member_id"]).any():
            raise ValueError("Missing ECC weather field or duplicate target/member record")
        values = values[["target_time", "member_id"]+list(column_map)].rename(columns=column_map)
        pivot = values.pivot(index="target_time", columns="member_id", values=fields)
        pivot = pivot.swaplevel(0, 1, axis=1)
        pivot.columns.names = ["member_id", "feature"]
        columns = pd.MultiIndex.from_product([range(1, settings.ecc_members+1), fields], names=["member_id", "feature"])
        pivot = pivot.reindex(index=index, columns=columns)
        array = pivot.to_numpy(dtype=float)
        if np.isinf(array).any():
            raise ValueError("Infinite member weather requires audit")
        complete_zone.append(np.isfinite(array).all(axis=1))
        frames[zone] = pivot
    complete = np.all(np.column_stack(complete_zone), axis=1)
    return frames, complete, {"schema": "zone_long" if "zone" in frame else "GEFS_zone_prefix_wide",
        "expected_members": settings.ecc_members, "expected_member_ids": sorted(expected_ids),
        "complete_member_target_hours": int(complete.sum()), "incomplete_member_target_hours": int((~complete).sum()),
        "weather_fields": fields, "time_identity": "each_member_initialization_and_issue_match_case_manifest"}


def run_joint_forward(shared, weather_features, timing_manifest, *, label_available_times,
                      marginal_provenance, settings, start="2022-01-01T00:00:00+00:00",
                      end="2024-01-01T00:00:00+00:00", qnames=("q90", "q95"), destination=None,
                      input_files=None, member_weather=None, allow_test=False, frozen_config=None,
                      weather_feature_spec=None, marginal_feature_spec=None,
                      ecc_calendar=None, ecc_calendar_feature_spec=None):
    """Issue C1--C6 each quarter; optionally C0/C9/C10 from available history.

    Predictions use query weather and shared threshold CDFs only, never query PIT
    or net-load completeness. Event-baseline labels are constructed only on the
    cutoff-selected past subset. C9 uses this runner's earlier-quarter issued C3
    probabilities; its alert-rank-score output is forcibly the original C3.
    All methods use the same complete three-zone training PIT/weather rows.
    """
    integration = settings.validate()
    begin, finish = _quarter(start), _quarter(end)
    if begin < pd.Timestamp("2022-01-01", tz="UTC") or finish > HISTORY_END or finish <= begin:
        raise ValueError("Use quarterly forecasts within 2022--2025; input history starts in 2021")
    if not qnames or any(q not in ("q90", "q95") for q in qnames):
        raise ValueError("Use explicitly declared q90/q95 target definitions")
    if not isinstance(shared, pd.DataFrame) or not isinstance(weather_features, pd.DataFrame):
        raise TypeError("Shared margins and weather features must be DataFrames")
    # Index/schema metadata precede every label/PIT/threshold column access.
    candidate_index = utc_index(shared.index, "shared margins index")
    weather_index = utc_index(weather_features.index, "weather features index")
    if ecc_calendar is not None:
        if not isinstance(ecc_calendar, pd.DataFrame):
            raise TypeError('Independent ECC calendar must be a DataFrame')
        calendar_index = utc_index(ecc_calendar.index, 'independent ECC calendar index')
        if not calendar_index.equals(candidate_index):
            raise ValueError('Independent ECC calendar rows must match the complete shared table in order')
    elif ecc_calendar_feature_spec is not None:
        raise ValueError('Independent ECC calendar spec requires an independent calendar DataFrame')
    test_required = (finish > TEST_START or any(candidate_index >= TEST_START) or any(weather_index >= TEST_START))
    if member_weather is not None:
        member_times = member_weather["target_time"] if "target_time" in member_weather else member_weather.index
        test_required = test_required or bool((pd.DatetimeIndex(member_times) >= TEST_START).any())
    calendar_schema = weather_feature_schema(ecc_calendar) if ecc_calendar is not None else None
    calendar_contract = ecc_calendar_contract(settings, calendar_schema, ecc_calendar_feature_spec, require_spec=test_required)
    freeze = check_joint_test_access(required=test_required, allow_test=allow_test, frozen_config=frozen_config,
        settings=settings, weather_schema=weather_feature_schema(weather_features),
        weather_feature_spec=weather_feature_spec, marginal_provenance=marginal_provenance,
        marginal_feature_spec=marginal_feature_spec, qnames=qnames,
        ecc_calendar_schema=calendar_schema, ecc_calendar_feature_spec=ecc_calendar_feature_spec)
    index = _development_index(shared, "shared margins", test_authorized=freeze is not None)
    if not _development_index(weather_features, "weather features", test_authorized=freeze is not None).equals(index):
        raise ValueError("Weather rows must match the complete shared table in order")
    calendar_source = weather_features if ecc_calendar is None else ecc_calendar
    if ecc_calendar is not None:
        _development_index(ecc_calendar, 'independent ECC calendar', test_authorized=freeze is not None)
    if not set(settings.ecc_calendar_columns).issubset(set(calendar_source.columns)):
        raise ValueError('ECC calendar columns must be present in the configured calendar source')
    if not np.isfinite(calendar_source.loc[:, list(settings.ecc_calendar_columns)].to_numpy(dtype=float)).all():
        raise ValueError('Incomplete ECC calendar requires upstream common-input repair; no silent C7-only mask')
    required = ["pit_"+zone for zone in ZONES]+["net_load_"+zone for zone in ZONES]
    for q in qnames:
        required += ["cdf_"+q+"_"+zone for zone in ZONES]
        required += ["threshold_"+q+"_"+zone for zone in ZONES]
    if any(column not in shared for column in required):
        raise ValueError("Missing shared PIT/CDF/label/threshold columns")
    feature_columns = list(settings.linear_columns)+list(settings.smooth_columns)
    if (set(weather_features.columns) != set(feature_columns) or len(feature_columns) != len(set(feature_columns)) or
            any(name.startswith(("net_load_", "pit_", "cdf_", "threshold_", "event_")) for name in feature_columns)):
        raise ValueError("Exact named forecast-only feature specification required")
    feature_values = weather_features.loc[:, feature_columns].to_numpy(dtype=float)
    if np.isinf(feature_values).any():
        raise ValueError("Infinite weather features require upstream audit")
    if not isinstance(label_available_times, pd.Series) or not utc_index(label_available_times.index).equals(index):
        raise ValueError("Supply an aligned conservative label-availability Series")
    availability = pd.DatetimeIndex([utc_timestamp(value, "label_available_time") for value in label_available_times])
    if (availability < index+pd.Timedelta(hours=1)).any():
        raise ValueError("A label cannot be available before its target interval ends")
    provenance = _validate_provenance(shared, marginal_provenance)
    # Validate feature chronology independently of labels; no D-3 assumption.
    from conditional_features import validate_timing_manifest
    timing = validate_timing_manifest(timing_manifest, index, require_initialization=True,
                                     minimum_release_delay_hours=settings.release_delay_hours,
                                     strict_issue_clock=True)
    member_frames, member_complete, member_schema = None, None, None
    if settings.include_ecc and member_weather is not None:
        member_frames, member_complete, member_schema = _prepare_member_weather(
            member_weather, index, timing, settings, test_authorized=freeze is not None)
    threshold_available = utc_timestamp(settings.threshold_available_time)
    output = pd.DataFrame(index=index[(index >= begin) & (index < finish)])
    output.index.name = "target_time"
    output["issue_time"] = output.index.normalize()-pd.Timedelta(hours=12)
    output["copula_origin"] = _origins(output.index)
    names = ["C1_independent", "C2_static_gaussian", "C3_dynamic_gaussian", "C4_static_t_shared_R",
             "C5_dynamic_t_shared_R", "C6_dynamic_t_refit_R"]
    names += ["C7_weather_checkerboard", "C8_analog_checkerboard"]
    dependence_names = names.copy()
    if settings.include_event_baselines:
        names += ["C0_climatology", "C9_event_calibrated_C3", "C10_direct_logistic"]
    for q in qnames:
        for name in names:
            output[name+"_"+q] = np.nan
            if name in dependence_names:
                output[name+"_"+q+"_all3"] = np.nan
        output["forecast_eligible_"+q] = False
        if settings.include_event_baselines:
            output["C9_"+q+"_alert_rank_score_from_C3"] = np.nan
    if settings.include_pit_diagnostics:
        for level in settings.pit_diagnostic_levels:
            level_name = "u"+str(int(round(100*level)))
            for name in dependence_names:
                column = "diagnostic_pit_"+name+"_"+level_name
                output[column] = np.nan
                output[column+"_all3"] = np.nan
            output["diagnostic_pit_forecast_eligible_"+level_name] = False
    if settings.include_pair_products:
        pair_defaults = {"pair_"+name+"_"+q+"_"+pair_name: np.nan
            for q in qnames for name in dependence_names for pair_name in PAIR_NAMES}
        output = pd.concat((output, pd.DataFrame(pair_defaults, index=output.index)), axis=1)
    code_hashes = _code_hashes()
    manifest = {"script_version": SCRIPT_VERSION, "created_utc": datetime.now(timezone.utc).isoformat(),
                "status": "running", "scope": "frozen_policy_2024_2025_authorized" if freeze else "development_2022_2023_test_not_opened",
                "frozen_configuration": freeze, "settings": _json_safe(asdict(settings)),
                "start": begin.isoformat(), "end_exclusive": finish.isoformat(), "qnames": list(qnames),
                "models": names, "code_sha256": code_hashes, "marginal_provenance_windows": provenance,
                "input_files": {str(path): sha256(path) for path in (input_files or [])}, "quarters": [],
                "timing_evidence": "supplied_manifest_checked_actual_historical_arrival_not_verified",
                "C9_alert_ranking_policy": "use_C9_q*_alert_rank_score_from_C3_column_original_C3_even_if_calibration_clips_or_ties",
                "upstream_shared_CDF_unchanged": True}
    manifest["marginal_calibration_mode"] = marginal_provenance.get("settings", {}).get("calibration_mode", "quarterly")
    manifest["upstream_forward_PIT_contract"] = marginal_provenance.get("forward_PIT_contract")
    manifest["physical_all3_policy"] = "C1_C8_secondary_from_same_ge2_result_no_additional_integration"
    manifest["physical_pair_policy"] = {"enabled": settings.include_pair_products,
        "pairs": list(PAIR_NAMES), "models": dependence_names, "thresholds": list(qnames),
        "no_additional_fit_or_parameter_selection": True, "shared_CDF_and_fitted_models_unchanged": True,
        "eligibility": "same_common_three_zone_physical_event_query_mask",
        "PIT_pair_diagnostics": False,
        "integration": "analytic_independent_checkerboard_otherwise_separate_pair_CDF_same_backend_no_primary_result_mutation"}
    manifest["PIT_diagnostic_policy"] = {
        "enabled": settings.include_pit_diagnostics, "levels": list(settings.pit_diagnostic_levels),
        "query": "constant_V_each_zone_equals_declared_u_same_fitted_C1_C8_models_on_declared_marginal_feature_complete_rows",
        "prediction_uses_future_PIT": False,
        "observed_labels": "evaluation_only_from_shared_issued_PIT_exceedances_not_constructed_by_runner",
        "physical_event_separation": "diagnostic_pit_prefix_not_seasonal_net_load_event_or_operational_warning",
        "conditional_marginal_boundary": "global_PIT_uniformity_does_not_prove_weather_conditional_PIT_uniformity"}
    manifest["forecast_input_eligibility"] = "Declared shared marginal feature_complete and threshold availability; incomplete known inputs may suppress every method; nonfinite CDF at declared available forecast is a hard failure; query outcome/PIT completeness never sets forecast eligibility"
    manifest["member_weather_schema"] = member_schema
    manifest["C7_history_policy"] = "same_two_year_PIT_complete_history_from_2021_only_not_extra_2019_observations"
    manifest["C7_calendar_contract"] = calendar_contract
    target_destination = None
    if destination is not None:
        target_destination = Path(destination).resolve()
        if not target_destination.resolve().is_relative_to(_work_path().resolve()) or not target_destination.name.startswith("joint_runner_") or target_destination.exists():
            raise ValueError("Choose a NEW D: joint_runner_* destination directory")
        target_destination.mkdir(parents=True)
    origin = begin
    try:
        while origin < finish:
            quarter_end = origin+pd.DateOffset(months=3)
            cutoff = origin-pd.Timedelta(days=settings.embargo_days)
            history_start = max(PIT_START, origin-pd.DateOffset(years=settings.history_years))
            past = ((index >= history_start) & (index+pd.Timedelta(hours=1) <= cutoff) & (availability <= cutoff))
            U = shared[["pit_"+zone for zone in ZONES]].to_numpy(dtype=float)
            complete = past & np.isfinite(U).all(axis=1) & np.isfinite(feature_values).all(axis=1)
            if complete.sum() < settings.minimum_pit_rows:
                raise ValueError("Insufficient common three-zone past PIT history for "+origin.isoformat())
            if ((U[complete] < 0) | (U[complete] > 1)).any():
                raise ValueError("Training PIT outside [0,1]")
            basis = ConditionalWeatherBasis(
                linear_columns=settings.linear_columns, smooth_columns=settings.smooth_columns,
                interactions=settings.interactions, n_knots=settings.n_knots, allow_missing=False,
                minimum_release_delay_hours=settings.release_delay_hours)
            X_train = basis.fit_transform(weather_features.loc[complete],
                                          timing_manifest=timing.loc[complete].reset_index(drop=True), fit_cutoff=cutoff).to_numpy()
            training_U = U[complete]
            common_options = dict(l2=settings.l2, maxiter=settings.maxiter, integration_settings=integration)
            c1 = IndependentCopula().fit(training_U)
            c2 = StaticGaussianCopula(**common_options).fit(training_U)
            c3 = DynamicGaussianCopula(**common_options).fit(training_U, X_train)
            c4 = StudentTCopula(df_candidates=settings.df_candidates, shared_model=c2,
                               integration_settings=integration).fit(training_U)
            c5 = StudentTCopula(df_candidates=settings.df_candidates, shared_model=c3,
                               integration_settings=integration).fit(training_U, X_train)
            c6 = StudentTCopula(df_candidates=settings.df_candidates, dynamic=True,
                               **common_options).fit(training_U, X_train)
            models = dict(zip(names[:6], (c1, c2, c3, c4, c5, c6)))
            c8 = AnalogCheckerboard(n_analogs=settings.n_analogs, seed=settings.checkerboard_seed).fit(
                training_U, X_train, availability[complete].tz_localize(None).to_numpy(dtype="datetime64[ns]"))
            models["C8_analog_checkerboard"] = c8
            c7, c7_result = None, None
            c7_record = {"available": False, "status": "member_weather_not_supplied_or_C7_disabled"}
            if member_frames is not None:
                if not member_complete[complete].all():
                    c7_record = {"available": False, "status": "incomplete_member_history_requires_upstream_common_mask",
                                 "missing_member_training_hours": int((~member_complete[complete]).sum())}
                else:
                    c7 = WeatherNetLoadRankTemplate(
                        weather_columns=settings.ecc_weather_columns, calendar_columns=settings.ecc_calendar_columns,
                        linear_columns=tuple(settings.ecc_calendar_columns)+tuple(settings.ecc_linear_weather_columns),
                        smooth_columns=settings.ecc_smooth_weather_columns, alpha=settings.ecc_ridge_alpha,
                        n_knots=settings.n_knots, minimum_training_rows=settings.minimum_pit_rows,
                        release_delay_hours=settings.release_delay_hours, require_selected_initialization=False).fit(
                        {zone: frame.loc[complete] for zone, frame in member_frames.items()},
                        calendar_source.loc[complete, list(settings.ecc_calendar_columns)],
                        shared.loc[complete, ["net_load_"+zone for zone in ZONES]].rename(columns={"net_load_"+zone: zone for zone in ZONES}),
                        timing_manifest=timing.loc[complete].reset_index(drop=True),
                        label_available_times=label_available_times.loc[complete], fit_cutoff=cutoff, quarter_start=origin)
                    c7_record = {"available": True, "status": "fitted", "fit_metadata": c7.fit_metadata_}
            query = ((index >= origin) & (index < quarter_end) & np.isfinite(feature_values).all(axis=1)
                     & declared_feature_complete(shared))
            X_query = basis.transform(weather_features.loc[query],
                                      timing_manifest=timing.loc[query].reset_index(drop=True)).to_numpy() if query.any() else np.empty((0, X_train.shape[1]))
            query_times = index[query]
            c7_query_mask = np.zeros(len(query_times), dtype=bool)
            if c7 is not None:
                c7_query_mask = member_complete[query]
                if c7_query_mask.any():
                    c7_full_mask = query & member_complete
                    c7_result = c7.predict({zone: frame.loc[c7_full_mask] for zone, frame in member_frames.items()},
                        calendar_source.loc[c7_full_mask, list(settings.ecc_calendar_columns)],
                        timing_manifest=timing.loc[c7_full_mask].reset_index(drop=True))
                    c7_record["forecast_rank_diagnostics"] = c7_result.diagnostics
                c7_record["complete_member_query_hours"] = int(c7_query_mask.sum())
            shared_dynamic_error = float(np.max(np.abs(c3.correlation_matrices(X_train)-c5.correlation_matrices(X_train))))
            shared_static_error = float(np.max(np.abs(c2.correlation_matrices(n=len(training_U))-c4.correlation_matrices(n=len(training_U)))))
            if shared_dynamic_error != 0 or shared_static_error != 0:
                raise AssertionError("Shared-R t comparator changed its Gaussian R")
            quarter = {"origin": origin.isoformat(), "prediction_end_exclusive": quarter_end.isoformat(),
                       "fit_cutoff": cutoff.isoformat(), "history_start": history_start.isoformat(),
                       "first_forecast_issue": (origin-pd.Timedelta(hours=12)).isoformat(),
                       "common_training_rows": int(complete.sum()), "past_rows_before_common_mask": int(past.sum()),
                       "training_target_start": index[complete][0].isoformat(),
                       "training_target_interval_end": (index[complete][-1]+pd.Timedelta(hours=1)).isoformat(),
                       "training_last_label_available": availability[complete].max().isoformat(),
                       "basis": basis.fit_metadata_, "model_fit_info": {name: getattr(model, "fit_info_", {"mode": "independent", "n_rows": len(training_U)}) for name, model in models.items()},
                       "shared_R_max_abs_difference": {"C4_vs_C2": shared_static_error, "C5_vs_C3": shared_dynamic_error},
                       "input_query_rows": int(((index >= origin) & (index < quarter_end)).sum()),
                       "complete_weather_query_rows": int(query.sum()),
                       "expected_calendar_quarter_hours": int((quarter_end-origin).total_seconds()/3600),
                       "targets": {}}
            quarter["model_fit_info"]["C8_analog_checkerboard"] = {
                "mode": "past_only_analog_checkerboard", "n_rows": len(training_U), "n_analogs": settings.n_analogs,
                "latest_label_available": availability[complete].max().isoformat()}
            c7_record["calendar_contract"] = calendar_contract
            quarter["C7"] = c7_record
            for q in qnames:
                V = shared.loc[query, ["cdf_"+q+"_"+zone for zone in ZONES]].to_numpy(dtype=float)
                eligible = declared_physical_eligible(shared.loc[query], V, threshold_available)
                if np.isfinite(V).any() and (((V[np.isfinite(V)] < 0) | (V[np.isfinite(V)] > 1)).any()):
                    raise ValueError("Shared threshold CDF outside [0,1]")
                selected_times = query_times[eligible]
                target_record = {"eligible_predictions": int(eligible.sum()), "models": {}, "event_baselines": {}}
                if settings.include_pair_products:
                    target_record["pair_products"] = {}
                output.loc[selected_times, "forecast_eligible_"+q] = True
                if eligible.any():
                    for name, model in models.items():
                        x_arg = X_query[eligible] if name in ("C3_dynamic_gaussian", "C5_dynamic_t_shared_R", "C6_dynamic_t_refit_R") else None
                        if name == "C8_analog_checkerboard":
                            prediction = model.predict_event_probabilities(V[eligible], X_query[eligible],
                                (selected_times.normalize()-pd.Timedelta(hours=12)).tz_localize(None).to_numpy(dtype="datetime64[ns]"))
                        else:
                            prediction = model.predict_event_probabilities(V[eligible], x_arg)
                        output.loc[selected_times, name+"_"+q] = prediction["ge2"]
                        output.loc[selected_times, name+"_"+q+"_all3"] = prediction["all3"]
                        target_record["models"][name] = _numerical_metadata(prediction, name, settings.integration_backend)
                        if settings.include_pair_products:
                            pair_prediction = fitted_model_pair_probabilities(name, model, V[eligible],
                                X_query[eligible] if name == "C8_analog_checkerboard" else x_arg,
                                (selected_times.normalize()-pd.Timedelta(hours=12)).tz_localize(None).to_numpy(dtype="datetime64[ns]"),
                                settings=integration)
                            for k, pair_name in enumerate(PAIR_NAMES):
                                output.loc[selected_times, "pair_"+name+"_"+q+"_"+pair_name] = pair_prediction["pairs"][:, k]
                            target_record["pair_products"][name] = pair_numerical_metadata(pair_prediction)
                    if c7_result is not None:
                        c7_eligible = eligible & c7_query_mask
                        # Template rows cover only complete member query hours.
                        template_eligible = eligible[c7_query_mask]
                        if c7_eligible.any():
                            prediction = checkerboard_event_probabilities(V[c7_eligible],
                                template=c7_result.values[template_eligible], seed=settings.checkerboard_seed)
                            output.loc[query_times[c7_eligible], "C7_weather_checkerboard_"+q] = prediction["ge2"]
                            output.loc[query_times[c7_eligible], "C7_weather_checkerboard_"+q+"_all3"] = prediction["all3"]
                            target_record["models"]["C7_weather_checkerboard"] = {
                                "eligible_predictions": int(c7_eligible.sum()), "integration_discrepancy_max": 0.,
                                "numerical_method": "analytic_checkerboard", "integration_order_max": 0,
                                "integration_maxpts_max": 0,
                                "rank_only_shared_CDFs_unchanged": True}
                            if settings.include_pair_products:
                                pair_prediction = checkerboard_pair_probabilities(V[c7_eligible],
                                    template=c7_result.values[template_eligible], seed=settings.checkerboard_seed)
                                for k, pair_name in enumerate(PAIR_NAMES):
                                    output.loc[query_times[c7_eligible], "pair_C7_weather_checkerboard_"+q+"_"+pair_name] = pair_prediction["pairs"][:, k]
                                target_record["pair_products"]["C7_weather_checkerboard"] = {
                                    "eligible_predictions": int(c7_eligible.sum()), **pair_numerical_metadata(pair_prediction)}
                if settings.include_event_baselines:
                    event_past = past & (index.normalize()-pd.Timedelta(hours=12) >= threshold_available)
                    past_table = shared.loc[event_past]
                    past_y = _event_labels(past_table, q)
                    usable = np.isfinite(past_y)
                    c0_available = int(usable.sum()) >= settings.minimum_event_rows
                    target_record["event_baselines"]["C0"] = {"available": c0_available,
                        "training_rows": int(usable.sum()), "event_count": int(np.nansum(past_y)),
                        "beta_prior": [1, 1], "status": "available" if c0_available else "insufficient_past_event_history"}
                    if c0_available:
                        output.loc[selected_times, "C0_climatology_"+q] = (np.nansum(past_y)+1)/(usable.sum()+2)
                    # Only earlier-quarter probabilities generated above qualify.
                    earlier = output.index < origin
                    prior_p = output.loc[earlier, "C3_dynamic_gaussian_"+q]
                    positions = index.get_indexer(prior_p.index)
                    allowed = ((prior_p.index+pd.Timedelta(hours=1) <= cutoff) &
                               (availability[positions] <= cutoff) & (prior_p.index >= history_start))
                    prior_p = prior_p.loc[allowed]
                    prior_y = _event_labels(shared.loc[prior_p.index], q)
                    good, status = _training_event_mask(prior_p.to_numpy()[:, None], prior_y,
                                                       settings.minimum_event_rows, settings.minimum_each_event_class)
                    status["source"] = "strictly_earlier_quarter_issued_C3_probabilities_from_this_runner"
                    target_record["event_baselines"]["C9"] = status
                    if status["available"]:
                        calibrator = EventLogisticCalibrator(nonnegative_slope=True).fit(prior_p.to_numpy()[good], prior_y[good])
                        gaussian_p = output.loc[selected_times, "C3_dynamic_gaussian_"+q].to_numpy()
                        status["fit_info"] = calibrator.fit_info_
                        if calibrator.coef_[1] <= 0:
                            status["available"] = False
                            status["status"] = "nonpositive_slope_not_issued_under_positive_monotone_policy"
                        elif len(gaussian_p):
                            output.loc[selected_times, "C9_event_calibrated_C3_"+q] = calibrator.predict(gaussian_p)
                            output.loc[selected_times, "C9_"+q+"_alert_rank_score_from_C3"] = gaussian_p
                    # Direct event model sees genuinely forward shared marginal
                    # CDFs and forecast weather on past target cases only.
                    past_X_raw = weather_features.loc[event_past]
                    finite_features = np.isfinite(past_X_raw.to_numpy(dtype=float)).all(axis=1)
                    V_past = past_table[["cdf_"+q+"_"+zone for zone in ZONES]].to_numpy(dtype=float)
                    finite_v = np.isfinite(V_past).all(axis=1)
                    usable_direct = finite_features & finite_v
                    X_past = (basis._transformed(past_X_raw.loc[usable_direct]).to_numpy()
                              if usable_direct.any() else np.empty((0, X_train.shape[1])))
                    direct_design = _event_design(V_past[usable_direct], X_past)
                    direct_y = past_y[usable_direct]
                    good, status = _training_event_mask(direct_design, direct_y,
                                                       settings.minimum_event_rows, settings.minimum_each_event_class)
                    status["source"] = "past_forward_shared_CDFs_and_issue_available_weather"
                    target_record["event_baselines"]["C10"] = status
                    if status["available"]:
                        scaler = StandardScaler().fit(direct_design[good])
                        logistic = LogisticRegression(C=settings.event_logistic_C, solver="lbfgs", max_iter=600).fit(
                            scaler.transform(direct_design[good]), direct_y[good])
                        if eligible.any():
                            output.loc[selected_times, "C10_direct_logistic_"+q] = logistic.predict_proba(
                                scaler.transform(_event_design(V[eligible], X_query[eligible])))[:, 1]
                        status["n_iterations"] = int(logistic.n_iter_[0])
                        status["feature_count"] = int(direct_design.shape[1])
                quarter["targets"][q] = target_record
            if settings.include_pit_diagnostics:
                quarter["pit_diagnostics"] = {}
                for level in settings.pit_diagnostic_levels:
                    level_name = "u"+str(int(round(100*level)))
                    V_diagnostic = np.full((len(query_times), len(ZONES)), level)
                    output.loc[query_times, "diagnostic_pit_forecast_eligible_"+level_name] = True
                    diagnostic_record = {"level": level, "weather_complete_predictions": len(query_times),
                                         "constant_threshold_CDF": [level]*len(ZONES), "models": {}}
                    if len(query_times):
                        for name, model in models.items():
                            x_arg = X_query if name in ("C3_dynamic_gaussian", "C5_dynamic_t_shared_R", "C6_dynamic_t_refit_R") else None
                            if name == "C8_analog_checkerboard":
                                prediction = model.predict_event_probabilities(V_diagnostic, X_query,
                                    (query_times.normalize()-pd.Timedelta(hours=12)).tz_localize(None).to_numpy(dtype="datetime64[ns]"))
                            else:
                                prediction = model.predict_event_probabilities(V_diagnostic, x_arg)
                            column = "diagnostic_pit_"+name+"_"+level_name
                            output.loc[query_times, column] = prediction["ge2"]
                            output.loc[query_times, column+"_all3"] = prediction["all3"]
                            diagnostic_record["models"][name] = _numerical_metadata(prediction, name, settings.integration_backend)
                        if c7_result is not None:
                            prediction = checkerboard_event_probabilities(V_diagnostic[c7_query_mask],
                                template=c7_result.values, seed=settings.checkerboard_seed)
                            column = "diagnostic_pit_C7_weather_checkerboard_"+level_name
                            output.loc[query_times[c7_query_mask], column] = prediction["ge2"]
                            output.loc[query_times[c7_query_mask], column+"_all3"] = prediction["all3"]
                            diagnostic_record["models"]["C7_weather_checkerboard"] = {
                                "eligible_predictions": int(c7_query_mask.sum()), "integration_discrepancy_max": 0.,
                                "numerical_method": "analytic_checkerboard", "integration_order_max": 0,
                                "integration_maxpts_max": 0,
                                "rank_template_same_as_physical_forecast": True}
                    quarter["pit_diagnostics"][level_name] = diagnostic_record
            manifest["quarters"].append(quarter)
            origin = quarter_end
        manifest["status"] = "completed"
        manifest["forecast_rows"] = len(output)
        if target_destination is not None:
            prediction_path = target_destination/"joint_runner_predictions.parquet"
            output.to_parquet(prediction_path)
            manifest["prediction_file"] = str(prediction_path)
            manifest["prediction_sha256"] = sha256(prediction_path)
    except Exception as error:
        manifest["status"] = "failed"
        manifest["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        if target_destination is not None:
            (target_destination/"joint_runner_manifest.json").write_text(
                json.dumps(_json_safe(manifest), indent=2), encoding="utf-8")
    return output, manifest


def declared_feature_complete(shared):
    """Known forecast information, never current target-label completeness."""
    if "feature_complete" not in shared:
        return np.ones(len(shared), dtype=bool)
    values = shared["feature_complete"]
    if values.isna().any() or not values.isin((True, False)).all():
        raise ValueError("feature_complete must be a complete declared boolean flag")
    return values.to_numpy(dtype=bool)


def declared_physical_eligible(shared_query, V, threshold_available):
    """Declared input/threshold availability precedes all model-value checks.

    Explicitly unavailable known features may exclude a forecast. A nonfinite
    CDF at a declared available case is a failure, never a candidate mask.
    Legacy synthetic tables without flags retain their established policy.
    """
    times = utc_index(shared_query.index)
    clock = times.normalize()-pd.Timedelta(hours=12) >= threshold_available
    if "threshold_available" in shared_query:
        values = shared_query["threshold_available"]
        if values.isna().any() or not values.isin((True, False)).all():
            raise ValueError("threshold_available must be a complete declared boolean flag")
        clock = clock & values.to_numpy(dtype=bool)
    if "feature_complete" in shared_query:
        eligible = clock & declared_feature_complete(shared_query)
        if not np.isfinite(V[eligible]).all():
            raise ValueError("Nonfinite shared threshold CDF at declared available forecast; no numerical-failure masking")
        return eligible
    return clock & np.isfinite(V).all(axis=1)


def _synthetic_provenance(index):
    records = []
    for origin in _origins(index).unique():
        cal_end = origin-pd.Timedelta(days=7)
        cal_start = cal_end-pd.Timedelta(days=90)
        records.append({"origin": origin.isoformat(), "calibration_start": cal_start.isoformat(),
                        "calibration_end": cal_end.isoformat(), "raw_fit_end_exclusive": cal_start.isoformat()})
    return {"quarters": records, "evidence_type": "synthetic_assertions_not_actual_margin_fit"}


def _synthetic_gefs_timing(index):
    """Synthetic policy clock; not authenticated historical public arrival."""
    index = utc_index(index)
    initialization = index.normalize()-pd.Timedelta(days=1)
    issue = initialization+pd.Timedelta(hours=12)
    return pd.DataFrame({"target_time": index, "initialization_time": initialization,
                         "issue_time": issue, "available_time": issue})


def _synthetic_member_weather(index, features, members=3):
    """Explicitly synthetic zone-prefix GEFS-schema fixture, never observations."""
    rng = np.random.default_rng(11207)
    timing = _synthetic_gefs_timing(index)
    frames = []
    for member in range(1, members+1):
        frame = timing[["target_time", "initialization_time", "issue_time"]].copy()
        frame["member_id"] = member
        for j, zone in enumerate(ZONES):
            frame[zone+"_t2m"] = 273.15+features.temperature.to_numpy()+j*.2+rng.normal(scale=.3, size=len(index))
            frame[zone+"_wind_speed"] = 3+np.abs(rng.normal(size=len(index)))
            frame[zone+"_marine_wind_speed"] = 5+np.abs(rng.normal(size=len(index)))
            frame[zone+"_dswrf"] = 200*np.maximum(0, np.sin(2*np.pi*(index.hour.to_numpy()-6)/24))+rng.uniform(0, 20, len(index))
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def synthetic_self_test(destination, backend="scipy_replicated"):
    rng = np.random.default_rng(20260930)
    index = pd.date_range("2021-01-01", "2022-01-01", freq="h", inclusive="left", tz="UTC")
    query = pd.date_range("2022-01-09", periods=48, freq="h", tz="UTC")
    index = index.append(query)
    features = pd.DataFrame({"temperature": rng.normal(size=len(index)),
                             "season_sin": np.sin(np.arange(len(index))/2000)}, index=index)
    eta = np.column_stack([.45+.08*features.temperature, np.full(len(index), .3), np.full(len(index), .2)])
    U = sample_elliptical_copula(correlation_from_eta(eta), seed=2309)
    shared = pd.DataFrame(index=index)
    shared["origin"] = _origins(index)
    for j, zone in enumerate(ZONES):
        shared["pit_"+zone] = U[:, j]
        shared["net_load_"+zone] = U[:, j]*100
        shared["threshold_q90_"+zone] = 90.
        shared["cdf_q90_"+zone] = .86+.02*np.sin(features.temperature+j)
        shared.loc[shared.index < query[0], "threshold_q90_"+zone] = np.nan
        shared.loc[shared.index < query[0], "cdf_q90_"+zone] = np.nan
    availability = pd.Series(index+pd.Timedelta(hours=1), index=index)
    settings = JointForwardSettings(df_candidates=(8,), linear_columns=("season_sin",),
        integration_backend=backend,
        smooth_columns=("temperature",), integration_tolerance=1e-3,
        integration_initial_maxpts=1024, integration_max_maxpts=16384,
        ecc_members=3, ecc_calendar_columns=("season_sin",))
    members = _synthetic_member_weather(index, features)
    first, manifest = run_joint_forward(shared, features, _synthetic_gefs_timing(index),
        label_available_times=availability, marginal_provenance=_synthetic_provenance(index),
        settings=settings, start="2022-01-01T00:00:00+00:00", end="2022-04-01T00:00:00+00:00", qnames=("q90",), member_weather=members)
    changed = shared.copy()
    future = changed.index >= pd.Timestamp("2022-01-01", tz="UTC")
    for zone in ZONES:
        changed.loc[future, "net_load_"+zone] = 1e7+rng.normal(size=future.sum())
        changed.loc[future, "pit_"+zone] = rng.uniform(size=future.sum())
    changed.loc[query[3], "net_load_FR"] = np.nan
    changed.loc[query[3], "pit_FR"] = np.nan
    second, altered_manifest = run_joint_forward(changed, features, _synthetic_gefs_timing(index),
        label_available_times=availability, marginal_provenance=_synthetic_provenance(index),
        settings=settings, start="2022-01-01T00:00:00+00:00", end="2022-04-01T00:00:00+00:00", qnames=("q90",), member_weather=members)
    prediction_columns = [name for name in first if name.startswith(("C", "diagnostic_pit_C"))]
    np.testing.assert_allclose(first[prediction_columns], second[prediction_columns], rtol=0, atol=0, equal_nan=True)
    assert first[[name for name in prediction_columns if name.startswith(tuple("C"+str(k)+"_" for k in range(1, 7)))]].notna().all().all()
    assert first[["C7_weather_checkerboard_q90", "C8_analog_checkerboard_q90"]].notna().all().all()
    quarter = manifest["quarters"][0]
    assert pd.Timestamp(quarter["training_target_interval_end"]) <= pd.Timestamp(quarter["fit_cutoff"])
    assert pd.Timestamp(quarter["training_last_label_available"]) <= pd.Timestamp(quarter["fit_cutoff"])
    assert quarter["shared_R_max_abs_difference"] == {"C4_vs_C2": 0., "C5_vs_C3": 0.}
    assert all(not row["available"] for row in quarter["targets"]["q90"]["event_baselines"].values())
    checks = ["all_eight_dependence_models_issued_on_same_48_forecast_hours",
              "changed_future_labels_and_PIT_leave_every_probability_identical",
              "unknown_target_label_does_not_remove_forecast",
              "training_label_and_target_cutoffs_precede_first_issue",
              "C4_C2_and_C5_C3_correlation_matrices_exactly_identical",
              "event_baselines_insufficient_forward_history_marked_unavailable"]
    checks.extend(["C7_member_rank_template_actually_fitted_same_past_mask",
                   "C8_50_analog_checkerboard_uses_same_past_PIT_and_basis",
                   "C7_C8_probabilities_unchanged_after_future_label_and_PIT_perturbation"])
    assert first.filter(regex="^C[1-8]_.*_all3$").notna().all().all()
    assert first.filter(regex="^diagnostic_pit_C[1-8]_").notna().all().all()
    checks.extend(["all_C1_C8_physical_all3_probabilities_issued_and_future_invariant",
                   "both_constant_V_PIT_diagnostic_levels_issued_and_future_invariant"])
    early = shared.copy()
    early.index = pd.DatetimeIndex([stamp-pd.Timedelta(days=8) if stamp >= query[0] else stamp for stamp in index])
    early_query = early.index >= pd.Timestamp("2022-01-01", tz="UTC")
    early_features = features.copy(); early_features.index = early.index
    early_availability = pd.Series(early.index+pd.Timedelta(hours=1), index=early.index)
    early["origin"] = _origins(early.index)
    before_reference, _ = run_joint_forward(early, early_features, _synthetic_gefs_timing(early.index),
        label_available_times=early_availability, marginal_provenance=_synthetic_provenance(early.index),
        settings=settings, start="2022-01-01T00:00:00+00:00", end="2022-04-01T00:00:00+00:00", qnames=("q90",),
        member_weather=_synthetic_member_weather(early.index, early_features))
    assert before_reference.filter(regex="^C[1-6]_").isna().all().all()
    assert not before_reference.forecast_eligible_q90.any()
    checks.append("pre_reference_availability_CDF_cannot_create_early_event_forecast")
    assert before_reference.filter(regex="^diagnostic_pit_C[1-8]_").notna().all().all()
    checks.append("PIT_diagnostics_do_not_depend_on_physical_threshold_availability")
    rejected_test = shared.iloc[-1:].copy(); rejected_test.index = pd.DatetimeIndex([TEST_START])
    try:
        _development_index(rejected_test, "sealed test")
    except ValueError:
        checks.append("2024_target_rows_rejected_before_label_inspection")
    else:
        raise AssertionError("Sealed test guard failed")
    # A separate two-quarter fixture actually exercises available C9/C10. Its
    # event pattern is deliberately synthetic and is not an empirical result.
    chain_index = pd.date_range("2021-01-01", "2022-04-03", freq="h", inclusive="left", tz="UTC")
    chain_features = pd.DataFrame({"temperature": rng.normal(size=len(chain_index)),
                                  "season_sin": np.sin(np.arange(len(chain_index))/2000)}, index=chain_index)
    chain_eta = np.column_stack([.45+.08*chain_features.temperature,
                                np.full(len(chain_index), .3), np.full(len(chain_index), .2)])
    chain_U = sample_elliptical_copula(correlation_from_eta(chain_eta), seed=774)
    chain_shared = pd.DataFrame(index=chain_index)
    chain_shared["origin"] = _origins(chain_index)
    second_query = pd.date_range("2022-04-01", periods=48, freq="h", tz="UTC")
    first_threshold_cdf = np.linspace(.3, .95, len(query))
    for j, zone in enumerate(ZONES):
        chain_shared["pit_"+zone] = chain_U[:, j]
        chain_shared["net_load_"+zone] = chain_U[:, j]*100
        chain_shared["threshold_q90_"+zone] = np.nan
        chain_shared.loc[chain_index >= query[0], "threshold_q90_"+zone] = 90.
        chain_shared["cdf_q90_"+zone] = np.nan
        chain_shared.loc[query, "cdf_q90_"+zone] = first_threshold_cdf
        chain_shared.loc[second_query, "cdf_q90_"+zone] = .75+.08*np.sin(chain_features.loc[second_query, "temperature"])
        chain_shared.loc[query, "net_load_"+zone] = np.where(first_threshold_cdf < np.median(first_threshold_cdf), 100., 0.)
    chain_availability = pd.Series(chain_index+pd.Timedelta(hours=1), index=chain_index)
    chain_settings = JointForwardSettings(**{
        **asdict(settings), "minimum_event_rows": 8, "minimum_each_event_class": 1,
        "include_pit_diagnostics": False})  # Avoid thousands of diagnostic-only cases in this branch fixture.
    chain_members = _synthetic_member_weather(chain_index, chain_features)
    chain_first, chain_manifest = run_joint_forward(chain_shared, chain_features, _synthetic_gefs_timing(chain_index),
        label_available_times=chain_availability, marginal_provenance=_synthetic_provenance(chain_index),
        settings=chain_settings, start="2022-01-01T00:00:00+00:00", end="2022-07-01T00:00:00+00:00", qnames=("q90",), member_weather=chain_members)
    baseline_records = chain_manifest["quarters"][1]["targets"]["q90"]["event_baselines"]
    assert all(row["available"] for row in baseline_records.values())
    assert chain_first.loc[second_query, ["C0_climatology_q90", "C9_event_calibrated_C3_q90", "C10_direct_logistic_q90"]].notna().all().all()
    np.testing.assert_array_equal(chain_first.loc[second_query, "C3_dynamic_gaussian_q90"],
                                  chain_first.loc[second_query, "C9_q90_alert_rank_score_from_C3"])
    assert baseline_records["C9"]["fit_info"]["slope"] > 0
    chain_changed = chain_shared.copy()
    for zone in ZONES:
        chain_changed.loc[second_query, "net_load_"+zone] = np.nan
        chain_changed.loc[second_query, "pit_"+zone] = rng.uniform(size=len(second_query))
    chain_second, _ = run_joint_forward(chain_changed, chain_features, _synthetic_gefs_timing(chain_index),
        label_available_times=chain_availability, marginal_provenance=_synthetic_provenance(chain_index),
        settings=chain_settings, start="2022-01-01T00:00:00+00:00", end="2022-07-01T00:00:00+00:00", qnames=("q90",), member_weather=chain_members)
    chain_prediction_columns = [name for name in chain_first if name.startswith("C")]
    np.testing.assert_allclose(chain_first[chain_prediction_columns], chain_second[chain_prediction_columns],
                               rtol=0, atol=0, equal_nan=True)
    checks.extend(["C0_C9_C10_available_branches_actually_fitted_on_synthetic_past",
                   "C9_strict_positive_slope_and_original_C3_alert_rank_score",
                   "second_quarter_future_label_changes_leave_all_eleven_models_identical"])
    report = {"status": "passed", "evidence_type": "SYNTHETIC_ONLY_NO_REAL_WEATHER_OR_MARGIN_FITS",
              "created_utc": datetime.now(timezone.utc).isoformat(), "checks": checks,
              "prediction_hours": len(first), "history_hours": int((index < query[0]).sum()),
              "integration_settings": _json_safe(asdict(settings)), "quarter": quarter,
              "two_quarter_event_baseline_fit": baseline_records,
              "code_sha256": manifest["code_sha256"],
              "limitation": "Actual weather, marginal calibration, model competitiveness and C9/C10 availability still require real development runs."}
    path = Path(destination).resolve()
    if not path.resolve().is_relative_to(_work_path().resolve()) or not path.name.startswith("joint_runner_") or path.exists():
        raise ValueError("Choose a NEW D: joint_runner_* self-test directory")
    path.mkdir(parents=True)
    first.to_parquet(path/"joint_runner_synthetic_predictions.parquet")
    (path/"joint_runner_synthetic_validation.json").write_text(json.dumps(_json_safe(report), indent=2), encoding="utf-8")
    return report


def synthetic_test_unlock(destination, backend="scipy_replicated", include_pairs=False):
    """Run only invented 2024/2025 fixtures; never access empirical test files."""
    from copy import deepcopy
    from prequential_margins import expected_frozen_settings
    folder = Path(destination).resolve()
    if not folder.resolve().is_relative_to(_work_path().resolve()) or not folder.name.startswith("joint_runner_") or folder.exists():
        raise ValueError("Choose a NEW D: joint_runner_* synthetic validation directory")
    folder.mkdir(parents=True)
    weather_spec = folder/"joint_runner_synthetic_weather_spec.json"
    marginal_spec = folder/"joint_runner_synthetic_margin_spec.json"
    weather_spec.write_text(json.dumps({"synthetic": True, "columns": ["season_sin", "temperature"],
                                      "source": "invented_GEFS_clock_not_real_weather"}), encoding="utf-8")
    marginal_spec.write_text(json.dumps({"synthetic": True, "features": ["season_sin", "temperature"]}), encoding="utf-8")
    settings = JointForwardSettings(df_candidates=(8,), linear_columns=("season_sin",),
        integration_backend=backend,
        include_pair_products=include_pairs,
        smooth_columns=("temperature",), minimum_pit_rows=96, maxiter=150, ecc_members=3,
        minimum_event_rows=16, minimum_each_event_class=2,
        integration_tolerance=1e-6 if backend == "elliptical_path" else 1e-3,
        integration_initial_maxpts=1024, integration_max_maxpts=16384)
    marginal_settings = expected_frozen_settings(marginal_spec, "additive", minimum_raw_rows=100,
        minimum_calibration_rows=50, reference_min_samples=50)
    fixture_counter = []
    rng = np.random.default_rng(731390)

    def fixture(year):
        fixture_counter.append(year)
        frozen = folder/f"joint_runner_SYNTHETIC_ONLY_frozen_{year}_{len(fixture_counter)}.json"
        history = pd.date_range("2023-01-01", periods=240, freq="h", tz="UTC")
        query = pd.date_range(f"{year}-01-09", periods=48, freq="h", tz="UTC")
        index = history.append(query)
        features = pd.DataFrame({"temperature": rng.normal(size=len(index)),
                                 "season_sin": rng.normal(size=len(index))}, index=index)
        U = sample_elliptical_copula(correlation_from_eta(np.tile([.4, .25, .2], (len(index), 1))), seed=year)
        shared = pd.DataFrame({"origin": _origins(index)}, index=index)
        for j, zone in enumerate(ZONES):
            shared["pit_"+zone] = U[:, j]
            shared["net_load_"+zone] = np.where(features.temperature > 0, 130.+j, 60.+j)
            shared["threshold_q90_"+zone] = 90.
            shared["cdf_q90_"+zone] = .8+.13/(1+np.exp(features.temperature+j*.1))
        available = pd.Series(index+pd.Timedelta(hours=1), index=index, name="label_available_time")
        timing = _synthetic_gefs_timing(index)
        member = _synthetic_member_weather(index, features, members=3)
        provenance = _synthetic_provenance(index)
        provenance["settings"] = marginal_settings
        contract = expected_joint_frozen_settings(settings, weather_schema=weather_feature_schema(features),
            weather_feature_spec=weather_spec, marginal_provenance=provenance,
            marginal_feature_spec=marginal_spec, qnames=("q90",))
        document = {"status": "frozen", "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
                    "test_outcomes_inspected": False, "evidence_type": "SYNTHETIC_ONLY",
                    "prequential_margins": marginal_settings, "joint_forward_runner": contract}
        frozen.write_text(json.dumps(_json_safe(document), indent=2), encoding="utf-8")
        provenance["frozen_configuration"] = {"path": str(frozen), "sha256": sha256(frozen),
                                               "frozen_at_utc": document["frozen_at_utc"]}
        kwargs = dict(label_available_times=available, marginal_provenance=provenance, settings=settings,
            start=f"{year}-01-01T00:00:00+00:00", end=f"{year}-04-01T00:00:00+00:00", qnames=("q90",),
            member_weather=member, allow_test=True, frozen_config=frozen,
            weather_feature_spec=weather_spec, marginal_feature_spec=marginal_spec)
        return shared, features, timing, kwargs, document

    checks, summaries = [], []
    for year in (2024, 2025):
        shared, features, timing, kwargs, document = fixture(year)
        prediction, manifest = run_joint_forward(shared, features, timing, **kwargs)
        changed = shared.copy()
        future = changed.index >= pd.Timestamp(f"{year}-01-01", tz="UTC")
        for zone in ZONES:
            changed.loc[future, "net_load_"+zone] = np.nan
            changed.loc[future, "pit_"+zone] = .999999
        corrupted, corrupt_manifest = run_joint_forward(changed, features, timing, **kwargs)
        pd.testing.assert_frame_equal(prediction, corrupted, check_exact=True)
        model_columns = [name for name in prediction if name.startswith("C") and "rank_score" not in name and not name.endswith("_all3")]
        available_models = [name for name in model_columns if prediction[name].notna().all()]
        assert len(available_models) == 10  # C9 needs earlier-quarter issued C3 history.
        all3_columns = [name for name in prediction if name.startswith("C") and name.endswith("_all3")]
        diagnostic_columns = [name for name in prediction if name.startswith("diagnostic_pit_C")]
        assert len(all3_columns) == 8 and prediction[all3_columns].notna().all().all()
        assert len(diagnostic_columns) == 32 and prediction[diagnostic_columns].notna().all().all()
        pair_columns = [name for name in prediction if name.startswith("pair_")]
        assert len(pair_columns) == (24 if include_pairs else 0)
        if include_pairs:
            assert prediction[pair_columns].notna().all().all()
            checks.append(f"{year}_24_physical_pair_probabilities_finite_and_future_label_invariant")
        for level in settings.pit_diagnostic_levels:
            level_name = "u"+str(int(round(100*level)))
            exceed = 1-level
            np.testing.assert_allclose(prediction["diagnostic_pit_C1_independent_"+level_name],
                                       3*exceed**2-2*exceed**3, rtol=0, atol=1e-14)
            np.testing.assert_allclose(prediction["diagnostic_pit_C1_independent_"+level_name+"_all3"],
                                       exceed**3, rtol=0, atol=1e-14)
        for column in all3_columns+diagnostic_columns:
            if column.endswith("_all3"):
                assert (prediction[column] <= prediction[column[:-5]]+1e-12).all()
        assert prediction.forecast_eligible_q90.all()
        quarter = manifest["quarters"][0]
        assert pd.Timestamp(quarter["training_last_label_available"]) <= pd.Timestamp(quarter["fit_cutoff"])
        assert quarter["shared_R_max_abs_difference"] == {"C4_vs_C2": 0., "C5_vs_C3": 0.}
        for name in ("C2_static_gaussian", "C3_dynamic_gaussian", "C4_static_t_shared_R", "C5_dynamic_t_shared_R", "C6_dynamic_t_refit_R"):
            numerical = quarter["targets"]["q90"]["models"][name]
            if backend == "elliptical_path":
                assert numerical["integration_maxpts_max"] == 0 and numerical["integration_order_max"] > 0
            else:
                assert numerical["integration_maxpts_max"] > 0 and numerical["integration_order_max"] == 0
        assert quarter["C7"]["available"] and quarter["C7"]["fit_metadata"]["past_available_hours"] == 240
        assert manifest["frozen_configuration"]["verification"].startswith("freeze_time_preinspection")
        checks += [f"{year}_all_available_model_probabilities_exactly_invariant_to_future_labels_and_PIT",
                   f"{year}_future_label_NaNs_do_not_change_prediction_mask",
                   f"{year}_actual_C7_C8_and_C0_C10_fits_cutoff_and_shared_R_guards",
                   f"{year}_eight_physical_all3_and_32_PIT_diagnostic_probabilities_finite_and_future_invariant",
                   f"{year}_independent_diagnostic_analytic_probability_and_all3_subset_bounds"]
        prediction.to_parquet(folder/f"joint_runner_synthetic_{year}_predictions.parquet")
        summaries.append({"year": year, "prediction_hours": len(prediction), "available_models": available_models,
                          "C9_status": quarter["targets"]["q90"]["event_baselines"]["C9"],
                          "training_rows": quarter["common_training_rows"], "fit_cutoff": quarter["fit_cutoff"]})

    # C9 must be tested with an actual positive fit, not with unavailable NaNs.
    shared, features, timing, kwargs, _ = fixture(2025)
    prior_times = pd.date_range("2024-10-01", periods=48, freq="h", tz="UTC")
    prior_features = pd.DataFrame({"temperature": rng.normal(size=48), "season_sin": rng.normal(size=48)}, index=prior_times)
    prior_table = pd.DataFrame({"origin": _origins(prior_times)}, index=prior_times)
    prior_U = sample_elliptical_copula(correlation_from_eta(np.tile([.4, .25, .2], (48, 1))), seed=2244)
    for j, zone in enumerate(ZONES):
        prior_table["pit_"+zone] = prior_U[:, j]
        prior_table["net_load_"+zone] = np.where(prior_features.temperature > 0, 130.+j, 60.+j)
        prior_table["threshold_q90_"+zone] = 90.
        prior_table["cdf_q90_"+zone] = .8+.13/(1+np.exp(prior_features.temperature+j*.1))
    shared = pd.concat([shared, prior_table]).sort_index()
    features = pd.concat([features, prior_features]).sort_index()
    timing = _synthetic_gefs_timing(shared.index)
    kwargs["label_available_times"] = pd.Series(shared.index+pd.Timedelta(hours=1), index=shared.index)
    provenance = _synthetic_provenance(shared.index)
    provenance.update(settings=marginal_settings, frozen_configuration=kwargs["marginal_provenance"]["frozen_configuration"])
    kwargs["marginal_provenance"] = provenance
    kwargs["member_weather"] = _synthetic_member_weather(shared.index, features, members=3)
    kwargs["start"] = "2024-10-01T00:00:00+00:00"
    first_kwargs = {**kwargs, "end": "2025-01-01T00:00:00+00:00"}
    first_output, _ = run_joint_forward(shared, features, timing, **first_kwargs)
    prior_p = first_output.loc[prior_times, "C3_dynamic_gaussian_q90"]
    prior_y = prior_p.to_numpy() > np.median(prior_p)
    for zone in ZONES:
        shared.loc[prior_times, "net_load_"+zone] = np.where(prior_y, 130., 60.)
    c9_output, c9_manifest = run_joint_forward(shared, features, timing, **kwargs)
    changed = shared.copy()
    future = changed.index >= TEST_START+pd.DateOffset(years=1)
    for zone in ZONES:
        changed.loc[future, "net_load_"+zone] = np.nan
        changed.loc[future, "pit_"+zone] = .000001
    c9_changed, _ = run_joint_forward(changed, features, timing, **kwargs)
    pd.testing.assert_frame_equal(c9_output, c9_changed, check_exact=True)
    c9_status = c9_manifest["quarters"][1]["targets"]["q90"]["event_baselines"]["C9"]
    assert c9_status["available"] and c9_status["fit_info"]["slope"] > 0
    forecast_2025 = c9_output.loc[c9_output.index.year == 2025]
    assert forecast_2025.C9_event_calibrated_C3_q90.notna().all()
    np.testing.assert_array_equal(forecast_2025.C9_q90_alert_rank_score_from_C3, forecast_2025.C3_dynamic_gaussian_q90)
    checks += ["2025_actual_positive_C9_from_earlier_2024_issued_probabilities_future_label_invariance",
               "2025_C9_fixed_alert_rank_score_exactly_equals_original_C3"]
    c9_output.to_parquet(folder/"joint_runner_synthetic_C9_2024Q4_2025Q1_predictions.parquet")
    summaries.append({"scope": "2024Q4_2025Q1_C9_positive_fit_invented_outcomes", "prediction_hours": len(c9_output),
                      "C9_2025_fit_info": c9_status})

    # Actual CLI refusal before pd.read_parquet can load a label-bearing table.
    shared, features, timing, kwargs, document = fixture(2024)
    frozen = kwargs["frozen_config"]
    paths = {}
    for name, frame in (("shared", shared), ("weather", features), ("timing", timing.set_index("target_time")),
                        ("available", kwargs["label_available_times"].to_frame())):
        path = folder/f"joint_runner_synthetic_{name}.parquet"
        frame.rename_axis("target_time").to_parquet(path)
        paths[name] = path
    provenance_file = folder/"joint_runner_synthetic_marginal_manifest.json"
    provenance_file.write_text(json.dumps(_json_safe(kwargs["marginal_provenance"]), indent=2), encoding="utf-8")
    settings_file = folder/"joint_runner_synthetic_settings.json"
    settings_file.write_text(json.dumps(_json_safe(asdict(settings)), indent=2), encoding="utf-8")
    arguments = ["--destination", str(folder/"joint_runner_never_created_rejection"), "--shared", str(paths["shared"]),
        "--weather-features", str(paths["weather"]), "--timing", str(paths["timing"]),
        "--label-availability", str(paths["available"]), "--marginal-manifest", str(provenance_file),
        "--settings", str(settings_file), "--start", "2024-01-01T00:00:00+00:00", "--end", "2024-04-01T00:00:00+00:00",
        "--qnames", "q90", "--weather-feature-spec", str(weather_spec), "--marginal-feature-spec", str(marginal_spec)]
    label_loads = []
    original_reader = pd.read_parquet

    def forbidden_reader(*args, **kw):
        label_loads.append(str(args[0]))
        raise AssertionError("Gate allowed data-column load before refusal")

    refusal_messages = {}
    try:
        pd.read_parquet = forbidden_reader
        cases = {"default_sealed_refusal": None, "wrong_runner_hash": "hash", "wrong_joint_settings": "settings",
                 "not_frozen": "status", "outcomes_preinspected": "inspected", "future_freeze_timestamp": "future",
                 "wrong_weather_spec_hash": "spec", "wrong_upstream_contract": "upstream",
                 "wrong_weather_schema": "schema", "future_marginal_fit_window": "window",
                 "wrong_upstream_freeze_sha": "upstream_sha", "wrong_PIT_diagnostic_flag": "diagnostic_flag",
                 "wrong_integration_backend": "backend", "wrong_pair_product_flag": "pair_flag",
                 "wrong_pair_helper_hash": "pair_hash"}
        for name, mutation in cases.items():
            bad = deepcopy(document)
            if mutation == "hash":
                bad["joint_forward_runner"]["code_sha256"]["joint_forward_runner.py"] = "0"*64
            elif mutation == "settings":
                bad["joint_forward_runner"]["settings"]["n_analogs"] = 49
            elif mutation == "status":
                bad["status"] = "provisional"
            elif mutation == "inspected":
                bad["test_outcomes_inspected"] = True
            elif mutation == "future":
                bad["frozen_at_utc"] = (pd.Timestamp.now(tz="UTC")+pd.Timedelta(days=1)).isoformat()
            elif mutation == "spec":
                bad["joint_forward_runner"]["weather_feature_spec_sha256"] = "0"*64
            elif mutation == "upstream":
                bad["prequential_margins"]["embargo_days"] = 8
            elif mutation == "schema":
                bad["joint_forward_runner"]["weather_feature_schema"][0]["type"] = "string"
            elif mutation == "diagnostic_flag":
                bad["joint_forward_runner"]["settings"]["include_pit_diagnostics"] = False
            elif mutation == "backend":
                bad["joint_forward_runner"]["settings"]["integration_backend"] = "elliptical_path" if backend == "scipy_replicated" else "scipy_replicated"
            elif mutation == "pair_flag":
                bad["joint_forward_runner"]["settings"]["include_pair_products"] = not settings.include_pair_products
            elif mutation == "pair_hash":
                bad["joint_forward_runner"]["code_sha256"]["joint_pair_products.py"] = "0"*64
            bad_path = folder/f"joint_runner_synthetic_reject_{name}.json"
            bad_path.write_text(json.dumps(_json_safe(bad), indent=2), encoding="utf-8")
            extra_arguments = []
            if mutation in ("window", "upstream_sha"):
                bad_provenance = deepcopy(kwargs["marginal_provenance"])
                if mutation == "window":
                    record = bad_provenance["quarters"][0]
                    record["calibration_end"] = (pd.Timestamp(record["origin"])+pd.Timedelta(hours=1)).isoformat()
                else:
                    bad_provenance["frozen_configuration"]["sha256"] = "0"*64
                bad_provenance_path = folder/f"joint_runner_synthetic_bad_provenance_{name}.json"
                bad_provenance_path.write_text(json.dumps(_json_safe(bad_provenance), indent=2), encoding="utf-8")
                extra_arguments = ["--marginal-manifest", str(bad_provenance_path)]
            try:
                main(arguments+([] if mutation is None else ["--allow-test", "--frozen-config", str(bad_path)])+extra_arguments)
            except PermissionError as error:
                refusal_messages[name] = str(error)
            else:
                raise AssertionError("Expected refusal was not raised: "+name)
        assert not label_loads
    finally:
        pd.read_parquet = original_reader
    checks.append("fifteen_actual_CLI_refusals_before_any_data_column_load_including_pair_flag_and_helper_hash")
    # Valid infinity candidates normalize to the same finite JSON contract token.
    inf_settings = JointForwardSettings(**{**asdict(settings), "df_candidates": (8, np.inf)})
    inf_doc = deepcopy(document)
    inf_doc["joint_forward_runner"] = expected_joint_frozen_settings(inf_settings,
        weather_schema=weather_feature_schema(features), weather_feature_spec=weather_spec,
        marginal_provenance=kwargs["marginal_provenance"], marginal_feature_spec=marginal_spec, qnames=("q90",))
    assert inf_doc["joint_forward_runner"]["settings"]["df_candidates"] == [8, "inf"]
    inf_path = folder/"joint_runner_synthetic_infinity_contract.json"
    inf_path.write_text(json.dumps(inf_doc, indent=2, allow_nan=False), encoding="utf-8")
    check_joint_test_access(required=True, allow_test=True, frozen_config=inf_path, settings=inf_settings,
        weather_schema=weather_feature_schema(features), weather_feature_spec=weather_spec,
        marginal_provenance=kwargs["marginal_provenance"], marginal_feature_spec=marginal_spec, qnames=("q90",))
    checks.append("infinite_df_candidates_have_exact_consistent_JSON_settings_contract")
    # Authorized CLI really loads and executes one complete synthetic quarter.
    valid_manifest = main(arguments+["--allow-test", "--frozen-config", str(frozen),
        "--destination", str(folder/"joint_runner_synthetic_valid_CLI")])
    assert valid_manifest["CLI_frozen_gate_before_any_label_column_read"]
    checks.append("authorized_actual_CLI_completed_after_frozen_metadata_gate")
    report = {"status": "passed", "evidence_type": "SYNTHETIC_ONLY_NOT_EMPIRICAL_TEST_RESULTS",
              "checks": checks, "prediction_hours": 96, "year_summaries": summaries,
              "refusal_messages": refusal_messages, "label_data_column_loads_in_refusal_tests": label_loads,
              "code_sha256": _code_hashes(), "formal_integration_tolerance": 1e-4,
              "synthetic_integration_tolerance": settings.integration_tolerance,
              "integration_backend": backend,
              "include_pair_products": include_pairs,
              "declaration_boundary": "preinspection_false_refers_to_freeze_time_not_a_claim_of_no_postunlock_test_observation_reads",
              "C9_boundary": "unavailable_without_earlier_quarter_issued_C3_history_use_2022_onward_run_to_supply_history"}
    (folder/"joint_runner_test_unlock_validation.json").write_text(json.dumps(_json_safe(report), indent=2), encoding="utf-8")
    return report


def synthetic_pair_products(destination, backend="scipy_replicated"):
    """Pair query guards and exact unchanged-primary comparison; invented data."""
    from dataclasses import replace
    from joint_pair_products import (PAIRS, independent_pair_probabilities,
                                    checkerboard_pair_probabilities, elliptical_pair_probabilities)
    from dependence import elliptical_event_probabilities
    rng = np.random.default_rng(274793)
    history = pd.date_range("2021-01-01", periods=240, freq="h", tz="UTC")
    query = pd.date_range("2022-01-09", periods=24, freq="h", tz="UTC")
    index = history.append(query)
    features = pd.DataFrame({"temperature": rng.normal(size=len(index)),
                             "season_sin": rng.normal(size=len(index))}, index=index)
    shared = pd.DataFrame({"origin": _origins(index)}, index=index)
    U = sample_elliptical_copula(correlation_from_eta(np.tile([.4, .25, .2], (len(index), 1))), seed=57402)
    for j, zone in enumerate(ZONES):
        shared["pit_"+zone] = U[:, j]
        shared["net_load_"+zone] = 100*U[:, j]
        for q, threshold in (("q90", .9), ("q95", .95)):
            shared["threshold_"+q+"_"+zone] = 100*threshold
            shared["cdf_"+q+"_"+zone] = threshold-.025+.02*np.sin(features.temperature+j)
    boundary = np.array([[0., .9, .95], [1., .9, .95], [.9, 0., 1.],
                         [1., 1., 1.], [0., 0., 0.], [.9, .95, 1.]])
    for q in ("q90", "q95"):
        shared.loc[query[:len(boundary)], ["cdf_"+q+"_"+zone for zone in ZONES]] = boundary
    settings = JointForwardSettings(df_candidates=(8,), linear_columns=("season_sin",),
        smooth_columns=("temperature",), minimum_pit_rows=96, maxiter=150, ecc_members=3,
        integration_backend=backend, integration_tolerance=1e-6 if backend == "elliptical_path" else 1e-3,
        integration_initial_maxpts=16384 if backend == "elliptical_path" else 1024,
        integration_max_maxpts=1048576 if backend == "elliptical_path" else 16384)
    kwargs = dict(label_available_times=pd.Series(index+pd.Timedelta(hours=1), index=index),
        marginal_provenance=_synthetic_provenance(index), start="2022-01-01T00:00:00+00:00",
        end="2022-04-01T00:00:00+00:00", qnames=("q90", "q95"),
        member_weather=_synthetic_member_weather(index, features, members=3))
    timing = _synthetic_gefs_timing(index)
    baseline, baseline_manifest = run_joint_forward(shared, features, timing, settings=settings, **kwargs)
    enabled_settings = replace(settings, include_pair_products=True)
    pairs, manifest = run_joint_forward(shared, features, timing, settings=enabled_settings, **kwargs)
    pd.testing.assert_frame_equal(baseline, pairs.loc[:, baseline.columns], check_exact=True)
    assert not baseline.filter(regex="^pair_").shape[1]
    pair_columns = list(pairs.filter(regex="^pair_").columns)
    assert len(pair_columns) == 48 and pairs[pair_columns].notna().all().all()
    changed = shared.copy()
    for zone in ZONES:
        changed.loc[query, "net_load_"+zone] = np.nan
        changed.loc[query, "pit_"+zone] = .999999
    altered, altered_manifest = run_joint_forward(changed, features, timing, settings=enabled_settings, **kwargs)
    pd.testing.assert_frame_equal(pairs, altered, check_exact=True)
    assert manifest["quarters"][0]["shared_R_max_abs_difference"] == {"C4_vs_C2": 0., "C5_vs_C3": 0.}
    checks = ["False_vs_True_all_primary_all3_diagnostic_baseline_columns_exactly_bit_identical",
              "48_C1_C8_three_pair_two_physical_quantile_outputs_finite",
              "future_label_NaNs_and_PIT_changes_leave_every_pair_and_primary_output_exact",
              "pair_queries_use_unchanged_shared_R_matrices_and_same_fitted_models"]
    for q in ("q90", "q95"):
        v = shared.loc[query, ["cdf_"+q+"_"+zone for zone in ZONES]].to_numpy()
        expected = independent_pair_probabilities(v)["pairs"]
        for k, pair_name in enumerate(PAIR_NAMES):
            np.testing.assert_array_equal(pairs["pair_C1_independent_"+q+"_"+pair_name], expected[:, k])
            for model_name in manifest["physical_pair_policy"]["models"]:
                probability = pairs["pair_"+model_name+"_"+q+"_"+pair_name].to_numpy()
                lo = np.maximum(0., 1-v[:, PAIRS[k][0]]-v[:, PAIRS[k][1]])
                hi = np.minimum(1-v[:, PAIRS[k][0]], 1-v[:, PAIRS[k][1]])
                assert np.all((probability >= lo-1e-12)&(probability <= hi+1e-12))
                tolerance = settings.integration_tolerance*2
                assert np.all(pairs[model_name+"_"+q+"_all3"].to_numpy() <= probability+tolerance)
                assert np.all(probability <= pairs[model_name+"_"+q].to_numpy()+tolerance)
    checks += ["independent_pairs_exact_analytic_products", "all_models_pair_Frechet_and_all3_ge2_subset_bounds_at_exact_V0_V1"]
    ranks = np.tile(np.arange(1, 7)[:, None], (1, 3))
    checker = checkerboard_pair_probabilities(boundary, ranks=ranks)["pairs"]
    a = np.clip(ranks[None, :, :]-6*boundary[:, None, :], 0, 1)
    manual = np.column_stack([(a[:, :, i]*a[:, :, j]).mean(axis=1) for i, j in PAIRS])
    np.testing.assert_array_equal(checker, manual)
    checks.append("checkerboard_pairs_exact_shared_rank_cell_fraction_analytic_formula")
    v = np.array([[.9, .92, .94], [.15, .85, .55], [.99, .8, .1]])
    R = correlation_from_eta(np.tile([.4, .25, .2], (len(v), 1)))
    reference_settings = IntegrationSettings(tolerance=1e-6, initial_maxpts=65536, max_maxpts=1048576)
    numerical_reference = []
    for df in (np.inf, 8.):
        candidate = elliptical_pair_probabilities(v, R, df, replace(reference_settings, backend=backend))
        for k, (i, j) in enumerate(PAIRS):
            # Third zone always exceeds when its CDF is exactly zero; original
            # all3 therefore equals the requested pair's exceedance probability.
            zero_third = v.copy(); zero_third[:, 3-i-j] = 0.
            reference = elliptical_event_probabilities(zero_third, R, df, reference_settings)["all3"]
            difference = float(np.max(np.abs(reference-candidate["pairs"][:, k])))
            assert difference < 2e-5
            numerical_reference.append({"df": "inf" if np.isinf(df) else df,
                                        "pair": PAIR_NAMES[k], "reference_max_difference": difference})
    checks.append("Gaussian_and_t_all_three_pairs_agree_with_original_exact_dimension_reduced_high_precision_reference")
    folder = Path(destination).resolve()
    if not folder.resolve().is_relative_to(_work_path().resolve()) or not folder.name.startswith("joint_runner_") or folder.exists():
        raise ValueError("Choose a NEW D: joint_runner_* pair-test output")
    folder.mkdir(parents=True)
    pairs.to_parquet(folder/"joint_runner_synthetic_pair_predictions.parquet")
    report = {"status": "passed", "evidence_type": "SYNTHETIC_ONLY_NOT_EMPIRICAL_COMBINATION_STABILITY",
        "checks": checks, "prediction_hours": len(query), "integration_backend": backend,
        "physical_pair_probabilities": len(pair_columns), "numerical_reference": numerical_reference,
        "code_sha256": _code_hashes(), "settings": _json_safe(asdict(enabled_settings)),
        "quarter": manifest["quarters"][0]}
    (folder/"joint_runner_pair_validation.json").write_text(json.dumps(_json_safe(report), indent=2), encoding="utf-8")
    return report


def inspect_parquet_target_times(path, *, member_rows=False):
    """Load the sole target-time column, never forecast values or label columns."""
    import pyarrow.parquet as pq
    times = pq.read_table(path, columns=["target_time"]).column("target_time").to_pandas()
    if member_rows:
        time_index = pd.DatetimeIndex(times)
        if time_index.tz is None or time_index.hasnans:
            raise ValueError("Member file requires nonmissing timezone-aware target times")
        time_index = time_index.tz_convert("UTC")
    else:
        time_index = utc_index(times)
    if len(time_index) == 0 or time_index.min() < PIT_START or time_index.max() >= HISTORY_END:
        raise ValueError("Input times must be in 2021--2025; values were not loaded")
    if not time_index.equals(time_index.floor("h")):
        raise ValueError("Input target times must be whole UTC hours")
    return time_index


def _read_development_parquet(path, test_authorized=False):
    """Refuse sealed time metadata before loading any data columns."""
    time_index = inspect_parquet_target_times(path)
    if time_index.max() >= TEST_START and not test_authorized:
        raise PermissionError("Input contains sealed targets; values were not loaded")
    frame = pd.read_parquet(path)
    if "target_time" in frame:
        frame = frame.set_index("target_time")
    return frame


def _read_member_development_parquet(path, test_authorized=False):
    """Time-only sealed guard allows repeated target/member rows by schema."""
    times = inspect_parquet_target_times(path, member_rows=True)
    if times.max() >= TEST_START and not test_authorized:
        raise PermissionError("Member file contains sealed targets; weather values were not loaded")
    return pd.read_parquet(path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--test-unlock-self-test", action="store_true", help="Synthetic 2024/2025 guard validation only")
    parser.add_argument("--pair-products-self-test", action="store_true", help="Synthetic pair mathematics and unchanged-primary guards")
    parser.add_argument("--enable-pairs-in-unlock-self-test", action="store_true", help="Enable pair scope in synthetic sealed fixtures only")
    parser.add_argument("--integration-backend", choices=("scipy_replicated", "elliptical_path"), default="scipy_replicated",
                        help="Synthetic validation backend; production backend is specified in frozen settings JSON")
    parser.add_argument("--destination", required=True)
    parser.add_argument("--shared")
    parser.add_argument("--weather-features")
    parser.add_argument("--timing")
    parser.add_argument("--label-availability")
    parser.add_argument("--marginal-manifest")
    parser.add_argument("--settings")
    parser.add_argument("--member-weather", help="Optional GEFS zone-prefix wide or zone-long member parquet")
    parser.add_argument("--ecc-calendar", help="Optional independent C7 known-calendar parquet; never added to R/C8 weather basis")
    parser.add_argument("--ecc-calendar-spec", help="Exact ordered C7 calendar specification JSON; required for independent-calendar test input")
    parser.add_argument("--allow-test", action="store_true")
    parser.add_argument("--frozen-config")
    parser.add_argument("--weather-feature-spec", help="Exact weather basis/aggregation specification; required for test input")
    parser.add_argument("--marginal-feature-spec", help="Actual upstream marginal feature-spec JSON; required for test input")
    parser.add_argument("--qnames", nargs="+", choices=("q90", "q95"), default=["q90", "q95"])
    parser.add_argument("--start", default="2022-01-01T00:00:00+00:00")
    parser.add_argument("--end", default="2024-01-01T00:00:00+00:00")
    args = parser.parse_args(argv)
    if args.self_test or args.test_unlock_self_test or args.pair_products_self_test:
        report = (synthetic_pair_products(args.destination, args.integration_backend) if args.pair_products_self_test else
                  synthetic_test_unlock(args.destination, args.integration_backend, args.enable_pairs_in_unlock_self_test) if args.test_unlock_self_test else
                  synthetic_self_test(args.destination, args.integration_backend))
        print(json.dumps({"status": report["status"], "evidence_type": report["evidence_type"],
                          "checks": report["checks"], "prediction_hours": report["prediction_hours"]}, indent=2))
        return report
    else:
        paths = [args.shared, args.weather_features, args.timing, args.label_availability, args.marginal_manifest, args.settings]
        if any(path is None for path in paths):
            parser.error("Require all shared/weather/timing/availability/provenance/settings inputs")
        config = json.loads(Path(args.settings).read_text(encoding="utf-8"))
        for name in ("linear_columns", "smooth_columns", "interactions", "df_candidates", "ecc_weather_columns",
                     "ecc_calendar_columns", "ecc_linear_weather_columns", "ecc_smooth_weather_columns", "pit_diagnostic_levels"):
            if name in config:
                config[name] = tuple(config[name])
        if "df_candidates" in config:
            config["df_candidates"] = tuple(np.inf if value == "inf" else value for value in config["df_candidates"])
        settings = JointForwardSettings(**config)
        marginal_provenance = json.loads(Path(args.marginal_manifest).read_text(encoding="utf-8-sig"))
        # Only timestamps and schema metadata are accessible until this gate.
        parquet_paths = [args.shared, args.weather_features, args.timing, args.label_availability]
        if args.ecc_calendar_spec and not args.ecc_calendar:
            parser.error('--ecc-calendar-spec requires --ecc-calendar')
        if args.ecc_calendar:
            parquet_paths.append(args.ecc_calendar)
            if not inspect_parquet_target_times(args.ecc_calendar).equals(inspect_parquet_target_times(args.shared)):
                raise ValueError('Independent ECC calendar rows must match shared timestamp metadata before values are loaded')
        test_required = _quarter(args.end) > TEST_START
        for path in parquet_paths:
            test_required = bool(inspect_parquet_target_times(path).max() >= TEST_START) or test_required
        if args.member_weather:
            test_required = bool(inspect_parquet_target_times(args.member_weather, member_rows=True).max() >= TEST_START) or test_required
        calendar_schema = weather_feature_schema(args.ecc_calendar) if args.ecc_calendar else None
        ecc_calendar_contract(settings, calendar_schema, args.ecc_calendar_spec, require_spec=test_required)
        freeze = check_joint_test_access(required=test_required, allow_test=args.allow_test,
            frozen_config=args.frozen_config, settings=settings, weather_schema=weather_feature_schema(args.weather_features),
            weather_feature_spec=args.weather_feature_spec, marginal_provenance=marginal_provenance,
            marginal_feature_spec=args.marginal_feature_spec, qnames=tuple(args.qnames),
            ecc_calendar_schema=calendar_schema, ecc_calendar_feature_spec=args.ecc_calendar_spec)
        authorized = freeze is not None
        shared = _read_development_parquet(args.shared, authorized)
        weather = _read_development_parquet(args.weather_features, authorized)
        timing = _read_development_parquet(args.timing, authorized).reset_index()
        available = _read_development_parquet(args.label_availability, authorized)["label_available_time"]
        member_weather = _read_member_development_parquet(args.member_weather, authorized) if args.member_weather else None
        ecc_calendar = _read_development_parquet(args.ecc_calendar, authorized) if args.ecc_calendar else None
        if args.ecc_calendar:
            paths.append(args.ecc_calendar)
        if args.member_weather:
            paths.append(args.member_weather)
        paths += [path for path in (args.frozen_config, args.weather_feature_spec, args.marginal_feature_spec, args.ecc_calendar_spec) if path]
        _, manifest = run_joint_forward(shared, weather, timing, label_available_times=available,
            marginal_provenance=marginal_provenance, settings=settings, start=args.start, end=args.end,
            qnames=tuple(args.qnames), destination=args.destination, input_files=paths, member_weather=member_weather,
            allow_test=args.allow_test, frozen_config=args.frozen_config,
            weather_feature_spec=args.weather_feature_spec, marginal_feature_spec=args.marginal_feature_spec,
            ecc_calendar=ecc_calendar, ecc_calendar_feature_spec=args.ecc_calendar_spec)
        manifest["CLI_frozen_gate_before_any_label_column_read"] = bool(authorized)
        (Path(args.destination)/"joint_runner_manifest.json").write_text(json.dumps(_json_safe(manifest), indent=2), encoding="utf-8")
        print(json.dumps({"status": manifest["status"], "forecast_rows": manifest["forecast_rows"],
                          "prediction_file": manifest["prediction_file"]}, indent=2))
        return manifest


if __name__ == "__main__":
    main()
