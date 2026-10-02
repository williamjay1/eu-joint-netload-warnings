"""Predeclared calendar-only threshold CANDIDATES; no automatic rule selection.

This module does not promote an adaptive threshold to the primary study target.
At each quarter origin, it fits q90 and q95 to the previous three calendar years,
ending origin minus seven days. Predictors are timestamp functions only: UTC
hour, annual harmonics interacted with four six-hour blocks, and optional slow
linear trends for those blocks. No weather or forecast covariates are accepted.
The paired quantiles use identical structure, then monotone rearrangement.

Callers must enforce their study's development/test access policy. The CLI runs
synthetic validation or explicitly requested 2019--2023 development evaluation;
the development evaluator rejects any input containing sealed test years.
"""

from __future__ import annotations
from release_paths import work_path as _work_path, raw_path as _raw_path

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import time
import warnings

import numpy as np
import pandas as pd
from scipy.sparse import csc_matrix
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import QuantileRegressor


SCRIPT_VERSION = "1.1-calendar-threshold-candidate-resumable"
QUANTILES = (.9, .95)
REGULARIZATION_ALPHA = 1e-4
DEFAULT_ZONES = ("DE_LU", "FR", "BE")


def _utc_timestamp(value):
    stamp = pd.Timestamp(value)
    if pd.isna(stamp):
        raise ValueError("Timestamp cannot be NaT")
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def _hourly_utc_index(value):
    index = pd.DatetimeIndex(value)
    if index.tz is None:
        raise ValueError("Use timezone-aware UTC timestamps; naive labels are ambiguous")
    index = index.tz_convert("UTC")
    if index.hasnans or index.has_duplicates or (index != index.floor("h")).any():
        raise ValueError("Expected unique UTC hourly interval-start timestamps")
    return index


def _quarter_origin(value):
    origin = _utc_timestamp(value)
    if origin.day != 1 or origin.month not in (1, 4, 7, 10) or origin != origin.normalize():
        raise ValueError("origin must be a UTC calendar-quarter boundary")
    return origin


def _disk_space():
    free = shutil.disk_usage(_work_path()).free
    if free < 2*1024**3:
        raise OSError("Less than 2 GiB free on D:; threshold computation was not started")
    return free


def _calendar_design(index, cutoff, trend):
    """Deterministic low-dimensional basis; no observed/forecast weather input."""
    hour = index.hour.to_numpy()
    block = hour//6
    # A leap-calendar mapping keeps each month/day at the same phase each year.
    days = np.array([pd.Timestamp(2000, stamp.month, stamp.day).dayofyear-1
                     for stamp in index], dtype=float)
    phase = (days+hour/24)/366
    columns = {f"utc_hour_{h}": (hour == h).astype(float) for h in range(1, 24)}
    for b in range(4):
        mask = (block == b).astype(float)
        for harmonic in (1, 2):
            angle = 2*np.pi*harmonic*phase
            columns[f"annual_sin_{harmonic}_block_{b}"] = mask*np.sin(angle)
            columns[f"annual_cos_{harmonic}_block_{b}"] = mask*np.cos(angle)
        if trend:
            years_from_cutoff = (index-cutoff).total_seconds().to_numpy()/(365.2425*86400)
            columns[f"linear_years_block_{b}"] = mask*years_from_cutoff
    return np.column_stack(list(columns.values())), list(columns)


def pinball_evaluation(observed, predicted, q, training_scales=None):
    """Explicit univariate evaluation only; no event counts or model selection.

    A zone's score uses only its own observed outcomes and finite predictions.
    `training_scales` optionally normalises MW loss by pre-origin training IQR.
    This function never changes thresholds or selects the trend specification.
    """
    if not 0 < q < 1:
        raise ValueError("q must be between zero and one")
    if not observed.index.equals(predicted.index) or list(observed) != list(predicted):
        raise ValueError("Observed/predicted timestamps and zone order must match exactly")
    result = []
    for zone in observed:
        y = observed[zone].to_numpy(dtype=float)
        p = predicted[zone].to_numpy(dtype=float)
        if np.isinf(y).any() or np.isinf(p).any():
            raise ValueError("Infinite values require an upstream data audit")
        valid = np.isfinite(y) & np.isfinite(p)
        residual = y[valid]-p[valid]
        loss = np.maximum(q*residual, (q-1)*residual)
        record = {"zone": zone, "q": float(q), "n": int(valid.sum()),
                  "pinball_mw": float(loss.mean()) if len(loss) else None,
                  "observed_exceedance_fraction": float((residual > 0).mean()) if len(loss) else None}
        if training_scales is not None:
            scale = float(training_scales[zone])
            if not np.isfinite(scale) or scale <= 0:
                raise ValueError("Normalisation scales must be positive pre-origin training scales")
            record["normalisation_training_scale_mw"] = scale
            record["pinball_training_scale_units"] = float(loss.mean()/scale) if len(loss) else None
        result.append(record)
    return result


class QuarterlyCalendarThresholds:
    """Quarter-frozen, calendar-only q90/q95 threshold candidate.

    `trend=False` and `trend=True` are the only two structural candidates.
    The regularisation level and harmonic order are fixed, not tuned here.
    At least 365 distinct observed dates per zone are required. No subsampling
    is performed: solver timeouts are recorded and raised rather than hidden.
    """

    def __init__(self, trend=True, zones=DEFAULT_ZONES, min_coverage=.8,
                 solver_time_limit=120.0):
        if type(trend) is not bool:
            raise ValueError("trend must be explicitly True or False")
        self.trend = trend
        self.zones = tuple(zones)
        if not self.zones or len(set(self.zones)) != len(self.zones):
            raise ValueError("Provide distinct zone response columns")
        if not 0 < min_coverage <= 1 or solver_time_limit <= 0:
            raise ValueError("Invalid coverage or solver time limit")
        self.min_coverage = float(min_coverage)
        self.solver_time_limit = float(solver_time_limit)

    def fit(self, net_load, origin):
        free = _disk_space()
        self.origin_ = _quarter_origin(origin)
        self.end_ = self.origin_+pd.DateOffset(months=3)
        self.cutoff_ = self.origin_-pd.Timedelta(days=7)
        self.history_start_ = self.cutoff_-pd.DateOffset(years=3)
        index = _hourly_utc_index(net_load.index)
        missing = [zone for zone in self.zones if zone not in net_load.columns]
        if missing:
            raise ValueError(f"Missing net-load response columns: {missing}")
        selected = (index >= self.history_start_) & (index < self.cutoff_)
        # Select historical rows BEFORE reading/checking numerical label values.
        # Future outcome values (including NaN/inf) cannot influence this fit.
        history = net_load.loc[selected, list(self.zones)].copy()
        history.index = index[selected]
        history = history.sort_index().apply(pd.to_numeric, errors="raise")
        if history.empty or np.isinf(history.to_numpy(dtype=float)).any():
            raise ValueError("Empty or infinite historical observations")
        self.metadata_ = {
            "status": "fitting", "script_version": SCRIPT_VERSION,
            "candidate_only_not_selected_primary": True,
            "trend": self.trend, "quantiles": list(QUANTILES),
            "origin": self.origin_.isoformat(), "prediction_end_exclusive": self.end_.isoformat(),
            "first_forecast_issue": (self.origin_-pd.Timedelta(hours=12)).isoformat(),
            "history_start_inclusive": self.history_start_.isoformat(),
            "history_end_exclusive": self.cutoff_.isoformat(), "embargo_days": 7,
            "minimum_distinct_observed_days": 365, "minimum_coverage": self.min_coverage,
            "D_free_bytes_before_fit": free, "weather_inputs_used": False,
            "predictor_source": "UTC timestamp functions only",
            "alpha": REGULARIZATION_ALPHA, "alpha_scale": "standardised response and feature columns",
            "solver": "scikit-learn QuantileRegressor / scipy linprog highs",
            "highs_threads": 2,
            "solver_time_limit_seconds_per_quantile_zone": self.solver_time_limit,
            "monotonicity_policy": "sort paired q90/q95 predictions at each timestamp and zone",
            "warnings": ["Calendar/trend quantiles do not guarantee future 10% or 5% exceedance rates.",
                         "Rearrangement may alter the individual fitted quantile functions.",
                         "Observation release/revision vintages require a separate upstream audit."],
            "zones": {},
        }
        started = time.monotonic()
        self.models_, self.training_scales_ = {}, {}
        try:
            for zone in self.zones:
                valid = history[zone].notna()
                times = history.index[valid]
                if len(times) == 0:
                    raise ValueError(f"{zone}: no observed history")
                days = len(times.normalize().unique())
                span_days = (times.max()+pd.Timedelta(hours=1)-times.min()).total_seconds()/86400
                expected_hours = int((self.cutoff_-max(self.history_start_, times.min())).total_seconds()/3600)
                coverage = len(times)/expected_hours
                hourly_counts = np.bincount(times.hour.to_numpy(), minlength=24)
                if days < 365 or span_days < 364 or coverage < self.min_coverage or hourly_counts.min() < 15:
                    raise ValueError(f"{zone}: insufficient history: days={days}, coverage={coverage:.4f}, min_hour_n={hourly_counts.min()}")
                if times.max()+pd.Timedelta(hours=1) > self.cutoff_:
                    raise AssertionError("Training label interval crosses the cutoff")
                X, names = _calendar_design(times, self.cutoff_, self.trend)
                x_scale = X.std(axis=0)
                x_scale[x_scale < 1e-10] = 1.0
                X = csc_matrix(X/x_scale)
                y = history.loc[valid, zone].to_numpy(dtype=float)
                center = float(np.median(y))
                scale = float(np.subtract(*np.quantile(y, [.75, .25])))
                scale_method = "training_IQR"
                if scale <= 1e-10:
                    scale, scale_method = float(np.std(y)), "training_SD_IQR_zero"
                if scale <= 1e-10:
                    scale, scale_method = 1.0, "one_MW_constant_response"
                self.training_scales_[zone] = scale
                records = []
                fits = []
                for q in QUANTILES:
                    tic = time.monotonic()
                    model = QuantileRegressor(quantile=q, alpha=REGULARIZATION_ALPHA,
                                              fit_intercept=True, solver="highs",
                                              solver_options={"time_limit": self.solver_time_limit,
                                                              "presolve": True, "threads": 2})
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always", ConvergenceWarning)
                        model.fit(X, (y-center)/scale)
                    elapsed = time.monotonic()-tic
                    if any(issubclass(w.category, ConvergenceWarning) for w in caught):
                        raise RuntimeError(f"{zone} q={q}: solver failed to certify convergence within {elapsed:.3f}s")
                    if not np.isfinite(model.coef_).all() or not np.isfinite(model.intercept_):
                        raise RuntimeError(f"{zone} q={q}: nonfinite coefficients")
                    record = {"q": q, "elapsed_seconds": elapsed,
                              "n_iter": int(model.n_iter_), "warnings": [str(w.message) for w in caught]}
                    if self.trend:
                        record["linear_trend_mw_per_year_by_utc_six_hour_block"] = {
                            str(block): float(model.coef_[names.index(f"linear_years_block_{block}")]
                                              *scale/x_scale[names.index(f"linear_years_block_{block}")])
                            for block in range(4)}
                    records.append(record)
                    fits.append(model)
                self.models_[zone] = {"models": fits, "x_scale": x_scale,
                                      "y_center": center, "y_scale": scale, "feature_names": names}
                self.metadata_["zones"][zone] = {
                    "n": len(y), "observed_days": days, "observed_span_days": span_days,
                    "coverage_from_first_training_observation_to_cutoff": coverage,
                    "first_target": times.min().isoformat(), "last_target": times.max().isoformat(),
                    "last_label_end": (times.max()+pd.Timedelta(hours=1)).isoformat(),
                    "training_scale_mw": scale, "training_scale_method": scale_method,
                    "response_center_mw": center, "number_predictors": len(names),
                    "predictors": names, "solver_fits": records,
                }
            self.metadata_.update(status="completed", elapsed_seconds=time.monotonic()-started)
        except Exception as error:
            self.metadata_.update(status="failed", elapsed_seconds=time.monotonic()-started,
                                  error_type=type(error).__name__, error=str(error))
            raise
        return self

    def predict_with_metadata(self, times):
        if not hasattr(self, "metadata_") or self.metadata_.get("status") != "completed":
            raise RuntimeError("Fit must finish successfully before prediction")
        index = _hourly_utc_index(times)
        if ((index < self.origin_) | (index >= self.end_)).any():
            raise ValueError("A fitted candidate is valid only inside its frozen target quarter")
        X, names = _calendar_design(index, self.cutoff_, self.trend)
        raw = np.empty((len(index), len(QUANTILES), len(self.zones)))
        for j, zone in enumerate(self.zones):
            bundle = self.models_[zone]
            if names != bundle["feature_names"]:
                raise AssertionError("Training/prediction calendar feature order mismatch")
            design = csc_matrix(X/bundle["x_scale"])
            for k, model in enumerate(bundle["models"]):
                raw[:, k, j] = bundle["y_center"]+bundle["y_scale"]*model.predict(design)
        crossings = raw[:, 0, :] > raw[:, 1, :]
        ordered = np.sort(raw, axis=1)
        values = {q: pd.DataFrame(ordered[:, k, :], index=index, columns=self.zones)
                  for k, q in enumerate(QUANTILES)}
        diagnostics = {"prediction_rows": len(index), "raw_crossing_zone_hours": int(crossings.sum()),
                       "raw_crossing_fraction": float(crossings.mean()) if crossings.size else 0.0,
                       "maximum_rearrangement_mw": float(np.max(np.abs(raw-ordered))) if raw.size else 0.0,
                       "rearrangement": "sort paired conditional quantiles", "uses_observed_outcomes": False}
        return values, diagnostics

    def predict(self, times, q=.9):
        if q not in QUANTILES:
            raise ValueError("Only the predeclared q90 and q95 thresholds are fitted")
        return self.predict_with_metadata(times)[0][q]


def synthetic_self_test(destination=None):
    """Small sparse synthetic execution; no observational threshold conclusions."""
    free = _disk_space()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    output = Path(destination or str(_work_path(f"results/adaptive_selftest_{stamp}"))).resolve()
    if not output.is_relative_to(_work_path().resolve()) or not output.name.startswith("adaptive_"):
        raise ValueError("Use a new adaptive_ output directory on D:")
    output.mkdir(parents=True, exist_ok=False)
    report = {"status": "running", "evidence_type": "SYNTHETIC_ONLY_NO_REAL_ENERGY_DATA",
              "created_utc": datetime.now(timezone.utc).isoformat(), "D_free_bytes_before_fit": free,
              "checks": [], "fits": {}, "interpretation": "Software/chronology checks only. No candidate was selected as primary."}
    report_path = output/"adaptive_selftest_report.json"
    started = time.monotonic()
    try:
        origin = pd.Timestamp("2023-01-01", tz="UTC")
        cutoff = origin-pd.Timedelta(days=7)
        # 4 observations per day, rotating across every UTC hour in six days.
        # Deliberately sparse to keep the exact LP test small; production default is 80% coverage.
        dates = pd.date_range(cutoff-pd.Timedelta(days=400), cutoff, freq="D", inclusive="left")
        times = pd.DatetimeIndex([day+pd.Timedelta(hours=i%6+6*b)
                                 for i, day in enumerate(dates) for b in range(4)])
        future = pd.date_range(origin, origin+pd.Timedelta(days=7), freq="h", inclusive="left")
        index = times.append(future).sort_values()
        rng = np.random.default_rng(270929)
        elapsed = (index-cutoff).total_seconds().to_numpy()/(365.2425*86400)
        y = (100+12*np.sin(2*np.pi*index.hour.to_numpy()/24)
             +8*np.cos(2*np.pi*index.dayofyear.to_numpy()/366)
             -6*elapsed+rng.normal(0, 3, len(index)))
        data = pd.DataFrame({"FR": y}, index=index)
        data.index.name = "target_time"
        data.to_parquet(output/"adaptive_synthetic_panel.parquet")
        fits = {}
        for trend in (False, True):
            model = QuarterlyCalendarThresholds(trend=trend, zones=("FR",), min_coverage=.15,
                                                solver_time_limit=30).fit(data, origin)
            prediction, details = model.predict_with_metadata(future)
            assert (prediction[.9].to_numpy() <= prediction[.95].to_numpy()).all()
            assert np.isfinite(prediction[.9].to_numpy()).all()
            meta = model.metadata_["zones"]["FR"]
            assert _utc_timestamp(meta["last_label_end"]) <= cutoff
            assert _utc_timestamp(model.metadata_["history_end_exclusive"]) < origin-pd.Timedelta(hours=12)
            assert meta["observed_days"] == 400
            fits[trend] = (model, prediction)
            report["fits"][str(trend)] = {"fit": model.metadata_, "prediction": details}
        report["checks"].append("both_predeclared_structures_fit_with_identical_q90_q95_structure")
        report["checks"].append("observed_label_intervals_and_seven_day_embargo_respected")
        report["checks"].append("predicted_q90_never_exceeds_q95_after_declared_rearrangement")
        changed = data.copy()
        changed.loc[changed.index >= cutoff, "FR"] = np.inf
        altered = QuarterlyCalendarThresholds(trend=True, zones=("FR",), min_coverage=.15,
                                               solver_time_limit=30).fit(changed, origin)
        for q in QUANTILES:
            np.testing.assert_allclose(altered.predict(future, q).to_numpy(), fits[True][1][q].to_numpy(),
                                       rtol=0, atol=1e-10)
        report["checks"].append("future_label_values_including_infinity_do_not_affect_thresholds")
        for invalid in (origin-pd.Timedelta(hours=1), origin+pd.DateOffset(months=3)):
            try:
                fits[True][0].predict(pd.DatetimeIndex([invalid]))
                raise AssertionError("Accepted target outside frozen quarter")
            except ValueError:
                pass
        report["checks"].append("quarter_prediction_boundaries_guarded")
        short_history = data.loc[data.index >= cutoff-pd.Timedelta(days=300)]
        try:
            QuarterlyCalendarThresholds(zones=("FR",), min_coverage=.15).fit(short_history, origin)
            raise AssertionError("Accepted fewer than 365 observed dates")
        except ValueError:
            pass
        report["checks"].append("less_than_one_year_of_observed_dates_is_rejected")
        # Directly verify sorting behavior even if normal fitting happens not to cross.
        crossing_model = fits[False][0]
        crossing_model.models_["FR"]["models"][0].intercept_ += 100
        _, crossing = crossing_model.predict_with_metadata(future)
        assert crossing["raw_crossing_zone_hours"] == len(future)
        assert crossing["maximum_rearrangement_mw"] > 0
        report["checks"].append("deliberate_quantile_crossing_is_reported_and_rearranged")
        scores = pinball_evaluation(data.loc[future], fits[True][1][.9], .9,
                                    fits[True][0].training_scales_)
        assert scores[0]["n"] == len(future) and scores[0]["pinball_mw"] >= 0
        # A tiny hand calculation checks the loss formula itself.
        hand_y = pd.DataFrame({"FR": [2., -1.]}, index=future[:2])
        hand_p = pd.DataFrame({"FR": [0., 0.]}, index=future[:2])
        assert abs(pinball_evaluation(hand_y, hand_p, .9)[0]["pinball_mw"]-.95) < 1e-12
        report["checks"].append("univariate_pinball_interface_and_hand_calculation")
        report.update(status="passed", elapsed_seconds=time.monotonic()-started,
                      n_checks=len(report["checks"]), synthetic_observations=len(data),
                      raw_training_rows_per_fit=len(times), evaluation_example=scores,
                      source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    except Exception as error:
        report.update(status="failed", elapsed_seconds=time.monotonic()-started,
                      error_type=type(error).__name__, error=str(error))
        report_path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
        raise
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"status": report["status"], "n_checks": report["n_checks"],
                      "elapsed_seconds": report["elapsed_seconds"], "report": str(report_path)}), flush=True)
    return report


def _resume_completed_candidates(resume_from, wide, input_hash):
    """Reuse only complete, validated quarter/candidate prediction artifacts.

    The prior incomplete report is kept untouched. An unsaved partial quarter
    is re-fitted because scores alone cannot recover its threshold predictions.
    The accepted predecessor hash identifies the same untouched model code.
    """
    prior_path = Path(resume_from)/"adaptive_development_report.json"
    prior = json.loads(prior_path.read_text(encoding="utf-8"))
    accepted_predecessor = "410f0164c7ec0ef05c11b2d30b84710490a9b9a797a513b821972ba0cca8ecc0"
    current = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if prior.get("input_sha256") != input_hash:
        raise ValueError("Resume input hash differs from the prior execution")
    if prior.get("script_sha256") not in (accepted_predecessor, current):
        raise ValueError("Resume requires the unchanged model implementation or documented predecessor")
    if prior.get("test_outcomes_inspected") is not False or prior.get("candidate_selected") is not None:
        raise PermissionError("Prior run cannot certify development-only, unselected candidates")
    complete = {}
    for path in sorted(Path(resume_from).glob("adaptive_????????_*.parquet")):
        frame = pd.read_parquet(path)
        if frame.empty:
            raise ValueError(f"Empty resume artifact: {path}")
        origin = _quarter_origin(frame.index[0])
        expected = pd.date_range(origin, origin+pd.DateOffset(months=3), freq="h", inclusive="left")
        candidate = path.stem.split("_", 2)[2]
        if candidate == "fixed_historical_reference":
            continue
        if candidate not in ("calendar_only", "calendar_plus_linear_trend"):
            raise ValueError(f"Unexpected resume candidate: {candidate}")
        columns = [f"q{int(q*100)}_{zone}" for zone in DEFAULT_ZONES for q in QUANTILES]
        if not frame.index.equals(expected) or set(frame.columns) != set(columns):
            raise ValueError(f"Incomplete resume timestamps or columns: {path}")
        if not np.isfinite(frame.to_numpy()).all() or any((frame[f"q90_{z}"] > frame[f"q95_{z}"]).any() for z in DEFAULT_ZONES):
            raise ValueError(f"Invalid resume quantiles: {path}")
        fits = [f for f in prior["fits"] if f["candidate"] == candidate and
                _utc_timestamp(f["metadata"]["origin"]) == origin and f["metadata"]["status"] == "completed"]
        scores = [s for s in prior["scores"] if s["candidate"] == candidate and _utc_timestamp(s["origin"]) == origin]
        if len(fits) != 3 or len(scores) != 6:
            raise ValueError(f"Resume artifact lacks all completed zone/quantile records: {path}")
        seen = set()
        for fit in fits:
            metadata = fit["metadata"]
            zones = list(metadata["zones"])
            if len(zones) != 1 or zones[0] in seen:
                raise ValueError("Ambiguous or duplicate fitted zones in resume")
            zone = zones[0]
            seen.add(zone)
            if metadata["trend"] != (candidate == "calendar_plus_linear_trend") or \
                    metadata["alpha"] != REGULARIZATION_ALPHA or metadata["embargo_days"] != 7 or \
                    _utc_timestamp(metadata["history_end_exclusive"]) != origin-pd.Timedelta(days=7):
                raise ValueError("Resume candidate specification differs")
            scale = metadata["zones"][zone]["training_scale_mw"]
            for q in QUANTILES:
                prediction = frame[[f"q{int(q*100)}_{zone}"]].rename(columns={f"q{int(q*100)}_{zone}": zone})
                recomputed = pinball_evaluation(wide.reindex(expected)[[zone]], prediction, q, {zone: scale})[0]
                saved = [s for s in scores if s["zone"] == zone and s["q"] == q]
                if len(saved) != 1 or any(not np.isclose(recomputed[k], saved[0][k], rtol=1e-12, atol=1e-12)
                                          for k in ("n", "pinball_mw", "pinball_training_scale_units", "observed_exceedance_fraction")):
                    raise ValueError(f"Resume artifact scores do not reproduce: {path} {zone} {q}")
        complete[(origin.isoformat(), candidate)] = {"frame": frame, "fits": fits, "scores": scores,
            "artifact": str(path.resolve()), "artifact_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    return complete, {"prior_report": str(prior_path.resolve()),
                      "prior_report_sha256": hashlib.sha256(prior_path.read_bytes()).hexdigest(),
                      "prior_input_sha256": prior["input_sha256"], "prior_script_sha256": prior["script_sha256"],
                      "prior_status": prior["status"], "prior_error_type": prior.get("error_type"),
                      "prior_error": prior.get("error"), "complete_candidate_quarters_reused": len(complete),
                      "prior_fits_not_reused_without_complete_prediction_artifact": len(prior["fits"])-3*len(complete),
                      "original_report_and_artifacts_modified": False}


def _annual_univariate_summary(scores):
    annual = []
    for keys, rows in pd.DataFrame(scores).groupby(["candidate", "zone", "q", "year"], sort=True):
        weights = rows.n.to_numpy(dtype=float)
        record = {"candidate": str(keys[0]), "zone": str(keys[1]), "q": float(keys[2]), "year": int(keys[3]),
                  "n": int(weights.sum()), "expected_hours": int(rows.expected_hours.sum())}
        for name in ("pinball_mw", "pinball_training_scale_units", "observed_exceedance_fraction"):
            record[name] = float(np.average(rows[name], weights=weights))
        annual.append(record)
    summaries = []
    for (candidate, year), rows in pd.DataFrame(annual).query("q == 0.9").groupby(["candidate", "year"]):
        summaries.append({"candidate": candidate, "year": int(year),
                          "equal_zone_mean_q90_normalised_pinball": float(rows.pinball_training_scale_units.mean())})
    return annual, summaries


def _fixed_reference_comparison(wide, output, result):
    """Compare threshold candidates only on common, issue-eligible observations."""
    from forecast_design import FixedSeasonalThreshold
    ref_end = pd.Timestamp("2022-01-01", tz="UTC")
    available = ref_end+pd.Timedelta(days=7)
    reference = wide.loc[wide.index < ref_end]
    models = {q: FixedSeasonalThreshold(q=q).fit(reference) for q in QUANTILES}
    common_scores = []
    for origin in pd.date_range("2022-01-01", "2024-01-01", freq="QS", inclusive="left", tz="UTC"):
        times = pd.date_range(origin, origin+pd.DateOffset(months=3), freq="h", inclusive="left")
        eligible = times.normalize()-pd.Timedelta(hours=12) >= available
        eligible_mask = pd.Series(eligible, index=times)
        outcomes = wide.reindex(times).where(eligible_mask, np.nan, axis=0)
        fixed_table = pd.DataFrame(index=times)
        fixed_table.index.name = "target_time"
        for q, model in models.items():
            forecast = model.predict(times).where(eligible_mask, np.nan, axis=0)
            for zone in DEFAULT_ZONES:
                fixed_table[f"q{int(q*100)}_{zone}"] = forecast[zone]
        fixed_table["threshold_available_at_issue"] = eligible
        fixed_table.to_parquet(output/f"adaptive_{origin.strftime('%Y%m%d')}_fixed_historical_reference.parquet")
        for candidate in ("calendar_only", "calendar_plus_linear_trend", "fixed_historical_reference"):
            table = fixed_table if candidate == "fixed_historical_reference" else \
                pd.read_parquet(output/f"adaptive_{origin.strftime('%Y%m%d')}_{candidate}.parquet")
            for zone in DEFAULT_ZONES:
                fit = next(f for f in result["fits"] if f["candidate"] == "calendar_only" and
                           f["metadata"]["origin"] == origin.isoformat() and zone in f["metadata"]["zones"])
                scale = fit["metadata"]["zones"][zone]["training_scale_mw"]
                for q in QUANTILES:
                    prediction = table[[f"q{int(q*100)}_{zone}"]].rename(columns={f"q{int(q*100)}_{zone}": zone})
                    score = pinball_evaluation(outcomes[[zone]], prediction, q, {zone: scale})[0]
                    score.update(candidate=candidate, origin=origin.isoformat(), year=origin.year,
                                 expected_hours=int(eligible.sum()), normalisation_shared_across_candidates=True)
                    common_scores.append(score)
    annual, summaries = _annual_univariate_summary(common_scores)
    return {"reference_start_inclusive": "2019-01-01T00:00:00+00:00", "reference_end_exclusive": ref_end.isoformat(),
            "assumed_label_embargo_days": 7, "reference_available_at": available.isoformat(),
            "first_issue_eligible_target_day": "2022-01-09T00:00:00+00:00",
            "actual_historical_release_vintages_verified": False,
            "comparison_policy": "All three candidates evaluated on the same issue-eligible hours; identical pre-origin training IQR by zone and quarter",
            "scores": common_scores, "annual_zone_scores": annual,
            "prespecified_metric_summary": summaries, "candidate_selected": None,
            "forecast_design_sha256": hashlib.sha256(Path(__file__).with_name("forecast_design.py").read_bytes()).hexdigest()}


def evaluate_development(panel_path, destination, max_runtime_seconds=480, resume_from=None):
    """Evaluate only corrected 2019--2023 long data; never select a candidate.

    Candidate comparison is fixed before fitting: q90 pinball loss in each
    zone's pre-origin training-scale units, hour-weighted within zone and then
    equal-weighted across zones. q95, coverage and exceedance frequencies are
    diagnostics. Joint events/copula improvements are not computed anywhere.
    The caller is responsible for approving the upstream observation variant.
    """
    free = _disk_space()
    output = Path(destination).resolve()
    if not output.is_relative_to(_work_path().resolve()) or not output.name.startswith("adaptive_"):
        raise ValueError("Use a new adaptive_ output directory on D:")
    if output.exists():
        raise FileExistsError(output)
    if max_runtime_seconds <= 0:
        raise ValueError("Runtime budget must be positive")
    panel_path = Path(panel_path)
    source = pd.read_parquet(panel_path)
    required = {"timestamp", "zone", "net_load_mw"}
    if not required.issubset(source):
        raise ValueError("Expected long timestamp/zone/net_load_mw development panel")
    source["timestamp"] = pd.to_datetime(source.timestamp, utc=True)
    if (source.timestamp < pd.Timestamp("2019-01-01", tz="UTC")).any() or \
            (source.timestamp >= pd.Timestamp("2024-01-01", tz="UTC")).any():
        raise PermissionError("This evaluator only accepts 2019--2023 input; sealed years are forbidden")
    source["zone"] = source.zone.replace({"DE-LU": "DE_LU"})
    if source.duplicated(["timestamp", "zone"]).any():
        raise ValueError("Duplicate timestamp-zone observations")
    wide = source.pivot(index="timestamp", columns="zone", values="net_load_mw")[list(DEFAULT_ZONES)]
    input_hash = hashlib.sha256(panel_path.read_bytes()).hexdigest()
    resumed, resume_audit = _resume_completed_candidates(resume_from, wide, input_hash) if resume_from else ({}, None)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    result = {
        "status": "running", "evidence_type": "OBSERVATIONAL_DEVELOPMENT_CANDIDATE_EVALUATION_ONLY",
        "test_outcomes_inspected": False, "candidate_selected": None,
        "created_utc": datetime.now(timezone.utc).isoformat(), "input": str(panel_path.resolve()),
        "input_sha256": input_hash,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "D_free_bytes_before_run": free, "threads": 2,
        "runtime_budget_seconds": max_runtime_seconds,
        "primary_comparison_if_later_adopted": "q90 pinball / pre-origin zone training IQR; hour-weighted within each zone; equal-weighted across zones",
        "selection_policy": "No automatic selection; no joint-event counts or copula improvements are inspected.",
        "source_quality_caveat": "Input is the v3 night-zero-PV candidate; its night-zero assumption is not validated by these model scores.",
        "drift_measure": "learned block-specific linear slope MW/year; absolute threshold summaries are not causal drift estimates",
        "fixed_historical_threshold_reference": "61-day cyclic same-hour empirical q90/q95 from 2019--2021; issue-gated from 2022-01-09",
        "fits": [], "scores": [],
        "resume_audit": resume_audit, "resumed_artifacts": [],
    }
    report_path = output/"adaptive_development_report.json"
    if "quality_flag" in source:
        result["input_quality_flag_counts"] = {str(key): int(value) for key, value in source.quality_flag.value_counts(dropna=False).items()}
    try:
        for origin in pd.date_range("2022-01-01", "2024-01-01", freq="QS", inclusive="left", tz="UTC"):
            times = pd.date_range(origin, origin+pd.DateOffset(months=3), freq="h", inclusive="left")
            outcomes = wide.reindex(times)
            for trend in (False, True):
                candidate = "calendar_plus_linear_trend" if trend else "calendar_only"
                reused = resumed.get((origin.isoformat(), candidate))
                if reused is not None:
                    result["fits"].extend(reused["fits"])
                    result["scores"].extend(reused["scores"])
                    artifact_path = output/f"adaptive_{origin.strftime('%Y%m%d')}_{candidate}.parquet"
                    shutil.copy2(reused["artifact"], artifact_path)
                    if hashlib.sha256(artifact_path.read_bytes()).hexdigest() != reused["artifact_sha256"]:
                        raise IOError("Copied resume prediction artifact hash differs")
                    result["resumed_artifacts"].append({k: reused[k] for k in ("artifact", "artifact_sha256")})
                    report_path.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
                    print(json.dumps({"stage": "completed_candidate_quarter_reused", "origin": origin.isoformat(),
                                      "candidate": candidate}), flush=True)
                    continue
                quarter_predictions = pd.DataFrame(index=times)
                quarter_predictions.index.name = "target_time"
                for zone in DEFAULT_ZONES:
                    if time.monotonic()-started >= max_runtime_seconds:
                        raise TimeoutError("Development time budget reached between fits; partial results retained without candidate selection")
                    candidate_model = QuarterlyCalendarThresholds(trend=trend, zones=(zone,),
                                                                  solver_time_limit=45)
                    try:
                        candidate_model.fit(wide, origin)
                    except Exception:
                        if hasattr(candidate_model, "metadata_"):
                            result["fits"].append({"candidate": candidate, "metadata": candidate_model.metadata_})
                        raise
                    forecasts, diagnostics = candidate_model.predict_with_metadata(times)
                    result["fits"].append({"candidate": candidate, "metadata": candidate_model.metadata_,
                                           "prediction_diagnostics": diagnostics})
                    for q in QUANTILES:
                        table = forecasts[q]
                        score = pinball_evaluation(outcomes[[zone]], table, q,
                                                   candidate_model.training_scales_)[0]
                        values = table[zone].to_numpy()
                        score.update(candidate=candidate, origin=origin.isoformat(), year=origin.year,
                                     expected_hours=len(times),
                                     threshold_mw_quantiles={str(p): float(np.quantile(values, p)) for p in (0, .1, .5, .9, 1)})
                        result["scores"].append(score)
                        quarter_predictions[f"q{int(q*100)}_{zone}"] = values
                    result["elapsed_seconds"] = time.monotonic()-started
                    report_path.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
                    print(json.dumps({"stage": "candidate_zone_quarter_completed", "origin": origin.isoformat(),
                                      "candidate": candidate, "zone": zone,
                                      "elapsed_seconds": result["elapsed_seconds"]}), flush=True)
                quarter_predictions.to_parquet(output/f"adaptive_{origin.strftime('%Y%m%d')}_{candidate}.parquet")
        annual, summaries = _annual_univariate_summary(result["scores"])
        result["annual_zone_scores"] = annual
        result["fixed_historical_common_reference_comparison"] = _fixed_reference_comparison(wide, output, result)
        result.update(status="completed", elapsed_seconds=time.monotonic()-started,
                      prespecified_metric_summary=summaries)
    except Exception as error:
        result.update(status="incomplete", elapsed_seconds=time.monotonic()-started,
                      error_type=type(error).__name__, error=str(error))
        report_path.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
        raise
    report_path.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"status": result["status"], "report": str(report_path),
                      "elapsed_seconds": result["elapsed_seconds"], "candidate_selected": None}), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--self-test", action="store_true")
    mode.add_argument("--development-panel")
    parser.add_argument("--destination")
    parser.add_argument("--max-runtime-seconds", type=float, default=480)
    parser.add_argument("--resume-from", help="Prior execution directory; only complete validated prediction artifacts are reused")
    arguments = parser.parse_args()
    if arguments.self_test:
        synthetic_self_test(arguments.destination)
    else:
        if not arguments.destination:
            parser.error("--destination is required for development evaluation")
        evaluate_development(arguments.development_panel, arguments.destination, arguments.max_runtime_seconds, arguments.resume_from)
