"""Finite-threshold margin sensitivity diagnostics, with explicit certification limits.

The real-data analysis is a constant-within-cell CDF-shift sensitivity analysis.
It does not convert cell-average calibration into pointwise conditional coverage.
The simulation has finite homogeneous observed states, so simultaneous binomial
intervals genuinely cover the state-specific threshold positions and event risk.
All new outputs are written under the owned upgrade_sensitivity_20261002 folder.
"""
from __future__ import annotations
from release_paths import work_path as _work_path, raw_path as _raw_path

import argparse
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy.stats import beta, norm, t

from dependence import correlation_from_eta, sample_elliptical_copula
from elliptical_event_fast import fast_event_probabilities

PROJECT = _work_path()
OUT = PROJECT / "results/upgrade_sensitivity_20261002"
ZONES = ("DE_LU", "FR", "BE")
DFS = (3., 5., 8., 15., 30., np.inf)
SEED = 2026100229


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [clean(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else "Gaussian_limit" if value > 0 else None
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(clean(value), handle, indent=2, ensure_ascii=False, allow_nan=False)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cp_interval(count, n, alpha):
    count, n = np.asarray(count), np.asarray(n)
    lo = np.where(count == 0, 0., beta.ppf(alpha / 2, count, n - count + 1))
    hi = np.where(count == n, 1., beta.ppf(1 - alpha / 2, count + 1, n - count))
    return np.asarray(lo, float), np.asarray(hi, float)


def probabilities(v, r, df=np.inf, tolerance=1e-7):
    # Numerical clipping is recorded in the analysis metadata. It avoids infinite
    # inverse CDF inputs; differences are at most 3e-10 by the uniform union bound.
    return fast_event_probabilities(np.clip(v, 1e-10, 1 - 1e-10), r, df,
                                    tolerance=tolerance)["ge2"]


def simulation_setup(scenario):
    states = np.array([[-1.], [1.]])
    slope = np.array([.4, .3, .2])
    eta = np.array([.55, .45, .30]) + states * np.array([.25, .20, .15])
    base_r = correlation_from_eta(eta)
    true_v = norm.cdf(norm.ppf(.9) - states * slope)
    if scenario == "omitted_weather_gaussian":
        r_minus = correlation_from_eta(eta - np.array([.65, .65, .65]))
        r_plus = correlation_from_eta(eta + np.array([.65, .65, .65]))
        # Observed-state Gaussian quasi-likelihood has the mixture covariance.
        r = (r_minus + r_plus) / 2
        truth = (probabilities(true_v, r_minus) + probabilities(true_v, r_plus)) / 2
    else:
        r, r_minus, r_plus = base_r, None, None
        truth = probabilities(true_v, r, 5. if scenario == "t5_correct_margins" else np.inf)
    bad = scenario == "gaussian_margin_error"
    issued_v = norm.cdf((norm.ppf(.9) - states * slope - (.25 if bad else 0.)) /
                       (.8 if bad else 1.))
    return dict(states=states, slope=slope, r=r, r_minus=r_minus, r_plus=r_plus,
                true_v=true_v, issued_v=issued_v, true_event=truth)


def synthetic_calibration(setup, scenario, n, rng):
    """True full-vector draws preserve joint marginal/event calibration covariance."""
    group = rng.integers(0, 2, n)
    state = setup["states"][group]
    if scenario == "omitted_weather_gaussian":
        hidden = rng.integers(0, 2, n)
        r = np.where(hidden[:, None, None] == 0, setup["r_minus"][group], setup["r_plus"][group])
    else:
        hidden = None
        r = setup["r"][group]
    u = sample_elliptical_copula(r, df=5. if scenario == "t5_correct_margins" else np.inf,
                               seed=int(rng.integers(1, 2**31 - 1)))
    y = state * setup["slope"] + norm.ppf(np.clip(u, 1e-12, 1 - 1e-12))
    below = y <= norm.ppf(.9)
    event = np.sum(~below, axis=1) >= 2
    counts = np.array([below[group == c].sum(axis=0) for c in (0, 1)])
    ns = np.array([np.sum(group == c) for c in (0, 1)])
    event_count = np.array([event[group == c].sum() for c in (0, 1)])
    # Six margin and two event intervals: familywise >=95% under this IID DGP.
    vlo, vhi = cp_interval(counts, ns[:, None], .05 / 8)
    elo, ehi = cp_interval(event_count, ns, .05 / 8)
    return dict(n=ns, counts=counts, event_counts=event_count,
                vhat=(counts + .5) / (ns[:, None] + 1), vlo=vlo, vhi=vhi,
                event_hat=event_count / ns, elo=elo, ehi=ehi)


def simulate_one(scenario, replicate, n_cal, n_select, n_test, setup):
    rng = np.random.default_rng(SEED + replicate * 104729 + n_cal)
    cal = synthetic_calibration(setup, scenario, n_cal, rng)
    # Chronology: prior marginal calibration -> separate dependence selection -> new test.
    selection_group = rng.integers(0, 2, n_select)
    selection_event = rng.random(n_select) < setup["true_event"][selection_group]
    records, losses = [], []
    for df in DFS:
        p0 = probabilities(setup["issued_v"], setup["r"], df)
        p1 = probabilities(cal["vhat"], setup["r"], df)
        loss = .5 * np.mean((p0[selection_group] - selection_event)**2 +
                             (p1[selection_group] - selection_event)**2)
        records.append((p0, p1))
        losses.append(loss)
    # Conservative predetermined simplicity tie preference for exact equal loss.
    selected = int(np.argmin(np.asarray(losses) + np.arange(len(DFS), 0, -1) * 1e-15))
    df = DFS[selected]
    p_g0, p_g1 = records[-1]
    p_d0, p_d1 = records[selected]
    test_group = rng.integers(0, 2, n_test)
    test_event = rng.random(n_test) < setup["true_event"][test_group]
    matrix = dict(M0=p_g0, M1=p_g1, M2=p_d0, M3=p_d1)
    brier = {name: float(np.mean((p[test_group] - test_event)**2)) for name, p in matrix.items()}
    paired_gain = ((p_g1[test_group] - test_event)**2 -
                   (p_d1[test_group] - test_event)**2)
    paired_se = float(paired_gain.std(ddof=1) / np.sqrt(n_test))
    paired_width = float(t.ppf(.975, n_test - 1) * paired_se)
    score_gain_interval = [float(paired_gain.mean() - paired_width),
                           float(paired_gain.mean() + paired_width)]
    # Monotonicity gives actual simultaneous model-probability ranges under the
    # homogeneous-state threshold confidence set, unlike generic mean calibration.
    g_lo = probabilities(cal["vhi"], setup["r"])
    g_hi = probabilities(cal["vlo"], setup["r"])
    d_lo = probabilities(cal["vhi"], setup["r"], df)
    d_hi = probabilities(cal["vlo"], setup["r"], df)
    gaussian_mismatch = (g_hi < cal["elo"]) | (g_lo > cal["ehi"])
    candidate_compatible = (d_hi >= cal["elo"]) & (d_lo <= cal["ehi"])
    shape_flag = gaussian_mismatch & candidate_compatible & np.isfinite(df)
    # This additional assay uses the independent newly generated test block.
    # Compatibility alone is deliberately not treated as a useful improvement.
    # IID t uncertainty is justified only in this synthetic independent-row DGP.
    score_validated_flag = bool(shape_flag.any() and score_gain_interval[0] > 0)
    marginal_cover = bool(np.all((cal["vlo"] <= setup["true_v"]) &
                                  (cal["vhi"] >= setup["true_v"])))
    event_cover = bool(np.all((cal["elo"] <= setup["true_event"]) &
                               (cal["ehi"] >= setup["true_event"])))
    # Population expected excess Brier is (forecast - known event probability)^2.
    excess = {name: float(np.mean((p - setup["true_event"])**2)) for name, p in matrix.items()}
    exact_margin_move = np.abs(probabilities(setup["issued_v"], setup["r"]) -
                               probabilities(setup["true_v"], setup["r"]))
    union_bound = np.minimum(1., np.sum(np.abs(setup["issued_v"] - setup["true_v"]), axis=1))
    return dict(scenario=scenario, replicate=replicate, n_calibration=n_cal,
                n_selection=n_select, n_test=n_test, selected_df=df, brier=brier,
                expected_excess_brier=excess, selected_finite_df=bool(np.isfinite(df)),
                gaussian_mismatch=gaussian_mismatch, candidate_compatible=candidate_compatible,
                shape_flag=shape_flag, abstain_all_states=bool(not shape_flag.any()),
                score_validated_flag=score_validated_flag,
                independent_test_paired_gain_95_interval=score_gain_interval,
                simultaneous_margin_coverage=marginal_cover,
                simultaneous_event_coverage=event_cover,
                simultaneous_joint_coverage=marginal_cover and event_cover,
                exact_margin_probability_movement=exact_margin_move,
                common_uniform_union_bound=union_bound,
                calibration=cal, probability_ranges=dict(g_lo=g_lo, g_hi=g_hi, d_lo=d_lo, d_hi=d_hi),
                population_truth=setup["true_event"], matrix_state_probabilities=matrix)


def mc_interval(values):
    values = np.asarray(values, float)
    n = len(values)
    se = float(values.std(ddof=1) / np.sqrt(n)) if n > 1 else None
    width = float(t.ppf(.975, n - 1) * se) if n > 1 else None
    return dict(n=n, mean=float(values.mean()), monte_carlo_se=se,
                mean_95pct_monte_carlo_interval=None if width is None else
                [float(values.mean()) - width, float(values.mean()) + width])


def simulation(args):
    started = time.perf_counter()
    scenarios = ("correct_gaussian_control", "gaussian_margin_error", "t5_correct_margins",
                 "omitted_weather_gaussian")
    output = OUT / args.label
    output.mkdir(parents=True, exist_ok=False)
    manifest = dict(status="running", type="SYNTHETIC", version="2.0-independent-score-gate",
                    created_utc=datetime.now(timezone.utc).isoformat(),
                    homogeneous_observed_states=2, repetitions=args.reps,
                    n_calibration=args.n_cal, n_selection=args.n_select, n_test=args.n_test,
                    seed=SEED, script_sha256=sha256(__file__),
                    correlation="Known state correlation; for hidden-state mixture its exact covariance. No estimated-R claim.",
                    interpretation="A finite-threshold copula discrepancy is conditional on observed information; it is not a physical tail mechanism.")
    write_json(output / "protocol.json", manifest)
    units = []
    for scenario in scenarios:
        setup = simulation_setup(scenario)
        for replicate in range(args.reps):
            unit = simulate_one(scenario, replicate, args.n_cal, args.n_select, args.n_test, setup)
            write_json(output / f"unit_{scenario}_{replicate:03}.json", unit)
            units.append(unit)
        print(json.dumps(dict(scenario=scenario, completed=args.reps,
                              elapsed_seconds=time.perf_counter() - started)), flush=True)
    summary = {}
    for scenario in scenarios:
        selected = [unit for unit in units if unit["scenario"] == scenario]
        flag_count = int(sum(u["score_validated_flag"] for u in selected))
        rate_lo, rate_hi = cp_interval(flag_count, len(selected), .05)
        summary[scenario] = dict(
            finite_df_fraction=float(np.mean([u["selected_finite_df"] for u in selected])),
            any_shape_flag_fraction=float(np.mean([bool(np.any(u["shape_flag"])) for u in selected])),
            shape_flag_fraction_by_state=np.mean([u["shape_flag"] for u in selected], axis=0),
            gaussian_mismatch_fraction_by_state=np.mean([u["gaussian_mismatch"] for u in selected], axis=0),
            abstention_fraction=float(np.mean([u["abstain_all_states"] for u in selected])),
            score_validated_flag_count=flag_count,
            score_validated_flag_fraction=float(flag_count / len(selected)),
            score_validated_rate_95_interval=[float(rate_lo), float(rate_hi)],
            simultaneous_joint_coverage=float(np.mean([u["simultaneous_joint_coverage"] for u in selected])),
            expected_margin_gain=mc_interval([u["expected_excess_brier"]["M0"] - u["expected_excess_brier"]["M1"] for u in selected]),
            expected_dependence_gain_before_margin=mc_interval([u["expected_excess_brier"]["M0"] - u["expected_excess_brier"]["M2"] for u in selected]),
            expected_dependence_gain_after_margin=mc_interval([u["expected_excess_brier"]["M1"] - u["expected_excess_brier"]["M3"] for u in selected]),
            test_dependence_gain_after_margin=mc_interval([u["brier"]["M1"] - u["brier"]["M3"] for u in selected]))
    manifest.update(status="completed", completed_units=len(units),
                    elapsed_seconds=time.perf_counter() - started, summary=summary,
                    common_uniform_bound_formula="|p_C(v)-p_C(v')| <= min(1,sum_j |v_j-v'_j|)",
                    no_new_theorem=True,
                    important_limit="The certification here relies on IID finite homogeneous synthetic states. Real weather cells do not satisfy this assumption by construction.")
    write_json(output / "summary.json", manifest)
    print(json.dumps(clean(manifest)), flush=True)


def complete_days(frame):
    day = frame.index.floor("D")
    counts = pd.Series(1, index=frame.index).groupby(day).sum()
    return frame.loc[day.isin(counts[counts == 24].index)].copy()


def synchronized_block_mean_intervals(frame, residual, cell, draws=2000, block=14):
    """Same day-block weights for all areas/event dimensions, separately by year-quarter.

    Pointwise intervals are paired with max-standardized simultaneous envelopes
    across the prespecified cells and diagnostic dimensions. Resampling days,
    never treating areas or persistent hours as independent.
    """
    days = pd.DatetimeIndex(frame.index.floor("D").unique()).sort_values()
    daypos = days.get_indexer(frame.index.floor("D"))
    cells = sorted(pd.unique(cell))
    d, c, k = len(days), len(cells), residual.shape[1]
    sums = np.zeros((d, c, k))
    counts = np.zeros((d, c))
    for ci, label in enumerate(cells):
        mask = np.asarray(cell == label)
        np.add.at(counts[:, ci], daypos[mask], 1)
        np.add.at(sums[:, ci], daypos[mask], residual[mask])
    mean = sums.sum(axis=0) / counts.sum(axis=0)[:, None]
    samples = np.empty((draws, c, k))
    rng = np.random.default_rng(SEED + block)
    quarters = [(year, quarter) for year, quarter in zip(days.year, days.quarter)]
    levels = sorted(set(quarters))
    positions = [np.array([i for i, label in enumerate(quarters) if label == level]) for level in levels]
    for draw in range(draws):
        weights = np.zeros(d)
        for pos in positions:
            # Circular moving calendar blocks retain gaps in day labels as gaps.
            first, last = days[pos[0]], days[pos[-1]]
            calendar = pd.date_range(first, last, freq="D", tz=days.tz)
            key = {int((date - first).days): i for i, date in enumerate(days[pos])}
            chosen = []
            while len(chosen) < len(calendar):
                start = int(rng.integers(0, len(calendar)))
                chosen.extend(((start + np.arange(block)) % len(calendar)).tolist())
            for j in chosen[:len(calendar)]:
                if j in key:
                    weights[pos[key[j]]] += 1
        denom = weights @ counts
        samples[draw] = np.einsum("d,dck->ck", weights, sums) / denom[:, None]
    sd = np.std(samples, axis=0, ddof=1)
    standardized = np.abs(samples - mean) / np.where(sd > 1e-12, sd, 1.)
    critical = float(np.nanquantile(np.nanmax(standardized, axis=(1, 2)), .95))
    point_lo, point_hi = np.nanquantile(samples, [.025, .975], axis=0)
    simultaneous_lo, simultaneous_hi = mean - critical * sd, mean + critical * sd
    return dict(cells=cells, mean=mean, point_lo=point_lo, point_hi=point_hi,
                simultaneous_lo=simultaneous_lo, simultaneous_hi=simultaneous_hi,
                max_standardized_critical=critical, draws=draws, block_days=block,
                n_days_by_cell=counts.astype(bool).sum(axis=0), n_hours_by_cell=counts.sum(axis=0))


def correlation_rows(frame):
    r = np.broadcast_to(np.eye(3), (len(frame), 3, 3)).copy()
    for column, i, j in (("rho_01", 0, 1), ("rho_02", 0, 2), ("rho_12", 1, 2)):
        r[:, i, j] = r[:, j, i] = frame[column].to_numpy(float)
    return r


def df_array(frame, q):
    candidates = (f"df_event_{q}", f"selected_df_{q}", f"df_{q}", "df_event_selected", "event_df", "selected_df")
    name = next((c for c in candidates if c in frame), None)
    if name is None:
        raise ValueError(f"Missing selected degrees of freedom, tried {candidates}")
    return frame[name].map(lambda v: np.inf if str(v) in ("Gaussian_limit", "infinity", "inf", "None")
                           else float(v)).to_numpy()


def mixed_probabilities(v, r, dfs):
    p = np.empty(len(v))
    for df in np.unique(dfs):
        keep = dfs == df
        p[keep] = probabilities(v[keep], r[keep], df, tolerance=1e-6)
    return p


def process_support(frame, event, labels):
    """Descriptive continuous supports, not independent meteorological events."""
    # pandas preserves parquet microsecond timestamps on current versions;
    # asi8 values therefore need an explicit nanosecond unit before comparison.
    times = frame.index.as_unit("ns")
    gaps = np.r_[True, np.diff(times.asi8) != pd.Timedelta(hours=1).value]
    label_change = np.r_[True, labels[1:] != labels[:-1]]
    cell_runs = np.cumsum(gaps | label_change)
    days = pd.DatetimeIndex(times.floor("D").unique()).sort_values()
    active = pd.Series(event, index=times).groupby(times.floor("D")).max().reindex(days).astype(bool)
    # Exactly the main one-inactive-complete-day proxy, with observed gaps breaking.
    process_by_day = {}
    process = -1
    previous_active = None
    for day, is_active in active.items():
        if not is_active:
            continue
        if previous_active is None or (day - previous_active).days > 2 or any(
            previous_active < missing < day for missing in pd.date_range(previous_active, day, freq="D", tz=times.tz)
            if missing not in set(days)):
            process += 1
        process_by_day[day] = process
        previous_active = day
    result = {}
    for cell in sorted(pd.unique(labels)):
        keep = labels == cell
        processes = {process_by_day[day] for day in pd.unique(times[keep & (event == 1)].floor("D"))
                     if day in process_by_day}
        result[cell] = dict(contiguous_hourly_weather_cell_runs=len(np.unique(cell_runs[keep])),
                            persistent_active_day_processes_with_an_event_hour_in_cell=len(processes),
                            support_interpretation="Overlapping descriptive process support; cells may share a process. Not an IID effective sample size.")
    return result


def real_analysis(args):
    started = time.perf_counter()
    source = Path(args.input)
    frame = pd.read_parquet(source)
    if not isinstance(frame.index, pd.DatetimeIndex):
        frame = frame.set_index("target_time")
    frame.index = pd.to_datetime(frame.index, utc=True)
    frame = frame.sort_index()
    output = OUT / args.label
    output.mkdir(parents=True, exist_ok=False)
    records = []
    for period, start, end in (("development", "2022-01-01", "2024-01-01"),
                               ("reanalysis", "2024-01-01", "2026-01-01")):
        period_frame = frame.loc[(frame.index >= pd.Timestamp(start, tz="UTC")) &
                                 (frame.index < pd.Timestamp(end, tz="UTC"))].copy()
        for q in ("q90", "q95"):
            columns = ([f"v0_{q}_{z}" for z in ZONES] + [f"v1_{q}_{z}" for z in ZONES] +
                       [f"exceed_{q}_{z}" for z in ZONES] + [f"M{i}_{q}" for i in range(4)] +
                       [f"y_{q}", "rho_01", "rho_02", "rho_12", "cell"])
            d = complete_days(period_frame.dropna(subset=columns))
            v0 = d[[f"v0_{q}_{z}" for z in ZONES]].to_numpy(float)
            v1 = d[[f"v1_{q}_{z}" for z in ZONES]].to_numpy(float)
            below = 1 - d[[f"exceed_{q}_{z}" for z in ZONES]].to_numpy(float)
            y = d[f"y_{q}"].to_numpy(float)
            ps = d[[f"M{i}_{q}" for i in range(4)]].to_numpy(float)
            residual = np.column_stack([below - v0, below - v1, y[:, None] - ps])
            ci = synchronized_block_mean_intervals(d, residual, d["cell"].to_numpy(), args.draws)
            r = correlation_rows(d)
            dfs = df_array(d, q)
            support = process_support(d, y, d["cell"].to_numpy())
            mean_rows = []
            for i, label in enumerate(ci["cells"]):
                mask = d["cell"].to_numpy() == label
                # Applying a cell-mean error as a constant to every hourly CDF is
                # an assumption for this path, not an inferred conditional bound.
                cell_records = dict(period=period, event=q, cell=label,
                    n_days=int(ci["n_days_by_cell"][i]), n_hours=int(mask.sum()),
                    event_hours=int(y[mask].sum()), event_days=int(pd.Series(y[mask], index=d.index[mask]).groupby(d.index[mask].floor("D")).max().sum()),
                    mean_residuals=ci["mean"][i], simultaneous_lo=ci["simultaneous_lo"][i],
                    simultaneous_hi=ci["simultaneous_hi"][i],
                    pointwise_lo=ci["point_lo"][i], pointwise_hi=ci["point_hi"][i],
                    mean_conditional_threshold_positions_v0=v0[mask].mean(axis=0),
                    mean_conditional_threshold_positions_v1=v1[mask].mean(axis=0),
                    mean_absolute_recalibration_movement=np.mean(np.abs(v1[mask] - v0[mask]), axis=0),
                    common_uniform_recalibration_bound_mean=float(np.minimum(1., np.sum(np.abs(v1[mask] - v0[mask]), axis=1)).mean()),
                    mean_probability_by_method=ps[mask].mean(axis=0), observed_event_fraction=float(y[mask].mean()),
                    process_support=support[label],
                    constant_cell_shift_paths={})
                for margin, v in ((0, v0), (1, v1)):
                    low = ci["simultaneous_lo"][i, 3 * margin:3 * margin + 3]
                    high = ci["simultaneous_hi"][i, 3 * margin:3 * margin + 3]
                    mean_shift = ci["mean"][i, 3 * margin:3 * margin + 3]
                    minus, plus = np.clip(v[mask] + low, 0, 1), np.clip(v[mask] + high, 0, 1)
                    g_lo, g_hi = probabilities(plus, r[mask], tolerance=1e-6), probabilities(minus, r[mask], tolerance=1e-6)
                    t_lo, t_hi = mixed_probabilities(plus, r[mask], dfs[mask]), mixed_probabilities(minus, r[mask], dfs[mask])
                    obs = y[mask]
                    # For binary obs, square loss monotonic on [0,1]. Outer
                    # intervals are valid conditional on the shift hypothesis;
                    # they do not rely on sampled corners finding the extrema.
                    g_loss_lo = np.minimum((g_lo - obs)**2, (g_hi - obs)**2)
                    g_loss_hi = np.maximum((g_lo - obs)**2, (g_hi - obs)**2)
                    t_loss_lo = np.minimum((t_lo - obs)**2, (t_hi - obs)**2)
                    t_loss_hi = np.maximum((t_lo - obs)**2, (t_hi - obs)**2)
                    gain_low, gain_high = g_loss_lo - t_loss_hi, g_loss_hi - t_loss_lo
                    # If the selected dependence is Gaussian, the two forecast
                    # functions are identical throughout the uncertainty set.
                    # An independent-box outer interval must not obscure this.
                    aliases = np.isinf(dfs[mask])
                    gain_low[aliases] = gain_high[aliases] = 0.
                    central = np.clip(v[mask] + mean_shift, 0, 1)
                    central_g = probabilities(central, r[mask], tolerance=1e-6)
                    central_t = mixed_probabilities(central, r[mask], dfs[mask])
                    cell_records["constant_cell_shift_paths"][f"margin_{margin}"] = dict(
                        gaussian_mean_probability_interval=[float(g_lo.mean()), float(g_hi.mean())],
                        event_selected_mean_probability_interval=[float(t_lo.mean()), float(t_hi.mean())],
                        dependence_brier_gain_outer_interval=[float(np.mean(gain_low)), float(np.mean(gain_high))],
                        mean_shift_scenario_brier_gain=float(np.mean((central_g - obs)**2 - (central_t - obs)**2)),
                        mean_shift_scenario_gaussian_event_bias=float(np.mean(central_g - obs)),
                        mean_shift_scenario_selected_event_bias=float(np.mean(central_t - obs)),
                        gaussian_alias_rows=int(aliases.sum()),
                        bound_interpretation="Valid only under the tested constant-within-cell CDF-shift hypothesis; not pointwise empirical calibration certification.")
                records.append(cell_records)
            write_json(output / f"{period}_{q}_cells.json", dict(status="completed", diagnostics=records[-len(ci["cells"]):],
                       residual_order=[f"margin{m}_{z}" for m in (0, 1) for z in ZONES] + [f"event_M{i}" for i in range(4)],
                       simultaneous_critical=ci["max_standardized_critical"], block_days=14, bootstrap_draws=args.draws))
            print(json.dumps(dict(period=period, q=q, cells=len(ci["cells"]), hours=len(d), elapsed_seconds=time.perf_counter() - started)), flush=True)
    write_json(output / "summary.json", dict(status="completed", source=str(source), source_sha256=sha256(source),
               script_sha256=sha256(__file__), diagnostics=records, elapsed_seconds=time.perf_counter() - started,
               claim_boundary="Real cell-average errors do not upper-bound pointwise conditional CDF errors. This is a documented constant-cell-shift sensitivity analysis and synchronized calibration diagnostic, not tail-mechanism identification.",
               clipping_effect_union_bound_max=3e-10, no_new_theorem=True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("simulation", "real"))
    parser.add_argument("--label", required=True)
    parser.add_argument("--reps", type=int, default=100)
    parser.add_argument("--n-cal", type=int, default=100000)
    parser.add_argument("--n-select", type=int, default=10000)
    parser.add_argument("--n-test", type=int, default=10000)
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--input", default=str(PROJECT / "results/upgrade_margin_dependence_20261002/upgrade_predictions.parquet"))
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    simulation(args) if args.mode == "simulation" else real_analysis(args)


if __name__ == "__main__":
    main()
