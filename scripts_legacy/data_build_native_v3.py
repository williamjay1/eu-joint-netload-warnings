"""Native-resolution observed targets with explicit archive and night assumptions.

Strict observed-data output is always preserved. A separate development-only
night reconstruction is a measurement assumption, never a claim of observed
truth. Sealed years receive only deterministic normalization and coverage audits.
"""
from release_paths import work_path as _work_path, raw_path as _raw_path
from pathlib import Path
from datetime import datetime, timezone
import argparse, hashlib, json, sys, zipfile
import numpy as np
import pandas as pd
from data_normalize_observations import unique_series, hourly

PROJECT = _work_path()
RAW = _raw_path()
ARCHIVE_ZIP = RAW / 'elia_historical_recovery_20260929T1345Z/jamesdeluk_ods032_snapshot.zip'
ARCHIVE_SHA256 = 'f3554629a7ace74a1dcc5cbfb01a0367a66e5ec594213db102b8996b49b27cad'
FACETS = [('Federal', 'Offshore', 'Elia'), ('Flanders', 'Onshore', 'Dso'),
          ('Flanders', 'Onshore', 'Elia'), ('Wallonia', 'Onshore', 'Dso'),
          ('Wallonia', 'Onshore', 'Elia')]
FIELDS = ['load_mw', 'solar_mw', 'wind_onshore_mw', 'wind_offshore_mw']


def find_file(pattern, required=True):
    paths = sorted(RAW.glob(pattern))
    if not paths:
        if required:
            raise FileNotFoundError(pattern)
        return None
    return paths[-1]


def native_csv(path, fields, year):
    d = pd.read_csv(path, sep=';', usecols=['datetime', 'resolutioncode', *fields])
    d['timestamp'] = pd.to_datetime(d.datetime, utc=True)
    d = d[d.timestamp.dt.year == year].set_index('timestamp')
    if len(d) and set(d.resolutioncode.dropna()) != {'PT15M'}:
        raise ValueError(f'Unexpected interval length in {path}')
    if ((d.index.minute % 15 != 0) | (d.index.second != 0)).any():
        raise ValueError(f'Unexpected timestamp alignment in {path}')
    return d


def load_archive():
    """Read the preserved raw ZIP directly; D-drive caches are derivative."""
    if hashlib.sha256(ARCHIVE_ZIP.read_bytes()).hexdigest() != ARCHIVE_SHA256:
        raise ValueError('Preserved PV archive SHA256 differs from the audited snapshot')
    columns = ['Datetime', 'Resolution code', 'Region', 'Measured & Upscaled']
    pieces = []
    with zipfile.ZipFile(ARCHIVE_ZIP) as archive:
        with archive.open('ods032.csv') as csv:
            for chunk in pd.read_csv(csv, sep=';', usecols=columns, chunksize=200000):
                chunk = chunk[chunk.Region == 'Belgium'].copy()
                chunk.index = pd.to_datetime(chunk.Datetime, utc=True)
                chunk = chunk[(chunk.index.year >= 2019) & (chunk.index.year <= 2023)]
                pieces.append(chunk)
    d = pd.concat(pieces).sort_index()
    if set(d['Resolution code']) != {'PT15M'} or set(d.Region) != {'Belgium'}:
        raise ValueError('Unexpected archive spatial or temporal schema')
    if d.index.has_duplicates:
        raise ValueError('Archive duplicate timestamps')
    result = pd.to_numeric(d['Measured & Upscaled'], errors='coerce').rename('solar_mw')
    result.index.name = 'timestamp'
    result.to_frame().to_parquet(PROJECT / 'datasets/data_be_pv_archive_2019_2023.parquet')
    return result


def build_be(years, archive):
    frames, overlaps, inputs, provenance = [], [], [], []
    for year in years:
        lf = find_file(f'observations_elia-load_*/elia_load_{year}.csv')
        wf = find_file(f'observations_elia-generation_*/elia_ods031_{year}.csv')
        sf = find_file(f'observations_elia-generation_*/elia_ods032_{year}.csv', required=year != 2019)
        inputs += [lf, wf]
        load = unique_series(pd.to_numeric(native_csv(lf, ['totalload'], year).totalload, errors='coerce'))
        wind = native_csv(wf, ['region', 'offshoreonshore', 'gridconnectiontype', 'measured'], year)
        observed_facets = set(map(tuple, wind[['region', 'offshoreonshore', 'gridconnectiontype']].drop_duplicates().to_numpy()))
        if observed_facets != set(FACETS):
            raise ValueError(f'Wind facet set changed in {year}: {observed_facets}')
        wc = {}
        for facet in FACETS:
            mask = (wind.region == facet[0]) & (wind.offshoreonshore == facet[1]) & (wind.gridconnectiontype == facet[2])
            wc['/'.join(facet)] = unique_series(pd.to_numeric(wind.loc[mask, 'measured'], errors='coerce'))
        wq = pd.DataFrame(wc)
        offshore = wq['Federal/Offshore/Elia']
        onshore = wq.drop(columns=['Federal/Offshore/Elia']).sum(axis=1, min_count=4)
        if sf:
            inputs.append(sf)
            solar = native_csv(sf, ['region', 'measured'], year)
            if len(solar) and set(solar.region) != {'Belgium'}:
                raise ValueError('Solar must use the Belgium aggregate only')
            solar = unique_series(pd.to_numeric(solar.measured, errors='coerce'))
        else:
            solar = pd.Series([], index=pd.DatetimeIndex([], tz='UTC'), dtype=float)
        idx = pd.date_range(f'{year}-01-01', f'{year+1}-01-01', freq='15min', inclusive='left', tz='UTC')
        solar = solar.reindex(idx)
        old = archive.reindex(idx) if year <= 2023 else pd.Series(np.nan, index=idx)
        pair = pd.concat([solar.rename('current'), old.rename('archive')], axis=1).dropna()
        maxdiff = float((pair.current - pair.archive).abs().max()) if len(pair) else None
        if maxdiff is not None and maxdiff > 1e-8:
            raise ValueError(f'Archive disagrees with current PV measurements in {year}: {maxdiff}')
        recovered = solar.isna() & old.notna()
        overlaps.append({'year': year, 'overlap_nonmissing_quarter_hours': len(pair),
                         'maximum_absolute_difference_mw': maxdiff,
                         'archive_recovered_quarter_hours': int(recovered.sum())})
        solar = solar.combine_first(old)
        d = pd.DataFrame({'load_mw': load, 'solar_mw': solar,
                          'wind_onshore_mw': onshore, 'wind_offshore_mw': offshore}).reindex(idx)
        d['solar_archive_recovered'] = recovered
        d['solar_night_reconstructed'] = False
        frames.append(d)
        provenance.append({'year': year, 'load': str(lf), 'wind': str(wf),
                           'solar_current': str(sf) if sf else None,
                           'solar_archive_if_current_missing': year <= 2023})
    return pd.concat(frames), overlaps, inputs, provenance


def night_candidates(native):
    """Explicit conditional reconstruction; never overwrites original values.

    A 0.5-degree grid covers a conservative rectangle around all Belgium and
    Sotel. Solar elevation is sampled every minute, including both endpoints.
    A +1 degree bound covers spatial nearest-grid and temporal sampling gaps;
    even this conservative upper bound must remain below -6 degrees.
    """
    sys.path.append(str(PROJECT / 'results/data_runtime'))
    from astral import Observer
    from astral.sun import elevation
    local = native.index.tz_convert('Europe/Brussels')
    candidate = native.solar_mw.isna() & (local.hour == 23) & (local.minute == 45)
    idx = native.index[candidate]
    rows = []
    observers = [Observer(float(lat), float(lon))
                 for lat in np.arange(49.0, 52.01, .5)
                 for lon in np.arange(2.0, 7.01, .5)]
    for timestamp in idx:
        previous = native.solar_mw.get(timestamp - pd.Timedelta(minutes=15), np.nan)
        following = native.solar_mw.get(timestamp + pd.Timedelta(minutes=15), np.nan)
        peak = max(elevation(observer, (timestamp + pd.Timedelta(minutes=minute)).to_pydatetime(),
                             with_refraction=False)
                   for observer in observers for minute in range(16))
        upper_bound = peak + 1.0
        eligible = previous == 0.0 and following == 0.0 and upper_bound < -6.0
        rows.append({'timestamp': timestamp, 'previous_pv_mw': previous, 'following_pv_mw': following,
                     'maximum_sampled_solar_elevation_deg': peak,
                     'conservative_upper_bound_deg': upper_bound,
                     'eligible_conditional_reconstruction': bool(eligible)})
    return pd.DataFrame(rows)


def to_hourly(native):
    d = pd.DataFrame({c: hourly(native[c], 15) for c in FIELDS})
    d['wind_total_mw'] = d.wind_onshore_mw + d.wind_offshore_mw
    d['net_load_mw'] = d.load_mw - d.solar_mw - d.wind_total_mw
    d['solar_archive_recovered'] = native.solar_archive_recovered.resample('h').max()
    d['solar_night_reconstructed'] = native.solar_night_reconstructed.resample('h').max()
    d['quality_flag'] = np.where(d.net_load_mw.notna(), 'complete', 'missing_component')
    d.loc[d.solar_night_reconstructed & d.net_load_mw.notna(), 'quality_flag'] = 'conditional_night_pv_zero'
    d['zone'] = 'BE'
    d['source_domain'] = 'Elia ODS001/ODS031/ODS032 PT15M; early PV archived original ODS032'
    d.index.name = 'timestamp'
    # Independent array calculation: all four native values, with NumPy NaN
    # propagation rather than pandas resample/count logic.
    qa = {}
    for field in FIELDS:
        independent = native[field].to_numpy().reshape(-1, 4).sum(axis=1) / 4.0
        actual = d[field].to_numpy()
        if not np.array_equal(np.isnan(actual), np.isnan(independent)):
            raise AssertionError(f'Independent missingness mismatch: {field}')
        diff = np.abs(actual - independent)
        qa[field] = float(np.nanmax(diff))
        if qa[field] > 1e-8:
            raise AssertionError(f'Independent aggregation mismatch: {field}')
    return d.reset_index(), qa


def other_zones(sealed=False):
    suffix = '2024_2025_sealed' if sealed else '2019_2023'
    d = pd.read_parquet(PROJECT / f'datasets/data_observations_{suffix}.parquet')
    d = d[d.zone != 'BE'].copy()
    d['solar_archive_recovered'] = False
    d['solar_night_reconstructed'] = False
    return d


def coverage(frame):
    annual = []
    for (zone, year), d in frame.groupby(['zone', frame.timestamp.dt.year]):
        daily = d.assign(date=d.timestamp.dt.floor('D')).groupby('date').net_load_mw.count()
        annual.append({'zone': zone, 'year': int(year), 'hours': len(d),
                       'complete_netload_hours': int(d.net_load_mw.notna().sum()),
                       'complete_utc_days': int((daily == 24).sum()),
                       'duplicate_hours': int(d.timestamp.duplicated().sum()),
                       'reconstructed_pv_hours': int(d.solar_night_reconstructed.sum())})
    wide = frame.pivot(index='timestamp', columns='zone', values='net_load_mw')
    common = wide.notna().all(axis=1)
    daily = common.groupby(common.index.floor('D')).sum()
    joint = [{'year': int(year), 'complete_common_hours': int(group.sum()),
              'complete_common_utc_days': int((daily[daily.index.year == year] == 24).sum())}
             for year, group in common.groupby(common.index.year)]
    return {'by_zone_and_year': annual, 'common_by_year': joint}


def save_table(be, other, filename):
    d = pd.concat([other, be], ignore_index=True).sort_values(['timestamp', 'zone'])
    target = PROJECT / 'datasets' / filename
    temp = target.with_suffix('.tmp.parquet')
    d.to_parquet(temp, index=False)
    temp.replace(target)
    return d, str(target)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--night-candidate', action='store_true',
                   help='Write a separate conditional reconstruction; strict remains primary archive.')
    p.add_argument('--include-sealed', action='store_true')
    a = p.parse_args()
    archive = load_archive()
    years = list(range(2019, 2024))
    native, overlap, inputs, provenance = build_be(years, archive)
    native.index.name = 'timestamp'
    native.to_parquet(PROJECT / 'datasets/data_be_native_v3_quarterhour_2019_2023.parquet')
    be, qa = to_hourly(native)
    strict, strict_path = save_table(be, other_zones(), 'data_observations_native_v3_strict_2019_2023.parquet')
    report = {'version': 'native_v3', 'built_at_utc': datetime.now(timezone.utc).isoformat(),
              'strict_output': strict_path, 'strict_coverage': coverage(strict),
              'independent_hour_aggregation_max_abs_difference_mw': qa,
              'archive_overlap_audit': overlap, 'provenance': provenance,
              'native_missing_quarter_hours_2019_2023': {c: int(native[c].isna().sum()) for c in FIELDS},
              'unresolved_boundaries': ['Elia does not explicitly certify gross-versus-inverter-night-consumption accounting in the reviewed ODS032 metadata.',
                                        'RTE exact half-hour timestamp positioning still needs documentary confirmation.',
                                        'Elia 2018--2023 offshore measured/upscaled versus validated-meter discrepancy warning.',
                                        'Published Elia load footprint includes Sotel.'],
              'test_policy': 'No test event frequency, cases, values, thresholds, or forecast scores examined.'}
    if a.night_candidate:
        audit = night_candidates(native)
        audit.to_csv(PROJECT / 'results/data_native_v3_night_candidates.csv', index=False)
        reconstructed = native.copy()
        eligible = audit.loc[audit.eligible_conditional_reconstruction, 'timestamp']
        reconstructed.loc[eligible, 'solar_mw'] = 0.0
        reconstructed.loc[eligible, 'solar_night_reconstructed'] = True
        bh, qa_candidate = to_hourly(reconstructed)
        candidate, candidate_path = save_table(bh, other_zones(), 'data_observations_native_v3_night_candidate_2019_2023.parquet')
        report['night_candidate'] = {
            'output': candidate_path, 'status': 'Conditional measurement reconstruction, not observed truth or certified gross PV accounting.',
            'rules': ['Only missing Belgian PV at Europe/Brussels 23:45.',
                      'Original immediately preceding and following 15-minute PV values both exactly zero.',
                      'Solar-elevation upper bound below -6 degrees across rectangle 49--52N, 2--7E and the full quarter-hour.',
                      'Observed PV, daytime, load, and wind never changed.'],
            'astronomy': 'Astral 3.2, geometric elevation; 0.5 degree spatial grid, 1 minute interval grid, +1 degree conservative sampling bound.',
            'eligible_quarter_hours': len(eligible), 'eligible_dates': len(set(eligible.dt.date)),
            'maximum_conservative_solar_elevation_bound_deg': float(audit.conservative_upper_bound_deg.max()),
            'coverage': coverage(candidate), 'independent_hour_aggregation_max_abs_difference_mw': qa_candidate,
            'sensitivity_plan': 'Use strict observed-hour comparison and perturb reconstructed quarter-hour PV by 0, 1, 5 MW. One 5 MW quarter-hour changes hourly residual load by 1.25 MW.'}
    if a.include_sealed:
        test_native, test_overlap, test_inputs, test_provenance = build_be([2024, 2025], archive)
        bt, test_qa = to_hourly(test_native)
        sealed, sealed_path = save_table(bt, other_zones(sealed=True), 'data_observations_native_v3_strict_2024_2025_sealed.parquet')
        report['sealed_structural_audit'] = {'output': sealed_path, 'coverage': coverage(sealed),
                                            'independent_hour_aggregation_max_abs_difference_mw': test_qa,
                                            'provenance': test_provenance}
        inputs += test_inputs
    inputs = sorted(set(inputs + [ARCHIVE_ZIP]))
    report['raw_inputs'] = [{'path': str(f), 'sha256': hashlib.sha256(f.read_bytes()).hexdigest()} for f in inputs]
    dependencies = [PROJECT / 'datasets/data_observations_2019_2023.parquet']
    if a.include_sealed:
        dependencies.append(PROJECT / 'datasets/data_observations_2024_2025_sealed.parquet')
    report['de_lu_fr_normalized_dependencies'] = [
        {'path': str(f), 'sha256': hashlib.sha256(f.read_bytes()).hexdigest(),
         'fields_used': 'DE-LU and FR rows only; former Belgian mirror rows excluded'}
        for f in dependencies]
    target = PROJECT / 'results/data_native_v3_quality.json'
    target.write_text(json.dumps(report, indent=2), encoding='utf8')
    print(json.dumps({'quality_report': str(target), 'strict_output': strict_path,
                      'strict_common_coverage': report['strict_coverage']['common_by_year'],
                      'night_candidate': report.get('night_candidate', {}).get('eligible_quarter_hours'),
                      'sealed_present': 'sealed_structural_audit' in report}))


if __name__ == '__main__':
    main()
