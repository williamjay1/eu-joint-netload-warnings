"""Assemble verified GEFS caches and audited native observations on D.

Development mode never opens the sealed observation file. Test assembly is
allowed only after a frozen preinspection protocol is supplied. No event or
performance summaries are computed here. Every output directory is new.
"""
from __future__ import annotations
from release_paths import work_path as _work_path, raw_path as _raw_path
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys

import holidays
import numpy as np
import pandas as pd

from forecast_design import calendar_features, REGIONS

ROOT = _work_path()
CACHE = ROOT/'cache/gefs_regional_timely_common1_v1'
DEV = ROOT/'datasets/data_observations_native_v3_night_candidate_2019_2023.parquet'
TEST = ROOT/'datasets/data_observations_native_v4_night_candidate_2024_2025_sealed.parquet'
PHYSICAL = ('t2m','wind_speed','marine_wind_speed','dswrf')
SCRIPT_VERSION = '1.1-prespecified-known-total-load-UTC-lags'
ALLOWED_LOAD_LAG_DAYS = (14,21)
TEST_START = pd.Timestamp('2024-01-01',tz='UTC')
CODE_FILES = ('weather_panel_builder.py','forecast_design.py')


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(2**20),b''): h.update(block)
    return h.hexdigest()


def validate_load_lag_days(load_lag_days=()):
    if not isinstance(load_lag_days,(tuple,list)):
        raise ValueError('load_lag_days must be a list/tuple of prespecified integer days')
    if any(isinstance(d,bool) or not isinstance(d,(int,np.integer)) or d not in ALLOWED_LOAD_LAG_DAYS for d in load_lag_days):
        raise ValueError('Only prespecified integer total-load lags 14 and 21 are allowed; 7d is unavailable')
    days=tuple(int(d) for d in load_lag_days)
    if len(days)!=len(set(days)):
        raise ValueError('Duplicate load lag days are forbidden')
    return tuple(sorted(days))


def load_lag_names(load_lag_days=()):
    days=validate_load_lag_days(load_lag_days)
    return [f'provider_load_lag{d}d_utc_{zone}' for zone in REGIONS for d in days]


def _utc_hour_index(index,label):
    result=pd.DatetimeIndex(index)
    if (result.tz is None or result.hasnans or result.has_duplicates or
            not result.is_monotonic_increasing):
        raise ValueError(label+' requires ordered unique timezone-aware timestamps')
    result=result.tz_convert('UTC')
    if not result.equals(result.floor('h')):
        raise ValueError(label+' requires exact UTC hourly interval starts')
    return result


def build_load_lags(index,loads,load_lag_days=()):
    """Known provider TOTAL load at exact UTC lags; never interpolate or fill."""
    days=validate_load_lag_days(load_lag_days)
    index=_utc_hour_index(index,'target')
    output=pd.DataFrame(index=index)
    if not days:
        return output,[]
    if not isinstance(loads,pd.DataFrame) or loads.columns.has_duplicates or not set(REGIONS).issubset(loads.columns):
        raise ValueError('Total-load source must have unique named DE_LU/FR/BE columns')
    source_index=_utc_hour_index(loads.index,'source load')
    loads=loads.loc[:,REGIONS].copy()
    loads.index=source_index
    loads=loads.apply(pd.to_numeric,errors='raise')
    if np.isinf(loads.to_numpy(dtype=float)).any():
        raise ValueError('Infinite provider load must be audited; NaN is retained')
    issue=index.normalize()-pd.Timedelta(hours=12)
    audit=[]
    for zone in REGIONS:
        for d in days:
            source=index-pd.Timedelta(days=d)
            interval_end=source+pd.Timedelta(hours=1)
            available=interval_end+pd.Timedelta(days=7)
            slack=(issue-available)/pd.Timedelta(hours=1)
            if (available>issue).any():
                raise PermissionError('Lag interval END plus seven days exceeds the actual daily issue')
            values=loads[zone].reindex(source).to_numpy(dtype=float)
            name=f'provider_load_lag{d}d_utc_{zone}'
            output[name]=values
            audit.append({'feature':name,'zone':zone,'lag_days':d,'rows':len(index),
                'finite_rows':int(np.isfinite(values).sum()),'missing_rows':int(np.isnan(values).sum()),
                'minimum_availability_margin_hours':float(slack.min()) if len(index) else None,
                'source_first_target':source.min().isoformat() if len(index) else None,
                'source_last_target':source.max().isoformat() if len(index) else None,
                'every_row_source_END_plus_7d_at_or_before_issue':True,'interpolation_used':False,
                'missing_value_imputation_used':False,'local_DST_hour_alignment_used':False})
    return output,audit


def lag_policy(load_lag_days=()):
    days=validate_load_lag_days(load_lag_days)
    return {'enabled':bool(days),'load_lag_days':list(days),'columns':load_lag_names(days),
        'source_variable':'same_native_provider_total_load_mw_not_net_load',
        'indexing':'target_UTC_interval_start_minus_exact_integer_days_no_interpolation',
        'source_interval_end':'source_interval_start_plus_1h',
        'availability':'source_interval_END_plus_7d_at_or_before_target_UTC_day_minus_12h',
        'revised_vintage_boundary':'Historical final/revised provider values; their publication vintages are not authenticated',
        'missing_values':'remain_NaN_no_backfill_or_interpolation',
        'column_order':'zone_major_DE_LU_FR_BE_then_ascending_days_append_after_base69',
        'additive_basis':'new_load_lags_linear_in_mean_and_scale_not_splines_unless_scale_explicitly_excluded_by_frozen_model',
        'comparator_use':'lags_only_in_shared_single_zone_F_all_models_use_the_same_issued_calibrated_margins',
        'C7_ECC_use':'independent_weather_plus_calendar_rank_template_applied_to_shared_F_no_member_specific_F_recomputation',
        'C8_analog_use':'existing_joint_weather_basis_and_shared_forward_PIT_not_a_new_lag_feature_schema',
        'member_weather_and_joint_feature_columns_unchanged':True}


def expected_frozen_settings(*,load_lag_days=(),development_path=DEV,sealed_path=TEST):
    """Metadata-only contract for review; does not freeze or authorize a run."""
    policy=lag_policy(load_lag_days)
    folder=Path(__file__).resolve().parent
    return {'script_version':SCRIPT_VERSION,'code_sha256':{n:sha256(folder/n) for n in CODE_FILES},
        'target_start_minimum':'2019-01-01T00:00:00+00:00','target_end_maximum_exclusive':'2026-01-01T00:00:00+00:00',
        'native_source_paths':{'development':str(Path(development_path).resolve()),'sealed':str(Path(sealed_path).resolve())},
        'native_source_sha256_policy':'source_file_hashes_recorded_in_assembly_manifest_after_test_gate',
        'base_margin_feature_count':69,'margin_feature_count':69+len(policy['columns']),
        'load_lag_policy':policy,'weather_members':list(range(1,21)),'grid':'common_integer_1deg',
        'weather_lead_base_hours':24,'weather_endpoint_steps':[24,30,36,42,48],
        'weather_release_policy':'initialization_plus_12h_retrospective_assumption',
        'holiday_library_version':holidays.__version__,
        'test_gate':'exact_builder_contract_status_frozen_test_outcomes_inspected_false_valid_clock_before_sealed_value_reads'}


def check_test_access(*,required,allow_test,frozen_config,expected):
    if not required:
        return None
    if not allow_test or not frozen_config:
        raise PermissionError('Test observation assembly requires preinspection frozen configuration')
    path=Path(frozen_config)
    access=json.loads(path.read_text(encoding='utf-8-sig'))
    if access.get('status')!='frozen' or access.get('test_outcomes_inspected') is not False:
        raise PermissionError('Configuration was not frozen before test inspection')
    freeze=pd.Timestamp(access.get('frozen_at_utc'))
    if pd.isna(freeze) or freeze.tzinfo is None or freeze>pd.Timestamp.now(tz='UTC'):
        raise PermissionError('Invalid preinspection freeze time')
    if access.get('weather_panel_builder')!=expected:
        raise PermissionError('Frozen weather_panel_builder code/source/lag/schema contract mismatch')
    return {'path':str(path.resolve()),'sha256':sha256(path),'frozen_at_utc':freeze.tz_convert('UTC').isoformat(),
        'exact_builder_contract_checked_before_values':True,'test_outcomes_inspected_at_freeze':False}


def _inspect_parquet_times(path,column,*,test_authorized):
    import pyarrow.parquet as pq
    values=pq.read_table(path,columns=[column]).column(column).to_pandas()
    times=pd.DatetimeIndex(values)
    if times.tz is None or times.hasnans or len(times)==0:
        raise ValueError('Source times must be nonempty and timezone-aware before value-column access')
    times=times.tz_convert('UTC')
    if times.min()<pd.Timestamp('2019-01-01',tz='UTC') or times.max()>=pd.Timestamp('2026-01-01',tz='UTC'):
        raise PermissionError('Parquet source target times outside supported observation range')
    if times.max()>=TEST_START and not test_authorized:
        raise PermissionError('Sealed timestamp detected before any value-column load')
    return times


def calendar_with_holidays(index):
    out = calendar_features(index)
    dates = pd.DatetimeIndex(index).tz_convert('Europe/Brussels').date
    years = list(range(2018,2027))
    for country in ('DE','LU','FR','BE'):
        schedule = holidays.country_holidays(country,years=years,observed=True,language='en_US')
        for offset,label in ((-1,'previous'),(0,'today'),(1,'next')):
            shifted = [(pd.Timestamp(d)+pd.Timedelta(days=offset)).date() for d in dates]
            out[f'holiday_{country}_{label}'] = np.asarray([d in schedule for d in shifted],float)
    return out


def feature_tables(summary):
    index = pd.DatetimeIndex(summary.index)
    calendar = calendar_with_holidays(index)
    margins = calendar.copy()
    for zone in REGIONS:
        for variable in PHYSICAL:
            for statistic in ('mean','std'):
                name = f'{zone}_{variable}_{statistic}'
                margins[name] = summary[name] - (273.15 if variable=='t2m' and statistic=='mean' else 0.)
    # Prespecified low-dimensional response interactions aid the additive
    # baseline and expose gradual renewable-capacity changes to both families.
    for zone in REGIONS:
        for harmonic in ('hour_sin_1','hour_cos_1'):
            margins[f'{zone}_temperature_x_{harmonic}'] = margins[f'{zone}_t2m_mean']*calendar[harmonic]
        for variable in ('wind_speed','marine_wind_speed','dswrf'):
            margins[f'{zone}_{variable}_x_slow_time'] = margins[f'{zone}_{variable}_mean']*calendar.years_since_2019
    joint = pd.DataFrame(index=index)
    for source,name in (('year_sin_1','season_sin'),('year_cos_1','season_cos'),
                        ('hour_sin_1','hour_sin'),('hour_cos_1','hour_cos')):
        joint[name] = calendar[source]
    joint['weekend'] = calendar.weekday_5+calendar.weekday_6
    for variable,name in (('t2m','temperature'),('wind_speed','land_wind'),
                          ('marine_wind_speed','marine_wind'),('dswrf','solar')):
        joint[f'common_{name}'] = margins[[f'{z}_{variable}_mean' for z in REGIONS]].mean(axis=1)
        joint[f'common_{name}_spread'] = margins[[f'{z}_{variable}_std' for z in REGIONS]].mean(axis=1)
        # Two independent contrasts encode all three zone means without an
        # outcome-dependent choice of geographical weights.
        joint[f'{name}_DE_minus_FR'] = margins[f'DE_LU_{variable}_mean']-margins[f'FR_{variable}_mean']
        joint[f'{name}_BE_minus_FR'] = margins[f'BE_{variable}_mean']-margins[f'FR_{variable}_mean']
    return margins,joint,calendar


def validate_cache(summary,member,manifest,init):
    target = init+pd.Timedelta(days=1)
    expected = pd.date_range(target,periods=24,freq='h')
    if manifest.get('status')!='complete' or manifest.get('lead_base')!=24 or manifest.get('members')!=20:
        raise ValueError('Incompatible or incomplete regional weather cache')
    if manifest.get('grid_mode')!='common_1deg' or manifest.get('steps')!=[24,30,36,42,48]:
        raise ValueError('Incorrect primary weather grid/forecast endpoints')
    if summary.target_time.duplicated().any() or not pd.DatetimeIndex(summary.target_time).equals(expected):
        raise ValueError('Weather summary must contain all 24 ordered target hours')
    if len(member)!=480 or member.duplicated(['target_time','member_id']).any():
        raise ValueError('Incomplete member weather table')
    for stamp,group in member.groupby('target_time'):
        if set(group.member_id)!=set(range(1,21)):
            raise ValueError('Incorrect common member identities')
    issue = target-pd.Timedelta(hours=12)
    for frame in (summary,member):
        if not (frame.initialization_time==init).all() or not (frame.issue_time==issue).all():
            raise ValueError('Decoded weather initialization/issue mismatch')
        if not (frame.feature_grid_degrees==1.).all():
            raise ValueError('Primary feature support is not common integer-degree nodes')
    columns = [f'{z}_{v}_{s}' for z in REGIONS for v in PHYSICAL for s in ('mean','std')]
    if not np.isfinite(summary[columns].to_numpy(float)).all():
        raise ValueError('Nonfinite regional forecast feature')


def assemble(cache,start,end,destination,*,development_path=DEV,sealed_path=TEST,
             allow_test=False,frozen_config=None,load_lag_days=()):
    start,end=pd.Timestamp(start),pd.Timestamp(end)
    start=start.tz_localize('UTC') if start.tzinfo is None else start.tz_convert('UTC')
    end=end.tz_localize('UTC') if end.tzinfo is None else end.tz_convert('UTC')
    if (start>=end or start!=start.normalize() or end!=end.normalize() or
            start<pd.Timestamp('2019-01-01',tz='UTC') or end>pd.Timestamp('2026-01-01',tz='UTC')):
        raise ValueError('Supported targets are complete UTC days in 2019--2025 with exclusive end')
    days=validate_load_lag_days(load_lag_days)
    expected=expected_frozen_settings(load_lag_days=days,development_path=development_path,sealed_path=sealed_path)
    access=check_test_access(required=end>TEST_START,allow_test=allow_test,frozen_config=frozen_config,expected=expected)
    destination=Path(destination).resolve()
    if not destination.resolve().is_relative_to(_work_path().resolve()) or destination.exists():
        raise ValueError('Use a new D output directory')
    free=shutil.disk_usage(_work_path()).free
    if free<2*1024**3:
        raise RuntimeError('Insufficient D capacity; no alternate drive used')
    summaries,members,sources=[],[],[]
    for target in pd.date_range(start,end,freq='D',inclusive='left'):
        init=target-pd.Timedelta(days=1)
        folder=Path(cache)/init.strftime('%Y%m%d')
        manifest_path=folder/'manifest.json'
        if not manifest_path.is_file():
            raise FileNotFoundError('Incomplete requested weather day: '+str(init))
        record=json.loads(manifest_path.read_text(encoding='utf8'))
        for filename in ('ensemble_summary.parquet','member_hourly.parquet'):
            _inspect_parquet_times(folder/filename,'target_time',test_authorized=access is not None)
        summary=pd.read_parquet(folder/'ensemble_summary.parquet')
        member=pd.read_parquet(folder/'member_hourly.parquet')
        validate_cache(summary,member,record,init)
        summaries.append(summary);members.append(member)
        sources.append({'initialization':str(init.date()),'cache_fingerprint':record['input_fingerprint'],
            'manifest_sha256':sha256(manifest_path),'summary_sha256':sha256(folder/'ensemble_summary.parquet'),
            'members_sha256':sha256(folder/'member_hourly.parquet')})
    summary=pd.concat(summaries).set_index('target_time').sort_index()
    member=pd.concat(members,ignore_index=True).sort_values(['target_time','member_id'])
    margin_features,joint_features,calendar=feature_tables(summary)
    base_names=list(margin_features)
    if len(base_names)!=69:
        raise ValueError('Established 69-feature base schema changed')
    _inspect_parquet_times(development_path,'timestamp',test_authorized=access is not None)
    observations=pd.read_parquet(development_path)
    observation_sources=[{'path':str(Path(development_path).resolve()),'sha256':sha256(development_path)}]
    if access is not None:
        _inspect_parquet_times(sealed_path,'timestamp',test_authorized=True)
        observations=pd.concat([observations,pd.read_parquet(sealed_path)],ignore_index=True)
        observation_sources.append({'path':str(Path(sealed_path).resolve()),'sha256':sha256(sealed_path)})
    observations['timestamp']=pd.to_datetime(observations.timestamp,utc=True)
    observations['zone']=observations.zone.replace({'DE-LU':'DE_LU'})
    source_start=start-pd.Timedelta(days=max(days,default=0))
    native_history=observations.loc[(observations.timestamp>=source_start)&(observations.timestamp<end)].copy()
    if native_history.duplicated(['timestamp','zone']).any():
        raise ValueError('Duplicate native observation hour including lag source history')
    lag_audit=[]
    if days:
        if 'load_mw' not in native_history:
            raise ValueError('Known total-load lags require native load_mw; never substitute net_load_mw')
        loads=native_history.pivot(index='timestamp',columns='zone',values='load_mw').sort_index().reindex(columns=REGIONS)
        lag_features,lag_audit=build_load_lags(summary.index,loads,days)
        margin_features=margin_features.join(lag_features)
    observations=native_history.loc[native_history.timestamp>=start].copy()
    wide=observations.pivot(index='timestamp',columns='zone',values='net_load_mw').reindex(summary.index)
    panel=margin_features.join(wide.reindex(columns=REGIONS).rename(columns={z:'net_load_'+z for z in REGIONS}))
    panel.index.name=joint_features.index.name=calendar.index.name='target_time'
    timing=summary[['initialization_time','issue_time','native_grid_degrees','feature_grid_degrees']].copy()
    timing['available_time']=timing.initialization_time+pd.Timedelta(hours=12)
    timing.index.name='target_time'
    label_available=pd.DataFrame({'label_available_time':panel.index+pd.Timedelta(hours=1)},index=panel.index)
    label_available.index.name='target_time'
    feature_names=list(margin_features)
    lag_names=load_lag_names(days)
    if feature_names!=base_names+lag_names:
        raise AssertionError('Lag schema must append after unchanged ordered base69')
    linear=['season_sin','season_cos','hour_sin','hour_cos','weekend']
    lean=linear+[f'common_{n}{suffix}' for n in ('temperature','land_wind','marine_wind','solar') for suffix in ('','_spread')]
    rich=list(joint_features)
    destination.mkdir(parents=True)
    paths={'panel':'panel.parquet','joint_features':'joint_features_rich.parquet',
        'joint_lean_features':'joint_features_lean.parquet','member_weather':'member_weather.parquet',
        'calendar':'calendar.parquet','timing':'timing.parquet','label_availability':'label_availability.parquet',
        'observation_quality':'observation_quality.parquet'}
    for name,frame in (('panel',panel),('joint_features',joint_features),('joint_lean_features',joint_features[lean]),
            ('member_weather',member),('calendar',calendar),('timing',timing),
            ('label_availability',label_available),('observation_quality',observations)):
        frame.to_parquet(destination/paths[name],index=name not in ('member_weather','observation_quality'))
    policy=lag_policy(days)
    margin_spec={'feature_columns':feature_names,'weather_source':'GEFS operational archive',
        'weather_age_at_issue_hours':12,'feature_support':'common integer-degree nodes',
        'holiday_library_version':holidays.__version__,'holiday_source_url':'https://github.com/vacanza/holidays',
        'temperature_units':'degrees Celsius in panel; Kelvin preserved in member weather',
        'lagged_observations_used':bool(days),'load_lag_policy':policy,
        'base_feature_count':len(base_names),'load_lag_linear_columns':lag_names,
        'load_lag_feature_indices':{name:feature_names.index(name) for name in lag_names},
        'response_interactions':'temperature with first hour harmonics; wind/solar means with deterministic slow time'}
    (destination/'margin_features.json').write_text(json.dumps(margin_spec,indent=2),encoding='utf8')
    for label,names in (('lean',lean),('rich',rich)):
        spec={'feature_columns':names,'linear_columns':linear,'smooth_columns':[n for n in names if n not in linear],
            'interactions':[['common_temperature','common_land_wind'],['common_temperature','common_solar']],
            'release_delay_hours':12.,'feature_definition':'Equal-zone regional means/spreads and prespecified zone contrasts; no observed outcomes',
            'shared_marginal_load_lags_not_added_to_joint_weather_basis':lag_names}
        (destination/f'joint_features_{label}.json').write_text(json.dumps(spec,indent=2),encoding='utf8')
    report={'status':'complete','script_version':SCRIPT_VERSION,'created_utc':datetime.now(timezone.utc).isoformat(),
        'target_start':start.isoformat(),'target_end_exclusive':end.isoformat(),'hours':len(panel),'weather_days':len(sources),
        'member_rows':len(member),'electricity_test_outcomes_opened':access is not None,'frozen_configuration':access,
        'developer_score_or_event_computed':False,'observation_sources':observation_sources,'weather_sources':sources,
        'D_free_bytes_before':free,'builder_sha256':sha256(__file__),'settings':expected,
        'paths':{name:str(destination/value) for name,value in paths.items()},
        'margin_feature_spec':str(destination/'margin_features.json'),'calendar_columns':list(calendar),
        'scale_exclude_indices':[feature_names.index('years_since_2019')],
        'load_lag_policy':policy,'load_lag_source_start_inclusive':source_start.isoformat(),'load_lag_audit':lag_audit,
        'shared_marginal_feature_count':len(feature_names),'load_lag_linear_feature_indices':[feature_names.index(n) for n in lag_names],
        'C7_ECC_lag_information_contract':policy['C7_ECC_use'],'C8_analog_lag_information_contract':policy['C8_analog_use'],
        'member_weather_lags_appended':False,'joint_weather_lags_appended':False,
        'label_release_assumption':'Interval-end earliest availability plus nominal model training cutoff origin minus seven days; historical revised-label arrival not certified',
        'weather_release_assumption':'Initialization plus 12 hours is a retrospective conservative policy; first public arrival not exhaustively verified',
        'artifacts_sha256':{p.name:sha256(p) for p in destination.iterdir() if p.is_file()}}
    (destination/'assembly_manifest.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    print(json.dumps({'status':'complete','destination':str(destination),'hours':len(panel),'weather_days':len(sources),
        'margin_features':len(feature_names),'load_lag_days':list(days)}),flush=True)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache',default=str(CACHE));p.add_argument('--start',default='2019-01-01');p.add_argument('--end',required=True)
    p.add_argument('--destination',required=True);p.add_argument('--allow-test',action='store_true');p.add_argument('--frozen-config')
    p.add_argument('--load-lag-days',nargs='*',type=int,default=[],help='Prespecified known total-load lags 14 21; default none')
    a=p.parse_args();assemble(a.cache,a.start,a.end,a.destination,allow_test=a.allow_test,frozen_config=a.frozen_config,
        load_lag_days=a.load_lag_days)
