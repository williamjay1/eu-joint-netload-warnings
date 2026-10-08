"""Time and event definitions shared by all forecasting methods."""
from __future__ import annotations
import numpy as np
import pandas as pd

REGIONS = ['DE_LU','FR','BE']


def forecast_clocks(target_times):
    target = pd.DatetimeIndex(pd.to_datetime(target_times,utc=True))
    day = target.normalize()
    issue = day-pd.Timedelta(hours=12)
    initialization = day-pd.Timedelta(days=3)
    return pd.DataFrame({'target_time':target,'issue_time':issue,
                         'weather_initialization':initialization,
                         'weather_policy_available':initialization+pd.Timedelta(hours=48),
                         'weather_age_at_issue_h':(issue-initialization).total_seconds()/3600,
                         'target_lead_h':(target-issue).total_seconds()/3600,
                         'weather_step_h':(target-initialization).total_seconds()/3600})


def calendar_features(target_times):
    idx=pd.DatetimeIndex(pd.to_datetime(target_times,utc=True))
    local=idx.tz_convert('Europe/Brussels')
    day=(local.dayofyear.to_numpy()-1)/365.2425
    hour=local.hour.to_numpy()/24
    columns={}
    for harmonic in (1,2,3):
        columns[f'year_sin_{harmonic}']=np.sin(2*np.pi*harmonic*day)
        columns[f'year_cos_{harmonic}']=np.cos(2*np.pi*harmonic*day)
    for harmonic in (1,2):
        columns[f'hour_sin_{harmonic}']=np.sin(2*np.pi*harmonic*hour)
        columns[f'hour_cos_{harmonic}']=np.cos(2*np.pi*harmonic*hour)
    for weekday in range(7):
        columns[f'weekday_{weekday}']=(local.dayofweek==weekday).astype(float)
    # A deterministic extrapolating time feature must not be inferred from future capacity.
    columns['years_since_2019']=(idx-pd.Timestamp('2019-01-01',tz='UTC')).total_seconds()/(365.2425*86400)
    return pd.DataFrame(columns,index=idx)


def leap_calendar_day(index):
    idx=pd.DatetimeIndex(index)
    return np.array([pd.Timestamp(year=2000,month=t.month,day=t.day).dayofyear-1 for t in idx])


class FixedSeasonalThreshold:
    """Same UTC hour, cyclic 61-day empirical quantile fitted on 2019--2021."""
    def __init__(self,q=.9,half_window=30,min_samples=100):
        self.q=q;self.half_window=half_window;self.min_samples=min_samples

    def fit(self, net_load):
        if not isinstance(net_load.index,pd.DatetimeIndex):
            raise ValueError('DatetimeIndex required.')
        if net_load.index.min()<pd.Timestamp('2019-01-01',tz='UTC') or net_load.index.max()>=pd.Timestamp('2022-01-01',tz='UTC'):
            raise ValueError('Threshold reference must be confined to 2019--2021.')
        self.columns_=list(net_load.columns)
        self.thresholds_=np.full((366,24,len(self.columns_)),np.nan)
        days=leap_calendar_day(net_load.index);hours=net_load.index.hour.to_numpy()
        for d in range(366):
            distance=np.minimum((days-d)%366,(d-days)%366)
            for h in range(24):
                selected=net_load.loc[(distance<=self.half_window)&(hours==h)]
                if (selected.count()<self.min_samples).any():
                    raise ValueError(f'Insufficient threshold data: calendar day {d}, UTC hour {h}.')
                self.thresholds_[d,h,:]=selected.quantile(self.q).to_numpy()
        return self

    def predict(self,times):
        idx=pd.DatetimeIndex(pd.to_datetime(times,utc=True))
        values=self.thresholds_[leap_calendar_day(idx),idx.hour.to_numpy(),:]
        return pd.DataFrame(values,index=idx,columns=self.columns_)


def joint_labels(net_load,thresholds,k=2):
    if not net_load.index.equals(thresholds.index) or list(net_load.columns)!=list(thresholds.columns):
        raise ValueError('Thresholds and observations must have identical ordering.')
    complete=net_load.notna().all(axis=1)&thresholds.notna().all(axis=1)
    y=(net_load.gt(thresholds).sum(axis=1)>=k).astype(float)
    return y.where(complete)


def assert_information_timing(feature_manifest):
    """Manifest rows include both physical valid times and publication availability."""
    required={'available_time','issue_time','target_time'}
    if not required.issubset(feature_manifest.columns):
        raise ValueError(f'Missing timing fields: {required-set(feature_manifest.columns)}')
    f=feature_manifest.copy()
    for col in required: f[col]=pd.to_datetime(f[col],utc=True)
    if (f.available_time>f.issue_time).any():
        raise ValueError('Future-available inputs detected.')
    if (f.issue_time>=f.target_time).any():
        raise ValueError('Forecast issue must precede target.')
