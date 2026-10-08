"""Forecast evaluation with daily budgets and calendar-preserving uncertainty."""
from __future__ import annotations
import numpy as np
import pandas as pd


def hourly_frame(times, y, predictions):
    index = pd.DatetimeIndex(pd.to_datetime(times, utc=True))
    if index.has_duplicates or not index.is_monotonic_increasing:
        raise ValueError('Unique increasing UTC timestamps required.')
    frame = pd.DataFrame({'event': np.asarray(y, float)}, index=index)
    for name, p in predictions.items():
        p = np.asarray(p, float)
        if len(p) != len(frame) or np.any(~np.isfinite(p)) or np.any((p<0)|(p>1)):
            raise ValueError(f'{name}: predictions invalid; model failures must be repaired, not masked.')
        frame[name] = p
    return frame


def daily_scores(frame, model_names):
    """Return full calendar including missing/incomplete days as NaN."""
    counts = frame['event'].groupby(frame.index.normalize()).count()
    valid = counts == 24
    daily = {}
    for model in model_names:
        loss = (frame[model]-frame.event)**2
        daily[model] = loss.groupby(frame.index.normalize()).mean().where(valid)
    out = pd.DataFrame(daily)
    return out.reindex(pd.date_range(out.index.min(),out.index.max(),freq='D',tz='UTC'))


def paired_block_ci(daily, candidate, reference, block_days=14, repetitions=2000,
                    seed=20260929, quarter_strata=True):
    """Paired contiguous moving blocks within calendar strata.

    NaN calendar days stay in the sampling lattice. Stratum means are weighted
    by their ORIGINAL valid-day counts. No hourly IID inference is used.
    """
    delta = (daily[candidate]-daily[reference]).to_numpy(float)
    labels = (daily.index.year*10+daily.index.quarter) if quarter_strata else daily.index.year
    strata = [np.flatnonzero(labels==s) for s in np.unique(labels)]
    weights = np.array([np.isfinite(delta[ix]).sum() for ix in strata],float)
    weights /= weights.sum()
    rng = np.random.default_rng(seed)
    samples = np.zeros(repetitions)
    for rep in range(repetitions):
        subtotal = 0.
        for weight, ix in zip(weights,strata):
            if weight==0: continue
            n=len(ix)
            for attempt in range(100):
                length=min(block_days,n)
                starts = rng.integers(0,n-length+1,size=int(np.ceil(n/length)))
                selected = (starts[:,None]+np.arange(length)[None,:]).ravel()[:n]
                values = delta[ix[selected]]
                if np.isfinite(values).any(): break
            else: raise RuntimeError('Bootstrap stratum has insufficient observed days.')
            subtotal += weight*np.nanmean(values)
        samples[rep]=subtotal
    return {'mean_difference':float(np.nanmean(delta)),
            'ci95':np.quantile(samples,[.025,.975]).tolist(),
            'bootstrap_replicates':repetitions,'block_days':block_days,
            'valid_days':int(np.isfinite(delta).sum()),
            'strata':'year_quarter' if quarter_strata else 'year'}


def top_k_alerts(frame, model, k=2):
    """Rank only the 24 hours of the next target day, at its common issue time."""
    alert = pd.Series(False,index=frame.index)
    for _, group in frame.groupby(frame.index.normalize()):
        if len(group)!=24 or not np.isfinite(group[model]).all(): continue
        # Stable sort: original timestamps are ascending and break probability ties.
        selected = group[model].sort_values(ascending=False,kind='stable').index[:k]
        alert.loc[selected]=True
    return alert


def monitoring_summary(frame, model, k=2):
    counts = frame.event.groupby(frame.index.normalize()).count()
    valid_days=counts.index[counts==24]
    valid=frame.index.normalize().isin(valid_days)
    alert=top_k_alerts(frame,model,k).to_numpy()[valid]
    truth=frame.event.to_numpy()[valid].astype(bool)
    tp=int((alert&truth).sum()); fn=int((~alert&truth).sum())
    fp=int((alert&~truth).sum()); tn=int((~alert&~truth).sum())
    return dict(k=k,tp=tp,fn=fn,fp=fp,tn=tn,
                miss_rate=fn/(tp+fn) if tp+fn else None,
                false_alert_fraction=fp/(tp+fp) if tp+fp else None,
                false_positive_rate=fp/(fp+tn) if fp+tn else None,
                alert_hours=int(alert.sum()),valid_days=int(len(valid_days)))


def event_processes(frame, max_empty_days=1):
    """Consecutive event-day runs, allowing specified intervening fully observed days.

    This is an event-clustering proxy, not meteorological process identification.
    An incomplete day splits a run and is flagged as an uncertain boundary.
    """
    calendar=pd.date_range(frame.index.min().normalize(),frame.index.max().normalize(),freq='D',tz='UTC')
    daily=frame.event.groupby(frame.index.normalize()).agg(['count','max']).reindex(calendar)
    observed=daily['count'].eq(24)
    active=daily['max'].eq(1)&observed
    episodes=[]; current=[]; last=None
    for day in calendar[active]:
        between=calendar[(calendar>last)&(calendar<day)] if last is not None else calendar[:0]
        connected=last is not None and len(between)<=max_empty_days and observed.loc[between].all()
        if not connected and current: episodes.append(current); current=[]
        current.append(day);last=day
    if current: episodes.append(current)
    output=[]
    for dates in episodes:
        start,end=dates[0],dates[-1]
        hours=frame.loc[(frame.index>=start)&(frame.index<end+pd.Timedelta(days=1))]
        event_hours=hours.index[hours.event.eq(1)]
        boundary_days=[start-pd.Timedelta(days=j) for j in range(1,max_empty_days+2)]
        boundary_days += [end+pd.Timedelta(days=j) for j in range(1,max_empty_days+2)]
        uncertain=any(not observed.get(day,False) for day in boundary_days)
        output.append(dict(onset=event_hours[0],end=event_hours[-1],
                           event_hours=event_hours,uncertain_boundary=bool(uncertain)))
    return output


def process_detection(frame, model, k=2, max_empty_days=1):
    alerts=top_k_alerts(frame,model,k)
    records=[]
    for e in event_processes(frame,max_empty_days):
        eh=e['event_hours']; hits=eh[alerts.loc[eh].to_numpy()]
        onset=e['onset']
        firstsix=hits[hits<onset+pd.Timedelta(hours=6)]
        firstday=hits[hits.normalize()==onset.normalize()]
        # Each target day's forecast is issued at previous day's 12 UTC.
        issues=hits.normalize()-pd.Timedelta(hours=12)
        leads=(onset-issues).total_seconds()/3600
        target_leads=(hits-issues).total_seconds()/3600
        records.append(dict(onset=onset.isoformat(),end=e['end'].isoformat(),
                            uncertain_boundary=e['uncertain_boundary'],event_hours=len(eh),
                            any_hit=bool(len(hits)),first_day_hit=bool(len(firstday)),
                            first_six_hours_hit=bool(len(firstsix)),
                            onset_lead_hours=float(np.max(leads)) if len(hits) else None,
                            first_alerted_event_target=hits[0].isoformat() if len(hits) else None,
                            first_alerted_target_lead_hours=float(target_leads[0]) if len(hits) else None,
                            detected_at_least_12h_early=bool(np.any(leads>=12))))
    return pd.DataFrame(records)
