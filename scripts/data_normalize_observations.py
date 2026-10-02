"""Normalize public observations, retaining missing labels and sealed test period.

Main sources: SMARD DE-LU, RTE France, Energy-Charts Belgium.
France eolien is total wind: no unsupported onshore/offshore split is fabricated.
"""
from release_paths import work_path as _work_path, raw_path as _raw_path
from pathlib import Path
import argparse, json
import numpy as np
import pandas as pd

ROOT=_raw_path()
OUT=_work_path('datasets')
RESULTS=_work_path() / 'results'
FIELDS=['load_mw','solar_mw','wind_onshore_mw','wind_offshore_mw','wind_total_mw']

def latest(pattern):
    paths=sorted(ROOT.glob(pattern))
    if not paths: raise FileNotFoundError(pattern)
    return paths[-1]

def unique_series(series):
    """Collapse exact duplicates; conflicting duplicates become missing."""
    if series.index.has_duplicates:
        nunique=series.groupby(level=0).nunique(dropna=False)
        series=series.groupby(level=0).first()
        series.loc[nunique>1]=np.nan
    return series.sort_index()

def hourly(series, minutes):
    series=unique_series(series)
    if minutes==60: return series.resample('h').first()
    mean=series.resample('h').mean()
    mean[series.resample('h').count() != 60//minutes]=np.nan
    return mean

def smard():
    root=latest('observations_smard_*')
    columns={}
    for field in FIELDS[:4]:
        rows=[]
        for p in sorted(root.glob(f'smard_{field}_[0-9]*.json')):
            rows.extend(json.loads(p.read_text())['series'])
        if not rows: continue
        x=pd.DataFrame(rows,columns=['timestamp',field])
        x['timestamp']=pd.to_datetime(x.timestamp,unit='ms',utc=True)
        # SMARD reports interval energy MWh; each requested hour is one hour.
        columns[field]=hourly(x.set_index('timestamp')[field].astype(float),60)/1.0
    df=pd.DataFrame(columns)
    if {'wind_onshore_mw','wind_offshore_mw'}<=set(df):
        df['wind_total_mw']=df.wind_onshore_mw+df.wind_offshore_mw
    df['zone']='DE-LU'; df['source_domain']='SMARD DE-LU / ENTSO-E'; return df

def rte():
    root=latest('observations_rte_*')
    cols={'consommation':'load_mw','eolien':'wind_total_mw','solaire':'solar_mw'}
    frames=[]
    for p in sorted(root.glob('rte_eco2mix_*.csv')):
        df=pd.read_csv(p,sep=';',encoding='utf-8-sig').rename(columns=cols)
        df.index=pd.to_datetime(df.date_heure,utc=True)
        # Dataset has quarter-hour forecast rows but half-hour actual values.
        df=df.dropna(subset=list(cols.values()),how='all')
        frames.append(pd.DataFrame({c:hourly(pd.to_numeric(df[c],errors='coerce'),30) for c in cols.values()}))
    df=pd.concat(frames).sort_index()
    df['wind_onshore_mw']=np.nan; df['wind_offshore_mw']=np.nan
    df['zone']='FR'; df['source_domain']='RTE eco2mix national consolidated/definitive'; return df

def energycharts(country):
    root=latest('observations_energycharts_*')
    names={'Load':'load_mw','Solar':'solar_mw','Wind onshore':'wind_onshore_mw','Wind offshore':'wind_offshore_mw'}
    frames=[]; metadata=[]
    for p in sorted(root.glob(f'energycharts_{country}_[0-9]*.json')):
        j=json.loads(p.read_text()); idx=pd.to_datetime(j['unix_seconds'],unit='s',utc=True)
        delta=pd.Series(idx).diff().dt.total_seconds().dropna()
        minutes=int(delta[delta>0].mode().iloc[0]/60)
        if minutes not in (15,30,60): raise ValueError(f'Unexpected cadence: {p.name}: {minutes}')
        data={names[x['name']]:hourly(pd.Series(x['data'],index=idx,dtype=float),minutes) for x in j['production_types'] if x['name'] in names}
        df=pd.DataFrame(data)
        metadata.append({'file':p.name,'interval_minutes_modal':minutes,'cadence_counts':delta.value_counts().to_dict(),'fields':list(data),'records':len(idx)})
        if country=='be':
            if not {'wind_onshore_mw','wind_offshore_mw'}<=set(df): raise ValueError(f'Missing BE wind component: {p.name}')
            df['wind_total_mw']=df.wind_onshore_mw+df.wind_offshore_mw
        frames.append(df)
    if not frames: raise FileNotFoundError(f'No Energy-Charts {country} annual files')
    df=pd.concat(frames).sort_index()
    df=pd.DataFrame({c:unique_series(df[c]) for c in df})
    df['zone']=country.upper(); df['source_domain']='Fraunhofer ISE Energy-Charts public_power'
    return df, metadata

def main():
    p=argparse.ArgumentParser(); p.add_argument('--allow-partial',action='store_true'); a=p.parse_args()
    frames=[]; metadata=[]; issues=[]
    for name,fn in [('DE-LU',smard),('FR',rte),('BE',lambda:energycharts('be'))]:
        try:
            result=fn()
            if isinstance(result,tuple): df,meta=result; metadata.extend(meta)
            else: df=result
            df=df[(df.index>=pd.Timestamp('2019-01-01',tz='UTC'))&(df.index<pd.Timestamp('2026-01-01',tz='UTC'))]
            for c in FIELDS:
                if c not in df: df[c]=np.nan
            df['net_load_mw']=df.load_mw-df.solar_mw-df.wind_total_mw
            df['quality_flag']=np.where(df.net_load_mw.notna(),'complete','missing_component')
            df.index.name='timestamp'; frames.append(df.reset_index())
        except Exception as e:
            issues.append({'zone':name,'error':str(e)})
            if not a.allow_partial: raise
    all_df=pd.concat(frames,ignore_index=True).sort_values(['timestamp','zone'])
    OUT.mkdir(parents=True,exist_ok=True)
    quality=[]
    for (zone,year),d in all_df.groupby(['zone',all_df.timestamp.dt.year]):
        expected=int((pd.Timestamp(year+1,1,1)-pd.Timestamp(year,1,1)).total_seconds()/3600)
        quality.append({'zone':zone,'year':int(year),'observed_rows':len(d),'expected_hours':expected,'complete_netload_hours':int(d.net_load_mw.notna().sum()),'complete_fraction':float(d.net_load_mw.notna().sum()/expected),'duplicate_timestamp_rows':int(d.timestamp.duplicated().sum())})
    suffix='_partial' if a.allow_partial else ''
    for start,end,split in [(2019,2023,'2019_2023'),(2024,2025,'2024_2025_sealed')]:
        df=all_df[(all_df.timestamp.dt.year>=start)&(all_df.timestamp.dt.year<=end)]
        target=OUT/f'data_observations_{split}{suffix}.parquet'; df.to_parquet(target,index=False)
    report={'sources':{'DE-LU':'SMARD exact bidding-zone key DE-LU, hourly MWh divided by 1h','FR':'RTE eCO2mix actual half-hour MW, all-source wind aggregated','BE':'Fraunhofer ISE Energy-Charts public_power MW; exact Load field'},'yearly_coverage_only':quality,'issues':issues,'api_cadences':metadata,'test_policy':'No test event frequencies, values, cases or model scores displayed; only structural quality counts.','negative_generation_policy':'Retain published net generation values, including small negative RTE solar readings; do not clip or impute.', 'publication_ready':False, 'publication_gate':'BE Energy-Charts historical hourly data are sampled first-quarter-hour observations, not verified hourly means. Full table is for pipeline development only until native-resolution replacement is completed. Also verify RTE timestamp interval semantics before formal aggregation.'}
    (RESULTS/f'data_observation_quality{suffix}.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    print(json.dumps({'outputs':str(OUT),'yearly_coverage':quality,'issues':issues}))

if __name__=='__main__': main()
