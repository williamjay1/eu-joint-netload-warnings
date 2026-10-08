"""Fixed warning-specification transfer to both Danish bidding-zone triples.

No Danish performance tunes gates, cooldowns, regional membership or weather
choice. Macro weather is an input sensitivity, local weather the primary input.
"""
from __future__ import annotations
from release_paths import work_path as _work_path, raw_path as _raw_path
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from upgrade_warning_policies_20261002 import policy,process_masks,summarise,onset_block_ci

ROOT=_work_path()
OUT=ROOT/'results/upgrade_external_warning_policies_20261002'

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def load(path,q):
    df=pd.read_parquet(path)
    f=pd.DataFrame({'p':df['M0_'+q],'event':df['y_'+q]},index=df.index)
    pred_complete=f.p.notna().groupby(f.index.normalize()).sum().eq(24)
    f.loc[~f.index.normalize().isin(pred_complete.index[pred_complete]),'p']=np.nan
    score_complete=f.notna().all(axis=1).groupby(f.index.normalize()).sum().eq(24)
    f.loc[~f.index.normalize().isin(score_complete.index[score_complete]),'event']=np.nan
    return f,df['y_'+q+'_DK_pivotal'].reindex(f.index)

def run(weather):
    cfgpath=ROOT/'config/upgrade_warning_policies_20261002.json'
    cfg=json.loads(cfgpath.read_text(encoding='utf-8'))
    assert cfg['forecast_gate']==.25 and cfg['priority_gate']==.19 and cfg['cooldown_hours']==6
    prefix={'macro':'macro_strict15d','local':'local_weather_strict15d'}[weather]
    destination=OUT/prefix;destination.mkdir(parents=True,exist_ok=True)
    records=[];intervals=[];sources={};started=time.perf_counter()
    for zone in ('DK1','DK2'):
        path=ROOT/'results/upgrade_external_transfer_20261002'/prefix/zone/'upgrade_predictions.parquet'
        sources[zone]={'path':str(path),'sha256':digest(path)}
        for q in ('q90','q95'):
            history,pivotal=load(path,q)
            mask=history.index.year>=2024
            plans={'P0_forced_top2':policy(history,0.)[mask],
                   'P1_risk_gate':policy(history,cfg['forecast_gate'])[mask],
                   'P2_risk_rise_cooldown':policy(history,cfg['priority_gate'],cfg['cooldown_hours'],True)[mask]}
            f=history.loc[mask];yp=pivotal.loc[mask].to_numpy(float)
            known=f.event.notna().to_numpy()
            B=int(min((plans['P1_risk_gate']&known).sum(),(plans['P2_risk_rise_cooldown']&known).sum()))
            for gap in (0,1,2):
                ps=process_masks(f,gap);recs={}
                for name,a in plans.items():
                    for mode,budget in [('native',None),('matched',B)]:
                        s,rec=summarise(f,a,ps,budget)
                        n=s['unstandardised_review_hours'];ratio=s['matched_review_hours']/n if n else 0.
                        s['expected_selected_DK_pivotal_hours']=float((a&known&(yp==1)).sum()*ratio)
                        s['DK_pivotal_event_hours']=int((known&(yp==1)).sum())
                        records.append({'weather':weather,'zone':zone,'q':q,'gap_days':gap,
                                        'policy':name,'mode':mode,**s})
                        if mode=='matched':
                            recs[name]=rec
                            rec.to_csv(destination/f'processes_{zone}_{q}_{gap}_{name}.csv',index=False)
                if gap==1:
                    for block in (7,14,28):
                        for ref in ('P0_forced_top2','P1_risk_gate'):
                            intervals.append({'zone':zone,'q':q,**onset_block_ci(f,recs,'P2_risk_rise_cooldown',ref,block)})
            saved=pd.DataFrame(plans,index=f.index)
            saved['p']=f.p;saved['event']=f.event;saved['DK_pivotal_event']=yp
            saved['evaluation_complete_day']=known
            saved.to_parquet(destination/f'alerts_{zone}_{q}.parquet')
    pd.DataFrame(records).to_csv(destination/'policy_results.csv',index=False)
    manifest={'status':'completed','weather':weather,'role':'primary' if weather=='local' else 'input sensitivity',
              'inputs':sources,'configuration':str(cfgpath),'configuration_sha256':digest(cfgpath),
              'script_sha256':digest(__file__),'elapsed_seconds':time.perf_counter()-started,
              'gates_retuned_in_DK':False,'priority_score':'p_t*(1-max previous six issued hourly risks)',
              'forecasts_allocated_before_truth_domain_restriction':True,
              'history_before_scoring_boundary_preserved':True,
              'primary_comparison':'P2 minus P1 first-six-hour process coverage at common total review exposure',
              'paired_intervals':intervals}
    (destination/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps({'weather':weather,'status':manifest['status'],
                      'destination':str(destination),'elapsed_seconds':manifest['elapsed_seconds']},indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('weather',choices=['macro','local'])
    run(parser.parse_args().weather)
