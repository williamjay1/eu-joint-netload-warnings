"""Portable score/coverage/inference reconstruction from exported aggregates.

No source observations, weather archive, credentials, or fitted model objects
are required. This reconstructs evaluation after frozen forecasts and plans;
it does not reconstruct their fitting or availability.
"""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd
REPS=2000
ZONES=('DE_LU_FR_BE','DK1','DK2')

def holm(p):
    p=np.asarray(p,float);order=np.argsort(p,kind='stable');adj=np.empty(len(p))
    adj[order]=np.minimum(1.,np.maximum.accumulate((len(p)-np.arange(len(p)))*p[order]));return adj
def row(kind,q,zone,contrast,point,draws):
    draws=draws[np.isfinite(draws)];ci=np.quantile(draws,[.025,.975])
    return {'family':kind+'_'+q,'zone':zone,'contrast':contrast,'estimate':float(point),
      'pointwise_ci_low':float(ci[0]),'pointwise_ci_high':float(ci[1]),
      'raw_centered_bootstrap_p':(1+int((np.abs(draws-point)>=abs(point)).sum()))/(len(draws)+1)}
def run(input_dir,output_dir):
    output_dir.mkdir(parents=True,exist_ok=True)
    records=[];score_rows=[];coverage_rows=[];checks=[];sources={}
    def read(name):
        p=input_dir/name;sources[name]=hashlib.sha256(p.read_bytes()).hexdigest()
        return pd.read_csv(p,float_precision='round_trip')
    for zone in ZONES:
        asset=read(f'daily_loss_sufficient_statistics_{zone}.csv')
        rng=np.random.default_rng(61002)
        for period in ('development','reanalysis'):
            for q in ('q90','q95'):
                d=asset[(asset.period==period)&(asset.q==q)]
                daily=d[[f'loss_sum_M{m}' for m in range(4)]].to_numpy();hours=d.hours.to_numpy();n=len(d)
                point=daily.sum(axis=0)/hours.sum()
                for m in range(4):score_rows.append({'zone':zone,'period':period,'q':q,'model':f'M{m}',
                  'Brier':float(point[m]),'hours':int(hours.sum()),'days':int((hours==24).sum()),'event_hours':float(d.event_hours.sum())})
                for extra in ('Independence','Direct_logistic'):
                    if 'loss_sum_'+extra in d:
                        score_rows.append({'zone':zone,'period':period,'q':q,'model':extra,
                          'Brier':float(d['loss_sum_'+extra].sum()/hours.sum()),'hours':int(hours.sum()),
                          'days':int((hours==24).sum()),'event_hours':float(d.event_hours.sum())})
                for block in (7,14,28):
                    picks=(rng.choice(np.arange(n),size=(REPS,int(np.ceil(n/block))))[:,:,None]+np.arange(block))%n
                    picks=picks.reshape(REPS,-1)[:,:n]
                    means=daily[picks].sum(axis=1)/hours[picks].sum(axis=1)[:,None]
                    if period=='reanalysis' and block==14:
                        for contrast,candidate,reference in (('margin_at_G',1,0),('dependence_at_calibrated',3,1)):
                            records.append(row('model',q,zone,contrast,point[candidate]-point[reference],means[:,candidate]-means[:,reference]))
        for q in ('q90','q95'):
            d=read(f'daily_onset_sufficient_statistics_{zone}_{q}.csv')
            calendar=pd.DatetimeIndex(pd.to_datetime(d.date,utc=True));labels=calendar.year*10+calendar.quarter
            strata=[np.flatnonzero(labels==x) for x in np.unique(labels)]
            values=d[['n','P1_minus_P0','P2_minus_P1']].to_numpy()
            for j in range(3):coverage_rows.append({'zone':zone,'q':q,'policy':f'P{j}',
                'processes':int(d.n.sum()),'expected_first6_covered_processes':float(d[f'P{j}_coverage_sum'].sum()),
                'matched_first6_coverage':float(d[f'P{j}_coverage_sum'].sum()/d.n.sum())})
            draws=np.empty((REPS,2));rng=np.random.default_rng(20261016)
            for rep in range(REPS):
                total=np.zeros(3)
                for ix in strata:
                    length=min(len(ix),14)
                    starts=rng.integers(0,len(ix)-length+1,size=int(np.ceil(len(ix)/length)))
                    picked=ix[(starts[:,None]+np.arange(length)).ravel()[:len(ix)]]
                    total+=values[picked].sum(axis=0)
                draws[rep]=total[1:]/total[0] if total[0] else np.nan
            point=values[:,1:].sum(axis=0)/values[:,0].sum()
            for j,contrast in enumerate(('P1_minus_P0','P2_minus_P1')):records.append(row('policy',q,zone,contrast,point[j],draws[:,j]))
    table=pd.DataFrame(records);table['holm_adjusted_p']=np.nan
    for family,idx in table.groupby('family',sort=False).groups.items():
        assert len(idx)==6;table.loc[idx,'holm_adjusted_p']=holm(table.loc[idx,'raw_centered_bootstrap_p'])
    expected=read('finite_family_inference.csv')
    keys=['family','zone','contrast'];numeric=['estimate','pointwise_ci_low','pointwise_ci_high','raw_centered_bootstrap_p','holm_adjusted_p']
    expected=expected.set_index(keys).sort_index();actual=table.set_index(keys).sort_index()
    errors=np.abs(expected[numeric].to_numpy()-actual[numeric].to_numpy())
    assert np.allclose(expected[numeric],actual[numeric],atol=1e-12,rtol=0),'Aggregate inference readback failed'
    checks.append({'name':'all_24_contrasts_and_Holm','status':'passed','max_absolute_error':float(errors.max())})
    pd.DataFrame(score_rows).to_csv(output_dir/'reconstructed_model_scores.csv',index=False)
    pd.DataFrame(coverage_rows).to_csv(output_dir/'reconstructed_matched_coverage.csv',index=False)
    table.to_csv(output_dir/'reconstructed_finite_family_inference.csv',index=False)
    (output_dir/'analysis_verification.json').write_text(json.dumps({'status':'passed','scope':'post-forecast aggregate reconstruction only',
      'model_refitting':False,'forecast_allocation_replayed':False,'checks':checks,'source_sha256':sources,
      'python_dependencies':['numpy','pandas'],'statistical_scope':'exploratory centered-bootstrap null approximation and four independent six-test Holm families'},indent=2),encoding='utf-8')
    print(json.dumps({'status':'passed','contrasts':len(table),'scores':len(score_rows),'coverage':len(coverage_rows),'max_readback_error':float(errors.max())}))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input-dir',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args();run(a.input_dir,a.output_dir)
