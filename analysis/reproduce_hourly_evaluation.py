"""Recompute scores, forecast-only plans and intact episodes from released data.

This executes evaluation of the original issued forecasts. It does not fit a
forecast model or recover source publication vintages.
"""
from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from upgrade_warning_policies_20261002 import policy, process_masks, summarise, causal_selftest
ZONES=('DE_LU_FR_BE','DK1','DK2')
POLS=('P0_forced_top2','P1_risk_gate','P2_risk_rise_cooldown')

def numeric_equal(actual,expected,columns,label,tolerance=1e-11):
    a=actual[columns].to_numpy(float);b=expected[columns].to_numpy(float)
    assert a.shape==b.shape,(label,a.shape,b.shape)
    error=np.abs(a-b)
    assert np.allclose(a,b,atol=tolerance,rtol=0,equal_nan=True),(label,float(np.nanmax(error)))
    return {'name':label,'rows':len(actual),'columns':columns,'max_absolute_error':float(np.nanmax(error)),'status':'passed'}

def run(output):
    output.mkdir(parents=True,exist_ok=True)
    checks=[];scores=[];policy_rows=[];sources={}
    for zone in ZONES:
        path=ROOT/f'data/hourly/{zone}.parquet'
        sources[path.relative_to(ROOT).as_posix()]=hashlib.sha256(path.read_bytes()).hexdigest()
        f=pd.read_parquet(path)
        assert f.index.tz is not None and str(f.index.tz)=='UTC'
        assert f.index.is_unique and f.index.is_monotonic_increasing
        assert ((f.issue_time==f.index.normalize()-pd.Timedelta(hours=12)) | f.issue_time.isna()).all()
        all_daily=[]
        for period,years in (('development',(2022,2023)),('reanalysis',(2024,2025))):
            calendar=pd.date_range(f'{years[0]}-01-01',f'{years[-1]}-12-31',freq='D',tz='UTC')
            for q in ('q90','q95'):
                names=[f'M{m}' for m in range(4)]+[name for name in ('Independence','Direct_logistic') if name+'_'+q in f]
                required=['y_'+q]+[f'M{m}_{q}' for m in range(4)]
                okay=f[required].notna().all(axis=1)
                complete=okay.groupby(f.index.normalize()).sum().eq(24)
                mask=okay & f.index.normalize().isin(complete.index[complete]) & f.index.year.isin(years)
                selected=f.loc[mask]
                for name in names:
                    p=selected[name+'_'+q]
                    assert p.notna().all() and p.between(0,1).all()
                loss=pd.DataFrame({f'loss_sum_{name}':(selected[name+'_'+q]-selected['y_'+q])**2 for name in names})
                daily=loss.groupby(loss.index.normalize()).sum().reindex(calendar,fill_value=0.)
                daily['hours']=pd.Series(1,index=selected.index).groupby(selected.index.normalize()).sum().reindex(calendar,fill_value=0)
                daily['event_hours']=selected['y_'+q].groupby(selected.index.normalize()).sum().reindex(calendar,fill_value=0)
                daily['date']=calendar.astype(str);daily['period']=period;daily['q']=q;daily['zone']=zone
                all_daily.append(daily.reset_index(drop=True))
                for name in names:
                    scores.append({'zone':zone,'period':period,'q':q,'model':name,'Brier':daily[f'loss_sum_{name}'].sum()/daily.hours.sum(),'hours':int(daily.hours.sum()),'days':int((daily.hours==24).sum()),'event_hours':int(daily.event_hours.sum())})
        daily=pd.concat(all_daily,ignore_index=True)
        expected=pd.read_csv(ROOT/f'data/aggregates/daily_loss_sufficient_statistics_{zone}.csv',float_precision='round_trip')
        assert daily[['date','period','q','zone']].equals(expected[['date','period','q','zone']])
        columns=[c for c in daily if c.startswith('loss_sum_')]+['hours','event_hours']
        checks.append(numeric_equal(daily,expected,columns,'hourly_to_daily_loss_'+zone))
        daily.to_csv(output/f'daily_loss_sufficient_statistics_{zone}.csv',index=False,float_format='%.17g')
        for q in ('q90','q95'):
            context=pd.DataFrame({'p':f['policy_p_'+q],'event':f['policy_event_'+q]},index=f.index)
            plans={'P0':policy(context,0.),'P1':policy(context,.25),'P2':policy(context,.19,6,True)}
            for short,plan in plans.items():
                assert np.array_equal(plan,f[short+'_'+q].to_numpy()),(zone,q,short,'stored plan mismatch')
            check_mask=f.index.year>=2024;scoring=context.loc[check_mask]
            alerts={long:plans[short][check_mask] for short,long in zip(('P0','P1','P2'),POLS)}
            known=scoring.event.notna().to_numpy()
            B=int(min((alerts[POLS[1]]&known).sum(),(alerts[POLS[2]]&known).sum()))
            reference=pd.read_csv(ROOT/('results/upgrade_warning_policies_20261002_v3/policy_results.csv' if zone=='DE_LU_FR_BE' else 'results/upgrade_external_warning_policies_20261002/local_weather_strict15d/policy_results.csv'),float_precision='round_trip')
            if zone=='DE_LU_FR_BE':reference=reference.rename(columns={'level':'q'})
            else:reference=reference[reference.zone==zone]
            for gap in (0,1,2):
                processes=process_masks(scoring,gap)
                ids=pd.Series(pd.NA,index=scoring.index,dtype='Int32')
                for j,e in enumerate(processes):ids.loc[scoring.index[e['ix']]]=j
                pd.testing.assert_series_equal(ids,f.loc[check_mask,f'event_process_id_{q}_gap{gap}'],check_names=False,check_exact=True)
                matched={}
                for long,a in alerts.items():
                    for mode,budget in (('native',None),('matched',B)):
                        result,rec=summarise(scoring,a,processes,budget)
                        row={'zone':zone,'q':q,'gap_days':gap,'policy':long,'mode':mode,**result}
                        policy_rows.append(row)
                        expected=reference[(reference.q==q)&(reference.gap_days==gap)&(reference.policy==long)&(reference['mode']==mode)]
                        assert len(expected)==1
                        checks.append(numeric_equal(pd.DataFrame([result]),expected,list(result),f'policy_{zone}_{q}_gap{gap}_{long}_{mode}',tolerance=1e-12))
                        if mode=='matched':matched[long]=rec
                if gap==1:
                    calendar=pd.date_range(scoring.index.min().normalize(),scoring.index.max().normalize(),freq='D',tz='UTC')
                    onset=pd.DataFrame(0.,index=calendar,columns=['n','P1_minus_P0','P2_minus_P1','P0_coverage_sum','P1_coverage_sum','P2_coverage_sum'])
                    for long in POLS:assert matched[long].onset.equals(matched[POLS[0]].onset)
                    for j,e in matched[POLS[0]].iterrows():
                        day=e.onset.normalize();onset.loc[day,'n']+=1
                        values=[float(matched[long].loc[j,'first_six_hours_hit_probability']) for long in POLS]
                        for k,value in enumerate(values):onset.loc[day,f'P{k}_coverage_sum']+=value
                        onset.loc[day,'P1_minus_P0']+=values[1]-values[0]
                        onset.loc[day,'P2_minus_P1']+=values[2]-values[1]
                    onset['date']=calendar.astype(str);onset['q']=q;onset['zone']=zone
                    onset=onset.reset_index(drop=True)
                    expected=pd.read_csv(ROOT/f'data/aggregates/daily_onset_sufficient_statistics_{zone}_{q}.csv',float_precision='round_trip')
                    assert onset[['date','q','zone']].equals(expected[['date','q','zone']])
                    checks.append(numeric_equal(onset,expected,[c for c in onset if c not in ('date','q','zone')],f'hourly_to_intact_onset_{zone}_{q}',tolerance=1e-12))
                    onset.to_csv(output/f'daily_onset_sufficient_statistics_{zone}_{q}.csv',index=False,float_format='%.17g')
        checks.append({'name':'all_issued_plans_and_six_event_process_ID_streams_'+zone,'status':'passed','stored_plans_match_exactly':True})
    pd.DataFrame(scores).to_csv(output/'hourly_recomputed_model_scores.csv',index=False,float_format='%.17g')
    pd.DataFrame(policy_rows).to_csv(output/'hourly_recomputed_policy_results.csv',index=False,float_format='%.17g')
    report={'status':'passed','scope':'frozen issued forecasts -> scores, forecast-only plans and realised episodes','forecast_model_refitted':False,'historical_vintages_recovered':False,'score_rows':len(scores),'policy_rows':len(policy_rows),'checks':checks,'input_sha256':sources,'synthetic_truth_invariance_test':causal_selftest()}
    (output/'hourly_verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({'status':'passed','score_rows':len(scores),'native_and_matched_policy_rows':len(policy_rows),'daily_assets_recomputed':9,'maximum_error':max(c.get('max_absolute_error',0) for c in checks)}))
    return report

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output-dir',type=Path,default=ROOT/'generated/hourly');args=parser.parse_args();run(args.output_dir)
