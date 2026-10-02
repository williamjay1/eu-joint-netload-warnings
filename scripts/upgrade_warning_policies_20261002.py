"""Forecast-only warning rules; delayed replay and workload-standardised comparisons.

No realised event or onset enters alert allocation. Development selection uses
2022 gates and 2023 process utility; the 2024--25 analysis is explicitly a
reanalysis. Uniform thinning is an evaluation standardisation, not a deployed
year-end budget controller. Exact expected detection integrates that randomisation.
"""
from __future__ import annotations
from release_paths import work_path as _work_path, raw_path as _raw_path
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import gammaln
from evaluation import event_processes

ROOT = _work_path()
OUT = ROOT / 'results/upgrade_warning_policies_20261002_v3'
CFG = ROOT / 'config/upgrade_warning_policies_20261002.json'
SEED = 20261002
ZONES = ('DE_LU', 'FR', 'BE')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inputs(level='q90', column=None):
    mf = ROOT/'results/shared_frozen_history_test_20261001/prequential_combined_shared_margins.parquet'
    pf = ROOT/'results/joint_runner_frozen_test_numerical_20261001/joint_runner_predictions.parquet'
    p = pd.read_parquet(pf)
    s = pd.read_parquet(mf).reindex(p.index)
    if column is None:
        column = 'C3_dynamic_gaussian_' + level
    a = s[['net_load_'+z for z in ZONES]].to_numpy(float)
    h = s[['threshold_'+level+'_'+z for z in ZONES]].to_numpy(float)
    known = np.isfinite(a).all(axis=1) & np.isfinite(h).all(axis=1)
    y = np.where(known, ((a>h).sum(axis=1)>=2).astype(float), np.nan)
    f = pd.DataFrame({'p': p[column], 'event': y}, index=p.index)
    f.loc[~p['forecast_eligible_'+level].to_numpy(bool), 'p'] = np.nan
    # Allocation sees forecast completeness only. Truth completeness defines
    # scoring afterward; unknown future labels cannot reset the cooldown state.
    pc = f.p.notna().groupby(f.index.normalize()).sum().eq(24)
    f.loc[~f.index.normalize().isin(pc.index[pc]), 'p'] = np.nan
    complete = f.notna().all(axis=1).groupby(f.index.normalize()).sum().eq(24)
    f.loc[~f.index.normalize().isin(complete.index[complete]), 'event'] = np.nan
    return f, {'margins': str(mf), 'margins_sha256': digest(mf),
               'probabilities': str(pf), 'probabilities_sha256': digest(pf),
               'forecast_column': column}


def policy(f, gate, cooldown=0, start_priority=False):
    """All candidates for D known at issue D-1 12UTC; carry prior planned alerts.

    Priority is a forecast score p_t*(1-max past-six issued hourly risks), not a
    probability of a temporal joint event. The past-day risks are forecasts
    already issued for that day. Missing predictions reset the past-risk score,
    and missing days do not induce an alert using future truth.
    """
    p = f.p.to_numpy(float)
    idx = f.index
    last6 = pd.Series(p, index=idx).shift(1).rolling(6, min_periods=1).max().fillna(0).to_numpy()
    priority = p*(1-np.clip(last6, 0, 1)) if start_priority else p
    alert = np.zeros(len(f), bool)
    selected = []
    cooldown_ns = int(cooldown*3600*1e9)
    tns = idx.as_unit('ns').asi8
    for _, ix in pd.Series(np.arange(len(f)), index=idx).groupby(idx.normalize()):
        ii = ix.to_numpy()
        if len(ii)!=24 or not np.isfinite(p[ii]).all():
            continue
        candidates = ii[p[ii] >= gate]
        order = np.lexsort((tns[candidates], -p[candidates], -priority[candidates]))
        picked = []
        recent = [t for t in selected if t >= tns[ii[0]]-cooldown_ns]
        for at in candidates[order]:
            if cooldown and any(abs(tns[at]-t)<cooldown_ns for t in recent + picked):
                continue
            alert[at] = True
            picked.append(tns[at])
            if len(picked)==2:
                break
        selected.extend(picked)
    return alert


def gate_from_count(f, target, cooldown=0, start_priority=False):
    # Fixed candidate grid chosen before reading reanalysis labels.
    candidates = np.r_[0., np.linspace(.01,.95,95), .99, 1.]
    counts = np.array([policy(f, g, cooldown, start_priority).sum() for g in candidates])
    okay = counts<=target
    best = np.flatnonzero(okay)[np.argmax(counts[okay])]
    return float(candidates[best]), int(counts[best])


def process_masks(f, gap=1):
    masks = []
    for e in event_processes(f, max_empty_days=gap):
        hit_ix = f.index.get_indexer(e['event_hours'])
        onset = e['onset']
        early = hit_ix[e['event_hours'] < onset+pd.Timedelta(hours=6)]
        masks.append({'onset': onset, 'end': e['end'], 'ix': hit_ix, 'early_ix': early,
                      'uncertain_boundary': e['uncertain_boundary']})
    return masks


def retained_hit(N, B, m):
    """P(at least one of m relevant slots retained), uniform B of N without replacement."""
    if m==0 or B==0: return 0.
    if N-m<B: return 1.
    if B==N: return 1.
    log_miss = (gammaln(N-m+1)-gammaln(N-m-B+1)
                -gammaln(N+1)+gammaln(N-B+1))
    return float(-np.expm1(min(0.,log_miss)))


def summarise(f, alerts, processes, budget=None):
    y = f.event.fillna(0).to_numpy().astype(bool)
    known = f.event.notna().to_numpy()
    alerts = alerts & known
    N = int(alerts.sum())
    B = N if budget is None else int(budget)
    if not 0<=B<=N: raise ValueError('Invalid matched workload')
    ratio = B/N if N else 0.
    tp = int((alerts&y&known).sum()); fp = int((alerts&~y&known).sum())
    records=[]
    for e in processes:
        m = int(alerts[e['early_ix']].sum())
        ma = int(alerts[e['ix']].sum())
        records.append({'onset':e['onset'], 'end':e['end'],
                        'uncertain_boundary':e['uncertain_boundary'],
                        'first_six_hours_hit_probability':retained_hit(N,B,m),
                        'any_hit_probability':retained_hit(N,B,ma),
                        'early_selected_event_hours_before_thinning':m})
    rec = pd.DataFrame(records, columns=[
        'onset', 'end', 'uncertain_boundary',
        'first_six_hours_hit_probability', 'any_hit_probability',
        'early_selected_event_hours_before_thinning'])
    days = f.event.notna().groupby(f.index.normalize()).sum().eq(24)
    complete_days = days.index[days]
    ac = pd.Series(alerts,index=f.index).groupby(f.index.normalize()).sum().reindex(complete_days)
    return {'unstandardised_review_hours':N, 'matched_review_hours':B,
            'expected_detected_event_hours':tp*ratio,
            'expected_selected_nonevent_hours':fp*ratio,
            'false_alert_fraction':fp/N if N else None,
            'zero_alert_days_before_thinning':int(ac.eq(0).sum()),
            'complete_days':len(complete_days), 'event_hours':int(y[known].sum()),
            'processes':len(processes),
            'expected_first_six_hours_covered_processes':float(rec.first_six_hours_hit_probability.sum()),
            'first_six_hours_coverage':float(rec.first_six_hours_hit_probability.mean()),
            'expected_any_covered_processes':float(rec.any_hit_probability.sum()),
            'any_coverage':float(rec.any_hit_probability.mean())}, rec


def select():
    OUT.mkdir(parents=True,exist_ok=True)
    if CFG.exists(): raise FileExistsError('Preserve previously frozen upgrade policy configuration')
    f, sources = inputs()
    tr = f.loc['2022']; va = f.loc['2023']
    ndays = int(tr.p.notna().groupby(tr.index.normalize()).sum().eq(24).sum())
    target = int(np.floor(.5*ndays))
    p1gate, _ = gate_from_count(tr, target)
    candidates=[]
    processes = process_masks(va)
    for cooldown in (6,12,24):
        gate,count = gate_from_count(tr,target,cooldown,True)
        a2 = policy(va,gate,cooldown,True)
        a1 = policy(va,p1gate)
        known=va.event.notna().to_numpy()
        B = int(min((a1&known).sum(),(a2&known).sum()))
        s2,_=summarise(va,a2,processes,B); s1,_=summarise(va,a1,processes,B)
        candidates.append({'cooldown_hours':cooldown,'gate':gate,
                           'training_alert_hours':count,'validation_alert_hours':int(a2.sum()),
                           'validation_common_review_hours':B,
                           'validation_first6_difference':s2['first_six_hours_coverage']-s1['first_six_hours_coverage']})
    best = max(candidates,key=lambda d:(d['validation_first6_difference'],-d['cooldown_hours']))
    cfg = {'status':'frozen', 'created_utc':pd.Timestamp.now(tz='UTC').isoformat(),
           'analysis_status':'new design applied to previously inspected 2024-2025; reanalysis',
           'policy_source':'C3 dynamic Gaussian q90; locked for primary analysis',
           'target_training_review_hours_per_complete_day':.5,
           'gate_candidates':[0.]+np.linspace(.01,.95,95).tolist()+[.99,1.],
           'gate_training_year':2022,'cooldown_selection_year':2023,
           'forecast_gate':p1gate, 'priority_gate':best['gate'],
           'cooldown_hours':best['cooldown_hours'],'max_alerts_per_day':2,
           'priority_score':'p_t*(1-max previous six hourly issued forecast risks); deterministic priority, not onset probability',
           'candidates':candidates,'inputs':sources,
           'primary_decision_endpoint':'first6 process coverage, P2 minus P1, at exact common total review hours',
           'primary_block_days':14,'secondary_block_days':[7,28],
           'random_thinning':'uniform B of N selected slots; B=min(N_P1,N_P2); exact hypergeometric expectations, no label-dependent choice',
           'thinning_interpretation':'exposure standardisation for evaluation, not a prospective exact annual-quota policy',
           'truth_not_used_for_alert_allocation':True,'not_preregistered':True}
    CFG.write_text(json.dumps(cfg,indent=2),encoding='utf-8')
    print(json.dumps(cfg,indent=2))


def onset_block_ci(f, recs, candidate, reference, block_days=14, reps=2000):
    days = pd.date_range(f.index.min().normalize(), f.index.max().normalize(),freq='D',tz='UTC')
    daily = pd.DataFrame(0.,index=days,columns=['n','delta'])
    a,b=recs[candidate],recs[reference]
    assert a.onset.equals(b.onset)
    for i,row in a.iterrows():
        day=row.onset.normalize()
        daily.loc[day,'n']+=1
        daily.loc[day,'delta']+=(row.first_six_hours_hit_probability-b.loc[i,'first_six_hours_hit_probability'])
    labels=days.year*10+days.quarter
    strata=[np.flatnonzero(labels==x) for x in np.unique(labels)]
    rng=np.random.default_rng(SEED+block_days)
    daily_values=daily[['n','delta']].to_numpy(float)
    vals=np.zeros(reps)
    for r in range(reps):
        n=delta=0.
        for ix in strata:
            length=min(len(ix),block_days)
            starts=rng.integers(0,len(ix)-length+1,size=int(np.ceil(len(ix)/length)))
            picked=ix[(starts[:,None]+np.arange(length)).ravel()[:len(ix)]]
            v=daily_values[picked].sum(axis=0)
            n+=v[0];delta+=v[1]
        vals[r]=delta/n if n else np.nan
    finite=vals[np.isfinite(vals)]
    return {'candidate':candidate,'reference':reference,'block_days':block_days,
            'difference':float(daily.delta.sum()/daily.n.sum()),
            'ci95':np.quantile(finite,[.025,.975]).tolist(),'replicates':reps,
            'assumption':'fixed policy, fixed common workload and intact realised processes, resampled by onset day within year/quarter; no episode reconnection'}


def run():
    OUT.mkdir(parents=True,exist_ok=True)
    cfg=json.loads(CFG.read_text(encoding='utf-8'))
    started=time.perf_counter()
    full={};table=[];cis=[]
    for level in ('q90','q95'):
        context,src=inputs(level)
        # Carry already-issued forecasts and planned targets across the analysis
        # boundary. An evaluation start must not create a new risk-rise signal.
        mask=context.index.year>=2024
        a0=policy(context,0.)[mask]
        a1=policy(context,cfg['forecast_gate'])[mask]
        a2=policy(context,cfg['priority_gate'],cfg['cooldown_hours'],True)[mask]
        f=context.loc[mask]
        alerts={'P0_forced_top2':a0,'P1_risk_gate':a1,'P2_risk_rise_cooldown':a2}
        known=f.event.notna().to_numpy()
        B=int(min((a1&known).sum(),(a2&known).sum()))
        for gap in (0,1,2):
            ps=process_masks(f,gap)
            recs={}
            for name,a in alerts.items():
                native,rawrec=summarise(f,a,ps)
                matched,rec=summarise(f,a,ps,B)
                recs[name]=rec
                table.append({'level':level,'gap_days':gap,'policy':name,'mode':'native',**native})
                table.append({'level':level,'gap_days':gap,'policy':name,'mode':'matched',**matched})
                rec.to_csv(OUT/f'processes_{level}_{gap}_{name}_matched.csv',index=False)
            if gap==1:
                for block in (7,14,28):
                    for ref in ('P0_forced_top2','P1_risk_gate'):
                        cis.append({'level':level,**onset_block_ci(f,recs,'P2_risk_rise_cooldown',ref,block)})
        ap=pd.DataFrame(alerts,index=f.index)
        ap['p']=f.p;ap['event']=f.event
        ap['evaluation_complete_day']=known
        ap.to_parquet(OUT/f'alerts_{level}.parquet')
        full[level]={'sources':src,'native_alert_hours_common_complete_day_domain':{n:int((a&known).sum()) for n,a in alerts.items()},
                     'planned_hours_all_forecast_complete_days':{n:int(a.sum()) for n,a in alerts.items()},'common_review_hours':B}
    pd.DataFrame(table).to_csv(OUT/'policy_results.csv',index=False)
    result={'status':'completed','scope':'2024-2025 reanalysis; not new sealed testing',
            'configuration':str(CFG),'configuration_sha256':digest(CFG),
            'source_sha256':digest(__file__),'elapsed_seconds':time.perf_counter()-started,
            'levels':full,'paired_primary_and_secondary_intervals':cis,
            'original_P0_review_hours':1450,
            'causality_repair':'allocation preserves forecast-only plans through outcome-missing days; scoring restricts domain after allocation',
            'initial_state_repair':'allocate on the entire 2022-2025 issued forecast history, then score 2024-2025; carry prior issued risk and cooldown plans across the analysis boundary',
            'parameters_retuned_after_reanalysis':False,
            'exact_exposure_matching_has_no_realised_label_selection':True,
            'new_rules_can_return_zero_alerts':True}
    (OUT/'manifest.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


def causal_selftest():
    idx=pd.date_range('2023-01-01',periods=5*24,freq='h',tz='UTC')
    p=np.where(idx.hour<4,.6,.05)
    f=pd.DataFrame({'p':p,'event':0.},index=idx)
    a=policy(f,.2,12,True)
    f['event']=np.where(idx.hour>=20,1.,0.)
    assert np.array_equal(a,policy(f,.2,12,True))
    f.loc[idx[24:48],'event']=np.nan
    assert np.array_equal(a,policy(f,.2,12,True))
    g=f.copy();g.loc[idx[-24:],'p']=.99
    assert np.array_equal(a[:-24],policy(g,.2,12,True)[:-24])
    assert pd.Series(a,index=idx).groupby(idx.normalize()).sum().max()<=2
    assert not policy(f,.9).any()
    assert abs(retained_hit(10,5,1)-.5)<1e-12
    assert abs(retained_hit(10,5,2)-(1-20/90))<1e-12
    # Boundary context matters: a persistently high forecast must not become a
    # fictitious fresh rise merely because scoring starts on the next day.
    steady=pd.DataFrame({'p':.8,'event':0.},index=idx)
    full=policy(steady,.2,6,True)
    repeat=policy(steady.copy(),.2,6,True)
    assert np.array_equal(full[24:],repeat[24:])
    return {'status':'passed','evidence_type':'SYNTHETIC only',
            'checks':['truth-invariant allocation','future forecast does not alter earlier daily plans','max-two daily cap','zero alert allowed','exact hypergeometric detection']}


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('mode',choices=['select','run','selftest'])
    mode=parser.parse_args().mode
    if mode=='select':select()
    elif mode=='run':run()
    else:
        result=causal_selftest();OUT.mkdir(parents=True,exist_ok=True)
        (OUT/'selftest.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
        print(json.dumps(result))
