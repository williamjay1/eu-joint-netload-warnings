"""ASMBI fixed-stream policy reanalysis, 2026-10-08.

Planning is forecast-only; future outcome availability defines scoring afterward.
No test labels tune gates. All new outputs are explicitly post hoc reanalysis.
"""
from pathlib import Path
import argparse
import hashlib
import json
import time
import numpy as np
import pandas as pd
from shared_eval import ROOT, load_stream, select, process_masks, summary

OUT = ROOT / 'results/policy_analysis'
GRID = np.r_[0., np.linspace(.01,.95,95), .99, 1.]
COSTS = np.unique(np.r_[0., np.linspace(0,.01,41), np.linspace(.012,.08,35), .1,.2,.5,1., 3/832])
HOUR_COSTS = np.unique(np.r_[0., np.linspace(0,1,51)])
SEED = 20261008


def clean(value):
    if isinstance(value, dict): return {str(k):clean(v) for k,v in value.items()}
    if isinstance(value, list): return [clean(v) for v in value]
    if isinstance(value, (np.integer,)): return int(value)
    if isinstance(value, (np.floating,)): return float(value) if np.isfinite(value) else None
    if isinstance(value, pd.Timestamp): return value.isoformat()
    return value


def threshold_from_counts(p, rho=.5):
    """Use forecast-complete 24h days only; count after the daily cap."""
    rows=[]
    for _, g in p.groupby(p.index.normalize()):
        v=g.to_numpy(float)
        if len(v)==24 and np.isfinite(v).all(): rows.append(np.sort(v)[-2:])
    values=np.asarray(rows).reshape(-1)
    n=len(rows); target=int(np.floor(rho*n))
    counts=(values[:,None]>=GRID).sum(axis=0) if n else np.zeros(len(GRID),int)
    okay=counts<=target
    candidates=np.flatnonzero(okay)
    chosen=candidates[np.argmax(counts[candidates])]
    return float(GRID[chosen]),dict(days=n,target=target,slots=int(counts[chosen]),rho=rho)


def rolling_gates(f, window=90, rho=.5, min_days=30, fallback=.25):
    dates=f.index.normalize(); gates=np.full(len(f),fallback,float)
    prior=[]; logs=[]
    for day, inds in pd.Series(np.arange(len(f)),index=f.index).groupby(dates):
        ii=inds.to_numpy(); p=f.p.iloc[ii].to_numpy(float)
        complete=len(ii)==24 and np.isfinite(p).all()
        gate=fallback; n=min(window,len(prior)); target=int(np.floor(rho*n))
        if n>=min_days:
            v=np.asarray([z[1] for z in prior[-window:]]).reshape(-1)
            counts=(v[:,None]>=GRID).sum(axis=0)
            okay=np.flatnonzero(counts<=target)
            gate=float(GRID[okay[np.argmax(counts[okay])]])
        gates[ii]=gate
        logs.append(dict(target_day=day,issue_time=day-pd.Timedelta(hours=12),gate=gate,
                         past_complete_days=n,history_first=prior[-n][0] if n else None,
                         history_last=prior[-1][0] if n else None,historical_target=target,
                         forecast_complete=complete,used_outcomes=False))
        # This vector was issued before the next daily planning time.
        if complete: prior.append((day,np.sort(p)[-2:]))
    return gates,pd.DataFrame(logs)


def make_plans(f, tuned_gate):
    rolling,log=rolling_gates(f)
    plans={'none':np.zeros(len(f),bool),'top1':select(f,0,k=1),'top2':select(f,0),
           'gate025':select(f,.25),'inherited_priority019':select(f,.19,6,True),
           'gate025_cool6':select(f,.25,6,False),'gate025_rise':select(f,.25,0,True),
           'gate025_rise_cool6':select(f,.25,6,True),
           'rolling90_target05':select(f,rolling),'development_count_gate':select(f,tuned_gate)}
    # Paired secondary ablation at the inherited priority threshold.
    plans.update({'gate019':select(f,.19),'gate019_cool6':select(f,.19,6),
                  'gate019_rise':select(f,.19,0,True)})
    return plans,log


def daily_components(f,plans,proc):
    days=pd.date_range(f.index.min().normalize(),f.index.max().normalize(),freq='D',tz='UTC')
    known=f.event.notna().to_numpy(); y=f.event.fillna(0).to_numpy(bool)
    result=np.zeros((len(days),len(plans),3))
    labels=f.index.normalize(); dayid=days.get_indexer(labels)
    for j,a in enumerate(plans.values()):
        scored=np.asarray(a,bool)&known
        result[:,j,0]=np.bincount(dayid,weights=scored,minlength=len(days))
        result[:,j,1]=np.bincount(dayid,weights=scored&y,minlength=len(days))
        for e in proc:
            result[days.get_loc(e['onset'].normalize()),j,2]+=int(scored[e['early_ix']].any())
    return days,result


def block_weights(days,reps=2000,block=14):
    rng=np.random.default_rng(SEED+block)
    groups=days.year*10+days.quarter
    weights=np.zeros((reps,len(days)),float)
    for g in np.unique(groups):
        ix=np.flatnonzero(groups==g); n=len(ix); length=min(block,n)
        starts=rng.integers(0,n-length+1,size=(reps,int(np.ceil(n/length))))
        chosen=ix[(starts[:,:,None]+np.arange(length)).reshape(reps,-1)[:,:n]]
        np.add.at(weights,(np.repeat(np.arange(reps),n),chosen.ravel()),1.)
    return weights


def cost_bootstrap(f,plans,proc,reps,zone,model,level):
    days,daily=daily_components(f,plans,proc)
    w=block_weights(days,reps); sims=np.einsum('rd,dpk->rpk',w,daily,optimize=True)
    totals=daily.sum(axis=0); names=list(plans); output=[]; contrasts=[]
    for endpoint,j,grid in [('process',2,COSTS),('hour',1,HOUR_COSTS)]:
        for p,name in enumerate(names):
            values=totals[p,j]-grid*totals[p,0]
            sample=sims[:,p,j,None]-sims[:,p,0,None]*grid
            lo,hi=np.quantile(sample,[.025,.975],axis=0)
            # Simultaneous band over the declared fixed cost grid, per policy.
            maximum=np.max(abs(sample-values),axis=1)
            band=np.quantile(maximum,.95)
            for q,r in enumerate(grid):
                output.append(dict(zone=zone,model=model,level=level,endpoint=endpoint,policy=name,
                                   cost_ratio=r,utility=values[q],lower_pointwise=lo[q],upper_pointwise=hi[q],
                                   lower_gridband=values[q]-band,upper_gridband=values[q]+band,
                                   slots=int(totals[p,0]),reward=int(totals[p,j]),replicates=reps))
        # Predefined comparisons, not winner selection from a cost curve.
        for candidate,reference in [('gate025','top2'),('rolling90_target05','gate025'),
                                    ('gate025_rise_cool6','gate025'),('inherited_priority019','gate025'),
                                    ('top1','gate025')]:
            ca,ref=names.index(candidate),names.index(reference)
            point=(totals[ca,j]-totals[ref,j])-grid*(totals[ca,0]-totals[ref,0])
            samples=(sims[:,ca,j]-sims[:,ref,j])[:,None]-grid*(sims[:,ca,0]-sims[:,ref,0])[:,None]
            lo,hi=np.quantile(samples,[.025,.975],axis=0)
            width=np.quantile(np.max(abs(samples-point),axis=1),.95)
            for q,r in enumerate(grid):
                contrasts.append(dict(zone=zone,model=model,level=level,endpoint=endpoint,candidate=candidate,
                                      reference=reference,cost_ratio=r,difference=point[q],lower=lo[q],upper=hi[q],
                                      lower_gridband=point[q]-width,upper_gridband=point[q]+width))
    # Native matched-pair endpoint contrasts use the same draws.
    native=[]
    for ca,ref in [('gate025','top2'),('rolling90_target05','gate025'),('gate025_cool6','gate025'),
                   ('gate025_rise','gate025'),('gate025_rise_cool6','gate025'),('inherited_priority019','gate025')]:
        i,j=names.index(ca),names.index(ref)
        for ix,metric in [(0,'slots'),(1,'event_hour_hits'),(2,'early_processes')]:
            delta=sims[:,i,ix]-sims[:,j,ix]; lo,hi=np.quantile(delta,[.025,.975])
            native.append(dict(zone=zone,model=model,level=level,candidate=ca,reference=ref,metric=metric,
                               difference=totals[i,ix]-totals[j,ix],lower=lo,upper=hi,replicates=reps))
    return output,contrasts,native


def categories(f,a,proc):
    a=np.asarray(a,bool)&f.event.notna().to_numpy(); y=f.event.fillna(0).to_numpy(bool)
    label=np.full(len(f),'not_selected',object); label[a&~y]='non_event'; label[a&y]='late_event'
    pid=np.full(len(f),-1,int)
    for i,e in enumerate(proc):
        pid[e['ix']]=i; selected=e['early_ix'][a[e['early_ix']]]
        if len(selected):
            label[selected[0]]='first_early'
            label[selected[1:]]='repeat_early'
    assert np.sum(label!='not_selected')==a.sum()
    return label,pid


def period_statistics(f,plans,proc,zone,model,level):
    rows=[]
    periods=[('all',np.ones(len(f),bool))]
    for year in np.unique(f.index.year): periods.append((str(year),f.index.year==year))
    for key in np.unique(f.index.year.astype(str)+'Q'+f.index.quarter.astype(str)):
        periods.append((str(key),(f.index.year.astype(str)+'Q'+f.index.quarter.astype(str))==key))
    for period,mask in periods:
        known=f.event.notna().to_numpy()&mask
        ds=f.loc[mask].event.notna().groupby(f.index[mask].normalize()).sum().eq(24)
        valid_days=ds.index[ds]; n=len(valid_days)
        for name,a in plans.items():
            scored=np.asarray(a,bool)&known
            counts=pd.Series(scored,index=f.index).groupby(f.index.normalize()).sum().reindex(valid_days)
            psel=[e for e in proc if mask[f.index.get_loc(e['onset'])]]
            C=sum(bool(a[e['early_ix']].any()) for e in psel)
            rows.append(dict(zone=zone,model=model,level=level,period=period,policy=name,
                             N=int(scored.sum()),hits=int((scored&f.event.fillna(0).to_numpy(bool)).sum()),
                             C=int(C),processes=len(psel),coverage=C/len(psel) if psel else np.nan,
                             complete_days=n,slots_per_day=scored.sum()/n if n else np.nan,
                             zero_review_days=int(counts.eq(0).sum()),zero_day_fraction=counts.eq(0).mean(),
                             daily_cap_days=int(counts.eq(2).sum()),daily_cap_fraction=counts.eq(2).mean(),
                             target_slots_per_day=.5))
    return rows


def danish_subset(f,plans,zone,level,proc):
    x=pd.read_parquet(ROOT/'datasets'/f'{zone}_marginal.parquet').reindex(f.index)
    # New input may expose pivotal labels separately; do not infer from probabilities.
    col=f'y_{level}_DK_pivotal'
    if col not in x: return [],dict(status='awaiting_pivotal_labels',zone=zone,level=level)
    piv=x[col].eq(1).to_numpy(); rows=[]
    opp=[e for e in proc if piv[e['early_ix']].any()]
    for name,a in plans.items():
        scored=a&f.event.notna().to_numpy()
        c=sum(bool((scored[e['early_ix']]&piv[e['early_ix']]).any()) for e in opp)
        rows.append(dict(zone=zone,level=level,policy=name,opportunities=len(opp),covered=c,
                         coverage=c/len(opp) if opp else np.nan,review_slots=int(scored.sum()),
                         pivotal_event_hours=int((piv&f.event.notna().to_numpy()).sum())))
    return rows,dict(status='computed',zone=zone,level=level)


def refresh_danish(reps=2000):
    """Nested original-onset endpoint and separately reclustered pivotal events.

    Costs include every slot in the original forecast-only watchlist, not only
    slots retrospectively identified as Danish-dependent.
    """
    rows=[]; curves=[]; contrasts=[]; common={}; statuses=[]
    for zone in ('DK1','DK2'):
        marginal=pd.read_parquet(ROOT/'datasets'/f'{zone}_marginal.parquet')
        for level in ('q90','q95'):
            for model in ('M0','M1','M2','M3'):
                full=load_stream(zone,model,level); mask=full.index.year>=2024; f=full.loc[mask]
                saved=pd.read_parquet(OUT/f'plans_{zone}_{model}_{level}.parquet').reindex(f.index)
                plans={c:saved[c].to_numpy(bool) for c in saved.columns if c not in ['issue_time','probability','event']}
                known=f.event.notna(); piv=marginal[f'y_{level}_DK_pivotal'].reindex(f.index)
                fs=f.copy(); fs['event']=piv.where(known)
                parent=process_masks(f); subset=process_masks(fs)
                nested=[]
                for e in parent:
                    early=e['early_ix'][fs.event.iloc[e['early_ix']].eq(1).to_numpy()]
                    if len(early):
                        ix=e['ix'][fs.event.iloc[e['ix']].eq(1).to_numpy()]
                        nested.append(dict(**e,))
                        nested[-1]['early_ix']=early; nested[-1]['ix']=ix
                shared=(marginal[f'exceed_{level}_DE_LU'].reindex(f.index).eq(1)&
                        marginal[f'exceed_{level}_FR'].reindex(f.index).eq(1)&known)
                if model=='M0': common[zone+'_'+level]=set(f.index[shared])
                for endpoint,procs in [('pivotal_parent_onset',nested),('pivotal_reclustered',subset)]:
                    for name,a in plans.items():
                        s=summary(fs,a,procs)
                        rows.append(dict(zone=zone,model=model,level=level,endpoint=endpoint,policy=name,**s,
                                         shared_DE_FR_event_hours=int(shared.sum()),parent_processes=len(parent)))
                    co,ct,_=cost_bootstrap(fs,plans,procs,reps,zone,model,level)
                    for rec in co:
                        rec['scope']=endpoint; curves.append(rec)
                    for rec in ct:
                        rec['scope']=endpoint; contrasts.append(rec)
    pd.DataFrame(rows).to_csv(OUT/'danish_specific_early.csv',index=False)
    pd.DataFrame(curves).to_csv(OUT/'danish_specific_cost_curves.csv',index=False)
    pd.DataFrame(contrasts).to_csv(OUT/'danish_specific_cost_contrasts.csv',index=False)
    for level in ('q90','q95'):
        a,b=common['DK1_'+level],common['DK2_'+level]
        statuses.append(dict(level=level,DK1_shared_hours=len(a),DK2_shared_hours=len(b),
                             shared_identical=a==b,common_hours=len(a&b)))
    (OUT/'danish_subset_manifest.json').write_text(json.dumps(dict(status='completed',overlap=statuses,
         endpoints='parentonset nested primary; separatelyreclustered pivotal onset secondary',
         cost_scope='allslots originalwatchlist; no retrospectiveregionalfilter',replicates=reps),indent=2),encoding='utf-8')
    old=json.loads((OUT/'manifest.json').read_text(encoding='utf-8'))
    old['danish_subset_status']='completed; see danish_subset_manifest.json and two explicit endpoint scopes'
    (OUT/'manifest.json').write_text(json.dumps(old,indent=2),encoding='utf-8')
    print(json.dumps(statuses,indent=2),flush=True)


def paired_simulations():
    """Paired Monte Carlo precision, distinct from observed-data confidence."""
    out=ROOT/'results/simulations'; df=pd.read_parquet(out/'replicates.parquet')
    metrics=[c for c in ['BS','probability_MAE','N','C','processes','utility_r005','hits','false',
                         'changed_slots','marginal_MAE','gate_crossings'] if c in df]
    records=[]
    for (persist,drift,policy),g in df.groupby(['persistence','drift','policy']):
        for ca,ref in [('margin_near','margin_far'),('wrong_dependence','oracle'),
                       ('margin_near','oracle'),('margin_far','oracle')]:
            a=g[g.model==ca].set_index('rep'); b=g[g.model==ref].set_index('rep')
            assert a.index.equals(b.index)
            for metric in metrics:
                diff=a[metric]-b[metric]; mean=diff.mean(); se=diff.std(ddof=1)/np.sqrt(len(diff))
                records.append(dict(contrast_type='prediction_stream',persistence=persist,drift=drift,
                    model='paired',policy=policy,candidate=ca,reference=ref,metric=metric,
                    difference_mean=mean,paired_MCSE=se,MC_interval_lower=mean-1.96*se,
                    MC_interval_upper=mean+1.96*se,replications=len(diff)))
    for (persist,drift,model),g in df.groupby(['persistence','drift','model']):
        for ca,ref in [('Gate','Top2'),('RiseCooldown','Gate')]:
            a=g[g.policy==ca].set_index('rep'); b=g[g.policy==ref].set_index('rep')
            assert a.index.equals(b.index)
            for metric in metrics:
                diff=a[metric]-b[metric]; mean=diff.mean(); se=diff.std(ddof=1)/np.sqrt(len(diff))
                records.append(dict(contrast_type='policy',persistence=persist,drift=drift,model=model,
                    policy='paired',candidate=ca,reference=ref,metric=metric,difference_mean=mean,
                    paired_MCSE=se,MC_interval_lower=mean-1.96*se,MC_interval_upper=mean+1.96*se,
                    replications=len(diff)))
    pd.DataFrame(records).to_csv(out/'paired_contrasts.csv',index=False)
    print(f'Paired simulation contrasts: {len(records)} rows',flush=True)


def run(reps=2000):
    started=time.perf_counter(); OUT.mkdir(parents=True,exist_ok=True)
    allstats=[]; devstats=[]; gates=[]; costs=[]; contrasts=[]; native=[]; category_rows=[]
    subsets=[]; subset_status=[]; sources={}; loss=[]; original=None
    for zone in ('DE_LU_FR_BE','DK1','DK2'):
        sources[zone]=hashlib.sha256((ROOT/'datasets'/f'{zone}.parquet').read_bytes()).hexdigest()
        for level in ('q90','q95'):
            for model in ('M0','M1','M2','M3'):
                full=load_stream(zone,model,level)
                gate,meta=threshold_from_counts(full.loc['2022'].p)
                gates.append(dict(zone=zone,level=level,model=model,gate=gate,**meta))
                plans,rolling=make_plans(full,gate)
                rolling['zone']=zone; rolling['model']=model; rolling['level']=level
                rolling.to_csv(OUT/f'rolling_gate_{zone}_{model}_{level}.csv',index=False)
                # Development validation is chronological and separate from 2024-25.
                dm=full.index.year==2023; dev=full.loc[dm]; dp={n:a[dm] for n,a in plans.items()}
                devstats.extend(period_statistics(dev,dp,process_masks(dev),zone,model,level))
                mask=full.index.year>=2024; f=full.loc[mask]; p={n:a[mask] for n,a in plans.items()}
                proc=process_masks(f); allstats.extend(period_statistics(f,p,proc,zone,model,level))
                ledger=pd.DataFrame({'issue_time':f.index.normalize()-pd.Timedelta(hours=12),
                                     'probability':f.p,'event':f.event},index=f.index)
                for name,a in p.items(): ledger[name]=a
                ledger.to_parquet(OUT/f'plans_{zone}_{model}_{level}.parquet')
                co,ct,na=cost_bootstrap(f,p,proc,reps,zone,model,level)
                costs.extend(co); contrasts.extend(ct); native.extend(na)
                if model=='M0':
                    for name,a in p.items():
                        labels,pid=categories(f,a,proc)
                        counts=pd.Series(labels).value_counts()
                        for category in ('first_early','repeat_early','late_event','non_event'):
                            category_rows.append(dict(zone=zone,level=level,policy=name,category=category,
                                                      slots=int(counts.get(category,0))))
                        ledger[name+'_category']=labels
                    ledger['process_id']=pid
                    ledger.to_parquet(OUT/f'selection_mechanism_{zone}_{level}.parquet')
                    if zone=='DE_LU_FR_BE' and level=='q90':
                        known=f.event.notna().to_numpy(); removed=p['top2']&~p['gate025']&known
                        assert not np.any(p['gate025']&~p['top2'])
                        original=dict(removed_slots=int(removed.sum()),removed_event_hours=int((removed&f.event.eq(1).to_numpy()).sum()),
                                      removed_nonevent_hours=int((removed&f.event.eq(0).to_numpy()).sum()),
                                      top2=summary(f,p['top2'],proc),gate=summary(f,p['gate025'],proc),
                                      priority=summary(f,p['inherited_priority019'],proc))
                        marginal=pd.read_parquet(ROOT/'datasets/marginal_predictions.parquet')
                        weather=pd.read_parquet(ROOT/'datasets/weather_features.parquet')
                        for i,e in enumerate(proc):
                            if p['top2'][e['early_ix']].any() and not p['gate025'][e['early_ix']].any():
                                z=dict(process_id=i,onset=e['onset'],end=e['end'],event_hours=len(e['ix']),
                                       elapsed_duration_hours=(e['end']-e['onset']).total_seconds()/3600+1,
                                       early_event_hours=len(e['early_ix']),max_early_p=float(f.p.iloc[e['early_ix']].max()),
                                       top2_selected_early_hours=int(p['top2'][e['early_ix']].sum()),
                                       gate_selected_early_hours=0,uncertain_boundary=e['uncertain_boundary'])
                                for c in marginal:
                                    if c in ['stratum','cell'] and e['onset'] in marginal.index: z[c]=str(marginal.loc[e['onset'],c])
                                for region in ('DE_LU','FR','BE'):
                                    ix=f.index[e['ix']]
                                    exceed=(marginal['net_load_'+region]-marginal['threshold_q90_'+region]).reindex(ix)
                                    z[region+'_max_excess_MW']=float(exceed.max())
                                    z[region+'_onset_excess_MW']=float(exceed.iloc[0])
                                for c in weather:
                                    if e['onset'] in weather.index and pd.api.types.is_numeric_dtype(weather[c]):
                                        z['weather_'+c]=float(weather.loc[e['onset'],c])
                                loss.append(z)
                        ledger.loc[removed].to_csv(OUT/'removed_832_slots.csv')
                    if zone!='DE_LU_FR_BE':
                        sub,st=danish_subset(f,p,zone,level,proc); subsets.extend(sub); subset_status.append(st)
                print(f'{zone} {level} {model}: gate={gate:.2f} elapsed={time.perf_counter()-started:.1f}s',flush=True)
    outputs={'policy_summary.csv':allstats,'development_2023_summary.csv':devstats,'development_gates.csv':gates,
             'cost_curves.csv':costs,'cost_contrasts.csv':contrasts,'native_contrasts.csv':native,
             'selection_categories.csv':category_rows,'lost_three_early_processes.csv':loss,
             'danish_specific_early.csv':subsets}
    for name,rows in outputs.items(): pd.DataFrame(rows).to_csv(OUT/name,index=False)
    manifest=dict(status='completed',analysis_status='post hoc reanalysis of previously inspected 2024-2025',
                  protocol_date='2026-10-08',sources=sources,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  gates_selected_on='2022 forecast-only capped counts; 2023 development validation',
                  rolling='preceding90forecastcomplete targetdays;min30warmup;pastissued probabilities only;gridcountcalibration;rho0.5',
                  cooldown_state='planned before labels; full2022-25 context carried to score2024',
                  primary_endpoint='native firstsixhour coverage of active-day spells with oneobservedinactiveday bridge',
                  cost_inference='synchronous14calendar-day year-quarter bootstrap; processrewards ononsetday costs ontargetday; fixedstreams rules andboundaries',
                  curve_bands='pointwisepercentile and perpolicy/contrast maxdeviationbands overdeclaredcostgrid; no acrosspolicy familywiseclaim',
                  original_reproduction=original,danish_subset_status=subset_status,replicates=reps,
                  elapsed_seconds=time.perf_counter()-started)
    (OUT/'manifest.json').write_text(json.dumps(clean(manifest),indent=2),encoding='utf-8')
    print(json.dumps(clean(original),indent=2),flush=True)
    if all((ROOT/'datasets'/f'{z}_marginal.parquet').exists() for z in ('DK1','DK2')):
        refresh_danish(reps)


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--reps',type=int,default=2000)
    parser.add_argument('--refresh-danish',action='store_true')
    parser.add_argument('--paired-sim',action='store_true')
    args=parser.parse_args()
    if args.paired_sim: paired_simulations()
    elif args.refresh_danish: refresh_danish(args.reps)
    else: run(args.reps)
