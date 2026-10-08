"""Post hoc target-aligned and price-control checks for the ASMBI revision.

No realized lags enter any feature. Regularization is selected on 2023 only.
All refits use interval-end maturation: 7 days in the core, 15 in Danish checks.
Prices are latest-history snapshots, subject to assumed first-release availability;
UTC tail hours belonging to the subsequent local auction day are never exposed.
"""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from shared_eval import ROOT, load_stream, process_masks, select, summary
from policy_analysis import threshold_from_counts, block_weights, daily_components

OUT=ROOT/'results/early_price'
CS=[.1,1.,10.]
COSTS=np.unique(np.r_[0.,np.linspace(.0005,.015,30),.02,.03,.05,.1,.2,.25,.5,1.])


def early_labels(f):
    y=pd.Series(0.,index=f.index).where(f.event.notna())
    pid=pd.Series(-1,index=f.index,dtype=int)
    boundary=pd.Series(False,index=f.index)
    for i,e in enumerate(process_masks(f)):
        y.iloc[e['early_ix']]=1.
        pid.iloc[e['ix']]=i
        boundary.iloc[e['ix']]=e['uncertain_boundary']
    return y,pid,boundary


def calendar_features(index):
    h=index.hour.to_numpy(); season=index.dayofyear.to_numpy()/365.25
    return pd.DataFrame({'hour_sin':np.sin(2*np.pi*h/24),'hour_cos':np.cos(2*np.pi*h/24),
                         'season_sin':np.sin(2*np.pi*season),'season_cos':np.cos(2*np.pi*season),
                         'weekend':(index.dayofweek>=5).astype(float)},index=index)


def forecast_features(f,weather):
    """Future-target features are confined to the common issued UTC-day vector."""
    x=calendar_features(f.index); p=f.p.clip(1e-6,1-1e-6)
    x['p']=p; x['logit_p']=np.log(p/(1-p))
    cols=['previous6_mean','previous6_max','following6_mean','following6_max','day_mean','day_max','rise']
    for c in cols: x[c]=np.nan
    for _,g in f.groupby(f.index.normalize()):
        a=g.p.to_numpy(float)
        for h,t in enumerate(g.index):
            before=a[max(0,h-6):h]; after=a[h+1:min(len(a),h+7)]
            bm=np.nanmean(before) if len(before) and np.isfinite(before).any() else a[h]
            am=np.nanmean(after) if len(after) and np.isfinite(after).any() else a[h]
            x.loc[t,cols]=[bm,np.nanmax(before) if len(before) and np.isfinite(before).any() else a[h],
                          am,np.nanmax(after) if len(after) and np.isfinite(after).any() else a[h],
                          np.nanmean(a) if np.isfinite(a).any() else np.nan,
                          np.nanmax(a) if np.isfinite(a).any() else np.nan,a[h]-bm]
    for c in weather.columns:
        if c.startswith('common_'): x[c]=weather[c].reindex(f.index)
    return x


def prices_frame(index,zone,issues):
    raw=pd.concat([pd.read_parquet(ROOT/'datasets/dk_prices_dev.parquet'),
                   pd.read_parquet(ROOT/'datasets/dk_prices_test.parquet')],ignore_index=True)
    raw=raw.loc[raw.PriceArea.eq(zone)].set_index('target_time').sort_index()
    if raw.index.has_duplicates: raise ValueError('Duplicate prices')
    p=raw.price_eur_mwh.reindex(index)
    delivery=pd.DatetimeIndex(index).tz_convert('Europe/Copenhagen').date
    nextday=(pd.DatetimeIndex(issues).tz_convert('Europe/Copenhagen').normalize()+pd.DateOffset(days=1)).date
    eligible=np.array([a<=b for a,b in zip(delivery,nextday)])
    return p.where(eligible),pd.Series(eligible,index=index),raw.reindex(index)


def price_features(f,price,eligible,kind):
    x=calendar_features(f.index)
    if kind in ('base','combined'):
        p=f.p.clip(1e-6,1-1e-6); x['p']=p; x['logit_p']=np.log(p/(1-p))
    if kind in ('price','combined'):
        x['price_eur_mwh']=price; x['price_missing']=price.isna().astype(float)
        # Daily summaries only contain auction-eligible prices, not future releases.
        group=price.groupby(price.index.normalize())
        x['price_day_mean']=group.transform('mean'); x['price_day_max']=group.transform('max')
        x['price_day_min']=group.transform('min'); x['price_delta_daily_mean']=price-x.price_day_mean
    return x


def model(c):
    return make_pipeline(SimpleImputer(strategy='median',add_indicator=True),StandardScaler(),
                         LogisticRegression(C=c,solver='lbfgs',max_iter=2000,random_state=20261008))


def fit_predictions(x,y,f,zone,name):
    issue=pd.read_parquet(ROOT/'datasets'/f'{zone}.parquet').issue_time.reindex(f.index)
    valid=f.p.notna()&y.notna()
    # The first 2023 prediction is issued on 2022-12-31 at noon.
    maturation_days=15 if zone.startswith('DK') else 7
    devcut=pd.Timestamp('2022-12-31T12:00Z')-pd.Timedelta(days=maturation_days)
    tr=valid&(f.index.year==2022)&(f.index+pd.Timedelta(hours=1)<=devcut)
    # Hyperparameter selection must use only labels matured before the first
    # 2024 issue, just like the first quarterly refit.  Outcome-free resource
    # counting below can still use all issued 2023 forecast vectors.
    validation_cutoff=pd.Timestamp('2023-12-31T12:00Z')-pd.Timedelta(days=maturation_days)
    va=valid&(f.index.year==2023)&(f.index+pd.Timedelta(hours=1)<=validation_cutoff)
    choices=[]
    for c in CS:
        m=model(c); m.fit(x.loc[tr],y.loc[tr]); pv=m.predict_proba(x.loc[va])[:,1]
        choices.append({'zone':zone,'model':name,'C':c,'train_n':int(tr.sum()),'validation_n':int(va.sum()),
                        'train_positives':int(y.loc[tr].sum()),'validation_positives':int(y.loc[va].sum()),
                        'maturation_days':maturation_days,'train_interval_end_cutoff':devcut.isoformat(),
                        'validation_interval_end_cutoff':validation_cutoff.isoformat(),
                        'latest_validation_target_end':(f.index[va].max()+pd.Timedelta(hours=1)).isoformat(),
                        'validation_brier':float(np.mean((pv-y.loc[va].to_numpy())**2)),
                        'validation_logloss':float(log_loss(y.loc[va],pv,labels=[0,1]))})
    chosen=min(choices,key=lambda q:(q['validation_brier'],q['validation_logloss'],q['C']))['C']
    pred=pd.Series(np.nan,index=f.index); m=model(chosen); m.fit(x.loc[tr],y.loc[tr])
    # The 2022 count gate is an explicitly in-sample forecast-distribution setting;
    # no 2024-25 labels determine it. 2023 predictions are held-out.
    early=f.index.year<=2023; pred.loc[early]=m.predict_proba(x.loc[early])[:,1]
    pd.DataFrame({'p':pred.loc[early].where(f.p.loc[early].notna()),'target':y.loc[early],'issue_time':issue.loc[early],
                  'training_distribution_prediction':f.index.year[early]==2022},index=f.index[early]).to_parquet(
                      OUT/f'development_prediction_{zone}_{name}.parquet')
    logs=[]
    periods=sorted(set(zip(f.index.year[f.index.year>=2024],f.index.quarter[f.index.year>=2024])))
    for year,q in periods:
        te=(f.index.year==year)&(f.index.quarter==q)
        first=issue.loc[te].min(); cutoff=first-pd.Timedelta(days=maturation_days)
        fit=valid&(f.index+pd.Timedelta(hours=1)<=cutoff)&(f.index.year>=2022)
        m=model(chosen); m.fit(x.loc[fit],y.loc[fit]); pred.loc[te]=m.predict_proba(x.loc[te])[:,1]
        logs.append(dict(zone=zone,model=name,quarter=f'{year}Q{q}',C=chosen,
                         first_issue=first.isoformat(),label_cutoff=cutoff.isoformat(),
                         latest_fit_target=f.index[fit].max().isoformat(),fit_n=int(fit.sum()),
                         latest_fit_target_end=(f.index[fit].max()+pd.Timedelta(hours=1)).isoformat(),maturation_days=maturation_days,
                         fit_positives=int(y.loc[fit].sum()),features=x.columns.tolist(),
                         past_matured_test_labels_used=bool(f.index[fit].max().year>=2024)))
    pred.loc[f.p.isna()]=np.nan
    return pred,choices,logs


def evaluate(zone,f,predictions,early,issues,reps):
    mask=f.index.year>=2024; ft=f.loc[mask]; yt=early.loc[mask]
    proc=process_masks(ft); metrics=[]; curve=[]; val=[]; allplans={}; records=[]
    for name,p in predictions.items():
        pf=ft.copy(); pf['p']=p.loc[mask]
        gate,gateinfo=threshold_from_counts(p.loc[p.index.year==2022],rho=.5)
        valgate,valgateinfo=threshold_from_counts(p.loc[p.index.year==2023],rho=.5)
        plans={'top1':select(pf,k=1),'top2':select(pf,k=2),
               'count2022_target05':select(pf,gate=gate,k=2),
               'count2023_target05':select(pf,gate=valgate,k=2),
               'fixed025':select(pf,gate=.25,k=2),'count2022_cool6':select(pf,gate=gate,k=2,cooldown=6)}
        plans['count2023_cool6']=select(pf,gate=valgate,k=2,cooldown=6)
        for rule,a in plans.items():
            allplans[name+'__'+rule]=a
            usedinfo=valgateinfo if 'count2023' in rule else gateinfo
            rulegate=valgate if 'count2023' in rule else (gate if 'count2022' in rule else (.25 if rule=='fixed025' else 0))
            out=summary(ft,a,proc); out.update(zone=zone,model=name,rule=rule,gate=rulegate,
                                               gate_training_slots=usedinfo['slots'],gate_training_days=usedinfo['days'],
                                               gate_training_year=2023 if 'count2023' in rule else 2022)
            metrics.append(out)
            for year in (2024,2025):
                fy=ft.loc[ft.index.year==year]; ay=a[ft.index.year==year]
                row=summary(fy,ay); row.update(zone=zone,model=name,rule=rule,year=year); records.append(row)
        for target,y in [('hour',ft.event),('early_hour',yt)]:
            good=y.notna()&pf.p.notna(); pv=pf.p.loc[good].to_numpy(); yy=y.loc[good].to_numpy()
            val.append(dict(zone=zone,model=name,target=target,n=int(good.sum()),positives=int(yy.sum()),
                            brier=float(np.mean((pv-yy)**2)),logloss=float(log_loss(yy,pv,labels=[0,1]))))
        pf[['p','event']].assign(early_target=yt,issue_time=issues.loc[mask]).to_parquet(OUT/f'prediction_{zone}_{name}.parquet')
    days,daily=daily_components(ft,allplans,proc)
    weights=block_weights(days,reps=reps); sim=np.einsum('rd,dpk->rpk',weights,daily,optimize=True); totals=daily.sum(axis=0)
    for j,key in enumerate(allplans):
        name,rule=key.split('__')
        for endpoint,k in [('process',2),('hour',1)]:
            util=totals[j,k]-COSTS*totals[j,0]; samples=sim[:,j,k,None]-COSTS*sim[:,j,0,None]
            lo,hi=np.quantile(samples,[.025,.975],axis=0)
            for h,c in enumerate(COSTS): curve.append(dict(zone=zone,model=name,rule=rule,endpoint=endpoint,cost_ratio=c,
                                                           utility=util[h],lower=lo[h],upper=hi[h],N=int(totals[j,0]),reward=int(totals[j,k]),replicates=reps))
    # Every model also receives the same cost-dependent gate, with common daily K.
    costgate=[]
    for name,p in predictions.items():
        pf=ft.copy(); pf['p']=p.loc[mask]
        for c in COSTS:
            a=select(pf,gate=c,k=2); row=summary(ft,a,proc)
            row.update(zone=zone,model=name,cost_ratio=c,hour_utility=row['hits']-c*row['N'],
                       process_utility=row['C']-c*row['N']); costgate.append(row)
    plans=pd.DataFrame(allplans,index=ft.index).assign(event=ft.event,early_target=yt,issue_time=issues.loc[mask])
    plans.to_parquet(OUT/f'review_masks_{zone}.parquet')
    # Paired bootstrap comparisons avoid treating different methods' bands as a test.
    pairs=[]; keys=list(allplans); base=keys.index('original_M0__top2')
    for j,key in enumerate(keys):
        name,rule=key.split('__')
        if key=='original_M0__top2':continue
        for endpoint,k in [('process',2),('hour',1)]:
            point=totals[j,k]-totals[base,k]-COSTS*(totals[j,0]-totals[base,0])
            sims=(sim[:,j,k]-sim[:,base,k])[:,None]-COSTS*(sim[:,j,0]-sim[:,base,0])[:,None]
            lo,hi=np.quantile(sims,[.025,.975],axis=0)
            for h,c in enumerate(COSTS):pairs.append(dict(zone=zone,model=name,rule=rule,endpoint=endpoint,cost_ratio=c,
                                                         reference='original_M0__top2',difference=point[h],lower=lo[h],upper=hi[h]))
    return metrics,records,val,curve,costgate,pairs


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--reps',type=int,default=1000);ap.add_argument('--zone',default='all');args=ap.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    weather=pd.read_parquet(ROOT/'datasets/weather_features.parquet')
    choices=[];logs=[];metrics=[];annual=[];scores=[];curves=[];gates=[];pairs=[];availability=[]
    zones=['DE_LU_FR_BE','DK1','DK2'] if args.zone=='all' else [args.zone]
    for zone in zones:
        print('zone',zone,flush=True);f=load_stream(zone,'M0','q90')
        meta=pd.read_parquet(ROOT/'datasets'/f'{zone}.parquet');issues=meta.issue_time.reindex(f.index)
        assert issues.groupby(f.index.normalize()).nunique().eq(1).all()
        assert (pd.DatetimeIndex(issues)==f.index.normalize()-pd.Timedelta(hours=12)).all()
        early,pid,boundary=early_labels(f);preds={'original_M0':f.p.copy()}
        if zone=='DE_LU_FR_BE':
            x=forecast_features(f,weather)
            for target,y in [('early',early),('hour',f.event)]:
                name='forecast_'+target+'_logit';p,ch,lg=fit_predictions(x,y,f,zone,name)
                preds[name]=p;choices.extend(ch);logs.extend(lg)
                print(name,'chosen_C',lg[0]['C'],flush=True)
            x.to_parquet(OUT/f'features_{zone}.parquet')
        else:
            price,eligible,raw=prices_frame(f.index,zone,issues)
            raw.assign(eligible_under_local_auction_day=eligible,price_exposed=price,issue_time=issues).to_parquet(OUT/f'price_exposure_{zone}.parquet')
            availability.append(dict(zone=zone,rows=len(f),exposed=int(price.notna().sum()),future_local_day_excluded=int((~eligible).sum()),
                                     first_release_availability='assumed; no historical publication snapshots',
                                     auction_local_timezone='Europe/Copenhagen',unavailable_price_handling='training median plus missing indicators'))
            for kind in ('base','price','combined'):
                x=price_features(f,price,eligible,kind);x.to_parquet(OUT/f'features_{zone}_{kind}.parquet')
                for target,y in [('early',early),('hour',f.event)]:
                    name=kind+'_'+target+'_logit';p,ch,lg=fit_predictions(x,y,f,zone,name)
                    preds[name]=p;choices.extend(ch);logs.extend(lg)
                    print(name,'chosen_C',lg[0]['C'],flush=True)
            # This score is not a probability. A positive affine transform preserves
            # the exact price order, including 2024 values above the 2022 maximum.
            trprice=price.loc[price.index.year==2022];lo,hi=trprice.min(),trprice.max()
            rawrank=((price-lo)/(hi-lo)+1.).fillna(-1.)
            rawrank.loc[f.p.isna()]=np.nan;preds['raw_price_rank']=rawrank
        early.to_frame('early_target').assign(process_id=pid,uncertain_boundary=boundary,event=f.event,issue_time=issues).to_parquet(OUT/f'targets_{zone}.parquet')
        # Raw price ranking is a score, not a probability: evaluate separately.
        rank=preds.pop('raw_price_rank',None)
        outputs=evaluate(zone,f,preds,early,issues,args.reps)
        for container,rows in zip([metrics,annual,scores,curves,gates,pairs],outputs):container.extend(rows)
        if rank is not None:
            mask=f.index.year>=2024;ft=f.loc[mask];pf=ft.copy();pf['p']=rank.loc[mask]
            for k in (1,2):
                a=select(pf,gate=0,k=k);row=summary(ft,a);row.update(zone=zone,model='raw_price_rank',rule=f'top{k}')
                metrics.append(row)
                pd.DataFrame({'rank_score':pf.p,'selected':a,'price_eligible':eligible.loc[mask],'event':ft.event},index=ft.index).to_parquet(OUT/f'raw_price_{zone}_top{k}.parquet')
        print('finished',zone,flush=True)
    for name,rows in [('regularization_validation',choices),('quarterly_refits',logs),('policy_metrics',metrics),
                      ('annual_policy_metrics',annual),('probability_scores',scores),('cost_curves',curves),
                      ('cost_gate_policies',gates),('paired_cost_contrasts',pairs),('price_availability',availability)]:
        df=pd.DataFrame(rows)
        for c in df.columns:
            if c=='features':df[c]=df[c].map(json.dumps)
        df.to_csv(OUT/(name+'.csv'),index=False)
    with (OUT/'analysis_manifest.json').open('w',encoding='utf-8') as h:
        json.dump(dict(status='completed post hoc reanalysis',zones=zones,level='q90',fit_year=2022,validation_year=2023,
                       test_years=[2024,2025],C_candidates=CS,selection_metric='2023 Brier, then logloss, then smaller C',
                       refit='quarterly; label interval end+7d<=issue in core; +15d<=issue in Danish checks; frozen C',bootstrap='paired 14-day moving blocks, year-quarter strata',
                       replicates=args.reps,primary_paired_replicates=2000,quarterly_score_replicates=2000,
                       secondary_full_grid_replicates=args.reps,
                       auxiliary_cooldown_initialization='empty state at start of 2024; not a warm-start comparison with the original policy analysis',
                       validation_label_maturity='interval end before first 2024 issue minus 7 days core / 15 days DK; all-2023 resource counts use only issued forecasts from 2022-fitted pipelines',
                       count_gate='primary: 0.5 slots/day on 2023 held-out predictions from models fitted only in 2022; secondary: 2022 in-sample fitted forecast counts',
                       count_grid=[0.]+np.linspace(.01,.95,95).tolist()+[.99,1.],
                       prices='assumed availability; exclude local auction day+2 UTC tail hours',
                       controls='price/base/combined share calendar; base and combined share identical p/logit controls',
                       feature_data='issued forecast probabilities and archived weather summaries; no realized lag labels'),h,indent=2)


def postprocess(reps=1000):
    """Prespecified target contrasts and nested Danish attribution on saved plans."""
    native=[];contrasts=[];pivotal=[];support=[];calibration=[]
    bins=np.array([0.,.005,.01,.02,.05,.1,.2,.5,1.000001])
    for zone in ('DE_LU_FR_BE','DK1','DK2'):
        f=load_stream(zone,'M0','q90');te=f.loc[f.index.year>=2024]
        saved=pd.read_parquet(OUT/f'review_masks_{zone}.parquet')
        plans={c:saved[c].to_numpy(bool) for c in saved if '__' in c}
        targets=pd.read_parquet(OUT/f'targets_{zone}.parquet')
        processes=process_masks(f)
        for year in (2022,2023,2024,2025):
            mask=(f.index.year==year)&f.event.notna().to_numpy()&f.p.notna().to_numpy()
            ep=[e for e in processes if e['onset'].year==year]
            support.append(dict(zone=zone,year=year,n_hours=int(mask.sum()),
                                event_hours=int(f.event.loc[mask].sum()),early_event_hours=int(targets.early_target.loc[mask].sum()),
                                early_prior=float(targets.early_target.loc[mask].mean()),processes=len(ep),
                                uncertain_boundary_processes=sum(e['uncertain_boundary'] for e in ep)))
        if zone=='DE_LU_FR_BE':
            pairs=[('forecast_early_logit__count2023_target05','original_M0__count2023_target05'),
                   ('forecast_early_logit__count2023_target05','forecast_hour_logit__count2023_target05'),
                   ('forecast_early_logit__count2023_target05','original_M0__fixed025'),
                   ('forecast_early_logit__count2022_target05','original_M0__fixed025'),
                   ('forecast_early_logit__count2022_target05','forecast_hour_logit__count2022_target05'),
                   ('forecast_early_logit__top2','forecast_hour_logit__top2'),
                   ('forecast_early_logit__top2','original_M0__top2')]
        else:
            pairs=[('combined_early_logit__top2','base_early_logit__top2'),
                   ('combined_hour_logit__top2','base_hour_logit__top2'),
                   ('combined_early_logit__count2022_target05','base_early_logit__count2022_target05'),
                   ('combined_hour_logit__count2022_target05','base_hour_logit__count2022_target05'),
                   ('combined_early_logit__count2023_target05','base_early_logit__count2023_target05'),
                   ('combined_hour_logit__count2023_target05','base_hour_logit__count2023_target05'),
                   ('price_hour_logit__top2','original_M0__top2')]
        days,daily=daily_components(te,plans,process_masks(te));keys=list(plans)
        w=block_weights(days,reps=reps); sims=np.einsum('rd,dpk->rpk',w,daily,optimize=True); totals=daily.sum(axis=0)
        for cand,ref in pairs:
            i,j=keys.index(cand),keys.index(ref)
            for k,label in enumerate(['slots','event_hour_hits','early_processes']):
                values=sims[:,i,k]-sims[:,j,k];lo,hi=np.quantile(values,[.025,.975])
                native.append(dict(zone=zone,candidate=cand,reference=ref,endpoint=label,
                                   difference=totals[i,k]-totals[j,k],lower=lo,upper=hi,replicates=reps,
                                   inference='conditional on saved rolling forecasts; no model-refitting bootstrap'))
            for endpoint,k in [('process',2),('hour',1)]:
                point=totals[i,k]-totals[j,k]-COSTS*(totals[i,0]-totals[j,0])
                sam=(sims[:,i,k]-sims[:,j,k])[:,None]-COSTS*(sims[:,i,0]-sims[:,j,0])[:,None]
                lo,hi=np.quantile(sam,[.025,.975],axis=0)
                width=np.quantile(np.max(abs(sam-point),axis=1),.95)
                for h,c in enumerate(COSTS):
                    contrasts.append(dict(zone=zone,candidate=cand,reference=ref,endpoint=endpoint,cost_ratio=c,
                                          difference=point[h],lower=lo[h],upper=hi[h],lower_gridband=point[h]-width,
                                          upper_gridband=point[h]+width,replicates=reps))
        for path in OUT.glob(f'prediction_{zone}_*.parquet'):
            modelname=path.stem.removeprefix(f'prediction_{zone}_')
            pred=pd.read_parquet(path).p
            y=targets.early_target.reindex(pred.index)
            for year in (2024,2025):
                good=(pred.index.year==year)&pred.notna().to_numpy()&y.notna().to_numpy()
                vals=pred.loc[good];truth=y.loc[good]
                code=pd.cut(vals,bins=bins,right=False,labels=False)
                for k in range(len(bins)-1):
                    at=code.eq(k)
                    calibration.append(dict(zone=zone,model=modelname,year=year,target='early_hour',
                                            bin_left=bins[k],bin_right=min(1.,bins[k+1]),n=int(at.sum()),
                                            mean_probability=float(vals.loc[at].mean()) if at.any() else np.nan,
                                            event_rate=float(truth.loc[at].mean()) if at.any() else np.nan))
        if zone.startswith('DK'):
            marginal=pd.read_parquet(ROOT/'datasets'/f'{zone}_marginal.parquet')
            piv=marginal.y_q90_DK_pivotal.reindex(te.index).where(te.event.notna())
            fs=te.copy();fs['event']=piv
            sub=process_masks(fs);orig=process_masks(te)
            for name,a in plans.items():
                scored=a&te.event.notna().to_numpy()
                opp=[e for e in orig if piv.iloc[e['early_ix']].eq(1).any()]
                covered=sum(bool((scored[e['early_ix']]&piv.iloc[e['early_ix']].eq(1).to_numpy()).any()) for e in opp)
                s=summary(fs,a,sub)
                pivotal.append(dict(zone=zone,model_policy=name,N=int(scored.sum()),
                                    pivotal_hits=s['hits'],nested_opportunities=len(opp),nested_covered=covered,
                                    reclustered_processes=len(sub),reclustered_covered=s['C'],
                                    reclustered_coverage=s['coverage'],pivotal_event_hours=int(piv.sum())))
    for name,rows in [('primary_paired_contrasts',native),('primary_paired_cost_contrasts',contrasts),
                      ('pivotal_policy_metrics',pivotal),('early_target_support',support),('early_calibration',calibration)]:
        pd.DataFrame(rows).to_csv(OUT/(name+'.csv'),index=False)


def fix_raw_price_rank():
    """Repair rankings produced by an already-running prior script version."""
    path=OUT/'policy_metrics.csv';metrics=pd.read_csv(path);metrics=metrics.loc[~metrics.model.eq('raw_price_rank')]
    annual=pd.read_csv(OUT/'annual_policy_metrics.csv')
    annual=annual.loc[~annual.model.eq('raw_price_rank')]
    rows=[];ars=[]
    for zone in ('DK1','DK2'):
        f=load_stream(zone,'M0','q90');meta=pd.read_parquet(ROOT/'datasets'/f'{zone}.parquet')
        price,eligible,_=prices_frame(f.index,zone,meta.issue_time)
        tr=price.loc[price.index.year==2022];rank=((price-tr.min())/(tr.max()-tr.min())+1).fillna(-1.)
        rank.loc[f.p.isna()]=np.nan;mask=f.index.year>=2024;ft=f.loc[mask];pf=ft.copy();pf['p']=rank.loc[mask]
        for k in (1,2):
            a=select(pf,gate=0,k=k);row=summary(ft,a);row.update(zone=zone,model='raw_price_rank',rule=f'top{k}');rows.append(row)
            for year in (2024,2025):
                take=ft.index.year==year;s=summary(ft.loc[take],a[take]);s.update(zone=zone,model='raw_price_rank',rule=f'top{k}',year=year);ars.append(s)
            pd.DataFrame({'rank_score':pf.p,'selected':a,'price_eligible':eligible.loc[mask],'event':ft.event},index=ft.index).to_parquet(OUT/f'raw_price_{zone}_top{k}.parquet')
    pd.concat([metrics,pd.DataFrame(rows)],ignore_index=True).to_csv(path,index=False)
    pd.concat([annual,pd.DataFrame(ars)],ignore_index=True).to_csv(OUT/'annual_policy_metrics.csv',index=False)


def quarter_scores(reps=2000):
    rows=[];contrasts=[]
    for zone in ('DE_LU_FR_BE','DK1','DK2'):
        f=load_stream(zone,'M0','q90');f=f.loc[f.index.year>=2024]
        early=pd.read_parquet(OUT/f'targets_{zone}.parquet').early_target.reindex(f.index)
        predictions={p.stem.removeprefix(f'prediction_{zone}_'):pd.read_parquet(p).p for p in OUT.glob(f'prediction_{zone}_*.parquet')}
        days=pd.date_range(f.index.min().normalize(),f.index.max().normalize(),freq='D',tz='UTC')
        weights=block_weights(days,reps=reps)
        for endpoint,y in [('hour',f.event),('early_hour',early)]:
            losses={}
            for name,p in predictions.items():
                loss=(p-y)**2;count=loss.groupby(loss.index.normalize()).count()
                losses[name]=loss.groupby(loss.index.normalize()).mean().where(count.eq(24)).reindex(days)
            for name,daily in losses.items():
                for year in (2024,2025):
                    for q in (1,2,3,4):
                        ix=(days.year==year)&(days.quarter==q)
                        z=daily.loc[ix].to_numpy();w=weights[:,ix];den=w@np.isfinite(z).astype(float)
                        samples=(w@np.nan_to_num(z))/den;lo,hi=np.quantile(samples,[.025,.975])
                        valid=(f.index.year==year)&(f.index.quarter==q)&y.notna().to_numpy()&predictions[name].notna().to_numpy()
                        yy=y.loc[valid];p=predictions[name].loc[valid]
                        rows.append(dict(zone=zone,model=name,quarter=f'{year}Q{q}',target=endpoint,
                                         n=int(valid.sum()),positives=int(yy.sum()),prior=float(yy.mean()),
                                         brier=float(np.nanmean(z)),brier_lower=lo,brier_upper=hi,
                                         logloss=float(log_loss(yy,p,labels=[0,1])),replicates=reps))
                        ref='original_M0'
                        if zone=='DE_LU_FR_BE' and name=='forecast_early_logit':ref='forecast_hour_logit'
                        if name.startswith('combined_'):ref=name.replace('combined_','base_')
                        if name==ref:continue
                        delta=(daily-losses[ref]).loc[ix].to_numpy();den=w@np.isfinite(delta).astype(float)
                        sam=w@np.nan_to_num(delta)/den;lo,hi=np.quantile(sam,[.025,.975])
                        contrasts.append(dict(zone=zone,candidate=name,reference=ref,quarter=f'{year}Q{q}',target=endpoint,
                                              brier_difference=float(np.nanmean(delta)),lower=lo,upper=hi,replicates=reps))
    pd.DataFrame(rows).to_csv(OUT/'quarterly_probability_scores.csv',index=False)
    pd.DataFrame(contrasts).to_csv(OUT/'quarterly_brier_contrasts.csv',index=False)


if __name__=='__main__':
    main()
    if all((OUT/f'review_masks_{z}.parquet').exists() for z in ('DE_LU_FR_BE','DK1','DK2')):
        fix_raw_price_rank()
        postprocess(reps=2000)
        quarter_scores(reps=2000)
