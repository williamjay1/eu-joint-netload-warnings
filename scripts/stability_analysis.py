"""Fixed-copula calibration sensitivity propagated to complete review paths.

Perturbations are transparent probability-scale scenarios, not confidence sets
for true hourly probabilities. Each eight-corner scenario is held constant in
regional sign over the entire issued history. Stateful path invariance across
these scenarios does not certify every possible time-varying perturbation.
"""
from __future__ import annotations
import itertools, json, time
from pathlib import Path
import numpy as np
import pandas as pd
from numerical_check import ROOT, ZONES, matrices, values, event_probability, complete_mask, plan, save_json

OUT=ROOT/'results/stability'

def episodes(index,truth,gap=1):
    f=pd.Series(truth,index=index)
    calendar=pd.date_range(index.min().normalize(),index.max().normalize(),freq='D',tz='UTC')
    daily=f.groupby(index.normalize()).agg(['count','max']).reindex(calendar)
    observed=daily['count'].eq(24);active=daily['max'].eq(1)&observed
    clusters=[];current=[];last=None
    for day in calendar[active]:
        between=calendar[(calendar>last)&(calendar<day)] if last is not None else calendar[:0]
        connected=last is not None and len(between)<=gap and observed.loc[between].all()
        if not connected and current: clusters.append(current);current=[]
        current.append(day);last=day
    if current:clusters.append(current)
    result=[]
    for days in clusters:
        hours=f.loc[(index>=days[0])&(index<days[-1]+pd.Timedelta(days=1))]
        event=hours.index[hours.eq(1)];onset=event[0]
        result.append({'onset':onset,'early':index.get_indexer(event[event<onset+pd.Timedelta(hours=6)]),'hours':index.get_indexer(event)})
    return result

def coverage(alert,processes):
    return np.asarray([bool(alert[e['early']].any()) for e in processes])

def metrics(alert,scoremask,y,processes):
    return {'slots':int((alert&scoremask).sum()),'event_hits':int((alert&scoremask&(y==1)).sum()),
        'early_processes_covered':int(coverage(alert,processes).sum()),'processes':len(processes)}

def ordinary_certificate(index,L,U,gate=.25,k=2):
    # Sufficient rank and threshold conditions for all probabilities in bounds.
    # Strict score separation avoids requiring any convention about ties.
    selected=np.zeros(len(L),bool);unselected=np.zeros(len(L),bool)
    for _,ix in pd.Series(np.arange(len(L)),index=index).groupby(index.normalize()):
        ii=ix.to_numpy()
        if len(ii)!=24 or not np.isfinite(L[ii]).all():continue
        for at in ii:
            others=ii[ii!=at]
            selected[at]=(L[at]>=gate) and int((U[others]>=L[at]).sum())<k
            unselected[at]=(U[at]<gate) or int((L[others]>U[at]).sum())>=k
    return selected,unselected

def conditional_residuals(full):
    f=full.loc[complete_mask(full,years=(2022,))]
    rows=[]
    for q in ('q90','q95'):
        for cell,g in f.groupby('cell'):
            for z in ZONES:
                valid=g[[f'exceed_{q}_{z}',f'v0_{q}_{z}']].notna().all(axis=1)
                a=g.loc[valid]
                rows.append({'year':2022,'q':q,'cell':cell,'zone':z,'hours':len(a),'days':len(a.index.normalize().unique()),
                    'observed_exceedance':float(a[f'exceed_{q}_{z}'].mean()),'predicted_exceedance':float((1-a[f'v0_{q}_{z}']).mean()),
                    'signed_mean_residual':float((a[f'exceed_{q}_{z}']-(1-a[f'v0_{q}_{z}'])).mean())})
    return pd.DataFrame(rows)

def main():
    start=time.perf_counter();OUT.mkdir(parents=True,exist_ok=True)
    full=pd.read_parquet(ROOT/'datasets/marginal_predictions.parquet')
    published=pd.read_parquet(ROOT/'datasets/DE_LU_FR_BE.parquet')
    f=full.loc[full.index.year>=2022].copy();idx=f.index;n=len(f)
    good=f[['M0_q90','M1_q90','M2_q90','M3_q90','y_q90']].notna().all(axis=1)
    goodday=good.groupby(idx.normalize()).sum().eq(24)
    truth=f.y_q90.copy();truth.loc[~idx.normalize().isin(goodday.index[goodday])]=np.nan
    scoremask=complete_mask(f).to_numpy();testidx=idx[scoremask];y=truth.to_numpy(float)
    # Reconstruct intact test-domain episodes exactly as the released evaluator.
    testtruth=truth.loc[idx.year>=2024]
    proctest=episodes(testtruth.index,testtruth.to_numpy())
    offset=np.flatnonzero(idx.year>=2024)[0]
    processes=[{'onset':e['onset'],'early':e['early']+offset,'hours':e['hours']+offset} for e in proctest]
    assert len(processes)==71,'Frozen common-domain process definition changed'
    p=f.M0_q90.to_numpy(float)
    # Numerical clipping explains at most 2e-12 difference vs original policy p.
    p=p.copy();p[published.policy_p_q90.isna().to_numpy()]=np.nan
    rules=[('ordinary_gate',.25,0,False),('rise_cooldown',.19,6,True)]
    baseline={name:plan(idx,p,gate,cool,rise) for name,gate,cool,rise in rules}
    baseline_top2=plan(idx,p,0.,0,False)
    assert np.array_equal(baseline['ordinary_gate'],published.P1_q90.to_numpy())
    assert np.array_equal(baseline['rise_cooldown'],published.P2_q90.to_numpy())
    finite=np.isfinite(p);uf=values(f.loc[finite],'q90');R=matrices(f.loc[finite])
    signs=list(itertools.product((-1,1),repeat=3))
    scene_rows=[];summary_rows=[];hour_rows=[];certificate_rows=[]
    for delta in (.01,.03):
        probabilities=[];selections={name:[] for name,*_ in rules}
        for sign in signs:
            uu=np.clip(uf+delta*np.asarray(sign),1e-12,1-1e-12)
            changed=np.full(n,np.nan)
            changed[finite]=event_probability(uu,R,np.inf,1e-9)
            changed_top2=plan(idx,changed,0.,0,False)
            probabilities.append(changed)
            for name,gate,cool,rise in rules:
                a=plan(idx,changed,gate,cool,rise);selections[name].append(a)
                b=baseline[name];cover=coverage(a,processes);basecover=coverage(b,processes)
                scene_rows.append({'delta_each_region':delta,'regional_signs':''.join('+' if x>0 else '-' for x in sign),'rule':name,
                    **metrics(a,scoremask,y,processes),
                    'list_symmetric_difference':int(((a!=b)&scoremask).sum()),
                    'selection_days_changed':int(pd.Series((a!=b)&scoremask,index=idx).groupby(idx.normalize()).max().sum()),
                    'new_early_processes':int((cover&~basecover).sum()),'lost_early_processes':int((~cover&basecover).sum()),
                    'gate_crossing_hours':int((((changed>=gate)!=(p>=gate))&scoremask).sum()),
                    'probability_top2_symmetric_difference':int(((changed_top2!=baseline_top2)&scoremask).sum()),
                    'max_probability_change':float(np.nanmax(abs(changed[scoremask]-p[scoremask]))),
                    'mean_probability_change':float(np.nanmean(abs(changed[scoremask]-p[scoremask]))),
                    'Brier_change':float(np.mean((changed[scoremask]-y[scoremask])**2)-np.mean((p[scoremask]-y[scoremask])**2))})
            print('sensitivity corner',delta,sign,flush=True)
        Ps=np.stack(probabilities);lower=Ps.min(axis=0);upper=Ps.max(axis=0)
        # For a fixed copula, event probability is monotone in each regional u.
        # All-plus/all-minus corners give its exact rectangle extrema per hour.
        assert np.nanmax(Ps[-1]-lower)<1e-9 and np.nanmax(upper-Ps[0])<1e-9
        unionL=np.maximum(0,p-3*delta);unionU=np.minimum(1,p+3*delta)
        certS,certU=ordinary_certificate(idx,unionL,unionU)
        exactS,exactU=ordinary_certificate(idx,lower,upper)
        for label,S,U in [('union_bound',certS,certU),('monotone_probability_envelope',exactS,exactU)]:
            amb=~(S|U)
            certificate_rows.append({'delta_each_region':delta,'bound':label,'test_hours':int(scoremask.sum()),
                'certified_selected':int((S&scoremask).sum()),'certified_unselected':int((U&scoremask).sum()),
                'unresolved':int((amb&scoremask).sum()),'unresolved_fraction':float(amb[scoremask].mean()),
                'baseline_selected_certified_fraction':float(S[baseline['ordinary_gate']&scoremask].mean()),
                'baseline_early_coverage_with_at_least_one_certified_selection':int(coverage(S,processes).sum())})
        for name,gate,cool,rise in rules:
            corners=np.stack(selections[name])
            # Add the centre because interior rank changes need not be present at
            # any corner. This remains a finite-scenario assessment, not a full
            # continuous uncertainty-set certificate for a stateful path.
            cornerS=corners.all(axis=0);cornerU=~corners.any(axis=0)
            cornercritical=~(cornerS|cornerU)
            aa=np.vstack([baseline[name][None,:],corners])
            allS=aa.all(axis=0);noneS=~aa.any(axis=0);critical=~(allS|noneS)
            cover=np.stack([coverage(a,processes) for a in aa]);basecover=coverage(baseline[name],processes)
            row={'delta_each_region':delta,'rule':name,'test_hours':int(scoremask.sum()),
                'scenario_invariant_selected':int((allS&scoremask).sum()),'scenario_invariant_unselected':int((noneS&scoremask).sum()),
                'scenario_critical_hours':int((critical&scoremask).sum()),'scenario_critical_fraction':float(critical[scoremask].mean()),
                'corner_only_critical_hours':int((cornercritical&scoremask).sum()),
                'centre_opposes_all_corners_hours':int((((baseline[name]&cornerU)|(~baseline[name]&cornerS))&scoremask).sum()),
                'baseline_selected_scenario_critical_fraction':float(critical[baseline[name]&scoremask].mean()),
                'baseline_early_coverage':int(basecover.sum()),'minimum_scenario_coverage':int(cover.sum(axis=1).min()),
                'maximum_scenario_coverage':int(cover.sum(axis=1).max()),'early_coverage_changes_in_any_scenario':int((cover.any(axis=0)&~cover.all(axis=0)).sum()),
                'baseline_covered_processes_depending_entirely_on_critical_hours':int((basecover&~coverage(allS,processes)).sum()),
                'gate_threshold_union_bound_ambiguous_hours':int((((p-3*delta)<gate)&((p+3*delta)>=gate)&scoremask).sum()),
                'min_slots':int(min(r['slots'] for r in scene_rows if r['delta_each_region']==delta and r['rule']==name)),
                'max_slots':int(max(r['slots'] for r in scene_rows if r['delta_each_region']==delta and r['rule']==name))}
            summary_rows.append(row)
            hour_rows.append(pd.DataFrame({'target_time':testidx,'delta_each_region':delta,'rule':name,
                'baseline_selected':baseline[name][scoremask],'scenario_invariant_selected':allS[scoremask],
                'scenario_invariant_unselected':noneS[scoremask],'scenario_critical':critical[scoremask],
                'corner_only_critical':cornercritical[scoremask],
                'p':p[scoremask],'p_min':lower[scoremask],'p_max':upper[scoremask]}))
    # Conditional positions of the seasonal thresholds, by season/model/zone.
    positions=[]
    for q in ('q90','q95'):
        fq=full.loc[complete_mask(full,q)]
        for margin in (0,1):
            for z in ZONES:
                for season,sub in [('all',fq),('cool',fq.loc[fq.index.month.isin([10,11,12,1,2,3])]),('warm',fq.loc[fq.index.month.isin([4,5,6,7,8,9])])]:
                    v=sub[f'v{margin}_{q}_{z}'];v=v[np.isfinite(v)]
                    positions.append({'q':q,'margin':margin,'zone':z,'season':season,'hours':len(v),
                        **{f'quantile_{s:g}':float(v.quantile(s)) for s in (.01,.05,.1,.25,.5,.75,.9,.95,.99)},
                        'fraction_below_050':float((v<.5).mean()),'fraction_below_090':float((v<.9).mean()),
                        'fraction_above_099':float((v>.99).mean())})
    # Dependence-family change near operational thresholds.
    copula_rows=[]
    for m0,m2,margin in [('M0','M2',0),('M1','M3',1)]:
        pg=f[m0+'_q90'].to_numpy();pt=f[m2+'_q90'].to_numpy()
        for name,gate,cool,rise in rules:
            G=plan(idx,pg,gate,cool,rise);T=plan(idx,pt,gate,cool,rise)
            near=scoremask&(abs(pg-gate)<=.02)
            copula_rows.append({'margin':margin,'rule':name,'near_gate_hours':int(near.sum()),
                'mean_abs_probability_difference':float(np.nanmean(abs(pt[scoremask]-pg[scoremask]))),
                'near_gate_mean_abs_probability_difference':float(np.nanmean(abs(pt[near]-pg[near]))),
                'near_gate_max_abs_probability_difference':float(np.nanmax(abs(pt[near]-pg[near]))),
                'gate_crossing_hours':int(((pg>=gate)!=(pt>=gate))[scoremask].sum()),
                'list_symmetric_difference_hours':int(((G!=T)&scoremask).sum()),
                'gaussian_slots':int((G&scoremask).sum()),'t_slots':int((T&scoremask).sum()),
                'gaussian_early_processes':int(coverage(G,processes).sum()),'t_early_processes':int(coverage(T,processes).sum())})
    pd.DataFrame(scene_rows).to_csv(OUT/'stability_corner_paths.csv',index=False)
    pd.DataFrame(summary_rows).to_csv(OUT/'stability_summary.csv',index=False)
    pd.DataFrame(certificate_rows).to_csv(OUT/'stability_certificates.csv',index=False)
    pd.concat(hour_rows,ignore_index=True).to_parquet(OUT/'stability_hourly.parquet',index=False)
    pd.DataFrame(positions).to_csv(OUT/'stability_threshold_positions.csv',index=False)
    pd.DataFrame(copula_rows).to_csv(OUT/'stability_copula_gate_contrasts.csv',index=False)
    conditional_residuals(full).to_csv(OUT/'stability_2022_cell_residuals.csv',index=False)
    report={'status':'completed','hours':int(scoremask.sum()),'processes':len(processes),
        'marginal_perturbation_deltas':[.01,.03],'corner_scenarios_each_delta':8,'paths_in_summary_each_delta':9,
        'copula':'fixed hourly R Gaussian; all regional u shifted on probability scale and clipped to interior',
        'full_path':'all2022-2025issuedforecasts before2024-2025complete-label scoring; cooldown state fully recomputed',
        'scenario_bound':'Eight constant-regional-sign full-history scenarios plus the central baseline; invariance here does not certify interior or arbitrary time-varying perturbations of a stateful rule.',
        'certificate_bound':'Ordinary capacity-two gate only: coupling sum3delta and monotone pointwise rectangle-envelope sufficient threshold/rank certificates, strict rank gaps.',
        'not_a_confidence_set':True,'not_an_hourly_probability_confidence_interval':True,
        'baseline':{name:metrics(a,scoremask,y,processes) for name,a in baseline.items()},
        'sensitivity':summary_rows,'certificate':certificate_rows,'copula_contrasts':copula_rows,
        'seconds':time.perf_counter()-start}
    save_json(OUT/'stability_manifest.json',report);print(json.dumps(report),flush=True)

if __name__=='__main__': main()
