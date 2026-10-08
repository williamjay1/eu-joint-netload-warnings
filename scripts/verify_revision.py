"""Recompute fixed-result identities without downloading, refitting or plotting."""
from __future__ import annotations
import ast,hashlib,importlib.metadata,inspect,json,platform,sys,time
from pathlib import Path
import numpy as np
import pandas as pd
from shared_eval import ROOT,load_stream,select,process_masks,summary
import policy_analysis as pa
import stability_analysis as sa
import early_price_analysis as ep

VERSION='asmbi-evaluation-20261008.1'
CHECKS=[];INPUTS=[];SIM={}

def clean(x):
    if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [clean(v) for v in x]
    if isinstance(x,np.generic):return x.item()
    if isinstance(x,Path):return x.as_posix()
    return x

def check(name,condition,**details):
    CHECKS.append(dict(name=name,ok=bool(condition),**clean(details)))
    if not condition:print('FAILED:',name,details,flush=True)

def near(a,b,tol=1e-10):return np.allclose(a,b,rtol=0,atol=tol,equal_nan=True)

def csv(path):return pd.read_csv(ROOT/path)

def js(path):return json.loads((ROOT/path).read_text(encoding='utf8'))

def inventory_and_clocks():
    files=sorted((ROOT/'datasets').glob('*.parquet'))
    check('nine derivative input tables',len(files)==9,count=len(files))
    for p in files:
        f=pd.read_parquet(p)
        item={'path':p.relative_to(ROOT).as_posix(),'rows':len(f),'columns':list(f.columns),
              'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
        if isinstance(f.index,pd.DatetimeIndex):item.update(start=f.index.min().isoformat(),end=f.index.max().isoformat(),timezone=str(f.index.tz))
        INPUTS.append(item)
    for zone in ('DE_LU_FR_BE','DK1','DK2'):
        f=pd.read_parquet(ROOT/'datasets'/f'{zone}.parquet')
        check(zone+' UTC hourly grid',f.index.is_unique and str(f.index.tz)=='UTC' and len(f)==35064)
        check(zone+' daily issue clock',bool((pd.DatetimeIndex(f.issue_time)==f.index.normalize()-pd.Timedelta(hours=12)).all()))
        for level in ('q90','q95'):
            common=f[[f'M{m}_{level}' for m in range(4)]].notna()
            check(zone+' '+level+' identical four-model availability',all(common.iloc[:,0].equals(common.iloc[:,j]) for j in range(1,4)))
            stream=load_stream(zone,'M0',level);test=stream.loc[stream.index.year>=2024]
            check(zone+' '+level+' complete scoring domain',int(test.event.notna().sum())==17400,
                hours=int(test.event.notna().sum()),days=int(test.event.notna().sum()/24))
    check('analysis scripts have no old workspace path',all('D:\\MLWork\\' not in p.read_text(encoding='utf8') for p in (ROOT/'scripts').glob('*.py')))

def fixed_policies():
    gates=csv('results/policy_analysis/development_gates.csv')
    stored=csv('results/policy_analysis/policy_summary.csv')
    comparisons=0;maximumdifference=0
    for zone in ('DE_LU_FR_BE','DK1','DK2'):
        for level in ('q90','q95'):
            for model in ('M0','M1','M2','M3'):
                f=load_stream(zone,model,level)
                r=gates.loc[gates.zone.eq(zone)&gates.level.eq(level)&gates.model.eq(model)].iloc[0]
                computed,meta=pa.threshold_from_counts(f.loc['2022'].p)
                check(f'{zone} {model} {level} development forecast-count gate',near(computed,r.gate) and meta['slots']==int(r.slots))
                plans,logs=pa.make_plans(f,computed)
                check(f'{zone} {model} {level} rolling history is past only',bool((pd.to_datetime(logs.history_last.dropna(),utc=True)<pd.to_datetime(logs.loc[logs.history_last.notna(),'target_day'],utc=True)).all()) and not logs.used_outcomes.any())
                mask=f.index.year>=2024;ft=f.loc[mask]
                saved=pd.read_parquet(ROOT/'results/policy_analysis'/f'plans_{zone}_{model}_{level}.parquet')
                for name,a in plans.items():
                    changed=int((saved[name].to_numpy(bool)!=a[mask]).sum())
                    comparisons+=1;maximumdifference=max(maximumdifference,changed)
                rows=pd.DataFrame(pa.period_statistics(ft,{n:a[mask] for n,a in plans.items()},process_masks(ft),zone,model,level))
                ref=stored.loc[stored.zone.eq(zone)&stored.model.eq(model)&stored.level.eq(level)]
                merged=rows.merge(ref,on=['zone','model','level','period','policy'],suffixes=('_new','_old'),validate='one_to_one')
                fields=['N','hits','C','processes','complete_days','zero_review_days','daily_cap_days','coverage','slots_per_day']
                check(f'{zone} {model} {level} all-period fixed result cells',len(merged)==len(rows) and all(near(merged[c+'_new'],merged[c+'_old']) for c in fields),rows=len(rows))
    check('all frozen forecast-only plans reproduce',maximumdifference==0,comparisons=comparisons,maximum_changed_hours=maximumdifference)
    f=load_stream();gate,_=pa.threshold_from_counts(f.loc['2022'].p)
    original,_=pa.make_plans(f,gate);altered=f.copy();altered['event']=np.nan
    changed,_=pa.make_plans(altered,gate)
    check('allocation invariant to deleting all realised outcomes',all(np.array_equal(original[n],changed[n]) for n in original))
    ftest=f.loc[f.index.year>=2024];proc=process_masks(ftest)
    info={n:summary(ftest,a[f.index.year>=2024],proc) for n,a in original.items()}
    check('native fixed identities',info['top2']['N']==1450 and info['top2']['hits']==341 and info['top2']['C']==35 and info['gate025']['N']==618 and info['gate025']['hits']==322 and info['gate025']['C']==32 and info['inherited_priority019']['N']==572 and info['inherited_priority019']['C']==34,counts=info)
    removed=original['top2']&~original['gate025']&(f.index.year>=2024)&f.event.notna().to_numpy()
    check('removed-slot identity',int(removed.sum())==832 and int((removed&f.event.eq(1).to_numpy()).sum())==19 and int((removed&f.event.eq(0).to_numpy()).sum())==813)
    check('all three lost early spells retained',len(csv('results/policy_analysis/lost_three_early_processes.csv'))==3)
    for function in (select,pa.rolling_gates,ep.forecast_features):
        tree=ast.parse(inspect.getsource(function))
        accesses=[x.attr for x in ast.walk(tree) if isinstance(x,ast.Attribute)]
        check(function.__name__+' contains no realised-outcome access',not set(accesses)&{'event','exceed','y','early_target'},attributes=sorted(set(accesses)))

def scores_and_numerics():
    s=csv('results/forecast_diagnostics/scores.csv')
    for r in s.itertuples():
        f=load_stream(r.zone,r.model,r.level)
        f=f.loc[f.index.year.isin((2022,2023) if r.period=='development' else (2024,2025))]
        valid=f.p.notna()&f.event.notna();bs=float(np.mean((f.p[valid]-f.event[valid])**2))
        check(f'score {r.zone} {r.model} {r.level} {r.period}',int(valid.sum())==r.hours and near(bs,r.BS,1e-12))
    report=js('results/numerical/numerical_summary.json');conv=csv('results/numerical/numerical_full_convergence.csv')
    refs=csv('results/numerical/numerical_independent_cdf.csv');lists=csv('results/numerical/numerical_plan_comparison.csv')
    check('numerical input checksum',report['source_sha256']==hashlib.sha256((ROOT/'datasets/marginal_predictions.parquet').read_bytes()).hexdigest())
    check('numerical full sample and independent references',report['full_q90_hours']==17400 and len(refs)==96 and refs.weather_cell.nunique()==6 and bool((refs.minimum_R_eigenvalue>0).all()))
    calculated=refs[['scipy_seed_0','scipy_seed_1','scipy_seed_2']].mean(axis=1)
    check('independent reference arithmetic',near(calculated,refs.scipy_mean,1e-12) and near(abs(calculated-refs.deterministic),refs.abs_difference,1e-12))
    check('numerical scores and list precision',float(abs(conv.stored_Brier-conv.tighter_Brier).max())<1e-12 and int(lists.symmetric_difference_test.sum())==0)
    check('negative-count correction/normalisation logged',{'roundoff_corrected_hours','maximum_normalised_count_sum_discrepancy'}.issubset(conv.columns) and float(conv.maximum_normalised_count_sum_discrepancy.max())<1e-12)

def stability():
    a=pd.read_parquet(ROOT/'results/stability/stability_hourly.parquet')
    s=csv('results/stability/stability_summary.csv');cert=csv('results/stability/stability_certificates.csv')
    manifest=js('results/stability/stability_manifest.json')
    check('centre included in nine finite paths',manifest['paths_in_summary_each_delta']==9 and manifest['corner_scenarios_each_delta']==8 and manifest['not_an_hourly_probability_confidence_interval'])
    expected={(0.01,'ordinary_gate'):61,(0.01,'rise_cooldown'):116,(0.03,'ordinary_gate'):219,(0.03,'rise_cooldown'):333}
    for key,g in a.groupby(['delta_each_region','rule']):
        r=s.loc[s.delta_each_region.eq(key[0])&s.rule.eq(key[1])].iloc[0]
        critical=~(g.scenario_invariant_selected|g.scenario_invariant_unselected)
        okay=len(g)==17400 and np.array_equal(critical,g.scenario_critical)
        okay &= not bool((g.baseline_selected&g.scenario_invariant_unselected).any())
        okay &= not bool((~g.baseline_selected&g.scenario_invariant_selected).any())
        okay &= int(critical.sum())==expected[key]==r.scenario_critical_hours
        okay &= near(float(critical[g.baseline_selected].mean()),r.baseline_selected_scenario_critical_fraction)
        check('nine-path classification '+str(key),okay,critical_hours=int(critical.sum()),selected_critical_fraction=float(critical[g.baseline_selected].mean()))
        if key[1]=='ordinary_gate':
            idx=pd.DatetimeIndex(g.target_time);p=g.p.to_numpy()
            for label,L,U in [('union_bound',np.maximum(0,p-3*key[0]),np.minimum(1,p+3*key[0])),
                ('monotone_probability_envelope',g.p_min.to_numpy(),g.p_max.to_numpy())]:
                cs,cu=sa.ordinary_certificate(idx,L,U)
                r=cert.loc[cert.delta_each_region.eq(key[0])&cert.bound.eq(label)].iloc[0]
                check('ordinary sufficient rank certificate '+str(key[0])+' '+label,int(cs.sum())==r.certified_selected and int(cu.sum())==r.certified_unselected and int((~(cs|cu)).sum())==r.unresolved)

def simulations():
    metrics=['BS','probability_MAE','N','C','processes','utility_r005','hits','false','changed_slots','marginal_MAE','gate_crossings']
    for sub,amplitude,shift in [('',.012,-.035),('wide_boundary',.08,.035)]:
        base=Path('results/simulations')/sub
        f=pd.read_parquet(ROOT/base/'replicates.parquet');table=csv(base/'summary.csv');design=js(base/'ADEMP.json')
        SIM[sub]=f
        # The original narrow-run ADEMP predates the added amplitude field.
        # Its explicit archived shift and current reproducible default are kept
        # distinct from metadata actually present in that archived record.
        configured_amplitude=design.get('boundary_amplitude',.012 if sub=='' else np.nan)
        check('simulation fixture '+(sub or 'narrow'),len(f)==28800 and design['replications_per_primary_DGP']==400 and design['days']==120 and near(configured_amplitude,amplitude) and near(design['marginal_shift'],shift),
            amplitude_field_in_archived_ADEMP='boundary_amplitude' in design,configured_amplitude=configured_amplitude)
        grouped=f.groupby(['persistence','drift','model','policy'])
        mean=grouped[metrics].mean();se=grouped[metrics].std(ddof=1)/np.sqrt(400)
        table=table.set_index(['persistence','drift','model','policy']).sort_index();mean=mean.sort_index();se=se.sort_index()
        check('simulation means and MCSE '+(sub or 'narrow'),len(table)==72 and all(near(mean[c],table[c+'_mean'],1e-10) and near(se[c],table[c+'_MCSE'],1e-10) for c in metrics))
        nearfar=f.loc[f.model.isin(['margin_near','margin_far'])].pivot(index=['persistence','drift','policy','rep'],columns='model',values='marginal_MAE')
        check('equal near/far marginal perturbation scale '+(sub or 'narrow'),near(nearfar.margin_near,nearfar.margin_far,1e-14))
        check('simulation accounting '+(sub or 'narrow'),bool((f.N==f.hits+f.false).all()) and bool((f.C<=f.processes).all()) and bool((f.C<=f.N).all()))
        temporal=csv(base/'temporal_fixture.csv')
        check('temporal joint-probability fixture '+(sub or 'narrow'),bool((abs(temporal.estimate-temporal.truth)<=4*temporal.MCSE).all()))
        phase=csv(base/'phase_fixture.csv')
        mcse=phase.noise_phase_coverage.std(ddof=1)/np.sqrt(len(phase))
        check('explicit phase-information fixture '+(sub or 'narrow'),len(phase)==1000 and bool(phase.known_phase_coverage.eq(1).all()) and abs(phase.noise_phase_coverage.mean()-2/24)<=4*mcse)
    paired=csv('results/simulations/paired_contrasts.csv');groups={k:g for k,g in SIM[''].groupby(['persistence','drift'])}
    good=True
    for r in paired.itertuples():
        g=groups[(r.persistence,r.drift)]
        if r.contrast_type=='prediction_stream':
            a=g.loc[g.policy.eq(r.policy)&g.model.eq(r.candidate)].set_index('rep')[r.metric]
            b=g.loc[g.policy.eq(r.policy)&g.model.eq(r.reference)].set_index('rep')[r.metric]
        else:
            a=g.loc[g.model.eq(r.model)&g.policy.eq(r.candidate)].set_index('rep')[r.metric]
            b=g.loc[g.model.eq(r.model)&g.policy.eq(r.reference)].set_index('rep')[r.metric]
        diff=a-b;good &= near(diff.mean(),r.difference_mean) and near(diff.std(ddof=1)/np.sqrt(len(diff)),r.paired_MCSE)
    check('all narrow paired simulation contrasts',good,rows=len(paired))
    if (ROOT/'results/figures/figure4_simulation_mechanisms_source.csv').exists():
        fig=csv('results/figures/figure4_simulation_mechanisms_source.csv');good=True
        for r in fig.loc[fig.panel.isin(['a','b'])].itertuples():
            f=SIM['wide_boundary' if r.DGP.startswith('Wide') else '']
            g=f.loc[f.persistence.eq(r.persistence)&f.drift.eq(r.drift)&f.policy.eq('Gate')]
            p=g.pivot(index='rep',columns='model',values=r.metric);diff=p.margin_near-p.margin_far
            good &= near(diff.mean(),r.mean) and near(diff.std(ddof=1)/np.sqrt(400),r.MCSE)
        check('figure4 contrasts trace to both corrected fixtures',good)

def early_and_price():
    refits=csv('results/early_price/quarterly_refits.csv')
    for zone,g in refits.groupby('zone'):
        lag=15 if zone.startswith('DK') else 7
        first=pd.to_datetime(g.first_issue,utc=True);cut=pd.to_datetime(g.label_cutoff,utc=True);latest=pd.to_datetime(g.latest_fit_target_end,utc=True)
        check(zone+' interval-end maturity cutoff',bool((first-cut==pd.Timedelta(days=lag)).all()) and bool((latest<=cut).all()) and bool(g.maturation_days.eq(lag).all()))
    val=csv('results/early_price/regularization_validation.csv')
    for zone,g in val.groupby('zone'):
        lag=15 if zone.startswith('DK') else 7
        issues=pd.read_parquet(ROOT/'datasets'/f'{zone}.parquet').issue_time
        first2024=pd.to_datetime(issues.loc[issues.index.year==2024],utc=True).min()
        first2023=pd.to_datetime(issues.loc[issues.index.year==2023],utc=True).min()
        validation_cut=pd.to_datetime(g.validation_interval_end_cutoff,utc=True)
        latest_validation=pd.to_datetime(g.latest_validation_target_end,utc=True)
        train_cut=pd.to_datetime(g.train_interval_end_cutoff,utc=True)
        check(zone+' regularisation validation interval-end maturity',
              bool((validation_cut==first2024-pd.Timedelta(days=lag)).all())
              and bool((latest_validation<=validation_cut).all())
              and bool((train_cut==first2023-pd.Timedelta(days=lag)).all())
              and bool(g.maturation_days.eq(lag).all()),rows=len(g),
              first_2024_issue=first2024.isoformat(),validation_cutoff=validation_cut.iloc[0].isoformat(),
              latest_validation_target_end=latest_validation.max().isoformat())
    tree=ast.parse(inspect.getsource(ep.fit_predictions))
    fits=[n for n in ast.walk(tree) if isinstance(n,ast.Call)
          and isinstance(n.func,ast.Attribute) and n.func.attr=='fit']
    fit_inputs=[tuple(ast.unparse(a) for a in n.args) for n in fits]
    train_masks=[ast.unparse(n.value) for n in ast.walk(tree) if isinstance(n,ast.Assign)
                 and any(isinstance(t,ast.Name) and t.id=='tr' for t in n.targets)]
    steps=[type(s).__name__ for s in ep.model(.1).named_steps.values()]
    check('preprocessing fitted with training pipeline; validation prediction only',
          steps==['SimpleImputer','StandardScaler','LogisticRegression'] and len(fits)==3
          and fit_inputs.count(('x.loc[tr]','y.loc[tr]'))==2
          and fit_inputs.count(('x.loc[fit]','y.loc[fit]'))==1
          and len(train_masks)==1 and 'f.index.year == 2022' in train_masks[0],
          verification='pipeline and actual fit-call source inspection, without refitting',
          pipeline_steps=steps,fit_inputs=fit_inputs,development_training_mask=train_masks)
    choicegood=True
    development_counts_good=True
    for (zone,model),g in val.groupby(['zone','model']):
        selected=g.sort_values(['validation_brier','validation_logloss','C']).iloc[0].C
        logged=refits.loc[refits.zone.eq(zone)&refits.model.eq(model),'C']
        choicegood &= bool(logged.eq(selected).all())
        dev=pd.read_parquet(ROOT/'results/early_price'/f'development_prediction_{zone}_{model}.parquet')
        valid=dev.p.notna()&dev.target.notna()
        train_cut=pd.Timestamp(g.iloc[0].train_interval_end_cutoff)
        validation_cut=pd.Timestamp(g.iloc[0].validation_interval_end_cutoff)
        tr=valid&(dev.index.year==2022)&(dev.index+pd.Timedelta(hours=1)<=train_cut)
        va=valid&(dev.index.year==2023)&(dev.index+pd.Timedelta(hours=1)<=validation_cut)
        development_counts_good &= bool(g.train_n.eq(int(tr.sum())).all())
        development_counts_good &= bool(g.validation_n.eq(int(va.sum())).all())
        development_counts_good &= bool(g.train_positives.eq(int(dev.target.loc[tr].sum())).all())
        development_counts_good &= bool(g.validation_positives.eq(int(dev.target.loc[va].sum())).all())
        development_counts_good &= bool((pd.to_datetime(g.latest_validation_target_end,utc=True)==dev.index[va].max()+pd.Timedelta(hours=1)).all())
    check('regularisation selected only on recorded 2023 validation',choicegood)
    check('all 42 candidate rows reproduce matured train/validation sample counts',
          len(val)==42 and development_counts_good,rows=len(val),models=val.groupby(['zone','model']).ngroups)
    scores=csv('results/early_price/probability_scores.csv');cache={};good=True
    for r in scores.itertuples():
        key=(r.zone,r.model)
        if key not in cache:cache[key]=pd.read_parquet(ROOT/'results/early_price'/f'prediction_{r.zone}_{r.model}.parquet')
        f=cache[key];y=f.event if r.target=='hour' else f.early_target
        valid=f.p.notna()&y.notna();bs=float(np.mean((f.p[valid]-y[valid])**2))
        good &= int(valid.sum())==r.n and int(y[valid].sum())==r.positives and near(bs,r.brier,1e-12)
    check('all early/price probability scores',good,rows=len(scores))
    metrics=csv('results/early_price/policy_metrics.csv');good=True
    for zone,g in metrics.groupby('zone'):
        masks=pd.read_parquet(ROOT/'results/early_price'/f'review_masks_{zone}.parquet');f=load_stream(zone).reindex(masks.index);proc=process_masks(f)
        for r in g.itertuples():
            key=r.model+'__'+r.rule
            if key not in masks:
                if r.model!='raw_price_rank':good=False
                continue
            out=summary(f,masks[key].to_numpy(bool),proc)
            good &= all(out[c]==getattr(r,c) for c in ('N','hits','false','C','processes'))
    check('all early/price stored decision metrics',good,rows=len(metrics))
    availability=csv('results/early_price/price_availability.csv')
    for zone in ('DK1','DK2'):
        e=pd.read_parquet(ROOT/'results/early_price'/f'price_exposure_{zone}.parquet')
        idx=e.index;issue=pd.DatetimeIndex(e.issue_time)
        cutoff=(issue.tz_convert('Europe/Copenhagen').normalize()+pd.DateOffset(days=1)).date
        eligible=np.asarray(idx.tz_convert('Europe/Copenhagen').date)<=cutoff
        check(zone+' local auction-day price eligibility',np.array_equal(eligible,e.eligible_under_local_auction_day) and not e.loc[~eligible,'price_exposed'].notna().any())
    check('price publication availability remains an assumption',bool(availability.first_release_availability.str.contains('assumed').all()))

def artifacts():
    stems=['figure1_native_allocation','figure2_workload_and_cost','figure3_decision_stability','figure4_simulation_mechanisms']
    from pypdf import PdfReader
    from PIL import Image
    for stem in stems:
        base=ROOT/'results/figures'/stem
        good=all(base.with_suffix('.'+ext).exists() for ext in ('pdf','svg','eps','png','tiff'))
        page=PdfReader(base.with_suffix('.pdf')).pages[0]
        size=[float(page.mediabox.width)/72*25.4,float(page.mediabox.height)/72*25.4]
        im=Image.open(base.with_suffix('.png'));ti=Image.open(base.with_suffix('.tiff'))
        fonts=[v.get_object() for v in page['/Resources']['/Font'].values()]
        tt=all(f['/Subtype']=='/Type0' and all(c.get_object()['/Subtype']=='/CIDFontType2' for c in f['/DescendantFonts']) for f in fonts)
        max_height=130 if stem.startswith(('figure3','figure4')) else 247
        check(stem+' vector/raster export',good and tt and size[0]<=183 and size[1]<=max_height and min(im.info['dpi'])>=1199 and ti.info.get('compression')=='tiff_lzw',pdf_mm=size,embedded_TrueType=tt,maximum_height_mm=max_height)
    ledger=ROOT/'audit/RESULT_LEDGER.csv.gz'
    if ledger.exists():
        a=pd.read_csv(ledger,usecols=['source'])
        check('formal ledger excludes debug simulation',not bool(a.source.str.contains('wide_boundary_initial').any()))

def main():
    started=time.perf_counter()
    for func in (inventory_and_clocks,fixed_policies,scores_and_numerics,stability,simulations,early_and_price,artifacts):
        print('Checking:',func.__name__,flush=True)
        try:func()
        except Exception as e:check(func.__name__+' completed',False,error=f'{type(e).__name__}: {e}')
    versions={}
    for name in ('numpy','pandas','scipy','scikit-learn','pyarrow','pypdf','Pillow'):
        try:versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:versions[name]=None
    manifest=[]
    excluded_names={'finish_manuscript.py','get_development_prices.py','package_revision.py','warning_reference.py','ARTICLE_BUILD_SPEC.md'}
    excluded_parts={'__pycache__','.git','wide_boundary_initial','temp','cache'}
    for directory in ('scripts','datasets','results','audit','manuscript'):
        for p in sorted((ROOT/directory).rglob('*')):
            if not p.is_file() or any(x in excluded_parts for x in p.parts):continue
            if p.name in excluded_names or p.suffix in {'.pyc','.zip'}:continue
            if p.name.endswith('_qa.json') or '_pages_' in p.name or p.name.startswith('document_build_'):continue
            # A report cannot meaningfully embed its own final checksum.
            if p==ROOT/'audit/reproducibility_checks.json':continue
            manifest.append({'path':p.relative_to(ROOT).as_posix(),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
    for name in ('REPRODUCIBILITY.md','requirements_revision.txt'):
        p=ROOT/name
        if p.is_file():manifest.append({'path':name,'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
    report={'package_version':VERSION,'status':'passed' if all(c['ok'] for c in CHECKS) else 'failed',
        'checks':CHECKS,'checks_count':len(CHECKS),'failed_checks_count':sum(not c['ok'] for c in CHECKS),
        'input_inventory':INPUTS,'formal_scientific_manifest':manifest,'excluded_debug_directory':'results/simulations/wide_boundary_initial',
        'manifest_scope':'Scientific source/results/audit and reproducibility instructions included in the formal package; submission artifacts receive the final package manifest',
        'manifest_excluded_names':sorted(excluded_names),
        'manifest_excluded_patterns':['*_qa.json','*_pages_*','document_build_*','*.zip','*.pyc','this report itself'],
        'python':platform.python_version(),'versions':versions,'elapsed_seconds':time.perf_counter()-started,
        'scope':'Fixed archived forecast evaluation, new early/price classifier results, simulations and scientific artifact identities; no original marginal refit, provider-vintage recovery or verification of the price-release assumption.',
        'temporal_scope_note':'Primary fixed policies are planned over all 2022–25 forecasts before test scoring. The early/price evaluation currently starts its secondary cooldown variants at the test boundary; no full-history warm-start claim is made for that separate stage.'}
    path=ROOT/'audit/reproducibility_checks.json';path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(clean(report),indent=2,ensure_ascii=False),encoding='utf8')
    print(report['status'],len(CHECKS),'checks,',report['failed_checks_count'],'failed,',round(report['elapsed_seconds'],2),'seconds',flush=True)
    raise SystemExit(0 if report['status']=='passed' else 1)

if __name__=='__main__':main()
