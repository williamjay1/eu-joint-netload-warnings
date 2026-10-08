"""Independent numerical convergence checks of frozen same-R copula forecasts.

No parameters are refit.  SciPy's randomized CDF and deterministic path
integration are distinct numerical implementations; replicate spread is not a
rigorous integration confidence interval. All outputs belong to this revision.
"""
from __future__ import annotations
import hashlib, itertools, json, os, sys, time
from pathlib import Path
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal, multivariate_t, norm, t

ROOT = Path(__file__).resolve().parents[1]
LEGACY = Path(__file__).parent / 'legacy_copula'
sys.path.insert(0, str(LEGACY))
from elliptical_event_fast import fast_event_probabilities
ZONES = ('DE_LU', 'FR', 'BE')
OUT = ROOT / 'results' / 'numerical'

def safe(x):
    if isinstance(x, dict): return {str(k): safe(v) for k, v in x.items()}
    if isinstance(x, (tuple, list)): return [safe(v) for v in x]
    if isinstance(x, (np.integer, np.floating, np.bool_)): return x.item()
    if isinstance(x, Path): return str(x)
    if isinstance(x, pd.Timestamp): return x.isoformat()
    return x

def save_json(path, value):
    path.write_text(json.dumps(safe(value), indent=2, ensure_ascii=False), encoding='utf8')

def matrices(frame):
    R=np.broadcast_to(np.eye(3),(len(frame),3,3)).copy()
    for i,j,c in ((0,1,'rho_01'),(0,2,'rho_02'),(1,2,'rho_12')):
        R[:,i,j]=R[:,j,i]=frame[c].to_numpy(float)
    return R

def values(frame, q, margin=0):
    return np.clip(frame[[f'v{margin}_{q}_{z}' for z in ZONES]].to_numpy(float),1e-12,1-1e-12)

def event_probability(u, R, df=np.inf, tolerance=1e-9):
    return fast_event_probabilities(np.clip(u,1e-12,1-1e-12),R,df,tolerance=tolerance,batch_size=512)['ge2']

def integration_diagnostics(u,R,df=np.inf,tolerance=1e-11):
    result=fast_event_probabilities(np.clip(u,1e-12,1-1e-12),R,df,tolerance=tolerance,batch_size=512)
    return {'roundoff_corrected_hours':int(result['roundoff_corrected'].sum()),
        'maximum_reported_convergence_proxy':float(np.max(result['numerical_error'])),
        'maximum_legendre_order':int(np.max(result['integration_order'])),
        'maximum_normalised_count_sum_discrepancy':float(np.max(abs(result['count_probabilities'].sum(axis=1)-1))),
        'normalisation':'Every count vector is normalised after clipping within-tolerance cancellation; correction flags identify pre-normalisation counts outside [0,1].'}

def complete_mask(frame, q='q90', years=(2024,2025)):
    fields=['y_'+q]+[f'M{m}_{q}' for m in range(4)]
    okay=frame[fields].notna().all(axis=1)
    days=okay.groupby(frame.index.normalize()).sum().eq(24)
    return okay & frame.index.normalize().isin(days.index[days]) & frame.index.year.isin(years)

def plan(index, probability, gate=.25, cooldown=0, rise=False, k=2):
    p=np.asarray(probability,float)
    prior=pd.Series(p,index=index).shift(1).rolling(6,min_periods=1).max().fillna(0).to_numpy()
    score=p*(1-np.clip(prior,0,1)) if rise else p
    ns=index.as_unit('ns').asi8
    selected=[]; output=np.zeros(len(p),bool); gap=int(cooldown*3600*1e9)
    for _, ix in pd.Series(np.arange(len(p)),index=index).groupby(index.normalize()):
        ii=ix.to_numpy()
        if len(ii)!=24 or not np.isfinite(p[ii]).all(): continue
        candidates=ii[p[ii]>=gate]
        order=np.lexsort((ns[candidates],-p[candidates],-score[candidates]))
        picked=[]; recent=[v for v in selected if v>=ns[ii[0]]-gap]
        for at in candidates[order]:
            if cooldown and any(abs(ns[at]-v)<gap for v in recent+picked): continue
            output[at]=True; picked.append(ns[at])
            if len(picked)==k: break
        selected.extend(picked)
    return output

def scipy_probability(u,R,df,seed,maxpts=524288):
    # Symmetry transforms upper-tail intersections into lower-tail CDFs.
    a=-(norm.ppf(u) if np.isinf(df) else t.ppf(u,df))
    probabilities=[]
    for n, pair in enumerate(((0,1),(0,2),(1,2),(0,1,2))):
        j=list(pair); C=R[np.ix_(j,j)]
        rng=np.random.default_rng(seed+1009*n)
        if np.isinf(df):
            value=multivariate_normal.cdf(a[j],cov=C,maxpts=maxpts,abseps=1e-9,releps=1e-9,rng=rng)
        else:
            value=multivariate_t.cdf(a[j],shape=C,df=df,maxpts=maxpts,random_state=rng)
        probabilities.append(float(value))
    return sum(probabilities[:3])-2*probabilities[3]

def sample_cases(frame,q):
    p=frame['M0_'+q].to_numpy(float);u=values(frame,q)
    rho=frame[['rho_01','rho_02','rho_12']].mean(axis=1).to_numpy(float)
    candidates={
        'near_gate':np.flatnonzero(np.abs(p-.25)<=.02),
        'conditional_tail':np.flatnonzero(np.min(u,axis=1)>=.95),
        'conditional_middle':np.flatnonzero((np.min(u,axis=1)>=.2)&(np.max(u,axis=1)<=.85)),
        'highest_rho':np.argsort(rho)[-150:],
        'lowest_rho':np.argsort(rho)[:150],
        'probability_middle':np.flatnonzero((p>=.1)&(p<=.5)),
    }
    rng=np.random.default_rng(202610081); rows=[]
    for label,ix in candidates.items():
        if len(ix):
            for j in rng.choice(ix,size=min(4,len(ix)),replace=False): rows.append((int(j),label))
    return rows

def main():
    begin=time.perf_counter(); OUT.mkdir(parents=True,exist_ok=True)
    path=ROOT/'datasets/marginal_predictions.parquet'; full=pd.read_parquet(path)
    convergence=[]; samples=[]; revised=full[[f'M{m}_{q}' for q in ('q90','q95') for m in range(4)]].copy()
    # Full test-domain q90 check against stored values at tighter tolerance.
    score=full.loc[complete_mask(full,'q90')];R=matrices(score)
    for margin,df,model in ((0,np.inf,'M0'),(1,np.inf,'M1'),(0,3.,'M2'),(1,3.,'M3')):
        u=values(score,'q90',margin)
        tight=event_probability(u,R,df,1e-9)
        tighter=event_probability(u,R,df,1e-11)
        stored=score[model+'_q90'].to_numpy(float);y=score.y_q90.to_numpy(float)
        revised.loc[score.index,model+'_q90']=tighter
        convergence.append({'model':model,'q':'q90','hours':len(score),
            'mean_abs_tight_stored':float(np.mean(abs(tight-stored))),
            'max_abs_tight_stored':float(np.max(abs(tight-stored))),
            'mean_abs_tighter_tight':float(np.mean(abs(tighter-tight))),
            'max_abs_tighter_tight':float(np.max(abs(tighter-tight))),
            'stored_Brier':float(np.mean((stored-y)**2)),
            'tight_Brier':float(np.mean((tight-y)**2)),
            'tighter_Brier':float(np.mean((tighter-y)**2)),
            **integration_diagnostics(u,R,df,1e-11),
            'gate_025_crossings_tighter_vs_stored':int(((stored>=.25)!=(tighter>=.25)).sum())})
        print('full numerical',model,convergence[-1],flush=True)
    # Numerical changes of issued lists, preserve all earlier forecast context.
    context=full.loc[full.index.year>=2022]
    mask=complete_mask(context)
    plans=[]
    for m in range(4):
        old=context[f'M{m}_q90'].to_numpy(float)
        new=revised.loc[context.index,f'M{m}_q90'].to_numpy(float)
        for rule,gate,cool,rise in [('gate',.25,0,False),('priority',.19,6,True)]:
            a=plan(context.index,old,gate,cool,rise);b=plan(context.index,new,gate,cool,rise)
            plans.append({'model':f'M{m}','rule':rule,'symmetric_difference_test':int(((a!=b)&mask.to_numpy()).sum())})
    for q in ('q90','q95'):
        f=full.loc[complete_mask(full,q)];R=matrices(f);u=values(f,q)
        for ix,stratum in sample_cases(f,q):
            for df in (np.inf,3.):
                deterministic=float(event_probability(u[ix:ix+1],R[ix:ix+1],df,1e-11)[0])
                refs=[scipy_probability(u[ix],R[ix],df,202610080+s*65537) for s in range(3)]
                mean=float(np.mean(refs));sd=float(np.std(refs,ddof=1));y=float(f.y_q90.iloc[ix]) if q=='q90' else float(f.y_q95.iloc[ix])
                samples.append({'target_time':f.index[ix],'q':q,'stratum':stratum,'df':str(df),
                    'weather_cell':str(f.cell.iloc[ix]),'minimum_R_eigenvalue':float(np.linalg.eigvalsh(R[ix]).min()),
                    'deterministic':deterministic,'scipy_mean':mean,'scipy_replicate_sd':sd,
                    'abs_difference':abs(mean-deterministic),'max_replicate_abs_difference':max(abs(x-deterministic) for x in refs),
                    'abs_sample_Brier_difference':abs((mean-y)**2-(deterministic-y)**2),
                    'u_min':float(u[ix].min()),'u_max':float(u[ix].max()),'rho_mean':float(np.mean(R[ix][np.triu_indices(3,1)])),
                    **{f'scipy_seed_{j}':v for j,v in enumerate(refs)}})
        print('independent CDF',q,len(samples),flush=True)
    pd.DataFrame(convergence).to_csv(OUT/'numerical_full_convergence.csv',index=False)
    pd.DataFrame(samples).to_csv(OUT/'numerical_independent_cdf.csv',index=False)
    pd.DataFrame(plans).to_csv(OUT/'numerical_plan_comparison.csv',index=False)
    tight_bs={r['model']:r['tighter_Brier'] for r in convergence};old_bs={r['model']:r['stored_Brier'] for r in convergence}
    s=pd.DataFrame(samples)
    report={'status':'completed','source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
        'full_q90_hours':len(score),'tolerances':[1e-9,1e-11],'randomized_reference_maxpts':524288,'randomized_reference_seeds':3,
        'reference_cases':len(samples),'maximum_reference_mean_difference':float(s.abs_difference.max()),
        'mean_reference_mean_difference':float(s.abs_difference.mean()),'maximum_reference_replicate_sd':float(s.scipy_replicate_sd.max()),
        'diagnostics':[{k:r[k] for k in ('model','roundoff_corrected_hours','maximum_reported_convergence_proxy',
            'maximum_legendre_order','maximum_normalised_count_sum_discrepancy')} for r in convergence],
        'reference_weather_cells':sorted(s.weather_cell.unique().tolist()),
        'reference_minimum_R_eigenvalue_range':[float(s.minimum_R_eigenvalue.min()),float(s.minimum_R_eigenvalue.max())],
        'stored_M1_minus_M3_Brier':old_bs['M1']-old_bs['M3'],
        'tight_M1_minus_M3_Brier':tight_bs['M1']-tight_bs['M3'],
        'change_in_M1_minus_M3_Brier':(tight_bs['M1']-tight_bs['M3'])-(old_bs['M1']-old_bs['M3']),
        'maximum_Brier_change_under_tightening':max(abs(r['stored_Brier']-r['tighter_Brier']) for r in convergence),
        'list_comparisons':plans,'seconds':time.perf_counter()-begin,
        'interpretation':'Independent numerical validation of frozen three-dimensional same-R calculations; not a mathematical error certificate or proof of statistical equivalence.'}
    save_json(OUT/'numerical_summary.json',report)
    print(json.dumps(safe(report)),flush=True)

if __name__=='__main__': main()
