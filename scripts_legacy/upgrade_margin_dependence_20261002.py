"""Controlled margin/dependence reanalysis; never rewrites the original study.

All choices are fixed in this file before examining its 2024--2025 outputs.
The repeated 2024--2025 evaluation is explicitly a reanalysis, not a new test.
R(X) is fitted to the original issued PITs and held identical in all four cells.
The extra margin map is a monotone cell-conditional Beta CDF, updated weekly
from previously issued PITs with a seven-day-before-issue label cutoff.
"""
from __future__ import annotations
from release_paths import work_path as _work_path, raw_path as _raw_path
import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import betaln, digamma
from scipy.stats import beta

from conditional_features import ConditionalWeatherBasis
from dependence import DynamicGaussianCopula, IntegrationSettings, elliptical_event_probabilities

ROOT = _work_path()
OUT = ROOT/'results/upgrade_margin_dependence_20261002'
ZONES = ('DE_LU', 'FR', 'BE')
QNAMES = ('q90', 'q95')
NU = (3., 5., 8., 15., 30., np.inf)
VERSION = '1.0-fixed-grid-weekly-beta-shared-R-event-selection'


def safe(value):
    if isinstance(value, dict): return {str(k):safe(v) for k,v in value.items()}
    if isinstance(value, (list, tuple)): return [safe(v) for v in value]
    if isinstance(value, np.generic): value = value.item()
    if isinstance(value, (pd.Timestamp, Path)): return str(value)
    if isinstance(value, float) and not np.isfinite(value): return 'inf' if value > 0 else None
    return value


def write_json(path, value):
    path.write_text(json.dumps(safe(value), ensure_ascii=False, indent=2), encoding='utf-8')


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(2**20), b''): digest.update(block)
    return digest.hexdigest()


def inputs():
    files={
        'shared': ROOT/'results/shared_frozen_history_test_20261001/prequential_combined_shared_margins.parquet',
        'weather': ROOT/'datasets/joint_auxiliary_frozen_20261001/joint_features_lean.parquet',
        'timing': ROOT/'datasets/joint_auxiliary_frozen_20261001/timing.parquet',
        'availability': ROOT/'datasets/joint_auxiliary_frozen_20261001/label_availability.parquet',
        'old_predictions': ROOT/'results/joint_runner_frozen_test_numerical_20261001/joint_runner_predictions.parquet',
        'old_manifest': ROOT/'results/joint_runner_frozen_test_numerical_20261001/joint_runner_manifest.json',
    }
    data={k:pd.read_parquet(p) for k,p in files.items() if p.suffix=='.parquet'}
    data['settings']=json.loads(files['old_manifest'].read_text(encoding='utf-8'))['settings']
    index=data['shared'].index
    assert all(data[k].index.equals(index) for k in ('weather', 'timing', 'availability'))
    return data, files


def fixed_grid(weather):
    """No net-load or event outcomes determine these six forecast-only cells."""
    idx=weather.index
    season=np.where(np.isin(idx.month, (10,11,12,1,2,3)), 'cool', 'warm')
    reference=(idx >= pd.Timestamp('2021-01-01',tz='UTC'))&(idx<pd.Timestamp('2022-01-01',tz='UTC'))
    cell=np.full(len(idx), None, dtype=object)
    cuts={}
    for label in ('cool','warm'):
        past=weather.loc[reference&(season==label), ['common_temperature','common_land_wind']].dropna()
        wind=float(past.common_land_wind.quantile(.5))
        temp=float(past.common_temperature.quantile(.25))
        cuts[label]={'land_wind_median_2021':wind,'temperature_q25_2021':temp,'reference_rows':len(past)}
        valid=(season==label)&np.isfinite(weather[['common_temperature','common_land_wind']].to_numpy()).all(axis=1)
        low=weather.common_land_wind.to_numpy()<=wind
        cold=weather.common_temperature.to_numpy()<=temp
        cell[valid & low & cold]=label+'_low_wind_cold'
        cell[valid & low & ~cold]=label+'_low_wind_other'
        cell[valid & ~low]=label+'_higher_wind'
    return pd.Series(cell,index=idx,name='cell'), cuts


def fit_beta_daily_equal(pits, index):
    """Shrink towards identity with fixed four-day strength; days equal weight.

    Clipping only affects fitting logs. The output CDF remains monotonic. Daily
    equal weights avoid a day with 24 rows dominating an incomplete day. This
    does not make days independent; uncertainty is assessed with time blocks.
    """
    valid=np.isfinite(pits)
    x=np.clip(np.asarray(pits)[valid],1e-6,1-1e-6)
    d=pd.DatetimeIndex(index)[valid].normalize()
    frame=pd.DataFrame({'logu':np.log(x),'log1mu':np.log1p(-x)},index=d).groupby(level=0).mean()
    days=len(frame)
    if days<28 or len(x)<168:
        return np.array([1.,1.]), {'days':days,'hours':len(x),'mode':'identity_sparse_cell'}
    lu,lv=frame.mean().to_numpy()
    def objective(theta):
        a,b=np.exp(theta)
        value=-days*((a-1)*lu+(b-1)*lv-betaln(a,b))+2*np.sum(theta**2)
        gradient=-days*np.array([a*(lu-digamma(a)+digamma(a+b)),b*(lv-digamma(b)+digamma(a+b))])+4*theta
        return value,gradient
    fit=minimize(objective, np.zeros(2), jac=True, method='L-BFGS-B', bounds=((-3,3),(-3,3)))
    if not fit.success: raise RuntimeError('Beta map fit unresolved: '+fit.message)
    shape=np.exp(fit.x)
    return shape, {'days':days,'hours':len(x),'mode':'penalized_beta_daily_equal','shape_a':shape[0],'shape_b':shape[1]}


def calibration(data, cell):
    shared=data['shared']; idx=shared.index
    out=pd.DataFrame(index=idx)
    out.index.name='target_time'
    out['cell']=cell
    out['stratum']=cell
    out['issue_time']=shared.issue_time
    out['feature_complete']=shared.feature_complete
    out['threshold_available']=shared.threshold_available
    out['period']=np.where(idx.year<2022,'prior_history',np.where(idx.year<=2023,'development','reanalysis'))
    for zone in ZONES:
        out['pit_'+zone]=shared['pit_'+zone]
        out['raw_pit_'+zone]=shared['raw_pit_'+zone]
        out['calibrated_pit_'+zone]=np.nan
        for q in QNAMES:
            out[f'v0_{q}_{zone}']=shared[f'cdf_{q}_{zone}']
            out[f'v1_{q}_{zone}']=np.nan
            out[f'threshold_{q}_{zone}']=shared[f'threshold_{q}_{zone}']
            out[f'net_load_{zone}']=shared[f'net_load_{zone}']
            values=shared[f'net_load_{zone}'].to_numpy()
            threshold=shared[f'threshold_{q}_{zone}'].to_numpy()
            out[f'exceed_{q}_{zone}']=np.where(np.isfinite(values)&np.isfinite(threshold),(values>threshold).astype(float),np.nan)
    for q in QNAMES:
        y=out[[f'exceed_{q}_{z}' for z in ZONES]].to_numpy()
        out['y_'+q]=np.where(np.isfinite(y).all(axis=1),(y.sum(axis=1)>=2).astype(float),np.nan)
    records=[]
    # Weekly Monday origin, including the Monday before 2022 begins. No future
    # data determine the map; row-level issue clock is checked against cutoff.
    starts=pd.date_range('2021-12-27','2025-12-29',freq='7D',tz='UTC')
    for start in starts:
        end=start+pd.Timedelta(days=7)
        query=(idx>=max(start,pd.Timestamp('2022-01-01',tz='UTC')))&(idx<end)
        if not query.any(): continue
        first_issue=out.loc[query,'issue_time'].min()
        cutoff=first_issue-pd.Timedelta(days=7)
        past=(idx+pd.Timedelta(hours=1)<=cutoff)&(idx>=cutoff-pd.Timedelta(days=365))
        available=data['availability'].label_available_time.to_numpy()
        past &= pd.DatetimeIndex(available)<=cutoff
        record={'target_week_start':start,'first_issue':first_issue,'label_cutoff':cutoff,'maps':{}}
        if first_issue-cutoff<pd.Timedelta(days=7): raise AssertionError('Label embargo violation')
        for c in sorted(cell.dropna().unique()):
            query_cell=query&(cell.to_numpy()==c)
            if not query_cell.any(): continue
            train=past&(cell.to_numpy()==c)
            for zone in ZONES:
                shape, meta=fit_beta_daily_equal(shared.loc[train,'pit_'+zone].to_numpy(),idx[train])
                meta['last_target_interval_end']=idx[train][-1]+pd.Timedelta(hours=1) if train.any() else None
                record['maps'][c+'_'+zone]=meta
                p=shared.loc[query_cell,'pit_'+zone].to_numpy()
                out.loc[query_cell,'calibrated_pit_'+zone]=beta.cdf(p,*shape)
                for q in QNAMES:
                    v=out.loc[query_cell,f'v0_{q}_{zone}'].to_numpy()
                    out.loc[query_cell,f'v1_{q}_{zone}']=beta.cdf(v,*shape)
        records.append(record)
    return out, records


def df_name(nu): return 'inf' if np.isinf(nu) else str(int(nu))


def quarter_probabilities(data, out, quarter, manifest):
    idx=data['shared'].index; settings=data['settings']; shared=data['shared']; weather=data['weather']
    cutoff=quarter-pd.Timedelta(days=7)
    begin=max(pd.Timestamp('2021-01-01',tz='UTC'),quarter-pd.DateOffset(years=2))
    U=shared[['pit_'+z for z in ZONES]].to_numpy()
    features=weather.to_numpy()
    avail=pd.DatetimeIndex(data['availability'].label_available_time)
    history=(idx>=begin)&(idx+pd.Timedelta(hours=1)<=cutoff)&(avail<=cutoff)&np.isfinite(U).all(axis=1)&np.isfinite(features).all(axis=1)
    basis=ConditionalWeatherBasis(linear_columns=tuple(settings['linear_columns']),smooth_columns=tuple(settings['smooth_columns']),interactions=tuple(tuple(v) for v in settings['interactions']),n_knots=settings['n_knots'],allow_missing=False,minimum_release_delay_hours=12.)
    xt=basis.fit_transform(weather.loc[history],timing_manifest=data['timing'].loc[history].reset_index(),fit_cutoff=cutoff).to_numpy()
    integration=IntegrationSettings(tolerance=1e-6,initial_maxpts=16384,max_maxpts=4194304,backend='elliptical_path')
    tick=time.perf_counter()
    model=DynamicGaussianCopula(l2=settings['l2'],maxiter=settings['maxiter'],integration_settings=integration).fit(U[history],xt)
    fit_seconds=time.perf_counter()-tick
    query=(idx>=quarter)&(idx<quarter+pd.DateOffset(months=3))&np.isfinite(features).all(axis=1)&shared.feature_complete.to_numpy(bool)&shared.threshold_available.to_numpy(bool)
    xq=basis.transform(weather.loc[query],timing_manifest=data['timing'].loc[query].reset_index()).to_numpy()
    R=model.correlation_matrices(xq)
    for col,i,j in (('rho_01',0,1),('rho_02',0,2),('rho_12',1,2)):
        out.loc[query,col]=R[:,i,j]
    out.loc[query,'copula_origin']=quarter
    diagnostics=[]
    for q in QNAMES:
        for margin in (0,1):
            V=out.loc[query,[f'v{margin}_{q}_{z}' for z in ZONES]].to_numpy()
            eligible=np.isfinite(V).all(axis=1)
            if not eligible.all(): raise ValueError('Available query has missing marginal threshold CDF')
            # Machine saturation in beta.cdf is regularised at 1e-12 only.
            # Coupling probability perturbation is <= 3e-12, not a calibration.
            V=np.clip(V,1e-12,1-1e-12)
            for nu in NU:
                tick=time.perf_counter()
                result=elliptical_event_probabilities(V,R,nu,settings=integration)
                sec=time.perf_counter()-tick
                name=f'candidate_v{margin}_nu{df_name(nu)}_{q}'
                out.loc[query,name]=result['ge2']
                diagnostics.append({'q':q,'margin':margin,'nu':df_name(nu),'seconds':sec,'rows':len(V),'max_numerical_discrepancy':float(np.max(result['numerical_error']))})
                if len(manifest['benchmark_first_10_calculations'])<10:
                    manifest['benchmark_first_10_calculations'].append({'quarter':quarter,**diagnostics[-1]})
    old=data['old_predictions']
    baseline_check={}
    for q in QNAMES:
        a=out.loc[query,f'candidate_v0_nuinf_{q}']
        b=old.reindex(a.index)['C3_dynamic_gaussian_'+q]
        err=float(np.nanmax(np.abs(a.to_numpy()-b.to_numpy())))
        baseline_check[q]=err
        if err>1e-7: raise AssertionError('Original C3 not reproduced: '+str(err))
    record={'quarter':quarter,'history_begin':begin,'cutoff':cutoff,'first_issue':quarter-pd.Timedelta(hours=12),'history_rows':int(history.sum()),'query_rows':int(query.sum()),'fit_seconds':fit_seconds,'fit_info':model.fit_info_,'original_C3_max_abs_probability_difference':baseline_check,'calculations':diagnostics}
    np.savez_compressed(OUT/f'R_{quarter.year}Q{quarter.quarter}.npz',target_time=idx[query].asi8,target_time_unit=np.array(idx.dtype.unit),R=R,coef=model._correlation.coef_)
    manifest['quarters'].append(record)
    write_json(OUT/'manifest.json',manifest)
    print(json.dumps(safe({'completed_quarter':quarter,'fit_seconds':fit_seconds,'probability_seconds':sum(v['seconds'] for v in diagnostics),'baseline_difference':baseline_check})),flush=True)


def complete_day_mask(frame, columns):
    good=np.isfinite(frame[columns].to_numpy(dtype=float)).all(axis=1)
    counts=pd.Series(good,index=frame.index).groupby(frame.index.normalize()).sum()
    return good&frame.index.normalize().isin(counts.index[counts==24])


def select_and_matrix(out):
    idx=out.index
    selection=[]
    # Primary q90 selects one nu by equal average of base and calibrated
    # marginal Brier scores. Same nu is reused unchanged for secondary q95.
    for quarter in pd.date_range('2022-01-01','2025-10-01',freq='QS',tz='UTC'):
        # Explicit seven-day waiting period is measured before first issue,
        # twelve hours before quarter origin, rather than before target time.
        history_end=min(quarter-pd.Timedelta(days=7,hours=12),pd.Timestamp('2024-01-01',tz='UTC')-pd.Timedelta(days=7,hours=12))
        allcolumns=['y_q90']+[f'candidate_v{m}_nu{df_name(nu)}_q90' for m in (0,1) for nu in NU]
        good=complete_day_mask(out,allcolumns)
        history=(idx>=pd.Timestamp('2022-01-01',tz='UTC'))&(idx+pd.Timedelta(hours=1)<=history_end)&good
        scores=[]
        for nu in NU:
            values=[]
            for m in (0,1):
                p=out.loc[history,f'candidate_v{m}_nu{df_name(nu)}_q90'].to_numpy()
                y=out.loc[history,'y_q90'].to_numpy()
                values.append(float(np.mean((p-y)**2)) if len(p) else np.nan)
            scores.append({'nu':df_name(nu),'mean_two_margin_Brier':float(np.mean(values)) if len(p) else np.nan,'base_Brier':values[0],'conditional_Brier':values[1]})
        # No usable new-development quarter defaults to the Gaussian limit.
        # Exact ties go to the simpler/larger nu, a declared ordering rule.
        selected=np.inf if history.sum()<168 else min(NU,key=lambda n:(next(s['mean_two_margin_Brier'] for s in scores if s['nu']==df_name(n)), -n))
        query=(idx>=quarter)&(idx<quarter+pd.DateOffset(months=3))
        out.loc[query,'df_event_selected']=df_name(selected)
        for q in QNAMES:
            out.loc[query,'M0_'+q]=out.loc[query,'candidate_v0_nuinf_'+q]
            out.loc[query,'M1_'+q]=out.loc[query,'candidate_v1_nuinf_'+q]
            out.loc[query,'M2_'+q]=out.loc[query,f'candidate_v0_nu{df_name(selected)}_{q}']
            out.loc[query,'M3_'+q]=out.loc[query,f'candidate_v1_nu{df_name(selected)}_{q}']
        selection.append({'quarter':quarter,'selection_start':'2022-01-01','last_label_interval_end':idx[history][-1]+pd.Timedelta(hours=1) if history.any() else None,'selection_cutoff':history_end,'rows':int(history.sum()),'complete_days':int(history.sum()/24),'nu':df_name(selected),'selection_average_of_two_margin_q90_Brier':scores,'scope':'rolling_chronological_development_selection' if quarter.year<=2023 else 'fixed_full_development_selection_reanalysis'})
    return selection


def block_summary(out, draws=2000):
    summaries=[]; contrasts=[]
    rng=np.random.default_rng(61002)
    for period, years in (('development',(2022,2023)),('reanalysis',(2024,2025))):
        for q in QNAMES:
            columns=['y_'+q]+['M'+str(m)+'_'+q for m in range(4)]
            good=complete_day_mask(out,columns)&out.index.year.isin(years)
            frame=out.loc[good,columns]
            y=frame['y_'+q].to_numpy()
            p=frame.iloc[:,1:].to_numpy()
            loss=(p-y[:,None])**2
            days=frame.index.normalize().unique()
            calendar=pd.date_range(str(years[0])+'-01-01',str(years[-1])+'-12-31',freq='D',tz='UTC')
            daily_sum=pd.DataFrame(loss,index=frame.index,columns=['M0','M1','M2','M3']).groupby(frame.index.normalize()).sum().reindex(calendar,fill_value=0)
            daily_hours=pd.Series(1.,index=frame.index).groupby(frame.index.normalize()).sum().reindex(calendar,fill_value=0).to_numpy()
            for m in range(4):
                summaries.append({'period':period,'q':q,'model':'M'+str(m),'hours':len(frame),'days':len(days),'event_hours':int(y.sum()),'Brier':float(loss[:,m].mean())})
            # Loss differences rather than unique causal attribution.
            operators={'margin_at_G':np.array([-1,1,0,0]),'dependence_at_base':np.array([-1,0,1,0]),'margin_at_selected_dependence':np.array([0,0,-1,1]),'dependence_at_calibrated':np.array([0,-1,0,1]),'interaction':np.array([1,-1,-1,1]),'combined':np.array([-1,0,0,1])}
            for block in (7,14,28):
                n=len(calendar); starts=np.arange(n)
                offsets=np.arange(block)
                count=int(np.ceil(n/block))
                sample=(rng.choice(starts,size=(draws,count))[:,:,None]+offsets)%n
                sample=sample.reshape(draws,-1)[:,:n]
                # Missing whole days retain their calendar positions. Blocks
                # therefore span elapsed days, not compressed available days.
                means=daily_sum.to_numpy()[sample].sum(axis=1)/daily_hours[sample].sum(axis=1)[:,None]
                for name,vec in operators.items():
                    draws_value=means@vec
                    contrasts.append({'period':period,'q':q,'contrast':name,'block_days':block,'loss_difference':float(loss.mean(axis=0)@vec),'ci_low':float(np.quantile(draws_value,.025)),'ci_high':float(np.quantile(draws_value,.975)),'draws':draws,'calendar_days':n,'common_days':len(days)})
    pd.DataFrame(summaries).to_csv(OUT/'matrix_scores.csv',index=False)
    pd.DataFrame(contrasts).to_csv(OUT/'matrix_contrasts.csv',index=False)
    return summaries,contrasts


def self_check():
    idx=pd.date_range('2021-01-01',periods=40*24,freq='h',tz='UTC')
    p=np.linspace(.001,.999,len(idx))
    shape,meta=fit_beta_daily_equal(p,idx)
    grid=np.linspace(0,1,101)
    transformed=beta.cdf(grid,*shape)
    assert transformed[0]==0 and transformed[-1]==1 and np.all(np.diff(transformed)>=0)
    assert fit_beta_daily_equal(p[:24],idx[:24])[1]['mode']=='identity_sparse_cell'
    assert np.allclose(fit_beta_daily_equal(np.linspace(.001,.999,len(idx)),idx)[0],[1,1],atol=.03)
    V=np.array([[.9,.9,.9],[.2,.4,.6]])
    R=np.tile(np.eye(3),(2,1,1))
    p0=elliptical_event_probabilities(V,R,settings=IntegrationSettings(backend='elliptical_path',tolerance=1e-6))['ge2']
    a=1-V
    exact=a[:,0]*a[:,1]+a[:,0]*a[:,2]+a[:,1]*a[:,2]-2*a.prod(axis=1)
    assert np.allclose(p0,exact,atol=1e-12)
    return {'beta_monotonic_endpoints':True,'sparse_identity':True,'uniform_identity_recovery':True,'independent_joint_formula':True}


def strict_selection_amendment():
    """Repair the release-clock boundary without reselecting design options."""
    manifest_path=OUT/'manifest.json'
    manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    old=manifest['event_df_selection']
    old_path=OUT/'upgrade_margin_selection_before_strict_clock.json'
    amendment_path=ROOT/'config/upgrade_margin_strict_selection_clock_amendment_20261002.json'
    if old_path.exists() or amendment_path.exists():raise FileExistsError('Preserve the existing clock amendment without rewriting it')
    original=pd.read_parquet(OUT/'upgrade_predictions.parquet');recomputed=original.copy()
    selection=select_and_matrix(recomputed)
    columns=[f'M{m}_{q}' for q in QNAMES for m in range(4)]
    if not np.array_equal(original[columns].to_numpy(),recomputed[columns].to_numpy(),equal_nan=True):raise AssertionError('Strict clock changes issued probability; preserve results and inspect before replacement')
    if not original['df_event_selected'].equals(recomputed['df_event_selected']):raise AssertionError('Strict clock changes a selected nu')
    audit=[]
    for previous,current in zip(old,selection):
        query=(original.index>=pd.Timestamp(current['quarter']))&(original.index<pd.Timestamp(current['quarter'])+pd.DateOffset(months=3))
        audit.append({'quarter':current['quarter'],'old_nu':previous['nu'],'strict_clock_nu':current['nu'],'old_selection_cutoff':previous['selection_cutoff'],'strict_selection_cutoff':current['selection_cutoff'],'strict_last_interval_end':current['last_label_interval_end'],'old_label_rows':previous['rows'],'strict_label_rows':current['rows'],'all_8_probability_columns_exactly_identical':True,'query_rows':int(query.sum())})
    write_json(old_path,old)
    snapshot=OUT/'upgrade_margin_source_strict_selection_clock.py'
    snapshot.write_bytes(Path(__file__).read_bytes())
    amendment={'created_utc':pd.Timestamp.now(tz='UTC'),'reason':'Quarter origin minus7days is only6.5days before the first issue at origin minus12hours. New nu selection uses first issue minus7days.','original_config_preserved':str(ROOT/'config/upgrade_margin_design_20261002.json'),'old_selection_preserved':str(old_path),'strict_selection_rule':'primary q90 equal mean Brier across two margins; same6 candidates and chronological selection; firstquarterissue minus7days labels; larger nu breaks exactties','strict_fixed_development_cutoff':'2023-12-24T12:00:00Z','design_options_changed':False,'nu_changed_any_quarter':False,'all_issued_M0_M3_q90_q95_probabilities_exactly_unchanged':True,'existing_probability_parquet_sha256':sha(OUT/'upgrade_predictions.parquet'),'existing_scores_and_CI_unchanged':True,'external_transfer_inherited_nu':3,'external_nu_same_under_strict_selection':True,'R_reproduction_clock_boundary':'Existing C3 R fits still use target quarter origin minus7days, equivalent6.5days at first issue. Retained for exact controlled baseline reproduction, explicitly not a strict7day claim.','code_snapshot':str(snapshot),'code_snapshot_sha256':sha(snapshot),'quarter_audit':audit}
    write_json(amendment_path,amendment)
    write_json(OUT/'upgrade_margin_strict_selection_clock_audit.json',amendment)
    manifest['event_df_selection']=selection
    manifest['nu_selection_clock_amendment']=str(amendment_path)
    manifest['nu_selection']='primary q90 mean across two margins; chronological past labels with firstquarterissue minus7days cutoff; unchanged candidates and tie rule'
    manifest['2024_2025_selection']='nu3 unchanged under strict7day cutoff ending2023-12-24T12Z; fixed for all eight quarters'
    manifest['R_release_clock_boundary']='Original C3 targetquarterminus7days ->6.5days before firstissue, retained for controlled reproduction'
    manifest['strict_selection_code_snapshot']=str(snapshot)
    write_json(manifest_path,manifest)
    print(json.dumps(safe({'amendment':str(amendment_path),'quarters_checked':len(audit),'all_probability_columns_exactly_unchanged':True})),flush=True)


def relative_dependence_gain(draws=2000):
    """Secondary ratio precision on exactly the original paired block draws."""
    out=pd.read_parquet(OUT/'upgrade_predictions.parquet')
    original=pd.read_csv(OUT/'matrix_contrasts.csv')
    rng=np.random.default_rng(61002);rows=[]
    # Consume the development draws in the same order as block_summary so the
    # reanalysis samples reproduce its absolute contrast intervals exactly.
    for period,years in (('development',(2022,2023)),('reanalysis',(2024,2025))):
        for q in QNAMES:
            columns=['y_'+q]+[f'M{m}_{q}' for m in range(4)]
            good=complete_day_mask(out,columns)&out.index.year.isin(years)
            frame=out.loc[good,columns];y=frame['y_'+q].to_numpy()
            loss=(frame.iloc[:,1:].to_numpy()-y[:,None])**2
            calendar=pd.date_range(str(years[0])+'-01-01',str(years[-1])+'-12-31',freq='D',tz='UTC')
            daily_sum=pd.DataFrame(loss,index=frame.index).groupby(frame.index.normalize()).sum().reindex(calendar,fill_value=0).to_numpy()
            daily_hours=pd.Series(1.,index=frame.index).groupby(frame.index.normalize()).sum().reindex(calendar,fill_value=0).to_numpy()
            n=len(calendar)
            for block in (7,14,28):
                sample=(rng.choice(np.arange(n),size=(draws,int(np.ceil(n/block))))[:,:,None]+np.arange(block))%n
                sample=sample.reshape(draws,-1)[:,:n]
                means=daily_sum[sample].sum(axis=1)/daily_hours[sample].sum(axis=1)[:,None]
                if period!='reanalysis':continue
                gain=(means[:,1]-means[:,3])/means[:,1]
                ci=np.quantile(gain,[.025,.975])
                absolute=np.quantile(means[:,3]-means[:,1],[.025,.975])
                reference=original[(original.period==period)&(original.q==q)&(original.block_days==block)&(original.contrast=='dependence_at_calibrated')].iloc[0]
                if not np.allclose(absolute,[reference.ci_low,reference.ci_high],atol=1e-14,rtol=0):raise AssertionError('Relative gain samples differ from original paired blocks')
                point=float((loss[:,1].mean()-loss[:,3].mean())/loss[:,1].mean())
                rows.append({'period':period,'q':q,'comparison':'shared-R selected t3 versus Gaussian at additional conditional margins (M3 versus M1)','block_days':block,'relative_Brier_gain':point,'ci_low':float(ci[0]),'ci_high':float(ci[1]),'gain_percent':100*point,'ci_low_percent':100*float(ci[0]),'ci_high_percent':100*float(ci[1]),'baseline_M1_Brier':float(loss[:,1].mean()),'hours':len(frame),'common_days':int(len(frame)/24),'calendar_days':n,'draws':draws,'same_absolute_contrast_draws_verified':True,'status':'secondary pointwise percentile CI with resampled denominator; not pointwise conditional mechanism attribution or universal equivalence'})
    path=OUT/'relative_dependence_gain_reanalysis.csv'
    pd.DataFrame(rows).to_csv(path,index=False)
    write_json(OUT/'relative_dependence_gain_reanalysis_scope.json',{'formula':'(BS_M1 - BS_M3)/BS_M1 recomputed within every paired elapsed-calendar draw','resampling':'exact same synchronous circular 7/14/28day draws as matrix_contrasts.csv, seed61002, missing day positions retained,2000draws','M1':'fixed extra conditional marginal map and Gaussian sameR','M3':'same marginals/sameR with nu3 selected from original development and confirmed strict7day clock','all_original_absolute_CI_exactly_reproduced':True,'primary_scores_unchanged':True,'source_probability_sha256':sha(OUT/'upgrade_predictions.parquet'),'source_contrasts_sha256':sha(OUT/'matrix_contrasts.csv'),'script_sha256':sha(__file__),'inference_scope':'precision for this selected shared-R candidate on reanalysed2024–25 forecast stream; not proof tail dependence absent or mechanism identified'})
    print(pd.DataFrame(rows)[['q','block_days','gain_percent','ci_low_percent','ci_high_percent']].to_string(index=False),flush=True)


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--stage',choices=('prepare','probabilities','finish','all','strict-selection-amendment','relative-gain'),default='all'); parser.add_argument('--quarters',type=int); args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    manifest_path=OUT/'manifest.json'
    if args.stage=='strict-selection-amendment':strict_selection_amendment();return
    if args.stage=='relative-gain':relative_dependence_gain();return
    if args.stage in ('prepare','all'):
        if (OUT/'calibration.parquet').exists(): raise FileExistsError('Preparation output already exists; use remaining stages')
        data,files=inputs(); cell,cuts=fixed_grid(data['weather']); out,records=calibration(data,cell)
        policy={'version':VERSION,'created_utc':pd.Timestamp.now(tz='UTC'),'6_cell_weather_cutpoints_from_2021_only':cuts,'grid':'cool October-March/warm April-September x low_wind_cold/low_wind_other/higher_wind','weather_cutpoints_frozen_for_2022_onward':True,'conditional_map':'monotone Beta CDF of already-calibrated issued PIT','map_update':'weekly Monday; first actual issue in week minus seven days label cutoff','map_history_days':365,'minimum_cell_days':28,'minimum_cell_hours':168,'fixed_logshape_penalty':4,'history_day_weight':'equal days; hourly mean log-PIT within day','sparse_cell':'identity; no outcome-driven adjacent pooling','R_policy':'original C3 original issued PIT quarter fits; exactly same R all four cells','nu_candidates':[df_name(n) for n in NU],'nu_selection':'primary q90 equal mean Brier loss across original and conditional-margin predictions; chronological accumulated development, weekly embargo; larger nu breaks ties','2024_2025_selection':'one development-selected nu fixed for all eight quarters; development labels end 2023-12-25 interval cutoff','q95':'same selected q90 nu, no separate secondary tuning','test_reuse':'2024-2025 already inspected; new evaluation is reanalysis','numerics':'existing elliptical path tolerance 1e-6; replicate/order discrepancies are convergence proxies, not certified exact bounds','cdf_machine_saturation_clip':1e-12,'pointwise_probability_perturbation_coupling_max':3e-12,'raw_files_modified':False}
        config=ROOT/'config/upgrade_margin_design_20261002.json'
        if config.exists(): raise FileExistsError('Upgrade configuration exists')
        write_json(config,policy)
        manifest={**policy,'status':'calibration_prepared','self_checks':self_check(),'input_hashes':{k:sha(p) for k,p in files.items()},'script_hash':sha(__file__),'weekly_calibration':records,'benchmark_first_10_calculations':[],'quarters':[]}
        out.to_parquet(OUT/'calibration.parquet'); write_json(manifest_path,manifest)
        print(json.dumps({'prepared_rows':len(out),'calibration_week_updates':len(records),'cells':safe(cuts)},ensure_ascii=False),flush=True)
    if args.stage in ('probabilities','all'):
        data,files=inputs(); manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
        cache=OUT/'candidate_probabilities.parquet'
        out=pd.read_parquet(cache if cache.exists() else OUT/'calibration.parquet')
        finished={pd.Timestamp(q['quarter']) for q in manifest['quarters']}
        remaining=[q for q in pd.date_range('2022-01-01','2025-10-01',freq='QS',tz='UTC') if q not in finished]
        for number,quarter in enumerate(remaining):
            if args.quarters is not None and number>=args.quarters: break
            quarter_probabilities(data,out,quarter,manifest)
            out.to_parquet(cache)
        manifest['status']='candidate_probabilities_completed' if len(manifest['quarters'])==16 else 'candidate_probabilities_partial'
        write_json(manifest_path,manifest)
    if args.stage in ('finish','all'):
        manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
        if len(manifest['quarters'])!=16: raise ValueError('Finish requires all 16 quarters')
        out=pd.read_parquet(OUT/'candidate_probabilities.parquet')
        manifest['event_df_selection']=select_and_matrix(out)
        scores,contrasts=block_summary(out)
        out.to_parquet(OUT/'upgrade_predictions.parquet')
        manifest['matrix_scores']=scores; manifest['matrix_contrasts']=contrasts
        manifest['score_uncertainty_policy']='synchronous circular elapsed-calendar-day blocks, missing complete days retained as zero-count positions, paired ratio of loss-sums to observed-hour counts'
        manifest['final_script_hash']=sha(__file__)
        manifest['output_schema']={'index':'target_time UTC','physical_CDF':'v0_q90_ZONE original / v1_q90_ZONE additional conditional map; q95 analogous','PIT':'pit_ZONE original / calibrated_pit_ZONE','labels':'exceed_q90_ZONE strict actual net_load>fixed_threshold; y_q90 ≥2; q95 analogous','R':'rho_01=DE_LU/FR, rho_02=DE_LU/BE, rho_12=FR/BE','event_selected_df':'df_event_selected numeric string or inf','cell':'cell=stratum; six fixed forecast-only cells','matrix':'M0 original+Gaussian; M1 conditional+sameGaussian; M2 original+sameR event-selected nu; M3 conditional+same selected nu'}
        manifest['status']='completed'; manifest['output_sha256']=sha(OUT/'upgrade_predictions.parquet'); write_json(manifest_path,manifest)
        print(pd.DataFrame(scores).to_string(index=False),flush=True)


if __name__=='__main__': main()
