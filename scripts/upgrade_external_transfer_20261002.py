"""Fixed-rule spatial transfer to two previously unscored Danish bidding zones.

New DK data are not used to select models, grids, nu, warning rules or windows.
Region-specific marginal and R coefficients may learn previously available
historical outcomes. This is spatial transfer of the fixed procedure, not an
unchanged-coefficient zero-shot forecast. DE/FR labels were already inspected;
the Danish component of the evaluation is new. Weather remains the original
macroregional proxy rather than falsely labelled Denmark-specific weather.
"""
from __future__ import annotations
from release_paths import work_path as _work_path, raw_path as _raw_path
import argparse
import json
import pickle
import time
import warnings
import holidays
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csc_matrix
from scipy.stats import beta
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import QuantileRegressor

from adaptive_thresholds import _calendar_design, REGULARIZATION_ALPHA
from conditional_features import ConditionalWeatherBasis
from dependence import DynamicGaussianCopula, IndependentCopula, IntegrationSettings, elliptical_event_probabilities
from joint_pair_products import elliptical_pair_probabilities
from margins import GaussianAdditiveMargin, BetaPITCalibration, raw_margin_cdf_sf, beta_cdf_sf
import upgrade_margin_dependence_20261002 as matrix

ROOT=_work_path()
BASEOUT=ROOT/'results/upgrade_external_transfer_20261002'
OUT=BASEOUT/'macro_strict15d'
CONFIG=ROOT/'config/upgrade_margin_external_transfer_clock_amendment_20261002.json'
MODE='macro'
LOCAL_WEATHER=ROOT/'results/upgrade_data_audit_20261002/DK_local_weather_summary_20181231_20251230.parquet'
LOCAL_WEATHER_BATCH=LOCAL_WEATHER.with_name('DK_local_weather_batch_20181231_20251230.json')
DK=('DK1','DK2')
OBS=ROOT/'results/upgrade_data_audit_20261002/DK_settlement_hourly_2019_2025.parquet'
PANEL=ROOT/'datasets/weather_panel_gefs_frozen_through_20260101/panel.parquet'
FEATURES=ROOT/'datasets/weather_panel_gefs_frozen_through_20260101/margin_features.json'
SHARED=ROOT/'results/shared_frozen_history_test_20261001/prequential_combined_shared_margins.parquet'
MANIFEST=ROOT/'results/shared_frozen_history_test_20261001/prequential_manifest.json'
AUX=ROOT/'datasets/joint_auxiliary_frozen_20261001'
EMBARGO=15
NU=3.
INTEGRATION=IntegrationSettings(tolerance=1e-6,initial_maxpts=16384,max_maxpts=4194304,backend='elliptical_path')


def quarter_cutoff(origin):
    return origin-pd.Timedelta(hours=12,days=EMBARGO)


def set_mode(mode):
    global OUT,CONFIG,MODE
    MODE=mode
    if mode=='local':
        OUT=BASEOUT/'local_weather_strict15d'
        CONFIG=ROOT/'config/upgrade_margin_external_local_weather_schema_amendment_20261002.json'


def freeze():
    if CONFIG.exists(): raise FileExistsError('The existing external design must not be overwritten')
    selection=json.loads((matrix.OUT/'manifest.json').read_text(encoding='utf-8'))
    assert selection['event_df_selection'][-1]['nu']=='3'
    model=json.loads(MANIFEST.read_text(encoding='utf-8'))['settings']['model_parameters']
    features=json.loads(FEATURES.read_text(encoding='utf-8'))['feature_columns'][:69]
    design={'version':'1.0-fixed-spatial-transfer-macro-weather-15day-label-cutoff','frozen_utc':pd.Timestamp.now(tz='UTC'),'unscored_DK_joint_results_before_this_freeze':True,'DE_FR_outcomes_already_inspected':True,'bidding_zone_combinations':['DE_LU-FR-DK1','DE_LU-FR-DK2'],'region_choice':'both Danish bidding zones required; no selection by performance','observation_file':str(OBS),'observation_file_sha256':matrix.sha(OBS),'historical_vintage':'current revised settlement values; nominal delayed replay, not authenticated original documents','settlement_current_official_first_publish_days':[9,15],'maximum_nominal_first_publish_delay_days':EMBARGO,'settlement_revision_horizon':'up to two years; metadata is current description rather than historical availability proof','net_load_definition':'GrossConsumptionMWh - uncertain LocalPowerSelfConMWh - sum(four solar generation categories) - sum(four wind generation categories), each one-hour MWh as mean MW','2019_measurement_boundary':'all historical solar grouped under SolarPowerSelfCon; grid categories zero do not establish physical absence; historical grouping flag retained','new_DK_margin_base_features':features,'weather':'unchanged original DE-LU/FR/BE macroregional GEFS proxy, no DK-specific weather inputs; local weather transfer limitation','new_DK_load_lags_days':[21,28],'lag_source':'corrected gross total consumption, not net load; exact UTC-hour start alignment; source interval end plus 15d checked before issue','new_DK_GAM_parameters':model,'new_DK_F_fit_start':'2019-01-01','raw_quarter_fit_end':'quarter origin - 15 days - 60 days','daily_G':'Beta PIT MLE from same quarterly F; trailing 60 days ending target issue minus15days; no raw-fit overlap','DK_thresholds':'same calendar trend q90/q95 structure and alpha1e-4; three-year history ending quarter origin minus15days; originalDE/FRcached physical thresholds unchanged','new_DK_issued_PIT_start':'2021-01-01','original_DE_FR_margins':'original issued cached forecasts, nominal 7-day label delay as in original study; no new tuning','joint_R':'new triple original issued PIT; original fixed13 macro-weather basis, l2.001 maxiter2000; quarterly two-year past history ending quarter origin minus15days; sameR for all four matrix cells','conditional_margin_grid':selection['6_cell_weather_cutpoints_from_2021_only'],'conditional_margin_grid_structure':selection['grid'],'conditional_margin_map':'same fixed Beta-PIT conditional map and365day window, weekly firstissue minus15days, ≥28days and168h support otherwiseidentity; logshape penalty4; originalDE/FR extra map also15days','matrix':'M0 base+Gaussian; M1 conditional+sameGaussianR; M2 base+sameR t3; M3 conditional+sameR t3','fixed_nu':NU,'nu_source':'originalDEFRBE development q90 average two-margin loss selected3; never reselected usingDK outcomes','primary_event':'at leasttwo zones exceed their frozen physical seasonalq90 thresholds within samehour','secondary_q95':'same nu3, no extra tuning','external_DK_specific_diagnostic':'DK-pivotal event =DKexceeds and exactlyone ofDE_LU/FRexceeds; report counts and propereventprobability, preventDEFRboth dominatinginterpretation','evaluation_years':[2024,2025],'calibration_and_history_years':[2019,2020,2021,2022,2023],'later_available_history':'2024 labels may update2025coefficients under frozen procedure with15daydelay; not used for hyperparameters/nu','primary_warning_transfer':'originalDEVfixedP1tau.25 andP2tau.19,cooldown6h,priorityp*(1-maxprevious6issuedrisks),max2perday; rootpolicyimplementation','uncertainty':'paired synchronous elapsedcalendar blocks7/14/28days; pointwise95CI; no independent-hour interpretation','files_unchanged':['all preupgrade files','allraw archives'],'running_resource':'CPU, BLASthreads1, HiGHSthreads1, no global GRIB retrieval','input_hashes':{str(p):matrix.sha(p) for p in (PANEL,FEATURES,SHARED,MANIFEST,AUX/'joint_features_lean.parquet',AUX/'timing.parquet')},'script_sha256':matrix.sha(__file__)}
    design.update({'profile':MODE,'clock_amendment':'All DK quarterly training/threshold/R label cutoffs are first forecast issue (origin minus12h) minus15days; daily G and extra map use issue minus15days. Earlier nominal-origin-minus15day unscored pilot is superseded.','raw_quarter_fit_end':'quarter origin -12hours -15days -60days','DK_thresholds':'same calendar trend q90/q95 structure and alpha1e-4; three-year history ending first quarter issue minus15days; originalDE/FRcached physical thresholds unchanged','joint_R':'new triple original issued PIT; original fixed13 macro-weather basis, l2.001 maxiter2000; quarterly two-year past history ending first quarter issue minus15days; sameR for all four matrix cells','initial_nominal15day_design':str(ROOT/'config/upgrade_margin_external_transfer_20261002.json')})
    if MODE=='local':
        design.update({'weather':'original DE/FR GEFS summaries plus new DK-specific geographical forecast summaries for DK margin weather slot and three-zone joint basis; common1degree grid20members5endpoints; no new tuning','local_weather_file':str(LOCAL_WEATHER),'local_weather_expected_index':'UTCtarget_time 2019-01-01T00 through2025-12-31T23,61368hours','local_weather_schema':{z:[z+'_'+v+'_'+s for v in ('t2m','wind_speed','marine_wind_speed','dswrf') for s in ('mean','std')] for z in DK},'local_weather_unit_transform':'t2m_mean Kelvin minus273.15 ->degreesCelsius; stdunchanged, winds m/s, radiation W/m2','new_DK_margin_weather_slot':'originalBEslot replaced by relevant DK1/DK2 physical means/spreads and deterministic temperature-hour/wind-solar-slowtime interactions; slot names remainBE internally, metadata explicitly remaps; Danish holiday flags replace Belgian slot with samecalendarrole','joint_weather_basis':'original5calendar columns unchanged; eight common physical means/spreads are equal mean ofDE_LU,FR,relevantDK local forecasts; spread is mean of within-zone ensembleSD, not geographicSD','original_DE_FR_margins':'original issued cached forecasts from originalweather/lagcontext retained as existing forecastmodules; only Danish newly fitted margin and dependencebasis receiveDKlocalweather; do not claim entiretriple margins refitted to one identicalfullinput schema','conditional_margin_grid_transfer_policy':'same numerical2021 originalstudy cuts applied to newforecastbasis, no DK-based recutting','weather_mask_boundary':'NaturalEarth Denmark polygon approximation: representative x<11.1 or y>56.6 DK1,otherDK2; fixed NorthSea/Baltic marine masks; geographicalproxy not officialelectricalpolygon','evaluation_rule':'localweather profile prespecified as mainspatialtransfer; macroprofile retained as input sensitivity. Both reported regardlesstestloss, no selecting profile by winner','schema_amendment':'Previous unscored local config accidentally stored explanatory string over numeric grid field; restore exact original fixed cutpoints under numeric grid, retaining explanation in separate field. No model/parameter choice changed.'})
    matrix.write_json(CONFIG,design)
    OUT.mkdir(parents=True,exist_ok=True)
    matrix.write_json(OUT/'manifest.json',{'status':'fixed_before_DK_forecast_scoring','design':design,'margin_quarters':[],'joint_quarters':[]})
    print('External design frozen before producing Danish probability/score outputs',flush=True)


def load_panel():
    if not CONFIG.exists(): raise PermissionError('Freeze the external procedure before reading the Danish numerical series')
    design=json.loads(CONFIG.read_text(encoding='utf-8'))
    if matrix.sha(OBS)!=design['observation_file_sha256']: raise ValueError('Danish input changed after design freeze')
    obs=pd.read_parquet(OBS)
    panel=pd.read_parquet(PANEL)
    columns=design['new_DK_margin_base_features']
    wx=None
    if MODE=='local':
        wx=pd.read_parquet(LOCAL_WEATHER)
        if len(wx)!=61368 or wx.index.has_duplicates or not wx.index.equals(panel.index):raise ValueError('Complete local Danish forecast time grid must exactly equal the original panel')
        required=[z+'_'+v+'_'+s for z in DK for v in ('t2m','wind_speed','marine_wind_speed','dswrf') for s in ('mean','std')]
        if not set(required).issubset(wx.columns) or not np.isfinite(wx[required].to_numpy()).all():raise ValueError('Incomplete regional weather must be repaired upstream, not scored as a complete profile')
        if not all(150<float(wx[z+'_t2m_mean'].median())<350 for z in DK):raise ValueError('Declared raw Kelvin temperature units do not match the regional weather')
    frames={}
    for zone in DK:
        region=obs[obs.PriceArea==zone].set_index('timestamp').reindex(panel.index)
        f=panel[columns].copy()
        if MODE=='local':
            for variable in ('t2m','wind_speed','marine_wind_speed','dswrf'):
                for statistic in ('mean','std'):
                    series=wx[f'{zone}_{variable}_{statistic}']
                    f[f'BE_{variable}_{statistic}']=series-(273.15 if variable=='t2m' and statistic=='mean' else 0.)
            for harmonic in ('hour_sin_1','hour_cos_1'):f[f'BE_temperature_x_{harmonic}']=f.BE_t2m_mean*f[harmonic]
            for variable in ('wind_speed','marine_wind_speed','dswrf'):f[f'BE_{variable}_x_slow_time']=f[f'BE_{variable}_mean']*f.years_since_2019
            schedule=holidays.country_holidays('DK',years=range(2018,2027),observed=True,language='en_US')
            dates=f.index.tz_convert('Europe/Copenhagen').date
            for offset,label in ((-1,'previous'),(0,'today'),(1,'next')):f[f'holiday_BE_{label}']=np.asarray([(pd.Timestamp(d)+pd.Timedelta(days=offset)).date() in schedule for d in dates],float)
        for lag in (21,28):
            name=f'provider_load_lag{lag}d_utc_{zone}'
            source_index=f.index-pd.Timedelta(days=lag)
            source=region.load_corrected_mw.reindex(source_index)
            f[name]=source.to_numpy()
            issue=f.index.normalize()-pd.Timedelta(hours=12)
            source_available=source_index+pd.Timedelta(hours=1,days=15)
            if (source_available>issue).any(): raise AssertionError('DK consumption lag cannot be available before this issue')
        f['net_load_'+zone]=region.net_load_mw
        f['historic_grouping_flag']=region.historic_grouping_flag
        frames[zone]=f
    return frames,columns,design


def thresholds(y, origin, query):
    cutoff=quarter_cutoff(origin)
    train=y.loc[(y.index>=cutoff-pd.DateOffset(years=3))&(y.index+pd.Timedelta(hours=1)<=cutoff)].dropna()
    if len(train.index.normalize().unique())<365: raise ValueError('Too few preceding threshold dates')
    X,names=_calendar_design(train.index,cutoff,True)
    xscale=X.std(axis=0); xscale[xscale<1e-10]=1.
    values=train.to_numpy(); center=float(np.median(values)); scale=float(np.subtract(*np.quantile(values,[.75,.25])))
    if scale<=1e-10: raise ValueError('Constant Danish net-load history')
    X=csc_matrix(X/xscale)
    xq,_=_calendar_design(query,cutoff,True); xq=csc_matrix(xq/xscale)
    predictions=[]; records=[]
    for q in (.9,.95):
        tick=time.perf_counter()
        model=QuantileRegressor(quantile=q,alpha=REGULARIZATION_ALPHA,fit_intercept=True,solver='highs',solver_options={'time_limit':120.,'presolve':True,'threads':1})
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always',ConvergenceWarning); model.fit(X,(values-center)/scale)
        if any(issubclass(w.category,ConvergenceWarning) for w in caught): raise RuntimeError('Threshold LP did not converge')
        predictions.append(center+scale*model.predict(xq))
        records.append({'q':q,'fit_seconds':time.perf_counter()-tick,'n_iter':int(model.n_iter_),'rows':len(train),'cutoff':cutoff,'last_interval_end':train.index.max()+pd.Timedelta(hours=1)})
    ordered=np.sort(np.column_stack(predictions),axis=1)
    return ordered,records


def margins(quarter_limit=None):
    frames,base_features,design=load_panel()
    manifest=json.loads((OUT/'manifest.json').read_text(encoding='utf-8'))
    if MODE=='local':
        manifest['actual_local_weather_input_sha256']=matrix.sha(LOCAL_WEATHER)
        manifest['actual_local_weather_complete_schema_checked']=True
    done={(r['zone'],pd.Timestamp(r['quarter'])) for r in manifest['margin_quarters']}
    result={z:pd.read_parquet(OUT/f'margins_{z}.parquet') if (OUT/f'margins_{z}.parquet').exists() else pd.DataFrame(index=frames[z].index[frames[z].index.year>=2021]) for z in DK}
    number=0
    for origin in pd.date_range('2021-01-01','2025-10-01',freq='QS',tz='UTC'):
        for zone in DK:
            if (zone,origin) in done: continue
            if quarter_limit is not None and number>=quarter_limit: return
            number+=1; f=frames[zone]; idx=f.index
            features=base_features+[f'provider_load_lag{lag}d_utc_{zone}' for lag in (21,28)]
            raw_end=quarter_cutoff(origin)-pd.Timedelta(days=60)
            train=(idx>=pd.Timestamp('2019-01-01',tz='UTC'))&(idx+pd.Timedelta(hours=1)<=raw_end)&np.isfinite(f[features+['net_load_'+zone]].to_numpy()).all(axis=1)
            tick=time.perf_counter()
            model=GaussianAdditiveMargin(**design['new_DK_GAM_parameters']).fit(f.loc[train,features].to_numpy(),f.loc[train,'net_load_'+zone].to_numpy())
            fit_sec=time.perf_counter()-tick
            query=idx[(idx>=origin)&(idx<origin+pd.DateOffset(months=3))]
            threshold_cache=BASEOUT/'macro_strict15d'/f'thresholds_{zone}_{origin.year}Q{origin.quarter}.parquet'
            threshold_meta=threshold_cache.with_suffix('.json')
            if MODE=='local' and threshold_cache.exists():
                h=pd.read_parquet(threshold_cache).reindex(query).to_numpy();threshold_records=json.loads(threshold_meta.read_text(encoding='utf-8'))
                if not np.isfinite(h).all():raise ValueError('Threshold cache incomplete')
            else:
                h,threshold_records=thresholds(f['net_load_'+zone],origin,query)
                threshold_cache.parent.mkdir(parents=True,exist_ok=True)
                pd.DataFrame(h,index=query,columns=['q90','q95']).to_parquet(threshold_cache);matrix.write_json(threshold_meta,threshold_records)
            table=pd.DataFrame(index=query); table.index.name='target_time'
            complete=np.isfinite(f.loc[query,features].to_numpy()).all(axis=1)
            table['feature_complete']=complete
            table['historic_grouping_flag']=f.loc[query,'historic_grouping_flag']
            table['issue_time']=query.normalize()-pd.Timedelta(hours=12)
            table['origin']=origin
            for k,q in enumerate(('q90','q95')): table[f'threshold_{q}_{zone}']=h[:,k]
            # Compute raw F for all future inputs without feeding future y to
            # training or beta calibration. Future y enters only issued PIT
            # construction and later evaluation of these issued distributions.
            allgood=np.isfinite(f[features+['net_load_'+zone]].to_numpy()).all(axis=1)
            available_history=(idx>=raw_end)&(idx<query.max()+pd.Timedelta(hours=1))&allgood
            rows=f.loc[available_history]
            tick=time.perf_counter()
            raw_pit,raw_sf=raw_margin_cdf_sf(model,rows[features].to_numpy(),rows['net_load_'+zone].to_numpy())
            cache=pd.DataFrame({'u':raw_pit,'sf':raw_sf},index=rows.index)
            records=[]
            for day in query.normalize().unique():
                issue=day-pd.Timedelta(hours=12); cutoff=issue-pd.Timedelta(days=EMBARGO); start=max(raw_end,cutoff-pd.Timedelta(days=60))
                prior=cache.loc[(cache.index>=start)&(cache.index+pd.Timedelta(hours=1)<=cutoff)]
                if len(prior)<100: raise ValueError('Insufficient raw-fit-disjoint daily G labels')
                calibrator=BetaPITCalibration().fit(prior.u.to_numpy())
                times=query[(query.normalize()==day)&complete]
                X=f.loc[times,features].to_numpy()
                for q in ('q90','q95'):
                    u,sf=raw_margin_cdf_sf(model,X,table.loc[times,f'threshold_{q}_{zone}'].to_numpy())
                    table.loc[times,f'raw_cdf_{q}_{zone}']=u
                    table.loc[times,f'cdf_{q}_{zone}']=beta_cdf_sf(calibrator,u,sf)[0]
                observed=times.intersection(cache.index)
                table.loc[observed,'raw_pit_'+zone]=cache.loc[observed,'u']
                table.loc[observed,'pit_'+zone]=beta_cdf_sf(calibrator,cache.loc[observed,'u'].to_numpy(),cache.loc[observed,'sf'].to_numpy())[0]
                records.append({'day':day,'issue':issue,'cutoff':cutoff,'start':start,'rows':len(prior),'last_interval_end':prior.index.max()+pd.Timedelta(hours=1),'a':calibrator.a_,'b':calibrator.b_})
            table['net_load_'+zone]=f.loc[query,'net_load_'+zone]
            table['threshold_available']=True
            result[zone]=result[zone].combine_first(table)
            result[zone].to_parquet(OUT/f'margins_{zone}.parquet')
            record={'zone':zone,'quarter':origin,'raw_fit_end':raw_end,'quarter_label_cutoff':quarter_cutoff(origin),'first_issue':origin-pd.Timedelta(hours=12),'strict15day_interval_end_cutoff':True,'raw_training_rows':int(train.sum()),'GAM_fit_seconds':fit_sec,'daily_G_and_predictions_seconds':time.perf_counter()-tick,'threshold_fits':threshold_records,'threshold_cache_reused':MODE=='local' and threshold_cache.exists(),'daily_G':records,'feature_count':len(features),'query_hours':len(query)}
            manifest['margin_quarters'].append(record)
            matrix.write_json(OUT/'manifest.json',manifest)
            print(json.dumps(matrix.safe({'zone':zone,'quarter':origin,'raw_fit_seconds':fit_sec,'threshold_seconds':sum(r['fit_seconds'] for r in threshold_records),'daily_G_seconds':record['daily_G_and_predictions_seconds']})),flush=True)
    manifest['status']='DK_margins_completed'; matrix.write_json(OUT/'manifest.json',manifest)


def conditional_maps(frame,zones,cell):
    idx=frame.index
    out=pd.DataFrame(index=idx); out.index.name='target_time'; out['cell']=cell; out['stratum']=cell
    out['issue_time']=frame.issue_time; out['period']=np.where(idx.year<2022,'prior_history',np.where(idx.year<=2023,'development_transfer_not_selection','spatial_transfer_evaluation'))
    for zone in zones:
        out['pit_'+zone]=frame['pit_'+zone]
        out['calibrated_pit_'+zone]=np.nan
        for q in ('q90','q95'):
            out[f'v0_{q}_{zone}']=frame[f'cdf_{q}_{zone}']; out[f'v1_{q}_{zone}']=np.nan
            out[f'threshold_{q}_{zone}']=frame[f'threshold_{q}_{zone}']
            y=frame['net_load_'+zone].to_numpy(); h=frame[f'threshold_{q}_{zone}'].to_numpy()
            out[f'exceed_{q}_{zone}']=np.where(np.isfinite(y)&np.isfinite(h),(y>h).astype(float),np.nan)
    for q in ('q90','q95'):
        a=out[[f'exceed_{q}_{z}' for z in zones]].to_numpy(); good=np.isfinite(a).all(axis=1)
        out['y_'+q]=np.where(good,(a.sum(axis=1)>=2).astype(float),np.nan)
        out['y_'+q+'_DK_pivotal']=np.where(good,((a[:,2]==1)&(a[:,:2].sum(axis=1)==1)).astype(float),np.nan)
    records=[]
    for start in pd.date_range('2021-12-27','2025-12-29',freq='7D',tz='UTC'):
        query=(idx>=max(start,pd.Timestamp('2022-01-01',tz='UTC')))&(idx<start+pd.Timedelta(days=7))
        if not query.any():continue
        issue=frame.loc[query,'issue_time'].min(); cutoff=issue-pd.Timedelta(days=EMBARGO)
        past=(idx>=cutoff-pd.Timedelta(days=365))&(idx+pd.Timedelta(hours=1)<=cutoff)
        record={'week':start,'first_issue':issue,'cutoff':cutoff,'maps':{}}
        for c in sorted(cell.dropna().unique()):
            qc=query&(cell.to_numpy()==c); train=past&(cell.to_numpy()==c)
            if not qc.any():continue
            for zone in zones:
                shape,meta=matrix.fit_beta_daily_equal(frame.loc[train,'pit_'+zone].to_numpy(),idx[train]); record['maps'][c+'_'+zone]=meta
                out.loc[qc,'calibrated_pit_'+zone]=beta.cdf(frame.loc[qc,'pit_'+zone],*shape)
                for q in ('q90','q95'):out.loc[qc,f'v1_{q}_{zone}']=beta.cdf(out.loc[qc,f'v0_{q}_{zone}'],*shape)
        records.append(record)
    return out,records


def joint():
    manifest=json.loads((OUT/'manifest.json').read_text(encoding='utf-8'))
    if len(manifest['margin_quarters'])!=40:raise ValueError('All forty DK quarterly margin fits must finish before transfer scoring')
    shared=pd.read_parquet(SHARED); weather=pd.read_parquet(AUX/'joint_features_lean.parquet'); timing=pd.read_parquet(AUX/'timing.parquet')
    settings=json.loads((ROOT/'results/joint_runner_frozen_test_numerical_20261001/joint_runner_manifest.json').read_text(encoding='utf-8'))['settings']
    cell,_=matrix.fixed_grid(weather)
    for dk in DK:
        dest=OUT/dk;dest.mkdir(exist_ok=True)
        local=pd.read_parquet(OUT/f'margins_{dk}.parquet').reindex(shared.index)
        zones=('DE_LU','FR',dk)
        existing_cols=[c for c in shared if c.endswith('_DE_LU') or c.endswith('_FR')]
        frame=shared[existing_cols+['issue_time','feature_complete','threshold_available']].copy()
        frame=frame.join(local[[c for c in local if c.endswith('_'+dk)]])
        frame['feature_complete']=frame.feature_complete&local.feature_complete.fillna(False)
        frame['threshold_available']=frame.threshold_available&local.threshold_available.fillna(False)
        area_weather=weather.copy()
        if MODE=='local':
            panel=pd.read_parquet(PANEL).reindex(shared.index);wx=pd.read_parquet(LOCAL_WEATHER).reindex(shared.index)
            for variable,name in (('t2m','temperature'),('wind_speed','land_wind'),('marine_wind_speed','marine_wind'),('dswrf','solar')):
                for statistic,suffix in (('mean',''),('std','_spread')):
                    new=wx[f'{dk}_{variable}_{statistic}']-(273.15 if variable=='t2m' and statistic=='mean' else 0.)
                    area_weather[f'common_{name}{suffix}']=(panel[f'DE_LU_{variable}_{statistic}']+panel[f'FR_{variable}_{statistic}']+new)/3
            # Keep original numerical forecast cutpoints, not the new region's
            # own 2021 quantiles. This is a deliberately fixed-rule transfer.
            cuts=json.loads(CONFIG.read_text(encoding='utf-8'))['conditional_margin_grid']
            season=np.where(np.isin(area_weather.index.month,(10,11,12,1,2,3)),'cool','warm')
            values=np.full(len(area_weather),None,dtype=object)
            valid=np.isfinite(area_weather.to_numpy()).all(axis=1)
            for label in ('cool','warm'):
                low=area_weather.common_land_wind.to_numpy()<=cuts[label]['land_wind_median_2021'];cold=area_weather.common_temperature.to_numpy()<=cuts[label]['temperature_q25_2021']
                mask=valid&(season==label);values[mask&low&cold]=label+'_low_wind_cold';values[mask&low&~cold]=label+'_low_wind_other';values[mask&~low]=label+'_higher_wind'
            cell=pd.Series(values,index=area_weather.index,name='cell')
        out,map_records=conditional_maps(frame,zones,cell)
        area_record={'zone':dk,'conditional_weekly_maps':map_records,'quarters':[]}
        for origin in pd.date_range('2022-01-01','2025-10-01',freq='QS',tz='UTC'):
            idx=frame.index;cutoff=quarter_cutoff(origin); begin=max(pd.Timestamp('2021-01-01',tz='UTC'),origin-pd.DateOffset(years=2))
            U=frame[['pit_'+z for z in zones]].to_numpy(); fx=area_weather.to_numpy()
            train=(idx>=begin)&(idx+pd.Timedelta(hours=1)<=cutoff)&np.isfinite(U).all(axis=1)&np.isfinite(fx).all(axis=1)
            basis=ConditionalWeatherBasis(linear_columns=tuple(settings['linear_columns']),smooth_columns=tuple(settings['smooth_columns']),interactions=tuple(tuple(v) for v in settings['interactions']),n_knots=4,allow_missing=False,minimum_release_delay_hours=12.)
            xt=basis.fit_transform(area_weather.loc[train],timing_manifest=timing.loc[train].reset_index(),fit_cutoff=cutoff).to_numpy()
            tick=time.perf_counter();model=DynamicGaussianCopula(l2=.001,maxiter=2000,integration_settings=INTEGRATION).fit(U[train],xt)
            fitsec=time.perf_counter()-tick
            query=(idx>=origin)&(idx<origin+pd.DateOffset(months=3))&np.isfinite(fx).all(axis=1)&frame.feature_complete.to_numpy(bool)&frame.threshold_available.to_numpy(bool)
            X=basis.transform(area_weather.loc[query],timing_manifest=timing.loc[query].reset_index()).to_numpy();R=model.correlation_matrices(X)
            for c,i,j in (('rho_01',0,1),('rho_02',0,2),('rho_12',1,2)):out.loc[query,c]=R[:,i,j]
            out.loc[query,'df_event_selected']='3';out.loc[query,'copula_origin']=origin
            integrals=[]
            for q in ('q90','q95'):
                for margin in (0,1):
                    V=out.loc[query,[f'v{margin}_{q}_{z}' for z in zones]].to_numpy()
                    if not np.isfinite(V).all():raise ValueError('Available DK forecast is missing a physical-threshold CDF')
                    V=np.clip(V,1e-12,1-1e-12)
                    for nu in (np.inf,NU):
                        m=margin+2*(not np.isinf(nu));tick=time.perf_counter()
                        result=elliptical_event_probabilities(V,R,nu,settings=INTEGRATION)
                        pairs=elliptical_pair_probabilities(V,R,nu,settings=INTEGRATION)
                        # Pair names use generic DE_LU_FR/DE_LU_BE/FR_BE ordering;
                        # first pair is unchanged DE/FR in this remapped triple.
                        pair=pairs['pairs'][:,0]
                        out.loc[query,f'M{m}_{q}']=result['ge2']
                        out.loc[query,f'M{m}_{q}_all3']=result['all3']
                        pivotal=result['ge2']-pair
                        pivotal_correction=float(max(0.,-pivotal.min(),pivotal.max()-1))
                        if pivotal_correction>1e-5:raise RuntimeError('Pivotal probability violates event nesting beyond numerical tolerance')
                        out.loc[query,f'M{m}_{q}_DK_pivotal']=np.clip(pivotal,0,1)
                        integrals.append({'q':q,'margin':margin,'nu':'inf' if np.isinf(nu) else 3,'seconds':time.perf_counter()-tick,'max_discrepancy':float(result['numerical_error'].max()),'pair_discrepancy_max':float(pairs['numerical_error'].max()),'DK_pivotal_roundoff_correction_max':pivotal_correction})
                V=out.loc[query,[f'v0_{q}_{z}' for z in zones]].to_numpy();out.loc[query,'independence_'+q]=IndependentCopula().predict_event_probabilities(V)['ge2']
            area_record['quarters'].append({'origin':origin,'fit_cutoff':cutoff,'history_rows':int(train.sum()),'query_rows':int(query.sum()),'fit_seconds':fitsec,'fit_info':model.fit_info_,'calculations':integrals})
            out.to_parquet(dest/'upgrade_predictions.parquet');matrix.write_json(dest/'manifest.json',area_record)
            print(json.dumps(matrix.safe({'joint_zone':dk,'quarter':origin,'fitsec':fitsec})),flush=True)
        # Reuse paired elapsed-calendar block code; its generic labels describe
        # origin years, while this manifest explains new spatial transfer.
        matrix.OUT=dest
        scores,contrasts=matrix.block_summary(out)
        area_record['matrix_scores']=scores;area_record['matrix_contrasts']=contrasts
        for q in ('q90','q95'):
            cols=['y_'+q]+[f'M{m}_{q}' for m in range(4)]
            mask=matrix.complete_day_mask(out,cols)&out.index.year.isin((2024,2025))
            area_record['evaluated_'+q]={'hours':int(mask.sum()),'days':int(mask.sum()/24),'event_hours':int(out.loc[mask,'y_'+q].sum()),'DK_pivotal_hours':int(out.loc[mask,'y_'+q+'_DK_pivotal'].sum()),'DK_pivotal_Brier':{f'M{m}':float(np.mean((out.loc[mask,f'M{m}_{q}_DK_pivotal']-out.loc[mask,'y_'+q+'_DK_pivotal'])**2)) for m in range(4)}}
        area_record['status']='completed_spatial_transfer';matrix.write_json(dest/'manifest.json',area_record)
        manifest['joint_quarters'].append({'zone':dk,'completed_quarters':16,'scores':scores,'evaluated_q90':area_record['evaluated_q90'],'evaluated_q95':area_record['evaluated_q95']})
        matrix.write_json(OUT/'manifest.json',manifest)
    manifest['status']='completed_spatial_transfer';manifest['final_script_hash']=matrix.sha(__file__);matrix.write_json(OUT/'manifest.json',manifest)


def audit():
    """Read-only outcome diagnostics; no selection or subsequent tuning."""
    manifest=json.loads((OUT/'manifest.json').read_text(encoding='utf-8'))
    if manifest['status']!='completed_spatial_transfer':raise ValueError('Finish both external combinations before audit')
    violations=[]; clock_rows=[]; report={}; schema={}; marginal=[]
    for r in manifest['margin_quarters']:
        first=pd.Timestamp(r['first_issue']); cutoff=pd.Timestamp(r['quarter_label_cutoff'])
        if first-cutoff!=pd.Timedelta(days=EMBARGO):violations.append('quarter_margin_cutoff')
        if cutoff-pd.Timestamp(r['raw_fit_end'])!=pd.Timedelta(days=60):violations.append('raw_G_overlap')
        for fit in r['threshold_fits']:
            if pd.Timestamp(fit['last_interval_end'])>cutoff:violations.append('threshold_future_label')
        for g in r['daily_G']:
            if pd.Timestamp(g['issue'])-pd.Timestamp(g['cutoff'])!=pd.Timedelta(days=EMBARGO):violations.append('daily_G_cutoff')
            if pd.Timestamp(g['last_interval_end'])>pd.Timestamp(g['cutoff']):violations.append('daily_G_future_label')
        clock_rows.append({'zone':r['zone'],'quarter':r['quarter'],'first_issue':first,'label_cutoff':cutoff,'daily_updates':len(r['daily_G'])})
    for dk in DK:
        dest=OUT/dk; area=json.loads((dest/'manifest.json').read_text(encoding='utf-8'))
        frame=pd.read_parquet(dest/'upgrade_predictions.parquet'); zones=('DE_LU','FR',dk)
        if len(frame)!=43824 or frame.index.has_duplicates:violations.append('external_time_grid')
        schema[dk]={'file':str(dest/'upgrade_predictions.parquet'),'rows':len(frame),'index_name':frame.index.name,'first_target':frame.index.min(),'last_target':frame.index.max(),'columns':{c:str(t) for c,t in frame.dtypes.items()},'primary_probability':'M0_q90','primary_event':'y_q90','pivotal_probability':'M0_q90_DK_pivotal','pivotal_event':'y_q90_DK_pivotal','forecast_history_used_for_policy_context':'2022-2025; plan all available issues before slicing 2024-2025 scoring','evaluation_period_column':'spatial_transfer_evaluation'}
        pivotal=frame.copy()
        for q in ('q90','q95'):
            pivotal['y_'+q]=frame['y_'+q+'_DK_pivotal']
            for m in range(4):pivotal[f'M{m}_{q}']=frame[f'M{m}_{q}_DK_pivotal']
        pivotal_dest=dest/'pivotal_event';pivotal_dest.mkdir(exist_ok=True)
        matrix.OUT=pivotal_dest
        pivotal_scores,pivotal_contrasts=matrix.block_summary(pivotal)
        matrix.write_json(pivotal_dest/'scope.json',{'label':'DK exceeds its fixed threshold and exactly one of DE-LU/France exceeds','probability':'joint >=2 probability minus unchanged DE-LU/France both-exceedance probability','evaluation':'new Danish event component under fixed procedure; shared DE/FR outcomes previously inspected','inference':'paired elapsed-calendar 7/14/28d resampling; pointwise 95% intervals; exploratory secondary diagnosis, no selection'})
        for r in area['quarters']:
            if pd.Timestamp(r['fit_cutoff'])!=quarter_cutoff(pd.Timestamp(r['origin'])):violations.append('R_cutoff')
            if not r['fit_info']['converged']:violations.append('R_optimizer_not_converged')
        for r in area['conditional_weekly_maps']:
            if pd.Timestamp(r['first_issue'])-pd.Timestamp(r['cutoff'])!=pd.Timedelta(days=EMBARGO):violations.append('extra_map_cutoff')
        rhos=frame[['rho_01','rho_02','rho_12']].dropna().to_numpy(); matrices=np.tile(np.eye(3),(len(rhos),1,1))
        for col,i,j in ((0,0,1),(1,0,2),(2,1,2)):matrices[:,i,j]=matrices[:,j,i]=rhos[:,col]
        min_eigenvalue=float(np.linalg.eigvalsh(matrices).min())
        if min_eigenvalue<=0:violations.append('R_not_positive_definite')
        calculations=[c for r in area['quarters'] for c in r['calculations']]
        report[dk]={'R_min_eigenvalue':min_eigenvalue,'R_all_quarters_optimizer_converged':all(r['fit_info']['converged'] for r in area['quarters']),'joint_fit_seconds':sum(r['fit_seconds'] for r in area['quarters']),'event_and_pair_CDF_seconds':sum(c['seconds'] for c in calculations),'max_numerical_convergence_proxy':max(c['max_discrepancy'] for c in calculations),'max_pair_numerical_convergence_proxy':max(c['pair_discrepancy_max'] for c in calculations),'max_pivotal_nesting_roundoff_clip':max(c['DK_pivotal_roundoff_correction_max'] for c in calculations),'evaluated_q90':area['evaluated_q90'],'evaluated_q95':area['evaluated_q95']}
        if (frame[f'threshold_q95_{dk}']<frame[f'threshold_q90_{dk}']).any():violations.append('physical_threshold_order')
        for q in ('q90','q95'):
            columns=['y_'+q]+[f'M{m}_{q}' for m in range(4)]
            selected=matrix.complete_day_mask(frame,columns)&frame.index.year.isin((2024,2025))
            for m in range(4):
                p=frame.loc[selected,f'M{m}_{q}']; p3=frame.loc[selected,f'M{m}_{q}_all3']; pivot=frame.loc[selected,f'M{m}_{q}_DK_pivotal']
                if (p<0).any() or (p>1).any() or (p3>p+1e-6).any() or (pivot>p+1e-6).any():violations.append('event_probability_bounds')
            if (frame.loc[selected,'y_'+q+'_DK_pivotal']>frame.loc[selected,'y_'+q]).any():violations.append('event_label_nesting')
            for cell in ['pooled']+sorted(frame.cell.dropna().unique()):
                use=selected if cell=='pooled' else selected&(frame.cell==cell)
                for zone in zones:
                    indicator=1-frame.loc[use,f'exceed_{q}_{zone}']
                    for v in (0,1):
                        cdf=frame.loc[use,f'v{v}_{q}_{zone}']; residual=cdf-indicator
                        if (cdf<0).any() or (cdf>1).any():violations.append('marginal_CDF_bounds')
                        marginal.append({'profile':MODE,'combination':dk,'q':q,'cell':cell,'zone':zone,'margin_version':v,'hours':int(use.sum()),'days':int(frame.index[use].normalize().nunique()),'observed_nonexceedance':float(indicator.mean()),'mean_threshold_CDF':float(cdf.mean()),'mean_CDF_minus_nonexceedance':float(residual.mean()),'status':'descriptive cell-average calibration diagnostic; not pointwise conditional calibration certification'})
    if violations:raise AssertionError('External audit invariants failed: '+str(sorted(set(violations))))
    matrix.write_json(OUT/'probability_schema.json',schema)
    matrix.write_json(OUT/'transfer_audit.json',{'profile':MODE,'all_invariants_passed':True,'margin_quarter_fits_checked':len(clock_rows),'daily_G_updates_checked':sum(r['daily_updates'] for r in clock_rows),'quarter_label_end_embargo_days':EMBARGO,'authenticated_historical_versions':False,'combination_reports':report,'source_config':str(CONFIG),'current_script_sha256':matrix.sha(__file__),'statement':'All observed timing checks use explicit nominal release delays with current revised observations; they do not recover actual historic versions. Numerical discrepancies are convergence proxies, not certified integration error bounds.'})
    pd.DataFrame(marginal).to_csv(OUT/'marginal_diagnostics.csv',index=False)
    print(json.dumps(matrix.safe({'profile':MODE,'audit':'passed','combinations':report})),flush=True)


def completed_weather_receipt():
    """Verify completed cycles, all members and the final merged input hash."""
    if not LOCAL_WEATHER_BATCH.exists() or not LOCAL_WEATHER.exists():return None
    try:
        record=json.loads(LOCAL_WEATHER_BATCH.read_text(encoding='utf-8'))
    except (FileNotFoundError,json.JSONDecodeError):
        # The upstream progress JSON is rewritten during the batch. An
        # unfinished write is pending input, not a failed completed profile.
        return None
    if record.get('status')!='completed' or 'summary_sha256' not in record:return None
    if record['requested_days']!=2557 or record['complete_days']!=2557 or record['failed_days']!=0 or record['summary_hours']!=61368:raise AssertionError('Upstream completed manifest does not certify all cycles/hours')
    if Path(record['summary_path']).resolve()!=LOCAL_WEATHER.resolve():raise AssertionError('Upstream receipt refers to another weather profile')
    actual=matrix.sha(LOCAL_WEATHER)
    if actual!=record['summary_sha256']:raise AssertionError('Complete upstream local weather file hash mismatch')
    versions=set();member_rows=0
    for date in pd.date_range('2018-12-31','2025-12-30',freq='D',tz='UTC'):
        directory=LOCAL_WEATHER.parent/'DK_local_weather_daily'/date.strftime('%Y%m%d')
        daily=json.loads((directory/'manifest.json').read_text(encoding='utf-8'))
        if daily['status']!='completed' or daily['members']!=20 or daily['source_receipts']!=100 or daily['member_hourly_rows']!=480 or daily['summary_rows']!=24 or daily['steps']!=[24,30,36,42,48]:raise AssertionError('Incomplete daily local weather receipt')
        members=pd.read_parquet(directory/'member_hourly.parquet',columns=['member_id','target_time'])
        sizes=members.groupby('member_id').size();per_hour=members.groupby('target_time').member_id.nunique()
        if len(members)!=480 or len(sizes)!=20 or not (sizes==24).all() or len(per_hour)!=24 or not (per_hour==20).all() or members.duplicated(['member_id','target_time']).any():raise AssertionError('Incomplete local ensemble member support')
        versions.add(daily['version']);member_rows+=len(members)
    if versions!={record['version']}:raise AssertionError('Regional weather mixes incompatible extraction definitions')
    load_panel() # Full index/schema/K units checks before models see the inputs.
    receipt={'status':'complete_upstream_input_independently_verified','cycles':2557,'target_hours':61368,'members_each_cycle':20,'member_hourly_rows':member_rows,'same_extractor_version_all_cycles':True,'summary_sha256':actual,'upstream_manifest':str(LOCAL_WEATHER_BATCH),'upstream_manifest_sha256':matrix.sha(LOCAL_WEATHER_BATCH),'checked_utc':pd.Timestamp.now(tz='UTC'),'new_model_or_threshold_choices':False}
    matrix.write_json(OUT/'local_weather_input_completion_audit.json',receipt)
    return receipt


def local_auto():
    """One authorised local run after the complete upstream file is stable."""
    if MODE!='local':raise ValueError('The local automatic runner requires mode=local')
    if not CONFIG.exists():raise ValueError('Prespecify the local transfer before waiting for its data')
    print('Waiting for the complete 61368-hour Danish local weather summary; no partial input will be scored',flush=True)
    last_report=time.monotonic()
    while True:
        if completed_weather_receipt() is not None:break
        if time.monotonic()-last_report>=300:
            print('Complete local weather summary is still pending',flush=True);last_report=time.monotonic()
        time.sleep(15)
    print('Complete local weather schema verified; starting the prespecified local transfer',flush=True)
    margins();joint();audit()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=('freeze','margins','joint','audit','local-auto'),required=True);parser.add_argument('--mode',choices=('macro','local'),default='macro');parser.add_argument('--quarter-limit',type=int);args=parser.parse_args()
    set_mode(args.mode)
    if args.stage=='freeze':freeze()
    elif args.stage=='margins':margins(args.quarter_limit)
    elif args.stage=='joint':joint()
    elif args.stage=='audit':audit()
    else:local_auto()


if __name__=='__main__':main()
