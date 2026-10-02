"""Reuse identical past-only threshold fits across shared marginal candidates.

This is memoisation, not a different estimator. The key includes every selected
historical label, its timestamp, settings, and both implementation hashes. Query
outcomes are selected out before key construction. Native OS lock serialises
same-key fits; cached artifacts are checked for byte integrity before use.
"""
from release_paths import work_path as _work_path, raw_path as _raw_path
from contextlib import contextmanager
import hashlib,json,os,pickle,time
from pathlib import Path
import numpy as np
import pandas as pd
from adaptive_thresholds import QuarterlyCalendarThresholds,_quarter_origin,_hourly_utc_index

ROOT=_work_path('cache/calendar_threshold_memo_v1')

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

@contextmanager
def key_lock(path):
    with path.open('a+b') as f:
        if os.name!='nt':raise RuntimeError('Native Windows threshold cache lock required')
        import msvcrt
        f.seek(0,os.SEEK_END)
        if not f.tell():f.write(b'1');f.flush()
        while True:
            try:f.seek(0);msvcrt.locking(f.fileno(),msvcrt.LK_NBLCK,1);break
            except OSError:time.sleep(.25)
        try:yield
        finally:f.seek(0);msvcrt.locking(f.fileno(),msvcrt.LK_UNLCK,1)

def quarter_thresholds(net_load,origin,*,trend,zones,min_coverage=.8,solver_time_limit=120.):
    origin=_quarter_origin(origin);cutoff=origin-pd.Timedelta(days=7)
    index=_hourly_utc_index(net_load.index)
    mask=(index>=cutoff-pd.DateOffset(years=3))&(index<cutoff)
    history=net_load.loc[mask,list(zones)].copy();history.index=index[mask];history=history.sort_index()
    if history.empty:raise ValueError('Threshold cache cannot use an empty history')
    folder=Path(__file__).resolve().parent
    spec={'origin':origin.isoformat(),'trend':trend,'zones':list(zones),'min_coverage':min_coverage,
        'solver_time_limit':solver_time_limit,'history_rows':len(history),
        'history_sha256':hashlib.sha256(pd.util.hash_pandas_object(history,index=True).to_numpy().tobytes()).hexdigest(),
        'code_sha256':{name:sha(folder/name) for name in ('adaptive_thresholds.py','adaptive_threshold_cache.py')}}
    key=hashlib.sha256(json.dumps(spec,sort_keys=True).encode()).hexdigest()
    ROOT.mkdir(parents=True,exist_ok=True)
    artifact=ROOT/(key+'.pickle');manifest=ROOT/(key+'.json')
    with key_lock(ROOT/(key+'.lock')):
        resumed=artifact.is_file() and manifest.is_file()
        if resumed:
            metadata=json.loads(manifest.read_text(encoding='utf8'))
            if metadata.get('spec')!=spec or metadata.get('artifact_sha256')!=sha(artifact):
                raise ValueError('Threshold memo provenance/integrity mismatch')
            with artifact.open('rb') as f:model=pickle.load(f)
        else:
            model=QuarterlyCalendarThresholds(trend=trend,zones=tuple(zones),
                min_coverage=min_coverage,solver_time_limit=solver_time_limit).fit(history,origin)
            temporary=artifact.with_suffix('.tmp.pickle')
            with temporary.open('xb') as f:pickle.dump(model,f)
            temporary.replace(artifact)
            manifest.write_text(json.dumps({'spec':spec,'artifact_sha256':sha(artifact)},indent=2),encoding='utf8')
        expected=pd.date_range(origin,origin+pd.DateOffset(months=3),freq='h',inclusive='left')
        thresholds,diagnostics=model.predict_with_metadata(expected)
        evidence={'key':key,'history_sha256':spec['history_sha256'],'artifact_sha256':sha(artifact),
            'cache_reused':resumed,'cache_artifact':str(artifact),'predictive_outcome_values_read':False}
        return thresholds,model.metadata_,diagnostics,evidence
