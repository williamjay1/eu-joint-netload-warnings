"""Common fixed-forecast evaluation. Allocation never receives outcomes."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(Path(__file__).parent/'legacy'))
from evaluation import event_processes

def load_stream(zone='DE_LU_FR_BE', model='M0', level='q90'):
    x=pd.read_parquet(ROOT/'datasets'/f'{zone}.parquet')
    p=x[f'{model}_{level}'].astype(float).copy()
    # Forecast incompleteness precedes allocation; truth incompleteness follows it.
    good=p.notna().groupby(x.index.normalize()).sum().eq(24)
    p.loc[~x.index.normalize().isin(good.index[good])]=np.nan
    y=x[f'policy_event_{level}'].astype(float)
    return pd.DataFrame({'p':p,'event':y},index=x.index)

def select(f, gate=0., cooldown=0, rise=False, k=2):
    p=f.p.to_numpy(float); idx=f.index; ns=idx.as_unit('ns').asi8
    last=pd.Series(p,index=idx).shift(1).rolling(6,min_periods=1).max().fillna(0).to_numpy()
    priority=p*(1-np.clip(last,0,1)) if rise else p
    gates=np.broadcast_to(np.asarray(gate),p.shape)
    a=np.zeros(len(p),bool); selected=[]; cool=int(cooldown*3600*1e9)
    for _,ix in pd.Series(np.arange(len(f)),index=idx).groupby(idx.normalize()):
        ii=ix.to_numpy()
        if len(ii)!=24 or not np.isfinite(p[ii]).all(): continue
        c=ii[p[ii]>=gates[ii]]
        order=np.lexsort((ns[c],-p[c],-priority[c])); picked=[]
        recent=[t for t in selected if t>=ns[ii[0]]-cool] if cool else []
        for at in c[order]:
            if cool and any(abs(ns[at]-t)<cool for t in recent+picked): continue
            a[at]=True; picked.append(ns[at])
            if len(picked)==k: break
        selected.extend(picked)
    return a

def process_masks(f,gap=1):
    records=[]
    for e in event_processes(f,max_empty_days=gap):
        ix=f.index.get_indexer(e['event_hours'])
        early=ix[e['event_hours']<e['onset']+pd.Timedelta(hours=6)]
        records.append(dict(onset=e['onset'],end=e['end'],ix=ix,early_ix=early,
                            uncertain_boundary=e['uncertain_boundary']))
    return records

def summary(f,a,processes=None):
    a=np.asarray(a,bool)&f.event.notna().to_numpy(); y=f.event.fillna(0).to_numpy(bool)
    proc=process_masks(f) if processes is None else processes
    hit=[bool(a[e['early_ix']].any()) for e in proc]
    return dict(N=int(a.sum()),hits=int((a&y).sum()),false=int((a&~y).sum()),
                C=int(sum(hit)),processes=len(proc),coverage=float(np.mean(hit)) if hit else np.nan,
                event_hours=int(y.sum()),complete_days=int(f.event.notna().groupby(f.index.normalize()).sum().eq(24).sum()))
