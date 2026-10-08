"""Fixed-stream score contrasts and descriptive marginal diagnostics."""
from pathlib import Path
import sys,json
import numpy as np
import pandas as pd
from shared_eval import ROOT,load_stream
sys.path.insert(0,str(Path(__file__).parent/'legacy'))
from evaluation import paired_block_ci
OUT=ROOT/'results/forecast_diagnostics';OUT.mkdir(parents=True,exist_ok=True)
scores=[];contrasts=[];bins=[]
for zone in ('DE_LU_FR_BE','DK1','DK2'):
 for level in ('q90','q95'):
  frames={m:load_stream(zone,m,level) for m in ('M0','M1','M2','M3')}
  for period,years in [('development',(2022,2023)),('reanalysis',(2024,2025))]:
   dloss={}
   for model,f in frames.items():
    f=f.loc[f.index.year.isin(years)];ok=f.p.notna()&f.event.notna();x=f.loc[ok]
    BS=float(np.mean((x.p-x.event)**2));scores.append(dict(zone=zone,level=level,period=period,model=model,hours=len(x),days=len(x)/24,BS=BS,event_rate=float(x.event.mean())))
    dloss[model]=((f.p-f.event)**2).groupby(f.index.normalize()).mean()
    for lo in np.arange(0,1,.1):
     sub=x.loc[(x.p>=lo)&(x.p<(lo+.1 if lo<.9 else 1.000001))]
     bins.append(dict(zone=zone,level=level,period=period,model=model,lo=lo,n=len(sub),mean_p=sub.p.mean(),event_rate=sub.event.mean(),event_days=sub.event.eq(1).groupby(sub.index.normalize()).any().sum()))
   daily=pd.DataFrame(dloss);daily['interaction']=daily.M3-daily.M2-daily.M1+daily.M0;daily['zero']=0.
   for a,b in [('M1','M0'),('M2','M0'),('M3','M1'),('interaction','zero')]:
    out=paired_block_ci(daily,a,b,block_days=14,repetitions=2000,seed=20261008)
    contrasts.append(dict(zone=zone,level=level,period=period,candidate=a,reference=b,delta=out['mean_difference'],low=out['ci95'][0],high=out['ci95'][1],valid_days=out['valid_days']))
pd.DataFrame(scores).to_csv(OUT/'scores.csv',index=False);pd.DataFrame(contrasts).to_csv(OUT/'contrasts.csv',index=False);pd.DataFrame(bins).to_csv(OUT/'reliability_bins.csv',index=False)
m=pd.read_parquet(ROOT/'datasets/marginal_predictions.parquet');v=[];cal=[]
for years,period in [((2022,2023),'development'),((2024,2025),'reanalysis')]:
 idx=m.index.year.isin(years)
 for level in ('q90','q95'):
  f=load_stream(level=level);ids=f.index[f.event.notna()];x=m.loc[idx&m.index.isin(ids)]
  for path in (0,1):
   for region in ('DE_LU','FR','BE'):
    u=x[f'v{path}_{level}_{region}'];y=x[f'exceed_{level}_{region}']
    for cell,ix in [('all',u.index),*[(c,g.index) for c,g in x.groupby('cell')]]:
     uu=u.loc[ix];yy=y.loc[ix];good=uu.notna()&yy.notna();uu=uu[good];yy=yy[good]
     cal.append(dict(period=period,level=level,path=path,region=region,cell=cell,hours=len(uu),mean_exceedance=float(yy.mean()),mean_predicted_exceedance=float((1-uu).mean()),mean_residual=float((yy-(1-uu)).mean()),PIT_coverage=float((x.loc[ix,f'calibrated_pit_{region}']<=float(level[1:])/100).mean())))
     if cell=='all':v.append(dict(period=period,level=level,path=path,region=region,below_half=float((uu<.5).mean()),below_90=float((uu<.9).mean()),q10=uu.quantile(.1),q50=uu.median(),q90=uu.quantile(.9)))
pd.DataFrame(v).to_csv(OUT/'conditional_thresholds.csv',index=False);pd.DataFrame(cal).to_csv(OUT/'marginal_residuals.csv',index=False)
(OUT/'manifest.json').write_text(json.dumps({'analysis':'post hoc fixed-stream diagnostics','bootstrap':'2000 synchronized 14-calendar-day within year/quarter; conditional on fitted probabilities','marginal_residual':'cell mean only, not hourly confidence or calibration certificate','reliability_bin_support':'count of event days supplied; fine-bin results descriptive'},indent=2))
print(pd.DataFrame(scores).query("zone=='DE_LU_FR_BE' and level=='q90'").to_string(index=False))
