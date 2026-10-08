"""Supplementary figures: threshold positions and year-specific early decisions."""
from pathlib import Path
import json,hashlib
import pandas as pd
import numpy as np
from figstyle import apply_house_style,save,audit,mm
plt=apply_house_style(fontsize=7)
plt.rcParams.update({'xtick.labelsize':7,'ytick.labelsize':7,'legend.fontsize':7,
                    'axes.linewidth':.6,'savefig.pad_inches':.025})
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/figures'
COLORS=['#0072B2','#D55E00','#009E73']

def prepare():
    import pyarrow.parquet as pq
    f=pd.DataFrame(pq.read_table(ROOT/'datasets/marginal_predictions.parquet').to_pydict())
    # Arrow materialization preserves the original datetime index in this file.
    time=pd.to_datetime(f['target_time'] if 'target_time' in f else f['__index_level_0__'],utc=True)
    hours=pd.read_parquet(ROOT/'datasets/DE_LU_FR_BE.parquet')
    eligible=hours.index[(hours.index.year>=2024)&hours.policy_event_q90.notna()]
    m=time.isin(eligible);f=f.loc[m].copy()
    assert len(f)==17400
    rows=[]
    for level in ['q90','q95']:
        for region in ['DE_LU','FR','BE']:
            values=np.sort(f[f'v0_{level}_{region}'].to_numpy(float))
            # All exact values are retained; raster export is not interpolated data.
            rows.extend({'level':level,'region':region,'u':u,'ecdf':(i+1)/len(values)} for i,u in enumerate(values))
    pd.DataFrame(rows).to_csv(OUT/'figureS1_source.csv',index=False)

def finish(fig,name,source):
    result=audit(fig,verbose=True);assert result['ok']
    files=save(fig,str(OUT/name),dpi=1200)
    for ext in ['eps','tiff']:
        kw={'dpi':1200,'pil_kwargs':{'compression':'tiff_lzw'}} if ext=='tiff' else {}
        fig.savefig(OUT/f'{name}.{ext}',bbox_inches='tight',**kw)
    fig.savefig(OUT/f'{name}_preview.png',dpi=150,bbox_inches='tight')
    result.update({'dpi':1200,'source':source,'source_sha256':hashlib.sha256((ROOT/source).read_bytes()).hexdigest(),
                   'visual_inspection':'pending','vector_formats':['pdf','svg','eps'],'raster_formats':['png','tiff']})
    (OUT/f'{name}_qa.json').write_text(json.dumps(result,indent=2,default=str),encoding='utf-8')
    plt.close(fig)

def draw():
    f=pd.read_csv(OUT/'figureS1_source.csv')
    fig,ax=plt.subplots(1,2,figsize=(mm(178),mm(88)))
    fig.subplots_adjust(left=.11,right=.975,bottom=.20,top=.74,wspace=.36)
    for a,level,lab in zip(ax,['q90','q95'],['a  Seasonal q90','b  Seasonal q95']):
        for region,col in zip(['DE_LU','FR','BE'],COLORS):
            x=f[(f.level==level)&(f.region==region)]
            a.plot(x.u,x.ecdf,color=col,lw=1,label=region.replace('_','–'))
        a.axvline(.9,color='#777777',lw=.6,ls='--',zorder=0)
        a.set(xlim=(0,1.02),ylim=(0,1.02),xlabel='Conditional marginal CDF at reference',ylabel='Fraction of scored hours')
        a.set_title(lab,loc='left',fontsize=8,pad=12,fontweight='bold')
        a.spines[['right','top']].set_visible(False)
        a.set_xticks([0,.25,.5,.75,1]);a.set_yticks([0,.25,.5,.75,1])
    fig.legend(*ax[0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.53,.98),ncol=3,frameon=False)
    finish(fig,'figureS1_conditional_thresholds','results/figures/figureS1_source.csv')
    f=pd.read_csv(ROOT/'results/early_price/annual_policy_metrics.csv')
    fig,ax=plt.subplots(1,2,figsize=(mm(178),mm(95)))
    fig.subplots_adjust(left=.085,right=.975,bottom=.20,top=.70,wspace=.35)
    rows=[]
    for a,rule,lab in zip(ax,['top2','count2023_target05'],['a  Equal daily Top2 workload','b  Common development count target']):
        for offset,model,label,col in zip([-.23,0,.23],['original_M0','forecast_hour_logit','forecast_early_logit'],['Original joint','Hourly target','Early target'],COLORS):
            vals=[]
            for y in [2024,2025]:
                r=f[(f.zone=='DE_LU_FR_BE')&(f.model==model)&(f.rule==rule)&(f.year==y)].iloc[0]
                vals.append(100*r.C/r.processes)
                rows.append({'rule':rule,'year':y,'model':model,'N':int(r.N),'H':int(r.hits),'C':int(r.C),'processes':int(r.processes),'coverage':100*r.C/r.processes})
            a.bar(np.array([0,1])+offset,vals,width=.205,color=col,label=label)
        a.set_xticks([0,1],['2024 (39 processes)','2025 (32 processes)'])
        a.set(ylim=(0,80),ylabel='Early process coverage (%)')
        a.set_title(lab,loc='left',fontsize=8,pad=13,fontweight='bold')
        a.spines[['right','top']].set_visible(False)
        a.grid(axis='y',color='#DDDDDD',lw=.4);a.set_axisbelow(True)
    fig.legend(*ax[0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.53,.98),ncol=3,frameon=False)
    pd.DataFrame(rows).to_csv(OUT/'figureS2_source.csv',index=False)
    finish(fig,'figureS2_annual_target_alignment','results/early_price/annual_policy_metrics.csv')

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    prepare() if args.prepare else draw()
