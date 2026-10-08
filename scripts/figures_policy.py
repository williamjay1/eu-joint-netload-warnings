"""Nature-width policy figures from executed reanalysis only.

Preparation uses the research Python (parquet); rendering uses dedicated scifig.
No fitted frontier or new deployment-optimality claim is drawn.
"""
from pathlib import Path
import argparse
import json
import sys
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/figures'
IDS=['top1','top2','gate025','original_count2023','inherited_priority019','gate025_rise_cool6',
     'rolling90_target05','early_logit_count','early_logit_top2']
LABELS={'top1':'Top-1','top2':'Top-2','gate025':'Static gate',
        'original_count2023':'Original: 2023 gate',
        'inherited_priority019':'Inherited priority','gate025_rise_cool6':'Rise + cooldown',
        'rolling90_target05':'Rolling gate','early_logit_count':'Early logit: count gate',
        'early_logit_top2':'Early logit: Top-2','none':'No review'}
COLORS={'top1':'#999999','top2':'#222222','gate025':'#0072B2',
        'original_count2023':'#4477AA',
        'inherited_priority019':'#AA4499','gate025_rise_cool6':'#009E73',
        'rolling90_target05':'#D55E00','early_logit_count':'#56B4E9',
        'early_logit_top2':'#332288','none':'#777777'}
MARKERS={'top1':'x','top2':'o','gate025':'s','inherited_priority019':'D',
         'original_count2023':'>',
         'gate025_rise_cool6':'P','rolling90_target05':'v',
         'early_logit_count':'^','early_logit_top2':'*','none':'o'}
LINES={'top1':(0,(1,2)),'top2':'-','gate025':'-',
       'original_count2023':(0,(4,1,1,1)),
       'inherited_priority019':(0,(5,2)),'gate025_rise_cool6':(0,(3,1,1,1)),
       'rolling90_target05':(0,(1,1)),'early_logit_count':'-',
       'early_logit_top2':(0,(7,2)),'none':(0,(3,3))}


def prepare():
    from shared_eval import load_stream,process_masks,summary
    from policy_analysis import categories,daily_components,block_weights,COSTS
    OUT.mkdir(parents=True,exist_ok=True)
    full=load_stream(); f=full.loc[full.index.year>=2024]; proc=process_masks(f)
    base=pd.read_parquet(ROOT/'results/policy_analysis/plans_DE_LU_FR_BE_M0_q90.parquet').reindex(f.index)
    early=pd.read_parquet(ROOT/'results/early_price/review_masks_DE_LU_FR_BE.parquet').reindex(f.index)
    plans={name:base[name].to_numpy(bool) for name in IDS if name in base}
    plans['original_count2023']=early['original_M0__count2023_target05'].fillna(False).to_numpy(bool)
    plans['early_logit_count']=early['forecast_early_logit__count2023_target05'].fillna(False).to_numpy(bool)
    plans['early_logit_top2']=early['forecast_early_logit__top2'].fillna(False).to_numpy(bool)
    plans={name:plans[name] for name in IDS}; plans['none']=np.zeros(len(f),bool)
    stats=[]; decomposition=[]; quarterly=[]
    for name,a in plans.items():
        stats.append(dict(policy=name,**summary(f,a,proc)))
        labels,_=categories(f,a,proc)
        for category in ('first_early','repeat_early','late_event','non_event'):
            decomposition.append(dict(policy=name,category=category,N=int(np.sum(labels==category))))
        for year in (2024,2025):
            for quarter in (1,2,3,4):
                m=(f.index.year==year)&(f.index.quarter==quarter)
                known=f.event.notna().to_numpy()&m
                ndays=int(f.loc[m].event.notna().groupby(f.index[m].normalize()).sum().eq(24).sum())
                quarterly.append(dict(policy=name,year=year,quarter=quarter,
                                      slots=int(np.sum(a&known)),days=ndays,
                                      slots_per_day=np.sum(a&known)/ndays))
    st=pd.DataFrame(stats); de=pd.DataFrame(decomposition)
    # Exact agreement with the executed, independently produced summaries;
    # the 2023 held-out count contract replaces the earlier training-in count.
    metrics=pd.read_csv(ROOT/'results/early_price/policy_metrics.csv')
    for name,model,rule in [('early_logit_count','forecast_early_logit','count2023_target05'),
                            ('early_logit_top2','forecast_early_logit','top2'),
                            ('original_count2023','original_M0','count2023_target05')]:
        check=metrics[(metrics.zone=='DE_LU_FR_BE')&
                      (metrics.model==model)&(metrics.rule==rule)]
        assert len(check)==1
        for field in ['N','hits','C']:
            assert st.set_index('policy').loc[name,field]==check.iloc[0][field],(name,field)
    for name in plans:
        assert de.loc[de.policy==name,'N'].sum()==st.set_index('policy').loc[name,'N']
    st.to_csv(OUT/'figure1_source.csv',index=False)
    de.to_csv(OUT/'figure1_decomposition.csv',index=False)
    pd.DataFrame(quarterly).to_csv(OUT/'figure2_quarterly.csv',index=False)
    days,daily=daily_components(f,plans,proc); W=block_weights(days,reps=2000)
    draws=np.einsum('rd,dpk->rpk',W,daily,optimize=True); totals=daily.sum(axis=0)
    curves=[]; contrasts=[]; names=list(plans)
    for k,name in enumerate(names):
        point=totals[k,2]-COSTS*totals[k,0]
        sims=draws[:,k,2,None]-COSTS*draws[:,k,0,None]
        low,high=np.quantile(sims,[.025,.975],axis=0)
        for j,cost in enumerate(COSTS):
            curves.append(dict(policy=name,cost_ratio=cost,utility=point[j],lower=low[j],upper=high[j]))
        for ref in ['gate025','original_count2023']:
            ri=names.index(ref)
            delta=(totals[k,2]-totals[ri,2])-COSTS*(totals[k,0]-totals[ri,0])
            paired=(draws[:,k,2]-draws[:,ri,2])[:,None]-COSTS*(draws[:,k,0]-draws[:,ri,0])[:,None]
            l,h=np.quantile(paired,[.025,.975],axis=0)
            for j,cost in enumerate(COSTS):
                contrasts.append(dict(policy=name,reference=ref,cost_ratio=cost,
                                      difference=delta[j],lower=l[j],upper=h[j]))
    pd.DataFrame(curves).to_csv(OUT/'figure2_cost_curves.csv',index=False)
    pd.DataFrame(contrasts).to_csv(OUT/'figure2_cost_contrasts.csv',index=False)
    (OUT/'figure2_bootstrap.json').write_text(json.dumps(dict(reps=2000,block=14,seed=20261022,
        synchronous=True,inference='fixedstreams fixedrules andboundaries; yearquartercalendarblocks',
        process_rewards='onsetday',slot_costs='targetday',bands='pointwise95percent',
        no_familywise_or_testselected_frontier=True),indent=2),encoding='utf8')
    print(st.to_string(index=False))


def draw():
    sys.path.insert(0,str(ROOT/'scripts'))
    from figstyle import apply_house_style,mm,panel_label,audit
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    plt=apply_house_style(fontsize=7)
    plt.rcParams.update({'xtick.labelsize':7,'ytick.labelsize':7,'legend.fontsize':7,
        'font.family':'Arial','axes.linewidth':.6,'lines.linewidth':1.,
        'xtick.major.width':.6,'ytick.major.width':.6,'xtick.minor.visible':False,
        'ytick.minor.visible':False,'savefig.pad_inches':.025})
    OUT.mkdir(parents=True,exist_ok=True)
    reports=[]

    def tidy(ax):
        ax.spines[['top','right']].set_visible(False)
        ax.tick_params(direction='out',length=2.5,pad=3,top=False,right=False)

    def output(fig,stem,description):
        report=audit(fig,max_width_mm=181.5,verbose=True)
        assert report['ok'] and report['min_font_pt']>=7
        for ext in ('pdf','svg','png'):
            fig.savefig(OUT/f'{stem}.{ext}',dpi=1200,bbox_inches='tight')
        fig.savefig(OUT/f'{stem}.tiff',dpi=1200,bbox_inches='tight',
                    pil_kwargs={'compression':'tiff_lzw'})
        # EPS cannot represent alpha. Flatten only uncertainty fills onto the
        # white page; the linework, markers and all lettering stay vector.
        restored=[]
        for ax in fig.axes:
            for collection in ax.collections:
                alpha=collection.get_alpha()
                if alpha is not None and alpha<1:
                    face=collection.get_facecolors().copy()
                    restored.append((collection,alpha,face))
                    opaque=face.copy(); opaque[:,:3]=alpha*face[:,:3]+(1-alpha)
                    opaque[:,3]=1.;collection.set_alpha(1.);collection.set_facecolors(opaque)
        fig.savefig(OUT/f'{stem}.eps',dpi=1200,bbox_inches='tight')
        for collection,alpha,face in restored:
            collection.set_alpha(alpha);collection.set_facecolors(face)
        fig.savefig(OUT/f'{stem}_preview.png',dpi=150,bbox_inches='tight')
        report['figure']=stem; report['dpi_png']=1200; report['description']=description
        report['formats']=['PDF','SVG','EPS','PNG1200','TIFF1200LZW','PNG150preview']
        report['eps_alpha']='uncertainty fill colors flattened onto white; vector lines/text retained'
        report['preview_dpi']=150; report['visual_inspection']='pending'
        (OUT/f'{stem}_audit.json').write_text(json.dumps(report,indent=2,
            default=lambda value:value.item() if isinstance(value,np.generic) else str(value)),encoding='utf8')
        plt.close(fig)

    st=pd.read_csv(OUT/'figure1_source.csv').set_index('policy')
    de=pd.read_csv(OUT/'figure1_decomposition.csv')
    fig,axes=plt.subplots(1,2,figsize=(mm(179),mm(112)),
                          gridspec_kw={'width_ratios':[1.0,1.12]})
    fig.subplots_adjust(left=.09,right=.985,bottom=.34,top=.91,wspace=.36)
    a,b=axes
    for name in IDS:
        a.scatter(st.loc[name,'N'],st.loc[name,'C'],s=27 if name!='early_logit_top2' else 46,
                  color=COLORS[name],marker=MARKERS[name],linewidths=.9,zorder=4)
    a.set(xlabel='Selected review windows',ylabel='Processes covered in first 6 h',
          xlim=(200,1580),ylim=(18,46),xticks=[400,800,1200,1600],yticks=[20,25,30,35,40,45])
    panel_label(a,'a'); a.set_title('Native workload and coverage',fontsize=7,pad=8)
    barids=['top2','gate025','original_count2023','inherited_priority019','rolling90_target05','early_logit_count']
    catnames=['first_early','repeat_early','late_event','non_event']
    catcolors=['#0072B2','#56B4E9','#DDAA66','#DDDDDD']
    catlabels=['First early hit','Repeated early hit','Late event hit','Non-event review']
    left=np.zeros(len(barids));yy=np.arange(len(barids))
    for cat,col in zip(catnames,catcolors):
        vals=[int(de[(de.policy==name)&(de.category==cat)].N.iloc[0]) for name in barids]
        b.barh(yy,vals,left=left,height=.58,color=col,edgecolor='white',linewidth=.4)
        left+=vals
    for y,n in zip(yy,left): b.text(n+24,y,f'{int(n):,}',va='center',fontsize=7)
    b.set(yticks=yy,yticklabels=['Top-2','Static gate','Original: 2023 gate','Inherited priority','Rolling gate','Early logit gate'],
          xlabel='Selected review windows',xlim=(0,1660),xticks=[0,500,1000,1500])
    b.invert_yaxis();panel_label(b,'b');b.set_title('Where review is spent',fontsize=7,pad=8)
    for ax in axes:tidy(ax)
    handles=[Line2D([],[],marker=MARKERS[n],color=COLORS[n],linestyle='',markersize=4.5,
                    label=LABELS[n]) for n in IDS]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.50,.115),ncol=3,
               columnspacing=1.5,handletextpad=.6,labelspacing=.6)
    fig.legend(handles=[Patch(facecolor=c,label=l) for c,l in zip(catcolors,catlabels)],
               loc='lower center',bbox_to_anchor=(.5,.018),ncol=2,
               columnspacing=2.8,handlelength=1.3,labelspacing=.55)
    output(fig,'figure1_native_allocation','Executed original-zone M0q90 plus chronologicalearlylogit; nativecounts only, nolinefrontier; categories sum toslots')

    q=pd.read_csv(OUT/'figure2_quarterly.csv')
    curves=pd.read_csv(OUT/'figure2_cost_curves.csv')
    contrasts=pd.read_csv(OUT/'figure2_cost_contrasts.csv')
    fig=plt.figure(figsize=(mm(179),mm(155)))
    gs=fig.add_gridspec(2,2,height_ratios=[.79,1.],left=.095,right=.985,bottom=.215,top=.95,
                        hspace=.53,wspace=.34)
    a=fig.add_subplot(gs[0,:]);b=fig.add_subplot(gs[1,0]);c=fig.add_subplot(gs[1,1])
    for name in ['gate025','inherited_priority019','rolling90_target05','early_logit_count']:
        z=q[q.policy==name].sort_values(['year','quarter'])
        a.plot(np.arange(8),z.slots_per_day,color=COLORS[name],marker=MARKERS[name],
               markersize=4,linestyle=LINES[name],linewidth=1.1,label=LABELS[name])
    a.axhline(.5,color='#777777',ls=(0,(2,2)),lw=.7,zorder=1)
    a.set(xticks=np.arange(8),xticklabels=['Q1','Q2','Q3','Q4','Q1','Q2','Q3','Q4'],
          ylabel='Review windows per complete day',ylim=(0,1.95))
    a.text(.22,-.25,'2024',transform=a.transAxes,ha='center',fontsize=7)
    a.text(.79,-.25,'2025',transform=a.transAxes,ha='center',fontsize=7)
    a.set_title('Quarterly workload; development target = 0.5 windows per day',fontsize=7,pad=8)
    panel_label(a,'a')
    # The full scenario grid is retained in source; plotted range reaches the
    # region where no review dominates all plotted fixed strategies.
    for name in IDS+['none']:
        z=curves[(curves.policy==name)&(curves.cost_ratio<=.08)]
        b.plot(z.cost_ratio,z.utility,color=COLORS[name],linestyle=LINES[name],lw=1.1)
    b.set(xlabel='Review cost / value of one early process',ylabel='Net value (process equivalents)',
          xlim=(0,.08),ylim=(-88,49),xticks=[0,.02,.04,.06,.08])
    b.set_title('Cost scenarios for fixed review rules',fontsize=7,pad=8);panel_label(b,'b')
    for name in ['early_logit_count','rolling90_target05','gate025_rise_cool6']:
        z=contrasts[(contrasts.policy==name)&(contrasts.reference=='original_count2023')&
                    (contrasts.cost_ratio<=.08)]
        if name in ['early_logit_count','rolling90_target05']:
            c.fill_between(z.cost_ratio,z.lower,z.upper,color=COLORS[name],alpha=.17,
                           linewidth=0,zorder=1)
        c.plot(z.cost_ratio,z.difference,color=COLORS[name],linestyle=LINES[name],lw=1.2,zorder=3)
    c.axhline(0,color='#777777',lw=.6,ls=(0,(3,3)))
    c.set(xlabel='Review cost / value of one early process',ylabel='Net-value difference from 2023 gate',
          xlim=(0,.08),ylim=(-18,24),xticks=[0,.02,.04,.06,.08])
    c.set_title('Paired contrasts to original: 2023 gate',fontsize=7,pad=8);panel_label(c,'c')
    for ax in [a,b,c]:tidy(ax)
    handles=[Line2D([],[],color=COLORS[n],linestyle=LINES[n],marker=MARKERS[n] if n!='none' else '',
                    markersize=3.5,label=LABELS[n]) for n in IDS+['none']]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.52,.026),ncol=3,
               columnspacing=1.35,handlelength=2.1,handletextpad=.55,labelspacing=.7)
    output(fig,'figure2_workload_and_cost','Quarterly counts and scenario utility C-rN; 2023 held-out forecast count contract for new logit; pointwise 95 percent paired calendar bootstrap bands for early count and rolling contrasts relative to original forecast 2023 gate; not optimal frontier')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');args=p.parse_args()
    if args.prepare:prepare()
    else:draw()
