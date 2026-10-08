"""Figures 3–4: exact result-table values, finite scenarios and Monte Carlo SEs.

All calculations are read-only transformations of executed results.  Panel
source tables, checksums and print-size previews accompany vector exports.
"""
from __future__ import annotations
import hashlib,json,sys
from pathlib import Path
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

sys.path.insert(0,str(Path(__file__).resolve().parent))
from figstyle import apply_house_style,save,panel_label,audit,mm
plt=apply_house_style(fontsize=7)
plt.rcParams.update({'xtick.labelsize':7,'ytick.labelsize':7,'legend.fontsize':7,
    'savefig.pad_inches':.025,'axes.linewidth':.6,'xtick.major.width':.6,
    'ytick.major.width':.6,'xtick.minor.visible':False,'ytick.minor.visible':False})
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/figures'
BLUE='#0072B2';ORANGE='#D55E00';GREEN='#009E73';GRAY='#53575B'
SOURCES=[]

def read(path,parquet=False):
    path=ROOT/path;SOURCES.append(path)
    if parquet:
        # Use the declared dependency in the selected Python environment.
        # No author-specific installed skill or secondary environment is read.
        import pyarrow.parquet as pq
        return pd.DataFrame(pq.read_table(path).to_pydict())
    return pd.read_csv(path)

def style(ax,grid=False):
    for spine in ('top','right'):ax.spines[spine].set_visible(False)
    ax.tick_params(direction='out',length=2.5,pad=2)
    ax.set_axisbelow(True)
    if grid:ax.grid(axis='x',color='#E2E4E6',linewidth=.45)

def title(ax,text):
    panel_label(ax,text,pad=28)

def export(fig,stem,source_rows,caption):
    OUT.mkdir(parents=True,exist_ok=True)
    result=audit(fig,verbose=True)
    if not result['ok']:raise RuntimeError(result)
    if result['tight_mm'][1]>128:raise RuntimeError('Figure exceeds 130 mm export height')
    paths=save(fig,str(OUT/stem),dpi=1200)
    eps=OUT/(stem+'.eps');tiff=OUT/(stem+'.tiff')
    fig.savefig(eps,bbox_inches='tight')
    fig.savefig(tiff,dpi=1200,bbox_inches='tight',pil_kwargs={'compression':'tiff_lzw'})
    paths.update({'eps':str(eps),'tiff':str(tiff)})
    preview=OUT/(stem+'_preview.png')
    fig.savefig(preview,dpi=150,bbox_inches='tight')
    pd.DataFrame(source_rows).to_csv(OUT/(stem+'_source.csv'),index=False)
    from PIL import Image
    from pypdf import PdfReader
    im=Image.open(paths['png']);ti=Image.open(tiff)
    page=PdfReader(paths['pdf']).pages[0]
    pdfmm=[float(page.mediabox.width)/72*25.4,float(page.mediabox.height)/72*25.4]
    if pdfmm[0]>183 or pdfmm[1]>130:raise RuntimeError('Actual PDF export exceeds 183 x 130 mm')
    fonts=[]
    for v in page['/Resources']['/Font'].values():
        font=v.get_object();fonts.append({'basefont':str(font['/BaseFont']),'subtype':str(font['/Subtype'])})
    result.update({'files':paths,'preview':str(preview),'raster_dpi':1200,
        'preview_dpi':150,'all_fonts_at_least_7pt':True,
        'actual_export_mm':pdfmm,'png_pixel_size':list(im.size),'png_dpi':im.info['dpi'],
        'tiff_pixel_size':list(ti.size),'tiff_compression':ti.info.get('compression'),
        'embedded_pdf_fonts':fonts,'actual_width_height_pass':True,
        'sources':[{'path':str(p.relative_to(ROOT)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in dict.fromkeys(SOURCES)],
        'caption':caption,'visual_inspection':'pending actual preview inspection'})
    (OUT/(stem+'_qa.json')).write_text(json.dumps(result,indent=2,
        default=lambda x:x.item() if isinstance(x,np.generic) else str(x)),encoding='utf8')
    plt.close(fig)

def figure3():
    SOURCES.clear()
    summary=read('results/stability/stability_summary.csv')
    cert=read('results/stability/stability_certificates.csv')
    corners=read('results/stability/stability_corner_paths.csv')
    fig=plt.figure(figsize=(mm(180),mm(111)))
    gs=fig.add_gridspec(2,2,left=.135,right=.978,bottom=.105,top=.84,
        wspace=.46,hspace=1.20,height_ratios=[1.,.80])
    a=fig.add_subplot(gs[0,0]);b=fig.add_subplot(gs[0,1]);c=fig.add_subplot(gs[1,:])
    rows=[]
    positions=[3,2,1,0]
    labels=[]
    for y,(_,r) in zip(positions,summary.iterrows()):
        labels.append(('Gate' if r.rule=='ordinary_gate' else 'Priority')+f'  δ = {r.delta_each_region:.2f}')
        for field,label,color,marker in [('scenario_critical_fraction','All eligible hours',GRAY,'o'),
            ('baseline_selected_scenario_critical_fraction','Baseline selections',BLUE,'s')]:
            val=100*r[field]
            a.plot(val,y,marker=marker,color=color,markersize=4,linestyle='none',markeredgewidth=.6)
            rows.append({'panel':'a','delta':r.delta_each_region,'rule':r.rule,'metric':label,'value':val,
                'source':'results/stability/stability_summary.csv','field':field})
    a.set_yticks(positions,labels);a.set_ylim(-.5,3.5);a.set_xlim(-.6,31)
    a.set_xticks([0,10,20,30]);a.set_xlabel('Selections that change (%)')
    a.legend([Line2D([],[],color=GRAY,marker='o',ls='none'),Line2D([],[],color=BLUE,marker='s',ls='none')],
        ['All hours','Selected hours'],ncol=2,loc='lower left',bbox_to_anchor=(-.03,1.01),
        borderaxespad=0,columnspacing=.8,handletextpad=.4)
    title(a,'a  Selection sensitivity');style(a,True)
    for y,delta in [(1,.01),(0,.03)]:
        for offset,bound,color in [(.17,'union_bound',GRAY),(-.17,'monotone_probability_envelope',BLUE)]:
            r=cert.loc[cert.delta_each_region.eq(delta)&cert.bound.eq(bound)].iloc[0]
            value=100*r.baseline_selected_certified_fraction
            b.barh(y+offset,value,height=.26,color=color,edgecolor='none')
            rows.append({'panel':'b','delta':delta,'rule':'ordinary_gate','metric':bound,'value':value,
                'source':'results/stability/stability_certificates.csv','field':'baseline_selected_certified_fraction'})
    b.set_yticks([1,0],['δ = 0.01','δ = 0.03']);b.set_ylim(-.5,1.5);b.set_xlim(0,100)
    b.set_xticks([0,25,50,75,100]);b.set_xlabel('Certified baseline selections (%)')
    b.legend([Patch(facecolor=GRAY),Patch(facecolor=BLUE)],['Coupling bound','Copula envelope'],
        loc='lower left',bbox_to_anchor=(-.04,1.01),borderaxespad=0,ncol=1,
        handlelength=1.3,handletextpad=.5,labelspacing=.2)
    title(b,'b  Ordinary gate certificates');style(b,True)
    for y,policy,field in [(2,'Raw Top-2','probability_top2_symmetric_difference'),
        (1,'Gate','list_symmetric_difference'),(0,'Priority + cooldown','list_symmetric_difference')]:
        subsetrule='rise_cooldown' if y==0 else 'ordinary_gate'
        for delta,offset,color,marker in [(.01,.12,BLUE,'o'),(.03,-.12,ORANGE,'s')]:
            s=corners.loc[corners.delta_each_region.eq(delta)&corners.rule.eq(subsetrule),field]
            lo,hi=float(s.min()),float(s.max());mid=(lo+hi)/2
            c.errorbar(mid,y+offset,xerr=np.array([[mid-lo],[hi-mid]]),color=color,
                marker=marker,markersize=3.6,linewidth=1.25,capsize=2,markeredgewidth=.6)
            rows.append({'panel':'c','delta':delta,'rule':policy,'metric':'eight-corner min/max',
                'value':mid,'minimum':lo,'maximum':hi,'source':'results/stability/stability_corner_paths.csv','field':field})
    c.set_yticks([2,1,0],['Raw Top-2','Gate','Priority + cooldown']);c.set_ylim(-.45,2.45)
    c.set_xlim(-8,750);c.set_xticks([0,150,300,450,600,750]);c.set_xlabel('List symmetric difference (delivery-hour slots)')
    c.legend([Line2D([],[],color=BLUE,marker='o',lw=1.2),Line2D([],[],color=ORANGE,marker='s',lw=1.2)],
        ['δ = 0.01','δ = 0.03'],loc='lower left',bbox_to_anchor=(0,1.02),ncol=2,borderaxespad=0,
        handlelength=1.6,handletextpad=.4,columnspacing=1.5)
    title(c,'c  Capacity filters ranking changes');style(c,True)
    caption=('Sensitivity of native decisions to specified marginal-CDF perturbations. '
        'Panels a–b use 17,400 complete test hours. Panel a includes the unperturbed centre plus '
        'eight full-history constant regional-sign shifts; denominators are all eligible hours '
        'or baseline selections. Priority denotes the inherited rise ranking, gate 0.19 and '
        'six-hour cooldown; ordinary gate is 0.25. Panel b gives sufficient ordinary-gate '
        'threshold/rank certificates from the 3δ coupling bound or the fixed-copula monotone '
        'rectangle envelope. Panel c ranges are minima and maxima across eight signed scenarios, '
        'not confidence intervals. Corner invariance does not certify every interior or '
        'time-varying stateful perturbation; δ values are sensitivity scales, not validated error coverage.')
    export(fig,'figure3_decision_stability',rows,caption)

def paired(frame,metric,policy='Gate'):
    x=frame.loc[frame.policy.eq(policy)&frame.model.isin(['margin_near','margin_far'])]
    x=x.pivot(index=['persistence','drift','rep'],columns='model',values=metric)
    if x.isna().any().any():raise RuntimeError('Paired simulation rows incomplete')
    d=x.margin_near-x.margin_far
    return d.groupby(level=[0,1]).agg(['mean','std','count']).assign(MCSE=lambda a:a['std']/np.sqrt(a['count']))

def figure4():
    SOURCES.clear()
    narrow=read('results/simulations/replicates.parquet',True)
    wide=read('results/simulations/wide_boundary/replicates.parquet',True)
    summary=read('results/simulations/summary.csv')
    fig=plt.figure(figsize=(mm(180),mm(126)))
    gs=fig.add_gridspec(2,2,left=.14,right=.985,bottom=.09,top=.85,
        wspace=.40,hspace=1.10,height_ratios=[1.,.95])
    axes=[fig.add_subplot(gs[i,j]) for i,j in [(0,0),(0,1),(1,0),(1,1)]]
    a,b,c,d=axes;rows=[]
    conditions=[(p,drift) for p in [0.,.65,.9] for drift in [False,True]]
    ys=np.arange(6)[::-1]
    labels=[f'{p:g}'+('  shifted' if drift else '  baseline') for p,drift in conditions]
    for ax,metric,letter,text in [(a,'gate_crossings','a','Gate crossings'),(b,'changed_slots','b','Final list changes')]:
        for frame,name,color,marker,offset in [(narrow,'Narrow boundary',BLUE,'o',.12),(wide,'Wide boundary',ORANGE,'s',-.12)]:
            table=paired(frame,metric)
            for y,key in zip(ys,conditions):
                r=table.loc[key];mean=float(r['mean']);error=1.96*float(r.MCSE)
                ax.errorbar(mean,y+offset,xerr=error,fmt=marker,color=color,markersize=3.6,
                    markeredgewidth=.6,capsize=1.8,linewidth=.8)
                rows.append({'panel':letter,'DGP':name,'persistence':key[0],'drift':key[1],
                    'metric':metric,'contrast':'margin_near minus margin_far','mean':mean,
                    'MCSE':float(r.MCSE),'replications':int(r['count']),
                    'source':f'results/simulations/{"wide_boundary/" if name.startswith("Wide") else ""}replicates.parquet'})
        ax.set_yticks(ys,labels);ax.set_ylim(-.55,5.55)
        ax.set_xlabel('Near − far (slots per replicate)')
        if metric=='gate_crossings':ax.set_xlim(-12,410);ax.set_xticks([0,100,200,300,400])
        else:ax.set_xlim(-15,68);ax.set_xticks([-10,0,20,40,60])
        ax.axvline(0,color=GRAY,lw=.6,zorder=0)
        title(ax,letter+'  '+text);style(ax,True)
    a.set_ylabel('State persistence and test distribution',labelpad=4)
    a.legend([Line2D([],[],color=BLUE,marker='o',ls='none'),Line2D([],[],color=ORANGE,marker='s',ls='none')],
        ['Narrow','Wide'],loc='lower left',bbox_to_anchor=(-.02,1.01),ncol=2,borderaxespad=0,
        handlelength=1,handletextpad=.4,columnspacing=.9)
    policies=[('Top2','Top-2',GRAY),('Gate','Gate',BLUE),('RiseCooldown','Priority',ORANGE)]
    for ax,metric,letter,text in [(c,'N','c','Review volume'),(d,'C','d','Early process coverage')]:
        for ip,(policy,label,color) in enumerate(policies):
            for drift in [False,True]:
                for ix,persistence in enumerate([0.,.65,.9]):
                    r=summary.loc[summary.model.eq('oracle')&summary.policy.eq(policy)&
                        summary.persistence.eq(persistence)&summary.drift.eq(drift)].iloc[0]
                    pos=ix+(ip*2+int(drift)-2.5)*.115
                    value=float(r[metric+'_mean']);se=float(r[metric+'_MCSE'])
                    ax.bar(pos,value,width=.10,color=color if not drift else 'white',
                        edgecolor=color,linewidth=.7,hatch='///' if drift else None,
                        yerr=1.96*se,error_kw={'elinewidth':.55,'capsize':1.2,'capthick':.55})
                    rows.append({'panel':letter,'DGP':'Narrow boundary','persistence':persistence,
                        'drift':drift,'policy':policy,'model':'oracle','metric':metric,
                        'mean':value,'MCSE':se,'source':'results/simulations/summary.csv'})
        ax.set_xticks([0,1,2],['0','0.65','0.9']);ax.set_xlim(-.5,2.5)
        ax.set_xlabel('State persistence')
        if metric=='N':ax.set_ylabel('Selected slots, N');ax.set_ylim(0,260);ax.set_yticks([0,60,120,180,240])
        else:ax.set_ylabel('Covered early processes, C');ax.set_ylim(0,9.7);ax.set_yticks([0,3,6,9])
        title(ax,letter+'  '+text);style(ax)
    c.legend([Patch(facecolor=color,edgecolor=color) for _,_,color in policies],
        [label for _,label,_ in policies],loc='lower left',bbox_to_anchor=(-.01,1.01),
        ncol=3,borderaxespad=0,handlelength=1,handletextpad=.35,columnspacing=.7)
    d.legend([Patch(facecolor=GRAY,edgecolor=GRAY),Patch(facecolor='white',edgecolor=GRAY,hatch='///')],
        ['Baseline','Shifted weather mix'],loc='lower left',bbox_to_anchor=(-.01,1.01),
        ncol=1,borderaxespad=0,handlelength=1.2,handletextpad=.4,labelspacing=.2)
    caption=('Known-probability mechanism simulations with 400 paired 120-day replicates per condition. '
        'Panels a–b show near-minus-far differences under equal regional marginal perturbation '
        'count and magnitude, for the ordinary 0.25 gate with capacity two. Points contrast '
        'the primary narrow-boundary negative marginal shift and the exploratory wide-boundary '
        'positive shift; these are separate DGP fixtures, not isolated amplitude comparisons. '
        'Gate crossings can increase while final list changes decrease. Panels c–d use oracle '
        'risks in the primary narrow fixture; priority uses rise ranking plus six-hour cooldown '
        'under the same gate 0.25. Solid bars use the baseline weather-state mixture and hatched '
        'bars the alternative stationary test mixture, not a within-run change. Error bars '
        'are ±1.96 Monte Carlo SE (paired for a–b), and describe simulation estimation precision. '
        'The figures are synthetic evidence and do not estimate operating-system outcomes.')
    export(fig,'figure4_simulation_mechanisms',rows,caption)

if __name__=='__main__':
    figure3();figure4()
