"""Nature-inspired Figures 1, 2 and 6, with preserved scientific meanings."""
from pathlib import Path
import json
import hashlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle,FancyArrowPatch
from matplotlib.lines import Line2D
from nature_visual_style_20261002 import apply_style,figure,export_figure,COLORS,sha

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'generated/nature_visual_redesign_20261002/figures1_2_6'
OLD=ROOT/'results/ese_revision_figures_20261002'

def arrow(ax,start,end,color=None,lw=.75,scale=6):
    ax.add_patch(FancyArrowPatch(start,end,arrowstyle='-|>',
                 mutation_scale=scale,linewidth=lw,color=color or COLORS['ink']))

def text(ax,x,y,s,**kwargs):
    return ax.text(x,y,s,fontsize=7,ha='left',va='top',linespacing=1.35,**kwargs)

def tag(ax,x,y,s):
    return ax.text(x,y,s,fontsize=8,fontweight='bold',ha='left',va='top')

def f1():
    fig=figure(height_mm=102)
    ax=fig.add_axes([.025,.02,.95,.96]);ax.set_xlim(0,1);ax.set_ylim(0,1);ax.axis('off')
    tag(ax,.015,.99,'a')
    text(ax,.053,.988,'Forecast-to-review workflow',fontweight='bold')
    xs=[.017,.267,.517,.767];w=.215;y=.725;h=.19
    heads=['Inputs','Marginal forecasts','Joint probabilities','Review watchlist']
    bodies=['Historical records\nArchived GEFS weather',
            'Original margins F0\nConditional map F1',
            'Gaussian / selected t\nFour streams M0–M3',
            'P0 quota · P1 gate\nP2 rise + cooldown']
    for x,head,body in zip(xs,heads,bodies):
        ax.add_patch(Rectangle((x,y),w,h,facecolor='white',edgecolor=COLORS['rule'],lw=.6))
        ax.add_patch(Rectangle((x,y+h-.010),w,.010,facecolor=COLORS['blue'],edgecolor='none'))
        text(ax,x+.011,y+h-.031,head,fontweight='bold')
        text(ax,x+.011,y+.102,body)
    for x in xs[:-1]: arrow(ax,(x+w+.004,y+h/2),(x+.25-.005,y+h/2))
    text(ax,.017,.672,'Issued forecasts precede target hours; realised labels enter only after allocation.')
    tag(ax,.015,.578,'b')
    text(ax,.053,.577,'Controlled 2 × 2 probability comparison',fontweight='bold')
    # Open matrix: readable row labels and separate column headings.
    text(ax,.180,.485,'Gaussian',fontweight='bold')
    text(ax,.338,.485,'Selected t',fontweight='bold')
    text(ax,.016,.392,'F0\nOriginal')
    text(ax,.016,.246,'F1\nMapped')
    for (cx,cy,s) in [(.154,.325,'M0'),(.312,.325,'M2'),(.154,.18,'M1'),(.312,.18,'M3')]:
        ax.add_patch(Rectangle((cx,cy),.14,.105,facecolor=COLORS['light'],edgecolor='none'))
        ax.text(cx+.07,cy+.0525,s,ha='center',va='center',fontsize=7,fontweight='bold')
    text(ax,.016,.12,'Common weather, thresholds, hours and R')
    tag(ax,.535,.578,'c')
    text(ax,.573,.577,'Separate evaluation targets',fontweight='bold')
    ypos=[.438,.308,.178]
    labels=[('Marginal support','Threshold errors and sensitivity'),
            ('Joint probability','Brier loss and reliability'),
            ('Review performance','Slots, event hits and early coverage')]
    for yy,(head,body) in zip(ypos,labels):
        ax.add_patch(Rectangle((.538,yy-.080),.007,.086,facecolor=COLORS['blue'],edgecolor='none'))
        text(ax,.562,yy,head,fontweight='bold')
        text(ax,.562,yy-.043,body)
    text(ax,.017,.03,'Delayed historical replay; fixed-specification transfer to the Danish combinations.')
    return export_figure(fig,OUT,'figure1_evaluation_workflow',
        'Conceptual redesign only; unchanged controlled matrix, warning rules and validation roles.',
        {'models':{'M0':['F0','Gaussian'],'M1':['F1','Gaussian'],
                   'M2':['F0','t'],'M3':['F1','t']},
         'main_policy_input':'M0','secondary_fixed_P1_models':['M0','M1','M2','M3']})

def f2():
    fig=figure(height_mm=102)
    ax=fig.add_axes([.05,.025,.91,.95]);ax.set_xlim(-26,26);ax.set_ylim(0,1);ax.axis('off')
    tag(ax,-25,.98,'a');text(ax,-23,.978,'Information and market clocks',fontweight='bold')
    yy=.72
    arrow(ax,(-24,yy),(24,yy),lw=.65,scale=6)
    for xx in (-24,-12,0,23):
        ax.plot([xx,xx],[yy-.025,yy+.025],lw=.65,color=COLORS['ink'])
    text(ax,-24,.875,'GEFS initialisation\nD−1 00 UTC')
    ax.text(-12,.875,'Forecast issue\nD−1 12 UTC',ha='center',va='top',fontsize=7,linespacing=1.35)
    ax.text(11.5,.875,'Delivery day D\n00–23 UTC',ha='center',va='top',fontsize=7,linespacing=1.35)
    ax.plot([0,23],[yy,yy],color=COLORS['blue'],lw=1)
    # Separate market ticks and their text; no leader line through any label.
    for xx in (-14,-13): ax.plot([xx,xx],[yy-.027,yy-.066],color=COLORS['vermillion'],lw=.8)
    text(ax,-22,.605,'Day-ahead gate\nD−1 12:00 CET / CEST\n11 / 10 UTC')
    ax.add_patch(FancyArrowPatch((-12,.455),(0,.455),arrowstyle='<->',mutation_scale=6,
                                color=COLORS['ink'],lw=.65))
    ax.text(-6,.5,'12 h to first target hour',ha='center',va='bottom',fontsize=7)
    ax.text(11.5,.61,'12–35 h forecast-to-target lead',ha='center',va='top',fontsize=7)
    text(ax,-24,.354,'12 h replay allowance does not reconstruct historical first availability.')
    tag(ax,-25,.245,'b');text(ax,-23,.243,'Proposed review before selected operating hours',fontweight='bold')
    steps=[(-24,'Supply + reserves'),(-7.5,'Imports + capacity'),(10,'Intraday preparation')]
    for xx,label in steps:
        ax.add_patch(Rectangle((xx,.08),14,.089,facecolor=COLORS['light'],edgecolor='none'))
        ax.text(xx+7,.1245,label,ha='center',va='center',fontsize=7)
    for a,b in [((-9.6,.1245),(-8,.1245)),((6.9,.1245),(9.4,.1245))]:arrow(ax,a,b)
    return export_figure(fig,OUT,'figure2_information_and_review_timeline',
        'Unchanged GEFS and issue clocks. Review actions are proposed, not observed; day-ahead auction closes before issue.',
        {'GEFS_init_UTC':'D-1 00:00','issue_UTC':'D-1 12:00',
         'day_ahead_gate_UTC':{'winter_CET':'D-1 11:00','summer_CEST':'D-1 10:00'},'target_hour_lead_h':[12,35]})

def f6():
    case=json.loads((OLD/'figure6_forecast_only_case.json').read_text(encoding='utf-8'))
    values=case['hourly_probabilities']
    hours=np.array([x['hour_UTC'] for x in values]);p=np.array([x['p'] for x in values])
    chosen=np.array(case['P0_selected_target_hours_UTC'])
    assert (p<.25).all() and np.isclose(p.max(),case['max_issued_probability'])
    assert chosen.tolist()==hours[np.lexsort((hours,-p))[:2]].tolist()
    fig=figure(height_mm=127)
    left=.19;right=.975;width=right-left
    a=fig.add_axes([left,.69,width,.225])
    b=fig.add_axes([left,.345,width,.23])
    c=fig.add_axes([left,.105,width,.12])
    for ax in (a,b,c):
        ax.set_xlim(-.6,23.6);ax.set_xticks([0,4,8,12,16,20,23])
    a.plot(hours,p,'o-',color=COLORS['blue'],markersize=2.5,lw=.8)
    a.axhline(.25,color=COLORS['ink'],ls=(0,(3,2)),lw=.65)
    a.set_ylim(-.012,.29);a.set_yticks([0,.1,.2,.25]);a.set_ylabel('Joint probability')
    a.tick_params(axis='x',labelbottom=False)
    b.plot(hours,p,'o-',color=COLORS['blue'],markersize=2.5,lw=.8)
    b.scatter(chosen,p[chosen],s=30,facecolors='white',edgecolors=COLORS['P0'],linewidths=.8,zorder=5)
    b.set_ylim(-.0003,.0062);b.set_yticks([0,.002,.004,.006]);b.set_ylabel('Joint probability')
    b.tick_params(axis='x',labelbottom=False)
    c.scatter(chosen,[1,1],s=20,marker='s',color=COLORS['P0'])
    c.set_ylim(-.5,1.5);c.set_yticks([1,0],['P0 · 2 slots','P1 · 0 slots'])
    c.spines['left'].set_visible(False);c.tick_params(axis='y',length=0,pad=9)
    c.set_xlabel('Target hour (UTC)')
    # All labels live in dedicated gutters, outside data axes.
    for letter,yy,label in [('a',.933,'Full probability range'),
                           ('b',.608,'Low-probability detail'),
                           ('c',.256,'Review allocation')]:
        fig.text(.045,yy,letter,ha='left',va='bottom',fontsize=8,fontweight='bold')
        fig.text(left,yy,label,ha='left',va='bottom',fontsize=7,fontweight='bold')
    handles=[Line2D([],[],color=COLORS['blue'],marker='o',ms=2.5,lw=.8,label='Issued M0 probability'),
             Line2D([],[],color=COLORS['ink'],ls=(0,(3,2)),lw=.65,label='P1 gate = 0.25'),
             Line2D([],[],color=COLORS['P0'],marker='o',mfc='white',ms=4,lw=0,label='P0 selected hours')]
    fig.legend(handles=handles,loc='upper left',bbox_to_anchor=(left,.99),ncol=3,
               borderaxespad=0,handlelength=1.6,columnspacing=1.5)
    casepath=OUT/'figure6_forecast_only_case.json';OUT.mkdir(parents=True,exist_ok=True)
    casepath.write_text(json.dumps(case,indent=2),encoding='utf-8')
    return export_figure(fig,OUT,'figure6_forecast_only_quiet_day',
        'Same saved forecast-only chosen day and exact 24 probabilities as the content revision. Panel b is explicitly zoomed, not a changed risk scale in panel a.',case)

def main():
    style=apply_style();OUT.mkdir(parents=True,exist_ok=True)
    records=[f1(),f2(),f6()]
    manifest={'status':'rendered_layout_passed_pending_visual_review','figures':records,
              'source_sha256':{str(OLD/'figure6_forecast_only_case.json'):sha(OLD/'figure6_forecast_only_case.json'),
                               str(ROOT/'scripts/nature_visual_style_20261002.py'):sha(ROOT/'scripts/nature_visual_style_20261002.py'),
                               str(Path(__file__)):sha(Path(__file__))},
              'style':style,'no_fitting_or_case_reselection':True}
    (OUT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'status':manifest['status'],'figures':len(records)}))

if __name__=='__main__': main()
