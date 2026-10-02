"""Publication redesign of Figures 3--5 and S1; no numerical/model changes.

The original deterministic inputs are read directly.  The layout separates
descriptive native allocation, equal-exposure evaluation and score contrasts.
"""
from __future__ import annotations
import ast
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'temp/nature_mplconfig'))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from nature_visual_style_20261002 import apply_style, figure, panel_label, export_figure

OUT = ROOT / 'generated/nature_visual_redesign_20261002/figures3_5_s1'
OUT.mkdir(parents=True, exist_ok=True)
POLICIES = ['P0_forced_top2', 'P1_risk_gate', 'P2_risk_rise_cooldown']
COLORS = dict(zip(POLICIES, ['#7A7A7A', '#0072B2', '#D55E00']))
DOMAINS = [('DE_LU_FR_BE', 'DE–LU / FR / BE'),
           ('DK1', 'DE–LU / FR / DK1'), ('DK2', 'DE–LU / FR / DK2')]
SOURCES = {}
OUTPUTS = []

def source(rel):
    p = ROOT / rel
    SOURCES[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
    return p

def csv(rel):
    return pd.read_csv(source(rel))

def save(fig, stem, note, values):
    result = export_figure(fig, OUT, stem, source_note=note, values=values)
    if stem == 'figure5_transfer_gate_and_dependence_contrasts':
        import pymupdf
        doc = pymupdf.open(OUT/(stem+'.pdf'))
        spans = [s for b in doc[0].get_text('dict')['blocks'] if 'lines' in b
                 for line in b['lines'] for s in line['spans']]
        text_report = {'minimum_pdf_text_size_pt':min(s['size'] for s in spans),
             'pdf_text_sizes_pt':sorted({round(s['size'],6) for s in spans}),
             'embedded_fonts':[f[3] for f in doc[0].get_fonts()],
             'raster_image_count':len(doc[0].get_images()),
             'label_text':[s['text'] for s in spans if 'units of 0.0001' in s['text']]}
        assert text_report['minimum_pdf_text_size_pt'] >= 7-1e-5
        assert text_report['raster_image_count'] == 0
        assert len(text_report['label_text']) == 1
        assert all('Arial' in f for f in text_report['embedded_fonts'])
        doc.close()
        result['pdf_text_validation'] = text_report
        (OUT/(stem+'_manifest.json')).write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
    OUTPUTS.append(result)
    plt.close(fig)

def legend_axis(fig, rect):
    ax = fig.add_axes(rect)
    ax.set_axis_off()
    return ax

def policy_tables():
    a = csv('results/upgrade_warning_policies_20261002_v3/policy_results.csv')
    a['zone'] = 'DE_LU_FR_BE'
    a = a.rename(columns={'level': 'q'})
    b = csv('results/upgrade_external_warning_policies_20261002/local_weather_strict15d/policy_results.csv')
    return pd.concat([a, b], ignore_index=True)

def figure3():
    table = policy_tables()
    fig = figure(width_mm=180, height_mm=124)
    labels = fig.add_axes([.02, .25, .235, .645])
    labels.set_xlim(0, 1); labels.set_ylim(-1.7, 10.8); labels.axis('off')
    count = fig.add_axes([.285, .25, .255, .645])
    cover = fig.add_axes([.625, .25, .335, .645], sharey=count)
    panel_label(count, 'a', x=-.15, y=1.05)
    panel_label(cover, 'b', x=-.10, y=1.05)
    count.set_title('Native workload', loc='left', pad=8)
    cover.set_title('Early process coverage', loc='left', pad=8)
    records = []
    for j, (zone, name) in enumerate(DOMAINS):
        ytop = 10 - 4*j
        labels.text(0, ytop + .23, name, fontweight='bold', va='center')
        d = table[(table.zone == zone) & (table.q == 'q90') & (table.gap_days == 1)]
        budget = None
        for k, p in enumerate(POLICIES):
            y = ytop - 1 - k
            n = d[(d.policy == p) & (d['mode'] == 'native')].iloc[0]
            m = d[(d.policy == p) & (d['mode'] == 'matched')].iloc[0]
            budget = int(m.matched_review_hours)
            labels.text(.75, y, p[:2], ha='right', va='center')
            count.plot(float(n.unstandardised_review_hours), y, 'o', color=COLORS[p], ms=4.5)
            # Vertical categorical offsets only distinguish the two symbols.
            cover.plot(100*float(n.first_six_hours_coverage), y+.14, 'o', color=COLORS[p], ms=4.3)
            cover.plot(100*float(m.first_six_hours_coverage), y-.14, '^', color=COLORS[p],
                       markerfacecolor='white', markeredgewidth=.9, ms=5.0)
            records.append({'zone':zone, 'policy':p,
                'native_slots':int(n.unstandardised_review_hours),
                'native_coverage':float(n.first_six_hours_coverage),
                'matched_slots':budget,
                'matched_expected_coverage':float(m.first_six_hours_coverage)})
        labels.text(0, ytop-.25, f'Matched budget: {budget} slots', va='center', color='black')
    for ax in (count, cover):
        ax.set_ylim(-1.7, 10.8)
        ax.set_yticks([])
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_linewidth(.6)
    count.set_xlim(0, 1600); count.set_xticks([0, 500, 1000, 1500])
    count.set_xlabel('Selected review slots')
    cover.set_xlim(15, 60); cover.set_xticks([20, 30, 40, 50, 60])
    cover.set_xlabel('Covered in first six hours (%)')
    leg1 = legend_axis(fig, [.02, .105, .96, .05])
    leg1.legend(handles=[Line2D([],[],marker='o',ls='none',ms=4.3,color=COLORS[p],label=s)
        for p,s in zip(POLICIES,['P0  Compulsory top 2','P1  Probability gate','P2  Rise + cooldown'])],
        ncol=3, loc='center', frameon=False, columnspacing=2.2, handletextpad=.7)
    leg2 = legend_axis(fig, [.02, .045, .96, .05])
    leg2.legend(handles=[Line2D([],[],marker='o',ls='none',ms=4.3,color='black',label='Native allocation'),
        Line2D([],[],marker='^',ls='none',ms=5,color='black',markerfacecolor='white',
        markeredgewidth=.9,label='Matched expected coverage')], ncol=2,loc='center',frameon=False,
        columnspacing=3.2, handletextpad=.8)
    save(fig, 'figure3_workload_early_coverage_tradeoff',
        'q90, one-day bridge, 2024–2025. Nine aligned policy rows; filled circles are native; '
        'hollow triangles are analytic expected early coverage at each fixed common budget. '
        'Categorical vertical symbol offsets carry no quantitative meaning.', records)

def figure4():
    a = csv('results/upgrade_warning_policies_20261002_v3/yearly_native_policy_results.csv')
    b = csv('results/upgrade_external_warning_policies_20261002/local_weather_strict15d/yearly_native_policy_results.csv')
    table = pd.concat([a,b], ignore_index=True)
    fig = figure(width_mm=180, height_mm=113)
    records = []
    axs = []
    for j, (zone,name) in enumerate(DOMAINS):
        xpos = [.10,.40,.70][j]
        top = fig.add_axes([xpos,.575,.245,.285])
        bottom = fig.add_axes([xpos,.22,.245,.235])
        axs += [top,bottom]
        top.set_title(name, pad=11)
        if j == 0:
            panel_label(top, 'a', x=-.31, y=1.18)
            panel_label(bottom, 'b', x=-.31, y=1.08)
        for k,p in enumerate(POLICIES[:2]):
            data = table[(table.zone == zone)&(table.q == 'q90')&(table.policy == p)].sort_values('year')
            assert len(data) == 2
            xs = data.year.to_numpy(float) + [-.17,.17][k]
            marker = ['o','s'][k]
            top.plot(xs, 100*data.first6_coverage, marker=marker, ls='none',
                     color=COLORS[p], ms=4.3)
            bottom.bar(xs, data.review_slots/data.complete_days, width=.26,
                       color=COLORS[p], linewidth=0)
            for x,(_,r) in zip(xs, data.iterrows()):
                h,n = int(r.processes_first6_hit),int(r.processes_by_onset_year)
                top.text(x,100*float(r.first6_coverage)+3.0,f'{h}/{n}',ha='center',va='bottom')
                records.append({'zone':zone,'policy':p,'year':int(r.year),
                    'slots':int(r.review_slots),'days':int(r.complete_days),
                    'early_hits':h,'processes':n,'coverage':float(r.first6_coverage)})
        top.set_ylim(30,78); top.set_yticks([40,50,60,70])
        bottom.set_ylim(0,2.15); bottom.set_yticks([0,1,2])
        for ax in (top,bottom):
            ax.set_xlim(2023.63,2025.37);ax.set_xticks([2024,2025])
        top.tick_params(axis='x',labelbottom=False)
        bottom.set_xlabel('Onset / scoring year')
        if j:
            top.tick_params(axis='y',labelleft=False)
            bottom.tick_params(axis='y',labelleft=False)
        else:
            top.set_ylabel('Native early coverage (%)')
            bottom.set_ylabel('Review slots per complete day')
    leg = legend_axis(fig,[.02,.035,.96,.07])
    leg.legend(handles=[Line2D([],[],marker=m,ls='none',ms=4.3,color=COLORS[p],label=s)
        for p,m,s in [('P0_forced_top2','o','P0  Compulsory top 2'),
                      ('P1_risk_gate','s','P1  Probability gate')]],
        ncol=2,loc='center',frameon=False,columnspacing=3,handletextpad=.8)
    save(fig, 'figure4_annual_native_workload_and_coverage',
        'q90 native P0 and P1, not matched exposure. Upper annotations are observed covered '
        'spells / onset-year spells; lower bars are complete-day average review slots. '
        'Small categorical offsets within year separate the policies.', records)

def figure5():
    gate = pd.concat([csv('results/upgrade_gate_vs_quota_intervals_20261002/original/intervals.csv'),
         csv('results/upgrade_gate_vs_quota_intervals_20261002/local/intervals.csv')])
    fig = figure(width_mm=180,height_mm=73)
    left = fig.add_axes([.24,.27,.285,.56])
    right = fig.add_axes([.655,.27,.29,.56])
    label = fig.add_axes([.02,.27,.20,.56]); label.axis('off')
    label.set_xlim(0,1);label.set_ylim(2.6,-.6)
    panel_label(left,'a',x=-.14,y=1.10)
    panel_label(right,'b',x=-.12,y=1.10)
    left.set_title('Review allocation',loc='left',pad=11)
    right.set_title('Dependence choice',loc='left',pad=11)
    vals = []
    for i,(zone,name) in enumerate(DOMAINS):
        rel = ('results/upgrade_margin_dependence_20261002/matrix_contrasts.csv'
               if zone=='DE_LU_FR_BE' else
               f'results/upgrade_external_transfer_20261002/local_weather_strict15d/{zone}/matrix_contrasts.csv')
        d = csv(rel)
        r = d[(d.period=='reanalysis')&(d.q=='q90')&
              (d.contrast=='dependence_at_calibrated')&(d.block_days==14)].iloc[0]
        v,lo,hi = float(r.loss_difference),float(r.ci_low),float(r.ci_high)
        g = gate[(gate.zone==zone)&(gate.q=='q90')&(gate.block_days==14)].iloc[0]
        gl,gh = ast.literal_eval(g.ci95); gv = float(g.difference)
        left.errorbar(100*gv,i,xerr=np.array([[gv-gl],[gh-gv]])*100,
                      fmt='o',color='#0072B2',ms=4.4,lw=1.1,capsize=0)
        right.errorbar(v*1e4,i,xerr=np.array([[v-lo],[hi-v]])*1e4,
                       fmt='o',color='#303030',ms=4.4,lw=1.1,capsize=0)
        label.text(1,i,name,ha='right',va='center')
        vals.append({'zone':zone,'gate_gain_pp':100*gv,'gate_ci_pp':[100*gl,100*gh],
                    'brier_difference':v,'brier_ci':[lo,hi]})
    for ax in (left,right):
        ax.axvline(0,color='#8A8A8A',lw=.65,ls=(0,(3,3)),zorder=0)
        ax.set_ylim(2.6,-.6);ax.set_yticks([]);ax.spines['left'].set_visible(False)
    left.set_xlim(0,31);left.set_xticks([0,10,20,30])
    right.set_xlim(-1.4,1.4);right.set_xticks([-1,0,1])
    left.set_xlabel('P1 − P0 coverage gain (percentage points)')
    # Arial on this host lacks the superscript-minus glyph.  Plain typography
    # retains the full 7 pt label and avoids undersized mathtext superscripts.
    right.set_xlabel('M3 − M1 Brier difference (units of 0.0001)')
    fig.text(.3825,.075,'Positive favours the probability gate',ha='center',va='center')
    fig.text(.80,.075,'Negative favours selected t',ha='center',va='center')
    save(fig, 'figure5_transfer_gate_and_dependence_contrasts',
        'q90, paired pointwise 95% intervals, 14-day calendar blocks. The allocation and '
        'Brier axes have different units and estimands. Danish local weather is primary; '
        'original-region analysis is reanalysis. No multiplicity-adjusted intervals implied.',vals)

def figure_s1():
    frame = pd.read_parquet(source('results/upgrade_warning_policies_20261002_v3/alerts_q90.parquet'))
    fig = figure(width_mm=180,height_mm=145)
    onset_list = [pd.Timestamp('2024-02-27T07:00Z'),pd.Timestamp('2025-01-13T00:00Z')]
    records = []
    for k,onset in enumerate(onset_list):
        base = onset.normalize();start = base-pd.Timedelta(hours=6);end = base+pd.Timedelta(hours=24)
        f = frame.loc[(frame.index>=start)&(frame.index<end)]
        x = (f.index-base).total_seconds()/3600
        ypos = [.685,.24][k]
        risk = fig.add_axes([.135,ypos,.82,.19])
        lanes = fig.add_axes([.135,ypos-.125,.82,.095],sharex=risk)
        panel_label(risk,['a','b'][k],x=-.14,y=1.14)
        fig.text(.135,ypos+.222,onset.strftime('%d %B %Y'),va='center',fontweight='bold')
        hour = (onset-base).total_seconds()/3600
        risk.axvspan(hour,hour+6,facecolor='#F2E8C8',edgecolor='none',zorder=0)
        lanes.axvspan(hour,hour+6,facecolor='#F2E8C8',edgecolor='none',zorder=0)
        risk.plot(x,f.p,color='#303030',lw=1.15)
        risk.set_ylim(-.025,1.025);risk.set_yticks([0,.5,1]);risk.set_ylabel('Issued probability')
        risk.tick_params(axis='x',labelbottom=False)
        rows = [('P1_risk_gate',2,'s','#0072B2'),('P2_risk_rise_cooldown',1,'^','#D55E00')]
        for p,y,m,c in rows:
            sel = f[p].to_numpy(bool)
            lanes.plot(np.asarray(x)[sel],np.full(sel.sum(),y),m,ls='none',ms=4.7,color=c)
        observed = f.event.eq(1).to_numpy()
        lanes.plot(np.asarray(x)[observed],np.zeros(observed.sum()),'|',ls='none',ms=7,
                   markeredgewidth=1.3,color='#303030')
        if k == 1:
            prior = onset-pd.Timedelta(hours=1)
            assert bool(frame.loc[prior,'P2_risk_rise_cooldown'])
            assert not bool(frame.loc[prior,'event'])
            lanes.add_patch(Rectangle((-1,.62),6,.76,facecolor='#E1E1E1',edgecolor='none',zorder=0))
        lanes.set_yticks([2,1,0],['P1','P2','Observed event'])
        lanes.set_ylim(-.5,2.5);lanes.set_xlim(-6,24)
        lanes.set_xticks([-6,0,6,12,18,24]);lanes.spines['left'].set_visible(False)
        lanes.tick_params(axis='y',length=0)
        if k == 1:lanes.set_xlabel('Target time relative to 00 UTC (h)')
        records.append({'onset':str(onset),'window_start':str(start),'window_end':str(end),
             'first6_true_hours':int(f.loc[(f.index>=onset)&(f.index<onset+pd.Timedelta(hours=6)),'event'].eq(1).sum()),
             'forecast_only_selections':{p:[str(t) for t in f.index[f[p]]] for p in POLICIES[1:]},
             'issued_risk':[{ 'timestamp_UTC':str(t),'p':float(v)} for t,v in f.p.items()]})
    leg = legend_axis(fig,[.045,.015,.935,.06])
    leg.legend(handles=[Line2D([],[],color='#303030',lw=1.15,label='Issued joint risk'),
        Patch(facecolor='#F2E8C8',edgecolor='none',label='First six hours'),
        Patch(facecolor='#E1E1E1',edgecolor='none',label='P2 cooldown')],
        ncol=3,loc='center',frameon=False,columnspacing=2.4,handletextpad=.7)
    save(fig,'figureS1_priority_examples',
        'Same two previously selected post hoc directional cases. Selections are forecast-only; '
        'truth bands and realised onset are annotations, not allocation triggers. The grey '
        'lane identifies the six-hour cooldown after the prior planned hour. No additional tests.',records)

def verify_values():
    previous = json.loads(source('results/ese_revision_figures_20261002/figures_manifest.json').read_text(encoding='utf-8'))
    old = {r['stem']:r['values'] for r in previous['outputs'] if r.get('stem')}
    now = {r['stem']:r['values'] for r in OUTPUTS}
    checks = []
    for stem in ['figure3_workload_early_coverage_tradeoff',
                 'figure4_annual_native_workload_and_coverage',
                 'figure5_transfer_gate_and_dependence_contrasts']:
        before,after = old[stem],now[stem]
        assert len(before) == len(after)
        for a,b in zip(before,after):
            assert a.keys() == b.keys(), (stem,a.keys(),b.keys())
            for key in a:
                if isinstance(a[key],str):assert a[key] == b[key]
                else:assert np.allclose(a[key],b[key],atol=1e-15,rtol=0), (stem,key,a[key],b[key])
        checks.append({'figure':stem,'rows_verified':len(before),'status':'exact_numeric_agreement_to_1e-15'})
    old_s1 = json.loads(source('results/manuscript_figures_upgrade_20261002/manifest.json').read_text(encoding='utf-8'))
    for old_case,new_case in zip(old_s1['mechanisms'],now['figureS1_priority_examples']):
        for key in old_case:assert old_case[key] == new_case[key],(key,old_case[key],new_case[key])
    checks.append({'figure':'figureS1_priority_examples','cases_verified':2,
                   'status':'same_onsets_windows_truth_counts_and_planned_hours'})
    report = {'status':'passed','checks':checks,'no_model_refitting':True,
              'layout_does_not_modify_data':True,'float_tolerance':1e-15}
    (OUT/'numeric_verification.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    return report

def main():
    apply_style()
    figure3();figure4();figure5();figure_s1()
    numeric = verify_values()
    captions = '''# Redesigned data figure captions

**Figure 3. Workload and early coverage for three fixed review rules.** Rows align policies within each area combination. (a) Filled circles show native selected-slot counts. (b) Filled circles show native q90 first-six-hour process coverage, and hollow triangles show exact expected coverage after uniform, label-blind retention to the indicated common total budget: 572 slots for DE–LU/FR/BE, 432 for DE–LU/FR/DK1 and 492 for DE–LU/FR/DK2. Small vertical offsets within each policy row distinguish the symbols and carry no quantitative meaning. The evaluation covers 2024–2025 with one inactive day allowed between active days. The three fixed policies are not an optimised performance frontier.

**Figure 4. Annual native coverage and workload for compulsory and gated review.** (a) Native q90 first-six-hour coverage; annotations give covered spells divided by spells whose onset falls in each year. (b) Selected review slots per complete scoring day. P0 and P1 are separated slightly within each year for legibility. Spell boundaries remain intact across years. These are native allocation comparisons, not equal-workload contrasts.

**Figure 5. Review allocation and dependence contrasts.** (a) P1 minus P0 first-six-hour coverage at matched selected-slot exposure; positive values favour the probability gate. (b) M3 minus M1 Brier loss under additionally mapped margins and shared correlation matrices; negative values favour selected t dependence. Points and bars are q90 estimates and paired, pointwise 95% intervals using 14-day calendar blocks. Review intervals resample intact spells by onset day; score intervals synchronously resample common calendar data. Intervals are not multiplicity-adjusted. Danish local-weather specifications are primary; the original-area analysis is reanalysis.

**Figure S1. Post hoc examples of gains and losses from risk-rise priority.** The same two previously selected cases illustrate directional differences between P1 and P2. Upper panels show issued joint risk; lower lanes separate forecast-only planned hours and realised event hours. The tan bands mark the first six elapsed hours after observed onset. The grey P2 band in the second case marks the cooldown after a planned hour at −1 h. Time is relative to 00 UTC on the date shown. Observed onset and events annotate the cases and do not trigger allocation. These illustrations are not independent predictive validation.
'''
    (OUT/'captions.md').write_text(captions,encoding='utf-8')
    manifest = {'status':'generated_pending_visual_review','script':str(Path(__file__).resolve()),
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'style_helper_sha256':hashlib.sha256((ROOT/'scripts/nature_visual_style_20261002.py').read_bytes()).hexdigest(),
        'source_sha256':SOURCES,'outputs':OUTPUTS,'model_fitting':False,
        'numeric_inputs_unchanged':True,'scope':'Data Figures 3–5 and supplementary Figure S1',
        'publication_master':'editable PDF / SVG vectors; raster companion at genuine 1200 dpi',
        'numerical_verification':numeric,
        'all_geometric_checks_passed':all(r['layout']['passed'] for r in OUTPUTS)}
    review_path = OUT/'visual_review.json'
    if review_path.exists():
        review = json.loads(review_path.read_text(encoding='utf-8'))
        current = {r['stem']:r['files']['preview']['sha256'] for r in OUTPUTS}
        if review.get('reviewed_preview_sha256') == current:
            manifest['status'] = 'completed_and_visually_reviewed'
            manifest['visual_review'] = {'path':str(review_path),
                'sha256':hashlib.sha256(review_path.read_bytes()).hexdigest(),
                'method':'All four 300 dpi final-size previews inspected with view_image; '
                         'current preview SHA-256 matches every already-inspected image byte for byte; '
                         'data verified by deterministic readback rather than reading raster digits.'}
    (OUT/'figures_manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps({'status':manifest['status'],'figures':4,'output_directory':str(OUT)},indent=2))

def figure5_only():
    """Refresh only Figure 5 after a typography repair, preserving other files."""
    apply_style()
    manifest_path = OUT/'figures_manifest.json'
    old = json.loads(manifest_path.read_text(encoding='utf-8'))
    SOURCES.update(old['source_sha256'])
    untouched = {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.iterdir()
        if p.is_file() and not p.name.startswith('figure5_') and
        p.name not in ('figures_manifest.json','visual_review.json')}
    prior = next(r for r in old['outputs'] if r['stem']=='figure5_transfer_gate_and_dependence_contrasts')
    figure5()
    fresh = OUTPUTS[0]
    assert len(prior['values']) == len(fresh['values'])
    for a,b in zip(prior['values'],fresh['values']):
        for k in a:
            if isinstance(a[k],str):assert a[k]==b[k]
            else:assert np.allclose(a[k],b[k],atol=1e-15,rtol=0)
    for p,expected in untouched.items():
        assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==expected,p
    old['outputs'] = [fresh if r['stem']==fresh['stem'] else r for r in old['outputs']]
    old['source_sha256'] = SOURCES
    old['script_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    old['style_helper_sha256'] = hashlib.sha256((ROOT/'scripts/nature_visual_style_20261002.py').read_bytes()).hexdigest()
    old['status'] = 'generated_pending_visual_review'
    old.pop('visual_review',None)
    old['last_single_figure_update'] = {'figure':fresh['stem'],
         'change':'Replace 4.9 pt math superscript by complete 7 pt Arial units label',
         'new_label':'M3 − M1 Brier difference (units of 0.0001)',
         'other_files_sha256_preserved':untouched,'numeric_agreement':True}
    manifest_path.write_text(json.dumps(old,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps({'status':old['status'],'updated_figure':fresh['stem'],
                      'other_files_preserved':len(untouched)},indent=2))

if __name__ == '__main__':
    if '--figure5-only' in sys.argv:figure5_only()
    else:main()
