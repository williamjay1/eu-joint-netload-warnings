"""Populate the article from the final executed result tables, without refitting."""
from pathlib import Path
import re
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
P=ROOT/'manuscript/asmbi_manuscript.md'

def markdown_table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,r))+' |' for r in rows])

def bib_entries(text):
    out={}
    for m in re.finditer(r'@\w+\s*\{\s*([^,]+),',text):
        pos=m.end(); depth=1; start=pos
        while depth and pos<len(text):
            depth+=(text[pos]=='{')-(text[pos]=='}'); pos+=1
        body=text[start:pos-1]; fields={}
        for f in re.finditer(r'(?m)^\s*(\w+)\s*=\s*\{',body):
            x=f.end(); y=x; d=1
            while d and y<len(body):
                d+=(body[y]=='{')-(body[y]=='}');y+=1
            fields[f.group(1).lower()]=body[x:y-1]
        out[m.group(1).strip()]=fields
    return out

def plain(s):
    s=s.replace(r'{\"u}','ü').replace(r'{\"o}','ö').replace('\\_','_').replace('\\ ',' ').replace('$','')
    return s.replace('{','').replace('}','').replace('--','–')

def references():
    old=Path('D:/MLWork/eu_joint_netload_EYENG7621_20261006/submission/latex/jee_references.bib').read_text(encoding='utf-8')
    new=(ROOT/'audit/verified_references.bib').read_text(encoding='utf-8')
    entries=bib_entries(old+'\n'+new)
    keys=['vanderWiel2019','Bloomfield2020','Kittel2024','Browell2021','Browell2022','Gioia2025','Hirsch2024','Richardson2000','Murphy1977','Ehm2016','Gneiting2007a','Gneiting2007b','Ziegel2014','Franc2023','Tatbul2018','Dimitriadis2021','Morris2019','Demarta2005','Energinet2026','EnerginetPrices','EnerginetTerms','NordPool2022']
    items=[]
    for key in keys:
        f=entries[key];author=plain(f['author']).replace(' and ','; ');year=f['year']
        if key=='Gneiting2007a': year='2007a'
        if key=='Gneiting2007b': year='2007b'
        title=plain(f['title']); journal=plain(f.get('journal',f.get('booktitle',f.get('howpublished',''))))
        num=f.get('volume','')+('('+f['number']+')' if 'number' in f else '')
        loc=(num+', '+plain(f.get('pages',''))).strip(', ')
        link='https://doi.org/'+f['doi'] if 'doi' in f else plain(f.get('url',''))
        label=f"{author} ({year}). {title}. {journal}"+(', '+loc if loc else '')+'.'
        if link:label+=f' [Source]({link}).'
        items.append(label)
    (ROOT/'manuscript/asmbi_references.bib').write_text('\n\n'.join(re.search(r'@\w+\s*\{\s*'+re.escape(k)+r',',old+'\n'+new).group(0) for k in []) if False else old+'\n'+new,encoding='utf-8')
    return '\n\n'.join(sorted(items,key=str.lower))

def main():
    s=P.read_text(encoding='utf-8')
    replacements={
    'ABSTRACT_FINAL': '''Probabilistic monitoring of interconnected electricity systems must allocate limited review capacity as well as estimate joint risk. We study hourly synchronous high net load in Germany–Luxembourg, France and Belgium, distinguishing event-hour hits, selected delivery windows and coverage during a process's first six hours. A chronological, post hoc reanalysis of 2024–2025 contains 17,400 scored hours and 71 processes. Static gating reduces selections from 1,450 to 618 and retains 322 of the 341 event hours found by daily Top2, but early coverage falls from 35 to 32 processes. A forecast-only rolling gate approaches a mean target of 0.5 selections per day at the cost of eight further early discoveries. Under the same Top2 allocation, a logistic forecast targeting early event hours covers 43 processes, compared with 38 for an otherwise identical hourly-target model and 35 for the original joint forecast; year-specific performance and conditional intervals temper the pooled advantage. A controlled marginal-map × copula-family comparison shows no clear decision gain from the selected t family. Yet local marginal-CDF stress scenarios alter 16.2% of ordinary and 28.0% of stateful baseline selections, illustrating why small score differences do not guarantee stable decisions. Known-mechanism simulations separate gate crossings from capacity competition. Danish-pivotal endpoints and assumed-availability price controls delimit transfer evidence. The contribution is an executable evaluation linking probability diagnostics, selection sensitivity and explicit process-cost contracts, with workload and early-discovery trade-offs reported together.''',
    'INTRODUCTION_FINAL': '''A monitoring desk can receive a credible hourly risk forecast and still examine the wrong part of a developing episode. Weather links electricity demand and renewable supply across borders, creating periods when several regions have high net load together. European weather-regime studies explain this coincidence and the duration of renewable shortages (van der Wiel et al., 2019; Bloomfield et al., 2020; Kittel and Schill, 2024). For a desk with limited review capacity, however, identifying another high-risk hour late in an episode has a different value from identifying its first few hours. The forecasting target, the review contract and the evaluation endpoint must reflect that difference.

Regional net-load forecasting has already moved beyond independent predictions. Browell and Fasiolo (2021) combine gridded weather forecasts with conditional extremes; Browell et al. (2022) study energy covariance structures. Gioia et al. (2025) develop additive covariance matrix models and evaluate regional net demand, aggregates and differences with operational relevance. Within electricity markets, Hirsch and Ziel (2024) model cross-product dependence for simulation-based intraday forecasting. These contributions make a new copula family an insufficient claim by itself. The practical question here is what a controlled dependence comparison changes when review slots are scarce, marginal probabilities remain uncertain and high-risk hours arrive in persistent clusters.

Proper scores assess the quality of probabilistic predictions (Gneiting and Raftery, 2007a), while calibration and sharpness describe complementary properties (Gneiting et al., 2007b). Neither determines the value of a particular action without its cost and consequence. Cost-loss evaluation formalizes this dependence (Murphy, 1977; Richardson, 2000), and different threshold weights can reverse forecast rankings (Ehm et al., 2016). Selective classification supplies an established language for allowing abstention (Franc et al., 2023). Our review decision is related but distinct: several named future hours can be selected from one issued trajectory, and repeated selections within a process may have little additional discovery value. Range-oriented detection measures recognize this temporal structure (Tatbul et al., 2018).

Three gaps motivate the evaluation. First, a gate can appear better after retrospective workload equalization even though its actual list is a subset of daily Top2 and therefore loses some discoveries. Native allocation and hypothetical exposure standardization need separate interpretation. Second, a tiny Brier-score difference can mask changed decisions around a gate or daily ranking boundary. Conditional marginal error must be propagated through selection, including the state of a cooldown rule; pooled calibration alone cannot certify that error is small. Third, ranking hourly event probabilities does not directly optimize discovery near onset. An elementary early-target classifier with the same available predictors can test whether the target mismatch matters before proposing a more complex temporal model.

We address these gaps using issued probability streams and source-provider records for Germany–Luxembourg, France and Belgium, followed by combinations replacing Belgium with either Danish zone. The analysis compares abstention, daily Top1/Top2, static and rolling gates, and separated temporal-ranking mechanisms. It crosses an additional conditional marginal map with Gaussian and selected t dependence under the same correlation matrix, derives ordinary-selection sufficient conditions from familiar probability bounds, and replays nine marginal-stress paths for stateful decisions. Known-probability simulations distinguish threshold crossing from capacity competition; direct early forecasting and price controls test two competing sources of decision information.

The resulting contribution is a decision evaluation with explicit benefit units, resource contracts and calibration-sensitivity conditions. It does not depend on a new theorem or on declaring a universal winner. All added 2024–2025 comparisons are identified as reanalyses of previously inspected outcomes. The empirical target is a regional pressure indicator: simultaneous seasonal-threshold exceedance warrants review, while reserve inadequacy, outages and realized response costs require additional system information.''',
    'EARLY_RESULT_FINAL': '''The direct early-target experiment changes the ranking without changing the daily Top2 workload. It covers 43 processes, against 38 for the same-feature hourly-target logistic model and 35 for the original joint forecast; hourly hits are 340, 349 and 341, respectively. The paired early-target minus hourly-target increment is five processes, with a conditional 95% interval [−3,14]. Relative to the original forecast, the increment is eight [−2,17]. The pooled difference is informative as a target-alignment check, but does not establish a stable superiority.

For the common 2023 development resource contract, the original, hourly-target and early-target gates are 0.23, 0.05 and 0.02. They selected 175, 173 and 175 slots in 357 validation forecast days. In reanalysis, they select 642, 823 and 735 slots and cover 32, 37 and 41 processes. Thus the early forecast adds nine discoveries [−1,18] relative to the original while adding 93 slots [54,130]. Relative to the same-feature hourly model, it uses 88 fewer slots [−113,−71] and adds four discoveries [−4,13]. A common development mean target does not produce equal realized workload under shift. In 2024, the early gate covers 25 of 39 processes versus 15 for the original; in 2025 it covers 16 of 32 versus 17. The same-feature hourly gate covers 19 and 18. Figure S2 and Table S7 retain these annual contrasts.

The early classifier's Brier loss for its own early-hour label is 0.012313. For the general hourly event label its loss is 0.109771, compared with 0.062902 for the hourly logistic control and 0.060191 for M0. These are different probability tasks: the early forecast is useful for examining target-aligned ranking, rather than replacing a general joint-event forecast. Quarterly scores, calibration support and conditional contrasts are included in Appendix S1.''',
    'STABILITY_RESULT_FINAL': '''Selection sensitivity is concentrated within chosen hours rather than the full hourly series (Figure 3). Across the central path and eight constant regional CDF-shift paths with δ=0.03, 219 ordinary-rule hours and 333 stateful-rule hours change membership. These are only 1.26% and 1.91% of all scored hours, but include 16.18% and 27.97% of the baseline selected lists. Ordinary early coverage ranges from 31 to 33 processes; inherited stateful coverage ranges from 34 to 36. With δ=0.01, the corresponding baseline-selected fractions are 4.37% and 9.79%.

Continuous-box sufficient conditions are stricter than observed invariance across nine paths. For the ordinary gate at δ=0.03, the coupling condition certifies 54 of 618 baseline selections and the monotone condition certifies 212; these selected subsets cover five and sixteen early processes. At δ=0.01, the counts are 264 and 447, covering nineteen and twenty-eight. The conditions offer usable conditional checks for ordinary allocation, but the declared boxes are not justified hour-specific error bounds. For the stateful rule, finite scenario replay reports sensitivity rather than a continuous-box certificate. Raw Top2 rankings also change on hours later excluded by gating, so counting raw ranking changes alone exaggerates allocation consequences.''',
    'PRICE_RESULT_FINAL': '''Under the stated price-availability assumption, raw price Top2 covers 24 of 71 DK1 processes and 22 of 75 DK2 processes, below the original forecast's 35 and 40. Comparing combined price/risk logistic forecasts with probability/calendar logistic controls isolates conditional price information more fairly. Combined hourly-target Top2 covers 40 versus 35 DK1 processes and 48 versus 41 DK2 processes at the same 1,450 slots. The paired increments are five [−1,12] and seven [2,13]. These auxiliary comparisons across zones and targets are not multiplicity-adjusted evidence of a general price advantage. They depend on assumed price availability and a current snapshot. The allocation excludes unavailable next-local-day-plus-one prices from direct sorting; 1,151 scored UTC evening hours per zone are masked in 2024–2025.

On Danish-pivotal hours within parent onsets, the combined hourly forecast covers 25 of 45 DK1 opportunities and 35 of 54 DK2 opportunities, versus 22 and 29 for the original forecast. Reclustering pivotal events instead gives 30 of 63 and 40 of 76, versus 26 and 36. The different denominators represent different discovery questions, with full-list cost retained in both.''',
    'REFERENCES_FINAL':references(),
    }
    # Exact operational methods, distinct legacy and newly learned contracts.
    s=s.replace('The baseline regional model is a regularized Gaussian location-scale additive model followed by daily beta-PIT recalibration over mature observations. The second marginal path adds a weekly beta map in six season/weather cells determined before the reanalysis.', 'The raw regional distribution G is a regularized Gaussian location-scale additive model. The two marginal paths are F⁰(y)=B_daily(G(y)) and F¹(y)=B_cell(F⁰(y)), where the first beta-PIT map is updated daily over mature observations and the second weekly within six fixed season/weather cells. The second map is fitted to PIT values already transformed by the first; insufficiently supported cells use B_cell(u)=u. The cell partition precedes this reanalysis.')
    s=s.replace('Native Top2 and 2022 count-selected gates evaluate each target, keeping thresholds appropriate to their different probability scales.', 'Native Top2 gives an equal-workload comparison. Primary count gates use the same issued 2023 validation days and a common threshold grid, keeping thresholds appropriate to the different probability scales. These validation forecasts use 2022 parameter fits, while 2023 also selects the penalty; the period is development, not an independent test. The 2022 count contract is secondary and its newly fitted classifier probabilities are explicitly training-distribution predictions.')
    s=s.replace('subsequent quarterly expanding updates apply the maturity cutoff.', 'subsequent quarterly expanding updates retain the selected penalty and apply hour_end+7 days≤issue in the original combination and hour_end+15 days≤issue for newly fitted Danish models.')
    s=s.replace('(Gneiting et al., 2007; Ziegel and Gneiting, 2014)', '(Gneiting et al., 2007b; Ziegel and Gneiting, 2014)')
    s=s.replace('approximately [−0.0001125,0.0000351]', '[−0.00011324,0.00003494]')
    # Tables use final machine-readable calculations, not typed alternative versions.
    pol=pd.read_csv(ROOT/'results/policy_analysis/policy_summary.csv')
    pol=pol[(pol.zone=='DE_LU_FR_BE')&(pol.model=='M0')&(pol.level=='q90')&(pol.period=='all')].set_index('policy')
    labels={'none':'No review','top1':'Top1','top2':'Top2','gate025':'Static gate 0.25','inherited_priority019':'Rise/cooldown 0.19','gate025_cool6':'Cooldown 0.25','gate025_rise':'Rise ranking 0.25','gate025_rise_cool6':'Rise/cooldown 0.25','rolling90_target05':'Rolling count gate'}
    # Resolve exact stored rolling name once, fail rather than silently omit.
    if 'rolling90_target05' not in pol.index:
        labels={('rolling90_target05' if k=='rolling90_target05' else k):v for k,v in labels.items()}
        rolling=[k for k in pol.index if 'rolling' in k][0]
        labels[rolling]=labels.pop('rolling90_target05')
    rows=[]
    for k,l in labels.items():
        r=pol.loc[k];rows.append([l,int(r.N),int(r.hits),int(r.C),f'{100*r.coverage:.1f}',f'{r.slots_per_day:.3f}'])
    ep=pd.read_csv(ROOT/'results/early_price/policy_metrics.csv')
    for model,label in [('original_M0','Original: 2023 gate'),('forecast_hour_logit','Hourly target: 2023 gate'),('forecast_early_logit','Early target: 2023 gate'),('forecast_hour_logit','Hourly target: Top2'),('forecast_early_logit','Early target: Top2')]:
        rule='top2' if 'Top2' in label else 'count2023_target05'
        r=ep[(ep.zone=='DE_LU_FR_BE')&(ep.model==model)&(ep.rule==rule)].iloc[0]
        rows.append([label,int(r.N),int(r.hits),int(r.C),f'{100*r.coverage:.1f}',f'{r.N/r.complete_days:.3f}'])
    replacements['TABLE1']='Table 1. Native q90 review outcomes in the original combination, 2024–2025.\n\n'+markdown_table(['Rule','Slots N','Event hits H','Early processes C','Coverage (%)','Slots/day'],rows)+'\n\nAll rules share 725 scored complete UTC days, 2,105 event hours and 71 processes. The cap is two slots per day; count gates share a development mean target, not an identical realized workload. Native early discovery never uses an observed onset to trigger selection.'
    scores=pd.read_csv(ROOT/'results/forecast_diagnostics/scores.csv')
    policyall=pd.read_csv(ROOT/'results/policy_analysis/policy_summary.csv')
    rows=[]
    for model in ['M0','M1','M2','M3']:
        a=scores[(scores.zone=='DE_LU_FR_BE')&(scores.level=='q90')&(scores.model==model)]
        p=policyall[(policyall.zone=='DE_LU_FR_BE')&(policyall.level=='q90')&(policyall.model==model)&(policyall.period=='all')&(policyall.policy=='gate025')].iloc[0]
        rows.append([model,{'M0':'Baseline / Gaussian','M1':'Mapped / Gaussian','M2':'Baseline / selected t','M3':'Mapped / selected t'}[model],f'{a[a.period=="development"].iloc[0].BS:.6f}',f'{a[a.period=="reanalysis"].iloc[0].BS:.6f}',int(p.N),int(p.hits),int(p.C)])
    replacements['TABLE2']='Table 2. Controlled marginal-map × copula-family comparison.\n\n'+markdown_table(['Model','Margins / family','Development BS','Reanalysis BS','Slots N','Hits H','Early C'],rows)+'\n\nBrier scores refer to the same hourly q90 event. Allocations use the common 0.25 gate. All configurations share the dynamic correlation matrix, weather covariates, event labels and observation masks. Development covers 2022–2023; reanalysis covers 2024–2025.'
    ds=pd.read_csv(ROOT/'results/policy_analysis/danish_specific_early.csv')
    rows=[]
    for zone in ['DK1','DK2']:
        for policy,label in [('top2','Original Top2'),('gate025','Static gate 0.25'),('inherited_priority019','Rise/cooldown 0.19'),('rolling90_target05','Rolling count gate')]:
            a=ds[(ds.zone==zone)&(ds.level=='q90')&(ds.policy==policy)]
            if a.empty and 'rolling' in policy:a=ds[(ds.zone==zone)&(ds.level=='q90')&ds.policy.str.contains('rolling')]
            x=a[a.endpoint=='pivotal_parent_onset'].iloc[0];y=a[a.endpoint=='pivotal_reclustered'].iloc[0]
            rows.append([zone,label,int(x.N),f'{int(x.C)}/{int(x.processes)}',f'{int(y.C)}/{int(y.processes)}'])
    ps=pd.read_csv(ROOT/'results/early_price/pivotal_policy_metrics.csv')
    for zone in ['DK1','DK2']:
        for key,label in [('base_hour_logit__top2','Risk/calendar Top2'),('combined_hour_logit__top2','Price/risk/calendar Top2')]:
            a=ps[(ps.zone==zone)&(ps.model_policy==key)].iloc[0]
            rows.append([zone,label,int(a.N),f'{int(a.nested_covered)}/{int(a.nested_opportunities)}',f'{int(a.reclustered_covered)}/{int(a.reclustered_processes)}'])
    replacements['TABLE3']='Table 3. Danish-pivotal early coverage under two process definitions.\n\n'+markdown_table(['Zone','Rule','All slots N','Parent-onset C/opportunities','Reclustered C/processes'],rows)+'\n\nThe parent-onset definition preserves the original joint-process start; the pivotal-reclustered definition forms new processes from Danish-pivotal hours. Cost includes every issued review slot. Price/risk comparisons assume the price availability stated in Section 2.3.'
    for key,value in replacements.items():
        if key not in s:raise ValueError(f'Missing article placeholder {key}')
        s=s.replace(key,value)
    figures={
    '### 3.2 Resource adaptation':('figure1_native_allocation','Figure 1. Native review workload and early coverage. Outcomes cover 725 complete UTC days and 71 q90 processes. First early hits count distinct processes; repeated early hits, late event hits and non-event selections are mutually exclusive and sum to N. Newly fitted count gates use 2023 validation forecast distributions; historical gates retain their earlier development contracts.'),
    '### 3.3 Average forecast':('figure2_workload_and_cost','Figure 2. Workload drift and process-cost scenarios. Quarterly allocations retain their native lists. Process utility is C−rN, including no review; the displayed range is a scenario comparison rather than a deployable optimal frontier. The paired-contrast reference is the original forecast with its 2023 count gate, 0.23. Shading gives pointwise conditional 95% intervals from 2,000 synchronized fourteen-day block draws; it does not include model refitting. Full cost grids and single-contrast simultaneous bands are archived.'),
    '### 3.4 Known mechanisms':('figure3_decision_stability','Figure 3. Marginal stress and decision stability. Scenario-critical hours are the union of changes across the central path and eight constant regional CDF-shift paths. Counts are shown relative to all scored hours and to the baseline selected list. Continuous-box sufficient conditions apply only to ordinary top-two gating, conditional on an assumed δ box. Raw Top2 ranking changes and final gated-list changes are separately shown; scenario invariance of the cooldown rule is not a continuous-box certificate.'),
    '### 3.5 Transfer and market':('figure4_simulation_mechanisms','Figure 4. Known-probability simulations separate threshold crossings from selected-list changes. Narrow and wide boundary regimes use matched regional marginal error and common random paths; paired contrasts show Monte Carlo SE. Oracle workload and early coverage vary with weather persistence and a separately changed stationary weather mixture. Each regime has 400 replications of 120 days. All panels are synthetic, with known probabilities; no empirical forecasting improvement is inferred from them.'),
    }
    for prefix,(name,caption) in figures.items():
        s=s.replace(prefix,f'![{caption.split(".")[0]}](../results/figures/{name}_preview.png)\n\n{caption}\n\n'+prefix,1)
    P.write_text(s,encoding='utf-8')
    abstract=re.search(r'## Abstract\n\n(.*?)\n\nKeywords:',s,re.S).group(1)
    print('Abstract words:',len(abstract.split()),'; article words:',len(s.split()))
    print('Written:',P)

if __name__=='__main__':main()
