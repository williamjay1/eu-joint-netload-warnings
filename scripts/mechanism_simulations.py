"""ADEMP fixtures with known shared-shock probabilities, never real evidence."""
from pathlib import Path
import json,time
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/simulations'; OUT.mkdir(parents=True,exist_ok=True)
SEED=20261008

def choose(p,gate=.25,rise=False,cool=0,k=2):
    n=len(p);a=np.zeros(n,bool);prev=[]
    if rise:
        before=np.zeros(n)
        for lag in range(1,7): before[lag:]=np.maximum(before[lag:],p[:-lag])
        priority=p*(1-before)
    else:priority=p
    for start in range(0,n,24):
        ix=np.arange(start,start+24);c=ix[p[ix]>=gate]
        order=np.lexsort((c,-p[c],-priority[c]));recent=[t for t in prev if t>=start-cool];picked=[]
        for j in c[order]:
            if cool and any(abs(j-t)<cool for t in recent+picked):continue
            a[j]=True;picked.append(j)
            if len(picked)>=k:break
        prev.extend(picked)
    return a

def process(y):
    active=np.flatnonzero(y.reshape(-1,24).any(axis=1));out=[];current=[];last=-999
    for d in active:
        if d-last>2 and current:out.append(current);current=[]
        current.extend(np.flatnonzero(y[d*24:(d+1)*24])+d*24);last=d
    if current:out.append(current)
    return [(np.asarray(ix),np.asarray(ix)[np.asarray(ix)<ix[0]+6]) for ix in out]

def stats(p,y,a,proc,truep):
    C=sum(a[e].any() for _,e in proc);N=int(a.sum())
    return [np.mean((p-y)**2),np.mean(abs(p-truep)),N,C,len(proc),C-.005*N,
            int((a&y).sum()),int((a&~y).sum())]

def run(reps=400,days=120,amplitude=.012,shift=-.035,outsub=None):
    global OUT
    if outsub:
        OUT=ROOT/'results/simulations'/outsub;OUT.mkdir(parents=True,exist_ok=True)
    started=time.perf_counter();rng=np.random.default_rng(SEED);records=[]
    for persistence in (0.,.65,.9):
      for drift in (False,True):
       for rep in range(reps):
        weights=np.array([.74,.21,.05]) if not drift else np.array([.56,.32,.12])
        state=np.empty(days,int);state[0]=rng.choice(3,p=weights)
        for d in range(1,days):state[d]=state[d-1] if rng.random()<persistence else rng.choice(3,p=weights)
        s=np.repeat(state,24);hour=np.tile(np.arange(24),days)
        z=np.where(s==0,.0002,np.where(s==1,.315,.60)+np.where(s==1,amplitude,.08)*np.cos((hour-10)/24*2*np.pi))
        w=.30;truep=w*z+(1-w)*(3*z*z-2*z*z*z)
        common=rng.random(len(z))<w;U=rng.random((len(z),3));Uc=rng.random(len(z));U[common]=Uc[common,None]
        y=((U<z[:,None]).sum(axis=1)>=2);proc=process(y)
        near=np.flatnonzero(abs(truep-.25)<.025);far=np.flatnonzero((abs(truep-.25)>.12)&((z>.035) if shift<0 else (z<.965)))
        n=min(len(near),len(far));near=rng.choice(near,n,False) if n else near;far=rng.choice(far,n,False) if n else far
        streams={'oracle':truep,'wrong_dependence':3*z*z-2*z*z*z}
        for name,ix in [('margin_near',near),('margin_far',far)]:
            zz=z.copy();zz[ix]=zz[ix]+shift
            streams[name]=w*zz+(1-w)*(3*zz*zz-2*zz*zz*zz)
        for model,p in streams.items():
          for policy,g,r,c in [('Top2',0,False,0),('Gate',.25,False,0),('RiseCooldown',.25,True,6)]:
            a=choose(p,g,r,c);v=stats(p,y,a,proc,truep)
            mae=n/len(z)*abs(shift) if model.startswith('margin_') else 0.
            records.append([persistence,drift,rep,model,policy,mae,*v,int(np.sum(a!=choose(truep,g,r,c))),int(np.sum((p>=g)!=(truep>=g)))])
    columns=['persistence','drift','rep','model','policy','marginal_MAE','BS','probability_MAE','N','C','processes','utility_r005','hits','false','changed_slots','gate_crossings']
    df=pd.DataFrame(records,columns=columns);df.to_parquet(OUT/'replicates.parquet',index=False)
    values=['BS','probability_MAE','N','C','processes','utility_r005','hits','false','changed_slots','marginal_MAE','gate_crossings']
    g=df.groupby(['persistence','drift','model','policy'])
    m=g[values].mean();se=g[values].std()/np.sqrt(reps)
    m.columns=[f'{c}_mean' for c in m];se.columns=[f'{c}_MCSE' for c in se]
    pd.concat([m,se],axis=1).reset_index().to_csv(OUT/'summary.csv',index=False)
    # Same marginals alone do not identify whether onset is predictable.
    # A two-hour fixture distinguishes a supplied phase signal from a noise score.
    temporal=[]
    for mode in ('shared_occurrence','independent_occurrence'):
      for r in range(1000):
        truth=rng.random((100,2))<.4
        if mode=='shared_occurrence':truth[:,1]=truth[:,0]
        temporal.append([mode,r,float(truth.any(axis=1).mean())])
    td=pd.DataFrame(temporal,columns=['mode','rep','at_least_one']);tg=td.groupby('mode').at_least_one
    tab=pd.DataFrame({'estimate':tg.mean(),'MCSE':tg.std()/np.sqrt(1000)})
    tab['truth']=tab.index.map({'independent_occurrence':.64,'shared_occurrence':.4});tab.to_csv(OUT/'temporal_fixture.csv')
    # Isolated one-hour hit opportunities, not the observed active-day process.
    # Unconditional hourly risks are equal; observing phase changes conditional risks.
    phase=[]
    for rep in range(1000):
        onset=rng.integers(0,24,100);noise=rng.integers(0,24,100)
        hit_known=1.;hit_noise=np.mean((noise==onset)|((noise+6)%24==onset))
        phase.append([rep,hit_known,hit_noise])
    ph=pd.DataFrame(phase,columns=['rep','known_phase_coverage','noise_phase_coverage']);ph.to_csv(OUT/'phase_fixture.csv',index=False)
    design={'seed':SEED,'replications_per_primary_DGP':reps,'days':days,'boundary_amplitude':amplitude,'mechanism':'shared-uniform mixture w=.30, true q=w*z+(1-w)*(3*z^2-2*z^3)','weather_state_stationary_weights_baseline':[.74,.21,.05],'drift_weights':[.56,.32,.12],'drift_interpretation':'different stationary test distribution, not a within-run regime change or a validation of rolling adaptation','persistence':[0,.65,.9],'marginal_shift':shift,'perturbation_location':'near gate abs(q-.25)<.025 versus abs(q-.25)>.12, interior z ensures equal number and magnitude of changed regional marginal probabilities','spatial_threshold':'abstract regional exceedance probability z, not an actual European q90 threshold','process':'active-day clusters bridging one empty day; first six actual hours','MCSE':'between-replicate SD / sqrt(replications); synchronized paired replicate identifiers','temporal_fixture_reps':1000,'phase_fixture':'isolated one-hour opportunities; observed phase changes conditional information; not observed-process coverage or proof of a real onset signal','elapsed_seconds':time.perf_counter()-started}
    (OUT/'ADEMP.json').write_text(json.dumps(design,indent=2));print(json.dumps(design),flush=True)
if __name__=='__main__':run()
