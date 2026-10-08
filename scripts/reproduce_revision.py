"""Portable fixed-forecast evaluation launcher; no raw download or GAM refit."""
from __future__ import annotations
import argparse,os,subprocess,sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
VERSION='asmbi-evaluation-20261008.1'
TASKS=('simulation-narrow','simulation-wide','simulation-contrasts','policies',
       'forecast-diagnostics','early-price','stability','numerical',
       'figures-policy','figures-stability','ledger')
ANALYSES=('simulation-narrow','simulation-wide','simulation-contrasts','policies',
          'forecast-diagnostics','early-price','stability','numerical')

def command(task,args):
    py=str(Path(args.python).resolve()) if args.python else sys.executable
    script=lambda name:[py,'-B',str(ROOT/'scripts'/name)]
    if task=='simulation-narrow':
        code='import mechanism_simulations as m; m.run(reps=400,days=120,amplitude=.012,shift=-.035)'
        return [py,'-B','-c',code]
    if task=='simulation-wide':
        code='import mechanism_simulations as m; m.run(reps=400,days=120,amplitude=.08,shift=.035,outsub="wide_boundary")'
        return [py,'-B','-c',code]
    if task=='simulation-contrasts':return script('policy_analysis.py')+['--paired-sim']
    if task=='policies':return script('policy_analysis.py')+['--reps',str(args.policy_reps)]
    if task=='forecast-diagnostics':return script('forecast_diagnostics.py')
    if task=='early-price':return script('early_price_analysis.py')+['--reps',str(args.early_reps),'--zone','all']
    if task=='stability':return script('stability_analysis.py')
    if task=='numerical':return script('numerical_check.py')
    if task=='figures-policy':return [py,'-B','-c','import figures_policy as f; f.prepare(); f.draw()']
    if task=='figures-stability':return script('figures_stability_simulation.py')
    if task=='ledger':return script('build_ledger.py')
    raise ValueError(task)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--version',action='version',version=VERSION)
    p.add_argument('--verify',action='store_true',help='Fast read-only scientific checks; writes audit/reproducibility_checks.json')
    p.add_argument('--analyses',action='store_true',help='Recompute the eight explicit analysis stages, then verify')
    p.add_argument('--task',choices=TASKS,action='append',default=[],help='Explicit stage; repeat to execute in given order')
    p.add_argument('--python',help='Optional Python executable for stages; defaults to the calling interpreter')
    p.add_argument('--policy-reps',type=int,default=2000)
    p.add_argument('--early-reps',type=int,default=1000,help='Main early/price stage; its existing postprocess uses 2000 replicates')
    p.add_argument('--dry-run',action='store_true',help='List commands without executing analyses or verification')
    args=p.parse_args()
    if args.analyses and args.task:p.error('Choose --analyses or explicit --task stages')
    stages=list(ANALYSES) if args.analyses else args.task
    if not stages and not args.verify:args.verify=True
    env=os.environ.copy();env['PYTHONDONTWRITEBYTECODE']='1'
    env.setdefault('OPENBLAS_NUM_THREADS','1');env.setdefault('OMP_NUM_THREADS','1')
    # Relative script imports work on any drive or operating system.
    env['PYTHONPATH']=str(ROOT/'scripts')+os.pathsep+env.get('PYTHONPATH','')
    for stage in stages:
        cmd=command(stage,args)
        print('Stage:',stage,' '.join(cmd),flush=True)
        if not args.dry_run:subprocess.run(cmd,cwd=ROOT,env=env,check=True)
    if args.verify or args.analyses:
        py=str(Path(args.python).resolve()) if args.python else sys.executable
        cmd=[py,'-B',str(ROOT/'scripts/verify_revision.py')]
        print('Stage: verify',flush=True)
        if not args.dry_run:subprocess.run(cmd,cwd=ROOT,env=env,check=True)

if __name__=='__main__':main()
