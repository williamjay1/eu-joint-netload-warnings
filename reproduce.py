"""One-command, clone-local evaluation reproduction; never downloads raw data."""
from __future__ import annotations
import argparse, hashlib, json, os, subprocess, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--figures',action='store_true',help='also redraw all seven figures and five tables; Arial required');args=parser.parse_args()
    env=os.environ.copy();env['PYTHONDONTWRITEBYTECODE']='1';env['MPLCONFIGDIR']=str(ROOT/'generated/mplconfig')
    env['OMP_NUM_THREADS']='1';env['OPENBLAS_NUM_THREADS']='1';env['MKL_NUM_THREADS']='1'
    jobs=[['analysis/reproduce_hourly_evaluation.py','--output-dir','generated/hourly'],['analysis/reproduce_core_analysis.py','--input-dir','generated/hourly','--output-dir','generated/inference']]
    # The core reconstruction checks against this finite-family reference.
    dst=ROOT/'generated/hourly/finite_family_inference.csv';dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes((ROOT/'data/aggregates/finite_family_inference.csv').read_bytes())
    if args.figures:jobs.extend([[f'scripts/{name}'] for name in ('nature_concept_figures_20261002.py','nature_data_figures_20261002.py','nature_core_tables_20261002.py')])
    results=[]
    for job in jobs:
        done=subprocess.run([sys.executable,'-B',*job],cwd=ROOT,env=env,check=True)
        results.append({'command':['python','-B',*job],'exit_code':done.returncode})
    report={'status':'passed','commands':results,'refitting_or_raw_acquisition':False,'plot_regeneration_requested':args.figures}
    (ROOT/'generated/reproduction_readback.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report))

if __name__=='__main__':main()
