"""Immutable, rate-aware public observation acquisition. No event-label exploration."""
from release_paths import work_path as _work_path, raw_path as _raw_path
from pathlib import Path
from datetime import datetime, timezone
import argparse, hashlib, json, os, time, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests

BASE = _work_path() / 'results'
RAW = _raw_path()
SMARD = {'load_mw':410,'solar_mw':4068,'wind_onshore_mw':4067,'wind_offshore_mw':1225}
LOCAL = threading.local()

def save_response(url, path, params=None):
    if path.exists():
        content=path.read_bytes()
        if path.suffix=='.json': json.loads(content)
        return {'url':url,'path':str(path),'bytes':len(content),'sha256':hashlib.sha256(content).hexdigest(),'reused_existing_immutable':True}
    if not hasattr(LOCAL,'session'): LOCAL.session=requests.Session()
    for attempt in range(4):
        try:
            r = LOCAL.session.get(url, params=params, timeout=(20,180))
        except requests.RequestException:
            if attempt==3: raise
            time.sleep(5*(attempt+1)); LOCAL.session=requests.Session(); continue
        if r.status_code == 429:
            delay = min(180,float(r.headers.get('Retry-After','35')))
            time.sleep(delay)
            continue
        r.raise_for_status()
        break
    else:
        raise RuntimeError('Rate-limited after four attempts')
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb') as f:
        f.write(r.content)
    os.chmod(path,0o444)
    return {'url':r.url,'path':str(path),'bytes':len(r.content),'sha256':hashlib.sha256(r.content).hexdigest(),'retrieved_utc':datetime.now(timezone.utc).isoformat(),'http_status':r.status_code}

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--mode',choices=['pilot','energycharts','smard','smard-be','elia-load','elia-generation','rte'],default='pilot')
    p.add_argument('--start-year',type=int,default=2019)
    p.add_argument('--end-year',type=int,default=2025)
    p.add_argument('--resume',type=Path)
    p.add_argument('--countries',nargs='+',default=['fr','be'])
    p.add_argument('--elia-datasets',nargs='+',choices=['ods031','ods032'],default=['ods031','ods032'])
    a=p.parse_args()
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out=a.resume or RAW/f'observations_{a.mode}_{stamp}'
    if not out.resolve().is_relative_to(RAW.resolve()): raise ValueError('Raw path outside authorized root')
    manifests=[]
    def get(url, name, params=None):
        m=save_response(url,out/name,params)
        manifests.append(m)
        (BASE/f'data_download_{a.mode}_{stamp}.json').write_text(json.dumps(manifests,indent=2),encoding='utf8')
        print(json.dumps({'saved':name,'bytes':m['bytes']}),flush=True)
        return json.loads(Path(m['path']).read_text(encoding='utf8')) if name.endswith('.json') else None
    if a.mode in ['pilot','smard','smard-be']:
        region = 'BE' if a.mode == 'smard-be' else 'DE-LU'
        jobs=[]
        lower=int(datetime(a.start_year,1,1,tzinfo=timezone.utc).timestamp()*1000)
        upper=int(datetime(a.end_year+1,1,1,tzinfo=timezone.utc).timestamp()*1000)
        for field,code in SMARD.items():
            j=get(f'https://www.smard.de/app/chart_data/{code}/{region}/index_hour.json',f'smard_{field}_index.json')
            times=j['timestamps']
            chosen=[t for i,t in enumerate(times) if t<upper and (i+1==len(times) or times[i+1]>lower)]
            if a.mode=='pilot': chosen=chosen[:1]
            for t in chosen:
                jobs.append((f'https://www.smard.de/app/chart_data/{code}/{region}/{code}_{region}_hour_{t}.json',f'smard_{field}_{t}.json'))
        with ThreadPoolExecutor(max_workers=4) as pool:
            fs={pool.submit(save_response,u,out/n):(u,n) for u,n in jobs}
            for f in as_completed(fs):
                _,name=fs[f]
                try:
                    m=f.result(); manifests.append(m)
                    print(json.dumps({'saved':name,'bytes':m['bytes']}),flush=True)
                except Exception as e:
                    manifests.append({'url':fs[f][0],'path':str(out/name),'error':str(e)})
                (BASE/f'data_download_{a.mode}_{stamp}.json').write_text(json.dumps(manifests,indent=2),encoding='utf8')
    if a.mode in ['pilot','energycharts']:
        get('https://api.energy-charts.info/openapi.json','energycharts_openapi.json')
        last=0
        for country in a.countries:
            years=range(a.start_year,a.end_year+1) if a.mode=='energycharts' else [2019]
            for year in years:
                time.sleep(max(0,31-(time.monotonic()-last)))
                end=f'{year+1}-01-01T00:00Z' if a.mode=='energycharts' else f'{year}-01-02T00:00Z'
                get('https://api.energy-charts.info/public_power',f'energycharts_{country}_{year}.json',{'country':country,'start':f'{year}-01-01T00:00Z','end':end})
                last=time.monotonic()
    if a.mode in ['pilot','elia-load']:
        get('https://opendata.elia.be/api/explore/v2.1/catalog/datasets/ods001','elia_ods001_metadata.json')
        years=range(a.start_year,a.end_year+1) if a.mode=='elia-load' else [2019]
        for year in years:
            end=f'{year+1}-01-01T00:00:00Z' if a.mode=='elia-load' else f'{year}-01-02T00:00:00Z'
            get('https://opendata.elia.be/api/explore/v2.1/catalog/datasets/ods001/exports/csv',f'elia_load_{year}.csv',{'where':f'datetime >= "{year}-01-01T00:00:00Z" AND datetime < "{end}"','select':'datetime,resolutioncode,totalload','order_by':'datetime'})
    if a.mode == 'elia-generation':
        for dataset in a.elia_datasets:
            get(f'https://opendata.elia.be/api/explore/v2.1/catalog/datasets/{dataset}',f'elia_{dataset}_metadata.json')
            for year in range(a.start_year,a.end_year+1):
                where=f'datetime >= "{year}-01-01T00:00:00Z" AND datetime < "{year+1}-01-01T00:00:00Z"'
                if dataset=='ods032': where+=' AND region = "Belgium"'
                fields='datetime,resolutioncode,region,measured'
                if dataset=='ods031': fields+=',offshoreonshore,gridconnectiontype'
                get(f'https://opendata.elia.be/api/explore/v2.1/catalog/datasets/{dataset}/exports/csv',f'elia_{dataset}_{year}.csv',{'where':where,'select':fields,'order_by':'datetime'})
    if a.mode=='rte':
        dataset='eco2mix-national-cons-def'
        get(f'https://odre.opendatasoft.com/api/explore/v2.1/catalog/datasets/{dataset}','rte_metadata.json')
        for year in range(a.start_year,a.end_year+1):
            get(f'https://odre.opendatasoft.com/api/explore/v2.1/catalog/datasets/{dataset}/exports/csv',f'rte_eco2mix_{year}.csv',{'where':f'date_heure >= "{year}-01-01T00:00:00Z" AND date_heure < "{year+1}-01-01T00:00:00Z"','select':'date_heure,consommation,eolien,solaire','order_by':'date_heure'})
    print(json.dumps({'mode':a.mode,'raw_directory':str(out),'files':len(manifests),'errors':sum('error' in x for x in manifests)}),flush=True)

if __name__=='__main__': main()
