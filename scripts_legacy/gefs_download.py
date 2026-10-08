"""Anonymous, resumable acquisition of four original NOAA GEFS GRIB messages.

Only byte ranges identified by each object's own .idx are retrieved. F raw
stores original encoded messages and the original index; decoding/caches live
on D. No ECMWF credentials, terms, or archive are used. Default is a small
measured pilot, not an unbounded full-archive download.
"""
from __future__ import annotations
from release_paths import work_path as _work_path, raw_path as _raw_path
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import threading
import time
import requests
from requests.adapters import HTTPAdapter
import pandas as pd

ROOT=_work_path()
RAW=_raw_path()
BASE='https://noaa-gefs-pds.s3.amazonaws.com/'
FIELDS=(('TMP','2 m above ground'),('UGRD','10 m above ground'),
        ('VGRD','10 m above ground'),('DSWRF','surface'))
LOCAL=threading.local()


def timestamp():return datetime.now(timezone.utc).isoformat()


def sha256(content):return hashlib.sha256(content).hexdigest()


def session():
    if not hasattr(LOCAL,'session'):
        s=requests.Session()
        s.mount('https://',HTTPAdapter(pool_connections=4,pool_maxsize=4,max_retries=0))
        LOCAL.session=s
    return LOCAL.session


def get(url,headers=None,retries=5):
    for attempt in range(retries):
        try:
            r=session().get(url,headers=headers,timeout=(15,90))
            if r.status_code in (429,500,502,503,504):
                raise requests.HTTPError(f'Retryable HTTP {r.status_code}')
            return r
        except (requests.RequestException,OSError):
            if attempt==retries-1:raise
            time.sleep(min(2**attempt,15))


def object_keys(day,member,step):
    prefix=f'gefs.{day}/00/'
    legacy=prefix+f'pgrb2a/gep{member:02}.t00z.pgrb2af{step}'
    modern=prefix+f'atmos/pgrb2ap5/gep{member:02}.t00z.pgrb2a.0p50.f{step:03}'
    return [legacy,modern] if day<'20200923' else [modern,legacy]


def select_ranges(text):
    rows=[line.split(':') for line in text.strip().splitlines()]
    selected=[]
    for variable,level in FIELDS:
        matches=[(i,row) for i,row in enumerate(rows) if len(row)>=6 and row[3:5]==[variable,level]]
        if len(matches)!=1:raise ValueError(f'Expected unique {variable}/{level}, found {len(matches)}')
        i,row=matches[0]
        if i+1>=len(rows):raise ValueError('Selected last message: source length must be resolved before use.')
        selected.append(dict(variable=variable,level=level,description=':'.join(row),
                             begin=int(row[1]),end=int(rows[i+1][1])-1))
    # Preserve original order. Adjacent U/V fields share one range transaction.
    ordered=sorted(selected,key=lambda x:x['begin'])
    groups=[]
    for item in ordered:
        if groups and item['begin']==groups[-1]['end']+1:
            groups[-1]['end']=item['end'];groups[-1]['items'].append(item)
        else:groups.append(dict(begin=item['begin'],end=item['end'],items=[item]))
    return selected,groups


def write_raw(path,content):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        if path.stat().st_size!=len(content) or sha256(path.read_bytes())!=sha256(content):
            raise ValueError(f'Immutable raw collision: {path}')
        return
    with path.open('xb') as f:f.write(content)
    path.chmod(0o444)


def acquire(job,snapshot):
    started=time.monotonic()
    day,member,step=job
    name=f'gefs_{day}_m{member:02}_f{step:03}'
    folder=Path(snapshot)/day[:4]/day
    target=folder/(name+'.grib2')
    receipt=folder/(name+'.receipt.json')
    if receipt.exists():
        record=json.loads(receipt.read_text(encoding='utf-8'))
        if not target.exists() or target.stat().st_size!=record['bytes']:
            raise ValueError('Completed receipt has missing/incomplete immutable raw.')
        return {**record,'resumed_existing':True}
    response=None;key=None
    for candidate in object_keys(day,member,step):
        r=get(BASE+candidate+'.idx')
        if r.status_code==404:continue
        r.raise_for_status()
        response=r;key=candidate;break
    if response is None:raise FileNotFoundError(f'No GEFS object index: {job}')
    index=response.content
    selected,groups=select_ranges(response.text)
    encoded={};requests_metadata=[]
    for group in groups:
        begin,end=group['begin'],group['end']
        r=get(BASE+key,{'Range':f'bytes={begin}-{end}'})
        if r.status_code!=206 or not r.headers.get('Content-Range','').startswith(f'bytes {begin}-{end}/'):
            raise ValueError(f'Incorrect range response {r.status_code}: {key}')
        if len(r.content)!=end-begin+1:raise ValueError('Range length mismatch.')
        for item in group['items']:
            data=r.content[item['begin']-begin:item['end']-begin+1]
            if data[:4]!=b'GRIB' or data[-4:]!=b'7777':raise ValueError('GRIB message boundary mismatch.')
            encoded[item['variable']]=data
            item['bytes']=len(data);item['sha256']=sha256(data)
        requests_metadata.append(dict(range=[begin,end],etag=r.headers.get('ETag'),
                                      source_last_modified=r.headers.get('Last-Modified')))
    content=b''.join(encoded[variable] for variable,_ in FIELDS)
    write_raw(folder/(name+'.idx'),index)
    write_raw(target,content)
    record=dict(status='complete',initialization_date=day,initialization_hour_utc=0,
                member_id=member,forecast_step_hours=step,source_url=BASE+key,
                index_url=BASE+key+'.idx',index_sha256=sha256(index),
                raw_path=str(target),bytes=len(content),sha256=sha256(content),
                messages=selected,range_requests=requests_metadata,
                retrieval_utc=timestamp(),elapsed_seconds=time.monotonic()-started,
                representation='Concatenation of four unchanged original GRIB messages obtained by anonymous HTTP byte ranges.')
    write_raw(receipt,json.dumps(record,indent=2).encode())
    return record


def run(args):
    root=ROOT/'results'
    root.mkdir(parents=True,exist_ok=True)
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    snapshot=Path(args.snapshot) if args.snapshot else RAW/('gefs_public_'+stamp)
    if not snapshot.resolve().is_relative_to(_raw_path().resolve()) or not snapshot.resolve().is_relative_to(RAW.resolve()):
        raise ValueError('New raw snapshots must be inside the configured immutable raw root.')
    dates=pd.date_range(args.start,args.end,freq='D')
    if args.members<2 or args.members>20 or args.workers<1 or args.workers>256:
        raise ValueError('Common-member acquisition supports 2--20 members and <=256 workers.')
    steps=tuple(int(x) for x in args.steps.split(','))
    if len(set(steps))!=len(steps) or not steps or any(s<6 or s%6 for s in steps):
        raise ValueError('Unique positive six-hour endpoints required.')
    jobs=[(d.strftime('%Y%m%d'),m,s) for d in dates for m in range(1,args.members+1) for s in steps]
    disk={'work':shutil.disk_usage(_work_path()).free,'raw':shutil.disk_usage(RAW).free}
    allowance=len(jobs)*1_200_000
    if disk['raw']<allowance*1.2 or disk['work']<500_000_000:
        raise RuntimeError('Insufficient checked F raw or D work capacity.')
    report=dict(created_utc=timestamp(),status='running',source='NOAA_GEFS_PUBLIC_OPERATIONAL_ARCHIVE',
                start_initialization=args.start,end_initialization=args.end,members=args.members,workers=args.workers,
                steps=list(steps),snapshot=str(snapshot),anonymous=True,
                checked_disk_free_bytes=disk,planning_bytes=allowance,
                estimate='1.2MB per selected four-field object; actual pilot old .254MB/new .894MB.',
                electricity_test_outcomes_inspected=False,complete_jobs=0,failed_jobs=0,
                bytes_added=0,elapsed_seconds=0,failures=[],script_sha256=sha256(Path(__file__).read_bytes()))
    manifest=root/('gefs_download_'+stamp+'.json')
    log=root/('gefs_download_'+stamp+'.jsonl')
    started=time.monotonic();completed=0
    # One JSONL record per successful job or failure; all raw receipts immutable.
    with ThreadPoolExecutor(max_workers=args.workers) as pool,log.open('x',encoding='utf-8') as handle:
        outstanding={};cursor=iter(jobs)
        def top_up():
            while len(outstanding)<args.workers*2:
                if time.monotonic()-started>args.max_run_seconds:return
                try:job=next(cursor)
                except StopIteration:return
                outstanding[pool.submit(acquire,job,snapshot)]=job
        top_up()
        while outstanding:
            future=next(as_completed(outstanding))
            job=outstanding.pop(future)
            try:
                record=future.result();report['complete_jobs']+=1
                if not record.get('resumed_existing'):report['bytes_added']+=record['bytes']
            except Exception as exc:
                record=dict(status='failed',job=job,error_type=type(exc).__name__,error=str(exc),utc=timestamp())
                report['failed_jobs']+=1;report['failures'].append(record)
            handle.write(json.dumps(record)+'\n');handle.flush()
            completed+=1
            report['elapsed_seconds']=time.monotonic()-started
            if completed%100==0:
                manifest.write_text(json.dumps(report,indent=2),encoding='utf-8')
                print(json.dumps({k:report[k] for k in ('complete_jobs','failed_jobs','bytes_added','elapsed_seconds')}),flush=True)
            top_up()
    report['status']='completed' if report['complete_jobs']==len(jobs) and not report['failed_jobs'] else 'partial_resumable'
    report['total_expected_jobs']=len(jobs)
    report['elapsed_seconds']=time.monotonic()-started
    report['finished_utc']=timestamp()
    manifest.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({'status':report['status'],'manifest':str(manifest),'snapshot':str(snapshot),
                      'complete_jobs':report['complete_jobs'],'failed_jobs':report['failed_jobs'],
                      'bytes_added':report['bytes_added'],'elapsed_seconds':report['elapsed_seconds']}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--start',default='2018-12-29',help='Initialization date, inclusive')
    p.add_argument('--end',default='2018-12-29',help='Initialization date, inclusive')
    p.add_argument('--members',type=int,default=20)
    p.add_argument('--workers',type=int,default=12)
    p.add_argument('--steps',default='24,30,36,42,48',help='Six-hour endpoints; main latest-cycle forecast, 72--96 sensitivity')
    p.add_argument('--snapshot',help='Existing immutable snapshot root for idempotent resume')
    p.add_argument('--max-run-seconds',type=float,default=600)
    run(p.parse_args())
