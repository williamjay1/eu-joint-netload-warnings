"""Fetch a new immutable official Danish day-ahead price snapshot."""
from pathlib import Path
from urllib.request import urlopen,Request
from urllib.parse import urlencode
import json,hashlib
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
RAW=Path(r'F:\AcademicData\eu_joint_netload\raw\asmbi_revision_20261008')
RAW.mkdir(parents=True,exist_ok=True)
params={'start':'2021-12-31','end':'2024-01-02','filter':json.dumps({'PriceArea':['DK1','DK2']}),'sort':'HourUTC ASC','limit':100000}
url='https://api.energidataservice.dk/dataset/Elspotprices?'+urlencode(params)
path=RAW/'Elspotprices_2022_2023_snapshot_20261008.json'
if path.exists(): body=path.read_bytes()
else:
    with urlopen(Request(url,headers={'User-Agent':'academic-reanalysis/1.0'}),timeout=90) as r: body=r.read()
    with path.open('xb') as f:f.write(body)
    (RAW/'Elspotprices_2022_2023_receipt_20261008.json').write_text(json.dumps({'url':url,'retrieved_utc':pd.Timestamp.now(tz='UTC').isoformat(),'bytes':len(body),'sha256':hashlib.sha256(body).hexdigest(),'vintage':'current snapshot; historical publication availability not established'},indent=2))
data=json.loads(body);df=pd.DataFrame(data['records']);df['target_time']=pd.to_datetime(df.HourUTC,utc=True)
df=df.rename(columns={'SpotPriceEUR':'price_eur_mwh'})
df=df.loc[df.target_time.dt.year.isin([2022,2023]),['target_time','PriceArea','price_eur_mwh']].sort_values(['target_time','PriceArea'])
assert not df.duplicated(['target_time','PriceArea']).any()
df.to_parquet(ROOT/'datasets/dk_prices_dev.parquet',index=False)
print(json.dumps({'rows':len(df),'min':str(df.target_time.min()),'max':str(df.target_time.max()),'missing':int(df.price_eur_mwh.isna().sum()),'price_areas':df.PriceArea.unique().tolist(),'output':str(ROOT/'datasets/dk_prices_dev.parquet')}))
