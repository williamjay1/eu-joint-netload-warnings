from pathlib import Path
import pandas as pd,json
ROOT=Path(__file__).resolve().parents[1]
rows=[]
for p in sorted((ROOT/'results').rglob('*.csv')):
    if 'wide_boundary_initial' in str(p):continue
    try:df=pd.read_csv(p)
    except Exception:continue
    for rownum,row in df.iterrows():
        for col,val in row.items():
            if isinstance(val,(int,float)) and pd.notna(val):
                rows.append(dict(source=p.relative_to(ROOT).as_posix(),row=int(rownum)+2,field=col,value=val))
pd.DataFrame(rows).to_csv(ROOT/'audit/RESULT_LEDGER.csv.gz',index=False)
print(json.dumps({'numerical_cells':len(rows),'sources':len(set(r['source'] for r in rows))}))
