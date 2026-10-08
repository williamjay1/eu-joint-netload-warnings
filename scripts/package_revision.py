"""Assemble the final local package; never publish or change archived inputs."""
from pathlib import Path
import zipfile,json,hashlib
ROOT=Path(__file__).resolve().parents[1]
TARGET=ROOT/'results/ASMBI_revision_20261008.zip'
SKIP_PARTS={'__pycache__','.git','wide_boundary_initial','temp','cache'}
SKIP_NAMES={'ASMBI_revision_20261008.zip','finish_manuscript.py','get_development_prices.py','package_revision.py','warning_reference.py'}

def files():
    roots=['submission','manuscript','scripts','datasets','results','audit']
    out=[]
    for folder in roots:
        for p in (ROOT/folder).rglob('*'):
            if not p.is_file():continue
            rel=p.relative_to(ROOT)
            if any(x in SKIP_PARTS for x in rel.parts) or p.name in SKIP_NAMES or p.suffix=='.pyc':continue
            if '_preview' in p.name or p.name=='ARTICLE_BUILD_SPEC.md':continue
            if p.name.endswith('_qa.json') or '_pages_' in p.name or p.name.startswith('document_build_'):continue
            out.append(p)
    for name in ['README_DELIVERY.md','REPRODUCIBILITY.md','REVISION_PROTOCOL.md','requirements_revision.txt','LICENSE','DATA_LICENSES.md']:
        out.append(ROOT/name)
    return sorted(set(out))

def main():
    fs=files();assert all(p.exists() for p in fs)
    # Markdown sources reference lightweight previews; retain them as source assets.
    fs+=list((ROOT/'results/figures').glob('*_preview.png'))
    items=[{'path':p.relative_to(ROOT).as_posix(),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in fs]
    manifest=ROOT/'PACKAGE_MANIFEST.json';manifest.write_text(json.dumps({'version':'asmbi-evaluation-20261008.1','files':items,'publication_status':'local revision; not a new public DOI'},indent=2),encoding='utf-8')
    with zipfile.ZipFile(TARGET,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in fs+[manifest]:z.write(p,p.relative_to(ROOT).as_posix())
    with zipfile.ZipFile(TARGET) as z:
        assert z.testzip() is None
        assert all(z.getinfo(x['path']).file_size==x['bytes'] for x in items)
    print(json.dumps({'zip':str(TARGET),'files':len(items)+1,'bytes':TARGET.stat().st_size,'crc_test':'passed'}))

if __name__=='__main__':main()
