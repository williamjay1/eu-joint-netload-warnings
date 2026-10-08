"""Shared Nature-inspired publication design and deterministic layout checks.

The GitHub SciencePlots source is used as a style base, with explicit local
overrides for editable text, no TeX, a colour-accessible palette and 180 mm.
This module never alters scientific inputs.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.text import Text
from matplotlib.lines import Line2D
from matplotlib.collections import Collection, PathCollection
from matplotlib.transforms import Bbox
from matplotlib.path import Path as MplPath
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
STYLE_ROOT=ROOT/'results/nature_visual_redesign_20261002'
COLORS={'P0':'#7A7A7A','P1':'#0072B2','P2':'#D55E00',
        'ink':'#222222','blue':'#0072B2','vermillion':'#D55E00',
        'green':'#009E73','light':'#F2F4F5','rule':'#91979B'}
STYLE_RECORD={}

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def apply_style():
    global STYLE_RECORD
    candidates=sorted((STYLE_ROOT/'style_reference').glob('*.mplstyle'))
    # Apply only known source styles; execute no downloaded Python code.
    ordered=[]
    for key in ('science','nature','no-latex'):
        ordered.extend(p for p in candidates if p.stem in (key,key+'-style'))
    plt.style.use('default')
    for path in ordered: plt.style.use(str(path))
    arial=font_manager.findfont(font_manager.FontProperties(family='Arial'),fallback_to_default=False)
    plt.rcParams.update({
        'font.family':'sans-serif','font.sans-serif':['Arial'],
        'font.size':7,'text.color':COLORS['ink'],
        'text.usetex':False,'mathtext.fontset':'custom',
        'mathtext.rm':'Arial','mathtext.it':'Arial:italic','mathtext.bf':'Arial:bold',
        'axes.labelsize':7,'axes.titlesize':7,'axes.titleweight':'normal',
        'axes.edgecolor':COLORS['ink'],'axes.labelcolor':COLORS['ink'],
        'axes.linewidth':0.6,'axes.spines.top':False,'axes.spines.right':False,
        'axes.grid':False,'axes.axisbelow':True,
        'xtick.labelsize':7,'ytick.labelsize':7,'xtick.color':COLORS['ink'],'ytick.color':COLORS['ink'],
        'xtick.direction':'out','ytick.direction':'out','xtick.top':False,'ytick.right':False,
        'xtick.major.size':2.5,'ytick.major.size':2.5,'xtick.major.width':0.6,'ytick.major.width':0.6,
        'xtick.minor.visible':False,'ytick.minor.visible':False,
        'xtick.major.pad':3,'ytick.major.pad':3,
        'legend.fontsize':7,'legend.frameon':False,'legend.labelspacing':0.55,
        'lines.linewidth':0.8,'lines.markersize':4,
        'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none',
        'savefig.facecolor':'white','figure.facecolor':'white','axes.facecolor':'white',
        'savefig.transparent':False,'savefig.bbox':None,'savefig.pad_inches':0,
        'figure.dpi':150,
    })
    STYLE_RECORD={'base_styles':[{'path':str(p),'sha256':sha(p)} for p in ordered],
                  'actual_arial_font':arial,'font_size_pt':7,'panel_label_pt':8,
                  'width_mm':180,'vector_text':'embedded TrueType PDF; editable SVG text',
                  'raster_dpi':1200,'preview_dpi':300,'palette':COLORS}
    return STYLE_RECORD

def figure(width_mm=180,height_mm=100,**kwargs):
    return plt.figure(figsize=(width_mm/25.4,height_mm/25.4),**kwargs)

def panel_label(ax,label,x=-0.12,y=1.08):
    return ax.text(x,y,label,transform=ax.transAxes,ha='left',va='bottom',
                   fontsize=8,fontweight='bold',clip_on=False)

def _bbox_hits(a,b,min_area=0.25):
    ix=max(0,min(a.x1,b.x1)-max(a.x0,b.x0))
    iy=max(0,min(a.y1,b.y1)-max(a.y0,b.y0))
    return ix*iy>min_area

def _stroke_hits(path,bbox):
    # Unlike Path.intersects_bbox, containment by a closed outline is not an
    # intersection. Text deliberately inside a diagram card is allowed;
    # a visible border passing through its text is not.
    def segment(a,b):
        if not np.isfinite([*a,*b]).all(): return False
        if max(a[0],b[0])<bbox.x0 or min(a[0],b[0])>bbox.x1 or max(a[1],b[1])<bbox.y0 or min(a[1],b[1])>bbox.y1: return False
        d=b-a;t0=0.;t1=1.
        for p,q in [(-d[0],a[0]-bbox.x0),(d[0],bbox.x1-a[0]),
                    (-d[1],a[1]-bbox.y0),(d[1],bbox.y1-a[1])]:
            if abs(p)<1e-12:
                if q<0:return False
            else:
                ratio=q/p
                if p<0:t0=max(t0,ratio)
                else:t1=min(t1,ratio)
                if t0>t1:return False
        return True
    previous=None;start=None
    for coords,code in path.iter_segments(curves=False,simplify=False):
        xy=np.asarray(coords[-2:],dtype=float)
        if code==MplPath.MOVETO:previous=xy;start=xy
        elif code==MplPath.CLOSEPOLY:
            if previous is not None and start is not None and segment(previous,start):return True
            previous=start
        elif code==MplPath.LINETO:
            if previous is not None and segment(previous,xy):return True
            previous=xy
    return False

def layout_report(fig):
    """All visible text, including left/right titles and legend/tick labels.

    Checks geometric text bounds against text, line paths, marker extents,
    scatter extents, patch borders, bar interiors and the canvas. It is a
    deterministic screen supplemented by actual-size visual inspection.
    """
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
    texts=[]
    for t in fig.findobj(match=Text):
        if not t.get_visible() or not t.get_text().strip() or t.get_alpha()==0: continue
        bbox=t.get_window_extent(renderer)
        if not np.isfinite(bbox.extents).all(): continue
        texts.append((t,bbox))
    outside=[]; overlaps=[]; linehits=[]; markhits=[]; patchhits=[]
    bounds=fig.bbox
    for t,b in texts:
        if b.x0<bounds.x0-0.5 or b.y0<bounds.y0-0.5 or b.x1>bounds.x1+0.5 or b.y1>bounds.y1+0.5:
            outside.append(t.get_text())
    for i,(ta,ba) in enumerate(texts):
        for tb,bb in texts[i+1:]:
            if _bbox_hits(ba,bb): overlaps.append([ta.get_text(),tb.get_text()])
    def clipped_bbox(artist,b):
        if artist.get_clip_on() and artist.get_clip_box() is not None:
            c=artist.get_clip_box()
            x0=max(b.x0,c.x0);y0=max(b.y0,c.y0);x1=min(b.x1,c.x1);y1=min(b.y1,c.y1)
            if x1<=x0 or y1<=y0: return None
            return Bbox.from_extents(x0,y0,x1,y1)
        return b
    for ax in fig.axes:
        for ln in ax.lines:
            if not ln.get_visible() or ln.get_alpha()==0: continue
            path=ln.get_path().transformed(ln.get_transform())
            linestyle=ln.get_linestyle()
            points=path.vertices[np.isfinite(path.vertices).all(1)]
            for t,b in texts:
                cb=clipped_bbox(ln,b)
                if cb is None: continue
                if linestyle not in ('None','none','',None) and len(points)>1 and _stroke_hits(path,cb):
                    linehits.append(t.get_text())
                marker=ln.get_marker()
                if marker not in ('None','none','',None,' '):
                    radius=ln.get_markersize()*fig.dpi/72/2
                    for x,y in points:
                        if _bbox_hits(Bbox.from_extents(x-radius,y-radius,x+radius,y+radius),cb):
                            markhits.append(t.get_text());break
        for coll in ax.collections:
            if not coll.get_visible() or coll.get_alpha()==0: continue
            if isinstance(coll,PathCollection):
                pts=coll.get_offset_transform().transform(coll.get_offsets())
                sizes=coll.get_sizes()
                for i,(x,y) in enumerate(pts):
                    if not np.isfinite([x,y]).all(): continue
                    radius=np.sqrt(sizes[i%len(sizes)])*fig.dpi/72/2 if len(sizes) else 2
                    mb=Bbox.from_extents(x-radius,y-radius,x+radius,y+radius)
                    for t,b in texts:
                        cb=clipped_bbox(coll,b)
                        if cb is not None and _bbox_hits(mb,cb): markhits.append(t.get_text())
            else:
                for p in coll.get_paths():
                    path=p.transformed(coll.get_transform())
                    for t,b in texts:
                        cb=clipped_bbox(coll,b)
                        if cb is not None and _stroke_hits(path,cb): linehits.append(t.get_text())
        for patch in [*ax.patches,*(ax.spines.values() if ax.axison else [])]:
            if not patch.get_visible() or patch.get_alpha()==0: continue
            edge=patch.get_edgecolor()
            if len(edge)==4 and edge[3]==0:continue
            path=patch.get_path().transformed(patch.get_transform())
            for t,b in texts:
                cb=clipped_bbox(patch,b)
                if cb is not None and _stroke_hits(path,cb): patchhits.append(t.get_text())
    report={'text_objects':len(texts),'outside_canvas':outside,'text_text_intersections':overlaps,
            'text_line_intersections':sorted(set(linehits)),
            'text_marker_intersections':sorted(set(markhits)),
            'text_patch_edge_intersections':sorted(set(patchhits)),
            'minimum_font_size_pt':min(t.get_fontsize() for t,b in texts) if texts else None,
            'screen_scope':'Bounding boxes, paths and markers; actual-size visual review is also required'}
    report['passed']=not any(report[k] for k in ('outside_canvas','text_text_intersections',
                              'text_line_intersections','text_marker_intersections','text_patch_edge_intersections'))
    return report

def export_figure(fig,outdir,stem,source_note='',values=None,check=True):
    outdir=Path(outdir);outdir.mkdir(parents=True,exist_ok=True)
    qa=layout_report(fig)
    qa_path=outdir/(stem+'_layout.json')
    qa_path.write_text(json.dumps(qa,ensure_ascii=False,indent=2),encoding='utf-8')
    fig.savefig(outdir/(stem+'_preview.png'),dpi=300)
    if check and not qa['passed']:
        raise AssertionError(f'{stem}: layout issues; see {qa_path}: '+json.dumps(qa,ensure_ascii=False))
    files={}
    for ext in ('pdf','svg'):
        p=outdir/(stem+'.'+ext)
        fig.savefig(p)
        files[ext]={'path':str(p),'sha256':sha(p)}
    for key,dpi in [('preview',300),('raster_1200dpi',1200)]:
        p=outdir/(stem+('_preview.png' if key=='preview' else '_1200dpi.png'))
        if key!='preview': fig.savefig(p,dpi=dpi)
        with Image.open(p) as im:
            dimensions=list(im.size);actual_dpi=list(im.info.get('dpi',[]))
        files[key]={'path':str(p),'sha256':sha(p),'pixels':dimensions,'dpi':actual_dpi}
    record={'stem':stem,'source_note':source_note,'values':values,'files':files,
            'layout':qa,'layout_path':str(qa_path),'style':STYLE_RECORD,
            'physical_mm':[float(x*25.4) for x in fig.get_size_inches()]}
    (outdir/(stem+'_manifest.json')).write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
    plt.close(fig)
    return record
