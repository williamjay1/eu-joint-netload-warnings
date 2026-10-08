"""Area-weighted historical GEFS weather features, reading immutable raw GRIB.

Primary representation: retain integer-degree forecast nodes and give each its
1-degree grid-cell footprint. This is node subsampling, not spatial averaging
or a claim that GEFS native forecast resolution remained unchanged. Native
0.5-degree footprints are available as a predeclared sensitivity.
"""
from __future__ import annotations
from release_paths import work_path as _work_path, raw_path as _raw_path
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import io
import json
import time
import zipfile
import numpy as np
import pandas as pd
import shapefile
from shapely.geometry import box, shape
from shapely.ops import transform, unary_union
from shapely import make_valid, segmentize
from pyproj import Transformer
import eccodes

ROOT = _work_path()
DEFAULT_BOUNDARY = _raw_path('weather_access_naturalearth_20260930T022005Z/ne_10m_admin_0_countries.zip')
ZONES = ('DE_LU', 'FR', 'BE')
STEPS = (24,30,36,42,48)
SHORTNAMES = ('2t','10u','10v','sdswrf')
UNITS = ('K','m s**-1','m s**-1','W m**-2')
TRANSFORM = Transformer.from_crs('EPSG:4326','EPSG:3035',always_xy=True).transform


def digest(data):
    return hashlib.sha256(data).hexdigest()


def country_geometries(boundary_zip):
    """Mainland France plus European coastal islands, excluding Corsica."""
    with zipfile.ZipFile(boundary_zip) as z:
        reader = shapefile.Reader(shp=io.BytesIO(z.read('ne_10m_admin_0_countries.shp')),
                                  shx=io.BytesIO(z.read('ne_10m_admin_0_countries.shx')),
                                  dbf=io.BytesIO(z.read('ne_10m_admin_0_countries.dbf')),
                                  encoding='utf8')
        all_geometries, countries = [], {}
        roi = box(-6,41,16,56)
        for item in reader.iterShapeRecords():
            props = item.record.as_dict()
            geom = make_valid(shape(item.shape.__geo_interface__))
            if geom.intersects(roi):
                all_geometries.append(geom.intersection(roi))
            code = props.get('ADM0_A3')
            if code in ('DEU','LUX','FRA','BEL'):
                countries[code] = geom
    if set(countries) != {'DEU','LUX','FRA','BEL'}:
        raise ValueError('Required Natural Earth country codes missing')
    france = countries['FRA'].intersection(roi)
    pieces = list(france.geoms) if hasattr(france,'geoms') else [france]
    kept = [p for p in pieces if not (p.representative_point().x > 8 and p.representative_point().y < 43.5)]
    france = unary_union(kept)
    if not france.contains(shape({'type':'Point','coordinates':[2.35,48.86]})):
        raise ValueError('Mainland France mask excludes Paris')
    if france.contains(shape({'type':'Point','coordinates':[8.74,41.93]})):
        raise ValueError('France mask still includes Corsica')
    land = unary_union(all_geometries)
    result = {'DE_LU_land': unary_union([countries['DEU'],countries['LUX']]),
              'FR_land': france, 'BE_land': countries['BEL']}
    # Fixed broad coast-weather windows; no observed power data chose these.
    result['BE_marine'] = box(2,51.3,4,52.1).difference(land)
    result['DE_LU_marine'] = unary_union([box(5.8,53.7,8.5,55.5),box(10,54,15,55.5)]).difference(land)
    result['FR_marine'] = box(-5,46.8,0,50.5).difference(land)
    return {name:segmentize(geom,.05) for name,geom in result.items()}


def cell_polygon(lon,lat,spacing):
    """Densify constant-longitude/latitude edges before equal-area projection."""
    half = spacing/2
    n = max(2,int(round(spacing/.1))+1)
    lo = np.linspace(lon-half,lon+half,n)
    la = np.linspace(lat-half,lat+half,n)
    ring = [(float(x),lat-half) for x in lo]
    ring += [(lon+half,float(y)) for y in la[1:]]
    ring += [(float(x),lat+half) for x in lo[-2::-1]]
    ring += [(lon-half,float(y)) for y in la[-2:0:-1]]
    from shapely.geometry import Polygon
    return Polygon(ring)


class RegionalWeights:
    def __init__(self,boundary_zip=DEFAULT_BOUNDARY,cache_dir=None,grid_mode='common_1deg'):
        if grid_mode not in ('common_1deg','native'):
            raise ValueError('Unknown grid mode')
        self.boundary_zip = Path(boundary_zip)
        self.boundary_sha = digest(self.boundary_zip.read_bytes())
        self.geometries = country_geometries(self.boundary_zip)
        self.grid_mode = grid_mode
        self.names = list(self.geometries)
        self.centroids = {zone: (self.geometries[zone+'_land'].centroid.y,
                                 self.geometries[zone+'_land'].centroid.x) for zone in ZONES}
        self.cache_dir = Path(cache_dir or ROOT/'cache/gefs_country_weights')
        if not self.cache_dir.resolve().resolve().is_relative_to(_work_path().resolve()):
            raise ValueError('Computed weights must live on D drive')
        self.cache_dir.mkdir(parents=True,exist_ok=True)
        self.memory = {}

    def for_grid(self,meta):
        spacing = float(meta['iDirectionIncrementInDegrees'])
        key = (int(meta['Ni']),int(meta['Nj']),spacing,self.grid_mode)
        if key in self.memory:
            return self.memory[key]
        signature = digest(json.dumps({'grid':key,'boundary':self.boundary_sha,
                            'marine_boxes':'fixed_v1','cell_edges':'0.1deg_densified','region_edges':'0.05deg_densified',
                            'geometry_sha':{name:digest(self.geometries[name].wkb) for name in self.names}},sort_keys=True).encode())[:24]
        cache = self.cache_dir/(signature+'.npz')
        if cache.exists():
            with np.load(cache,allow_pickle=False) as obj:
                result = {n:obj[n] for n in obj.files}
            self.memory[key] = result
            return result
        if spacing not in (.5,1.):
            raise ValueError('Only verified 0.5 or 1 degree GEFS grid supported')
        degree_step = 1. if self.grid_mode == 'common_1deg' else spacing
        latitude = 90-np.arange(int(meta['Nj']))*spacing
        longitude = np.arange(int(meta['Ni']))*spacing
        longitude[longitude>180] -= 360
        lat_keep = (latitude>=40)&(latitude<=57)
        lon_keep = (longitude>=-7)&(longitude<=17)
        if self.grid_mode == 'common_1deg':
            lat_keep &= np.isclose(latitude,np.round(latitude))
            lon_keep &= np.isclose(longitude,np.round(longitude))
        ijs = [(j,i) for j in np.flatnonzero(lat_keep) for i in np.flatnonzero(lon_keep)]
        matrix = np.zeros((len(self.names),len(ijs)))
        regions_area = np.array([transform(TRANSFORM,self.geometries[name]).area for name in self.names])
        for col,(j,i) in enumerate(ijs):
            cell = cell_polygon(float(longitude[i]),float(latitude[j]),degree_step)
            for row,name in enumerate(self.names):
                region = self.geometries[name]
                if cell.intersects(region):
                    overlap = cell.intersection(region)
                    if not overlap.is_empty:
                        matrix[row,col] = transform(TRANSFORM,overlap).area
        retained = matrix.sum(axis=0)>0
        matrix = matrix[:,retained]
        ijs = np.asarray(ijs,dtype=int)[retained]
        coverage = matrix.sum(axis=1)/regions_area
        if np.any(np.abs(coverage-1)>0.002):
            raise ValueError('Grid footprints fail coverage tolerance: '+repr(dict(zip(self.names,coverage.tolist()))))
        matrix /= matrix.sum(axis=1,keepdims=True)
        if not np.allclose(matrix.sum(axis=1),1,atol=1e-12):
            raise ValueError('Weights do not sum to one')
        result = {'indices': ijs[:,0]*int(meta['Ni'])+ijs[:,1], 'weights':matrix,
                  'latitude':latitude[ijs[:,0]], 'longitude':longitude[ijs[:,1]],
                  'area_km2':regions_area/1e6,'polygon_coverage':coverage,
                  'effective_grid_spacing':np.asarray(degree_step),'native_grid_spacing':np.asarray(spacing)}
        np.savez_compressed(cache,**result)
        self.memory[key] = result
        return result


def decode_object(path,weights,expected_day=None,expected_member=None,expected_step=None,receipt=None):
    """Verify four unchanged GRIB messages, then retain only weighted-region nodes."""
    content = Path(path).read_bytes()
    if receipt is not None:
        if len(content)!=receipt['bytes'] or digest(content)!=receipt['sha256']:
            raise ValueError('Original raw checksum/size differs from immutable receipt')
    arrays,metadata,cursor = {},[],0
    for shortname,units in zip(SHORTNAMES,UNITS):
        if content[cursor:cursor+4]!=b'GRIB' or content[cursor+7]!=2:
            raise ValueError('Expected GRIB edition2')
        length = int.from_bytes(content[cursor+8:cursor+16],'big')
        message = content[cursor:cursor+length]
        if len(message)!=length or message[-4:]!=b'7777':
            raise ValueError('Truncated original GRIB message')
        gid = eccodes.codes_new_from_message(message)
        try:
            keys = ('shortName','units','dataDate','dataTime','stepType','startStep','endStep',
                    'perturbationNumber','numberOfForecastsInEnsemble','Ni','Nj',
                    'iDirectionIncrementInDegrees','jDirectionIncrementInDegrees',
                    'latitudeOfFirstGridPointInDegrees','longitudeOfFirstGridPointInDegrees',
                    'jScansPositively','iScansNegatively','jPointsAreConsecutive','gridType')
            meta = {key:eccodes.codes_get(gid,key) for key in keys}
            if meta['shortName']!=shortname or meta['units']!=units:
                raise ValueError('GRIB order, variable or units mismatch')
            if meta['gridType']!='regular_ll' or meta['jScansPositively'] or meta['iScansNegatively'] or meta['jPointsAreConsecutive']:
                raise ValueError('Unsupported grid layout')
            if meta['latitudeOfFirstGridPointInDegrees']!=90 or meta['longitudeOfFirstGridPointInDegrees']!=0:
                raise ValueError('Unexpected global grid origin')
            if meta['iDirectionIncrementInDegrees']!=meta['jDirectionIncrementInDegrees']:
                raise ValueError('Unequal grid spacing')
            if expected_day is not None and str(meta['dataDate'])!=str(expected_day).replace('-',''):
                raise ValueError('Wrong forecast initialization')
            if meta['dataTime']!=0:
                raise ValueError('Only predeclared 00 UTC cycles supported')
            if expected_member is not None and meta['perturbationNumber']!=expected_member:
                raise ValueError('Wrong ensemble member')
            if expected_step is not None and meta['endStep']!=expected_step:
                raise ValueError('Wrong forecast step')
            if shortname=='sdswrf':
                if meta['stepType']!='avg' or meta['startStep']!=meta['endStep']-6:
                    raise ValueError('Solar field must be preceding six-hour average, not cumulative')
            elif meta['stepType']!='instant' or meta['startStep']!=meta['endStep']:
                raise ValueError('Temperature/wind must be instantaneous')
            area = weights.for_grid(meta)
            values = eccodes.codes_get_values(gid)[area['indices']]
            missing = eccodes.codes_get(gid,'missingValue')
            if not np.all(np.isfinite(values)) or np.any(values==missing):
                raise ValueError('Missing weather nodes inside required regional footprints')
            arrays[shortname] = values
            metadata.append(meta)
        finally:
            eccodes.codes_release(gid)
        cursor += length
    if cursor!=len(content):
        raise ValueError('Expected exactly four messages; extra bytes/messages found')
    base = metadata[0]
    for meta in metadata[1:]:
        for key in ('dataDate','dataTime','endStep','perturbationNumber','numberOfForecastsInEnsemble','Ni','Nj','iDirectionIncrementInDegrees'):
            if meta[key]!=base[key]:
                raise ValueError('Four fields disagree on initialization/member/grid')
    return arrays,base,area


def region_aggregates(arrays,area,weights):
    w = area['weights']
    means = {name:w@arrays[name] for name in SHORTNAMES}
    means['wind_speed'] = w@np.hypot(arrays['10u'],arrays['10v'])
    return {name:{key:float(value[row]) for key,value in means.items()}
            for row,name in enumerate(weights.names)}


def solar_cosine(timestamp,latitude,longitude):
    """Approximate NOAA solar geometry; used only to disaggregate forecast flux."""
    ts = pd.DatetimeIndex(timestamp)
    minute = ts.hour.to_numpy()*60+ts.minute.to_numpy()+ts.second.to_numpy()/60
    gamma = 2*np.pi/365*(ts.dayofyear.to_numpy()-1+(minute/60-12)/24)
    eqtime = 229.18*(.000075+.001868*np.cos(gamma)-.032077*np.sin(gamma)
                    -.014615*np.cos(2*gamma)-.040849*np.sin(2*gamma))
    declination = (.006918-.399912*np.cos(gamma)+.070257*np.sin(gamma)
                   -.006758*np.cos(2*gamma)+.000907*np.sin(2*gamma)
                   -.002697*np.cos(3*gamma)+.00148*np.sin(3*gamma))
    angle = np.deg2rad((minute+eqtime+4*longitude)/4-180)
    lat = np.deg2rad(latitude)
    return np.maximum(0,np.sin(lat)*np.sin(declination)+np.cos(lat)*np.cos(declination)*np.cos(angle))


def disaggregate_solar(six_hour_means,target_times,latitude,longitude):
    """Twenty-four hourly means conserve each six-hour forecast's total energy."""
    times = pd.DatetimeIndex(target_times)
    if len(times)!=24 or len(six_hour_means)!=4:
        raise ValueError('Requires 24 target hours and four six-hour means')
    # Five 12-minute quadrature points prevent short sunrise intervals disappearing.
    geometry = np.mean([solar_cosine(times+pd.Timedelta(minutes=float(offset)),latitude,longitude)
                        for offset in (6,18,30,42,54)],axis=0)
    hourly = np.zeros(24)
    for block,value in enumerate(six_hour_means):
        sl = slice(block*6,(block+1)*6)
        support = geometry[sl].sum()
        if value < -1e-6:
            raise ValueError('Negative mean downward solar radiation')
        value = max(float(value),0)
        if support>0:
            hourly[sl] = value*6*geometry[sl]/support
        elif value<=1e-6:
            hourly[sl] = 0
        else:
            # Forecast may contain tiny twilight flux. Keep its energy rather than
            # forcing an unverified physical zero when geometry says full night.
            hourly[sl] = value
    if not np.allclose(hourly.reshape(4,6).mean(axis=1),six_hour_means,atol=1e-6):
        raise ValueError('Solar disaggregation failed six-hour energy conservation')
    return hourly


def hourly_member(objects,weights,initialization_date,member,lead_base=24):
    if len(objects)!=5:
        raise ValueError('Requires all five six-hour endpoints')
    grid = objects[0][2]
    if any(not np.array_equal(o[2]['indices'],grid['indices']) for o in objects[1:]):
        raise ValueError('Grid changed inside one initialization/member')
    init = pd.Timestamp(str(initialization_date).replace('-',''),tz='UTC')
    steps = np.arange(5)*6+lead_base
    if lead_base not in (24,72):
        raise ValueError('Supported day-ahead weather ages use lead_base24/72')
    if [o[1]['endStep'] for o in objects]!=steps.tolist():
        raise ValueError('Endpoints disagree with configured lead base')
    target = init+pd.Timedelta(hours=lead_base)
    target_times = pd.date_range(target,periods=24,freq='h')
    midpoint_steps = np.arange(24)+lead_base+.5
    node_hours = {}
    for variable in ('2t','10u','10v'):
        values = np.asarray([o[0][variable] for o in objects])
        # Interpolation of components precedes local wind-speed calculation.
        pos = np.searchsorted(steps,midpoint_steps)-1
        pos = np.clip(pos,0,3)
        ratio = (midpoint_steps-steps[pos])/6
        node_hours[variable] = values[pos]*(1-ratio[:,None])+values[pos+1]*ratio[:,None]
    w = grid['weights']
    projected = {key:val@w.T for key,val in node_hours.items()}
    projected['wind_speed'] = np.hypot(node_hours['10u'],node_hours['10v'])@w.T
    radiation = np.asarray([o[0]['sdswrf'] for o in objects[1:]])@w.T
    out = pd.DataFrame({'target_time':target_times,'initialization_time':init,
                        'issue_time':target-pd.Timedelta(hours=12),'member_id':member,
                        'native_grid_degrees':float(grid['native_grid_spacing']),
                        'feature_grid_degrees':float(grid['effective_grid_spacing'])})
    for zone in ZONES:
        row = weights.names.index(zone+'_land')
        marine = weights.names.index(zone+'_marine')
        for key,label in (('2t','t2m'),('10u','u10'),('10v','v10'),('wind_speed','wind_speed')):
            out[f'{zone}_{label}'] = projected[key][:,row]
        lat,lon = weights.centroids[zone]
        out[f'{zone}_dswrf'] = disaggregate_solar(radiation[:,row],target_times,lat,lon)
        for key,label in (('10u','marine_u10'),('10v','marine_v10'),('wind_speed','marine_wind_speed')):
            out[f'{zone}_{label}'] = projected[key][:,marine]
    return out


def process_day(raw_day_directory,destination,boundary_zip=DEFAULT_BOUNDARY,members=20,grid_mode='common_1deg',weights=None,lead_base=24):
    directory, destination = Path(raw_day_directory),Path(destination)
    if not destination.resolve().resolve().is_relative_to(_work_path().resolve()):
        raise ValueError('Weather caches and results must be on D drive')
    day = directory.name
    if len(day)!=8 or not day.isdigit():
        raise ValueError('Raw day directory must be YYYYMMDD')
    if lead_base not in (24,72):
        raise ValueError('Supported lead_base values are24/72')
    steps = (np.arange(5)*6+lead_base).tolist()
    source = []
    for member in range(1,members+1):
        for step in steps:
            stem = directory/f'gefs_{day}_m{member:02}_f{step:03}'
            raw,receipt = stem.with_suffix('.grib2'),stem.with_suffix('.receipt.json')
            if not raw.exists() or not receipt.exists():
                raise FileNotFoundError(f'Incomplete raw day; missing member{member}/step{step}')
            record = json.loads(receipt.read_text(encoding='utf8'))
            if record['initialization_date']!=day or record['member_id']!=member or record['forecast_step_hours']!=step:
                raise ValueError('Receipt temporal/member mismatch')
            if raw.stat().st_size!=record['bytes']:
                raise ValueError('Raw size differs from immutable receipt')
            source.append((member,step,raw,record))
    boundary_hash = digest(Path(boundary_zip).read_bytes())
    fingerprint = digest(json.dumps({'sources':[r['sha256'] for _,_,_,r in source],
                      'boundary':boundary_hash,'grid_mode':grid_mode,
                      'script_sha256':digest(Path(__file__).read_bytes()),'members':members,'lead_base':lead_base},sort_keys=True).encode())
    outdir = destination/day
    manifest = outdir/'manifest.json'
    if manifest.exists():
        previous = json.loads(manifest.read_text(encoding='utf8'))
        if previous.get('input_fingerprint')==fingerprint and previous.get('status')=='complete':
            if all((outdir/name).exists() for name in ('member_hourly.parquet','ensemble_summary.parquet','member_sixhour.parquet')):
                return {**previous,'resumed_cache':True}
        raise ValueError('Existing derivative cache differs; choose new destination version')
    outdir.mkdir(parents=True,exist_ok=True)
    weights = weights or RegionalWeights(boundary_zip,grid_mode=grid_mode)
    if weights.grid_mode!=grid_mode or weights.boundary_sha!=boundary_hash:
        raise ValueError('Reused weights disagree with requested grid mode or boundary source')
    started = time.perf_counter()
    hour_frames,step_rows,actual_metadata = [],[],[]
    for member in range(1,members+1):
        objects = []
        for m,step,raw,receipt in source:
            if m!=member:
                continue
            decoded = decode_object(raw,weights,day,member,step,receipt)
            arrays,meta,area = decoded
            objects.append(decoded)
            actual_metadata.append(meta)
            aggregation = region_aggregates(arrays,area,weights)
            for region,values in aggregation.items():
                step_rows.append({'initialization_date':day,'member_id':member,'step':step,
                                  'region':region,'native_grid_degrees':float(area['native_grid_spacing']),
                                  'feature_grid_degrees':float(area['effective_grid_spacing']),**values})
        hour_frames.append(hourly_member(objects,weights,day,member,lead_base))
    hourly = pd.concat(hour_frames,ignore_index=True)
    if len({m['numberOfForecastsInEnsemble'] for m in actual_metadata})!=1:
        raise ValueError('Forecast ensemble metadata changed within initialization day')
    if len(hourly)!=24*members or hourly.duplicated(['target_time','member_id']).any():
        raise ValueError('Incomplete/duplicate member-hour table')
    feature_names = [c for c in hourly if any(c.startswith(zone+'_') for zone in ZONES)]
    if hourly[feature_names].isna().any().any():
        raise ValueError('Missing derived weather feature')
    summary = hourly.groupby('target_time')[feature_names].agg(['mean','std'])
    summary.columns = ['_'.join(c) for c in summary.columns]
    for quantile,label in ((.1,'p10'),(.9,'p90')):
        q = hourly.groupby('target_time')[feature_names].quantile(quantile)
        summary = summary.join(q.rename(columns={name:name+'_'+label for name in feature_names}))
    summary = summary.reset_index()
    for name in ('initialization_time','issue_time','native_grid_degrees','feature_grid_degrees'):
        summary[name] = hourly[name].iloc[0]
    hourly.to_parquet(outdir/'member_hourly.parquet',index=False)
    summary.to_parquet(outdir/'ensemble_summary.parquet',index=False)
    pd.DataFrame(step_rows).to_parquet(outdir/'member_sixhour.parquet',index=False)
    init = hourly['initialization_time'].iloc[0]
    report = {'status':'complete','initialization_date':day,'target_date':str((init+pd.Timedelta(hours=lead_base)).date()),
              'input_fingerprint':fingerprint,'members':members,'steps':steps,'lead_base':lead_base,'hourly_member_rows':len(hourly),
              'summary_rows':len(summary),'native_grid_degrees':hourly['native_grid_degrees'].iloc[0],
              'feature_grid_degrees':hourly['feature_grid_degrees'].iloc[0],'grid_mode':grid_mode,
              'elapsed_seconds':time.perf_counter()-started,'forecast_type':'actual archived perturbed operational forecasts',
              'boundary_source':'Natural Earth countries 1:10m 5.1.1 public domain','boundary_sha256':boundary_hash,
              'region_names':weights.names,'feature_names':feature_names,
              'polygon_area_km2':{n:float(a) for n,a in zip(weights.names,area['area_km2'])},
              'polygon_coverage':{n:float(a) for n,a in zip(weights.names,area['polygon_coverage'])},
              'issue_age_hours':float((hourly['issue_time'].iloc[0]-init)/pd.Timedelta(hours=1)),
              'warnings':['Political-country weather masks approximate electrical service domains; BE omits Sotel south Luxembourg.',
                          'Marine masks are fixed broad sea-weather proxy windows, not EEZ or installed-capacity weights.',
                          'Hourly temperature/wind use same-cycle linear interpolation at hourly midpoints; wind norm after component interpolation.',
                          'Solar flux is six-hour forecast mean, geometrically downscaled conserving block energy; full-night positive flux kept uniform.',
                          'common_1deg retains integer-degree nodes; it is not spatial-averaging downsampling and does not erase forecast model vintage.'],
              'created_at_utc':datetime.now(timezone.utc).isoformat(),'electricity_test_outcomes_read':False,
              'output_directory':str(outdir),'source_raw_directory':str(directory)}
    manifest.write_text(json.dumps(report,indent=2),encoding='utf8')
    return report


def run_cli(args):
    weights = RegionalWeights(args.boundaries,grid_mode=args.grid_mode)
    directory = Path(args.raw_snapshot)
    first,last = pd.Timestamp(args.start),pd.Timestamp(args.end)
    reports,failures = [],[]
    for day in pd.date_range(first,last,freq='D'):
        path = directory/day.strftime('%Y')/day.strftime('%Y%m%d')
        try:
            reports.append(process_day(path,args.destination,args.boundaries,args.members,args.grid_mode,weights,args.lead_base))
            print(json.dumps({'day':day.strftime('%Y%m%d'),'status':'complete','cache':reports[-1].get('resumed_cache',False)}),flush=True)
        except FileNotFoundError as exc:
            failures.append({'day':day.strftime('%Y%m%d'),'error':str(exc)})
            print(json.dumps({'day':day.strftime('%Y%m%d'),'status':'raw_incomplete'}),flush=True)
    result = {'completed_days':len(reports),'incomplete_days':len(failures),'failures':failures,
              'reports':reports,'status':'complete' if not failures else 'partial_raw_incomplete',
              'electricity_test_outcomes_read':False}
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    p = ROOT/'results'/('gefs_regional_run_'+stamp+'.json')
    p.write_text(json.dumps(result,indent=2),encoding='utf8')
    print(json.dumps({'result':str(p),'completed_days':len(reports),'incomplete_days':len(failures)}),flush=True)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-snapshot',required=True)
    parser.add_argument('--destination',default=str(ROOT/'cache/gefs_regional_common1_v1'))
    parser.add_argument('--boundaries',default=str(DEFAULT_BOUNDARY))
    parser.add_argument('--start',default='2018-12-29')
    parser.add_argument('--end',default='2018-12-29')
    parser.add_argument('--members',type=int,default=20)
    parser.add_argument('--grid-mode',choices=['common_1deg','native'],default='common_1deg')
    parser.add_argument('--lead-base',type=int,choices=[24,72],default=24)
    run_cli(parser.parse_args())
