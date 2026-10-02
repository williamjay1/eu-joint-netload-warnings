"""Upgrade data audit and compact, anonymous Danish transfer-data acquisition.

Owns only results/upgrade_data_audit_20261002 and this file. Original observations
and GRIB files remain immutable. Network snapshots use a new F-drive directory;
all normalized data and computed audits use the owned D-drive output directory.
"""
from __future__ import annotations
from release_paths import work_path as _work_path, raw_path as _raw_path
import argparse
from concurrent.futures import ProcessPoolExecutor, FIRST_COMPLETED, wait as wait_futures
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import shutil
import re
import sys
import time

import numpy as np
import pandas as pd
import requests

ROOT = _work_path()
OUT = ROOT / "results/upgrade_data_audit_20261002"
RAW = _raw_path()
PANEL = ROOT / "datasets/weather_panel_gefs_frozen_through_20260101"
OLD_RAW = RAW / "gefs_public_20260930T022939Z"
CACHE = ROOT / "cache/gefs_regional_timely_common1_v1"
WEEK_STARTS = ["2022-01-10", "2022-03-21", "2022-06-13", "2022-07-18",
               "2022-10-24", "2022-12-12", "2023-01-09", "2023-03-20",
               "2023-06-12", "2023-07-17", "2023-10-23", "2023-12-11"]
SETTLEMENT = ["GrossConsumptionMWh", "LocalPowerSelfConMWh", "SolarPowerSelfConMWh",
              "SolarPowerLt10kW_MWh", "SolarPowerGe10Lt40kW_MWh", "SolarPowerGe40kW_MWh",
              "OffshoreWindLt100MW_MWh", "OffshoreWindGe100MW_MWh",
              "OnshoreWindLt50kW_MWh", "OnshoreWindGe50kW_MWh"]
SCADA = ["TotalLoad", "SolarPower", "OnshoreWindPower", "OffshoreWindPower"]


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def jwrite(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def snapshot_write(path, content):
    path = Path(path)
    if not path.resolve().resolve().is_relative_to(_raw_path().resolve()) or not path.resolve().is_relative_to(RAW.resolve()):
        raise ValueError("Raw snapshot outside required F project raw directory")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(content)
    path.chmod(0o444)
    return {"path": str(path), "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}


def request_snapshot(url, params, path):
    path = Path(path)
    if path.exists():
        receipt = json.loads(path.with_suffix(path.suffix + ".receipt.json").read_text(encoding="utf-8"))
        content = path.read_bytes()
        if hashlib.sha256(content).hexdigest() != receipt["sha256"]:
            raise ValueError("Immutable snapshot hash differs")
        response = requests.Response()
        response.status_code = 200
        response._content = content
        response.encoding = "utf-8"
        response.url = receipt["request_url"]
        return response, receipt
    for attempt in range(8):
        try:
            response = requests.get(url, params=params, timeout=(20, 120))
            if response.status_code == 429:
                match = re.search(r"(?:in|after)\s+(\d+)\s+seconds", response.text, flags=re.I)
                delay = int(response.headers.get("Retry-After", 0) or 0)
                if match:
                    delay = max(delay, int(match.group(1)))
                delay = max(delay, 60)
                print(f"provider HTTP 429; waiting advertised interval {delay+1}s before next request", flush=True)
                time.sleep(delay + 1)
                continue
            response.raise_for_status()
            break
        except requests.RequestException:
            if attempt == 7:
                raise
            time.sleep(2 ** attempt)
    else:
        raise RuntimeError("Provider rate limit still active after bounded retries")
    receipt = {**snapshot_write(path, response.content), "request_url": response.url,
               "retrieval_utc": utcnow(), "status": response.status_code,
               "response_headers": {key: response.headers.get(key) for key in
                                    ["Date", "Last-Modified", "ETag", "Content-Type"]}}
    snapshot_write(Path(path).with_suffix(Path(path).suffix + ".receipt.json"),
                   json.dumps(receipt, indent=2).encode())
    return response, receipt


def native_audit():
    panel = pd.read_parquet(PANEL / "panel.parquet")
    timing = pd.read_parquet(PANEL / "timing.parquet")
    quality = pd.read_parquet(PANEL / "observation_quality.parquet")
    availability = pd.read_parquet(PANEL / "label_availability.parquet")
    feature_columns = [c for c in panel if not c.startswith("net_load_")]
    weather_columns = [c for c in panel if c.endswith("_mean") or c.endswith("_std")]
    rows, local_rows, receipt_rows = [], [], []
    for text in WEEK_STARTS:
        start = pd.Timestamp(text, tz="UTC")
        end = start + pd.Timedelta(days=7)
        expected = pd.date_range(start, end, freq="h", inclusive="left")
        sub = panel.reindex(expected)
        tm = timing.reindex(expected)
        ix = sub.index
        complete = sub[feature_columns + ["net_load_DE_LU", "net_load_FR", "net_load_BE"]].notna().all(axis=1)
        days = pd.Series(complete.to_numpy(), index=ix.normalize()).groupby(level=0).sum()
        row = {"week_start_UTC": str(start), "expected_hours": len(ix),
               "present_unique_hours": int(panel.index.isin(ix).sum()),
               "complete_feature_hours": int(sub[feature_columns].notna().all(axis=1).sum()),
               "complete_label_hours": int(sub[["net_load_DE_LU", "net_load_FR", "net_load_BE"]].notna().all(axis=1).sum()),
               "common_complete_hours": int(complete.sum()),
               "common_complete_UTC_days": int((days == 24).sum()),
               "weather_missing_hours": int(sub[weather_columns].isna().any(axis=1).sum()),
               "weather_age_min_h": float(((tm.issue_time - tm.initialization_time) / pd.Timedelta(hours=1)).min()),
               "weather_age_max_h": float(((tm.issue_time - tm.initialization_time) / pd.Timedelta(hours=1)).max()),
               "target_horizon_min_h": float(((ix - tm.issue_time) / pd.Timedelta(hours=1)).min()),
               "target_horizon_max_h": float(((ix - tm.issue_time) / pd.Timedelta(hours=1)).max()),
               "nominal_available_equals_issue": bool((tm.available_time == tm.issue_time).all()),
               "native_grid_degrees": sorted(tm.native_grid_degrees.unique().tolist()),
               "feature_grid_degrees": sorted(tm.feature_grid_degrees.unique().tolist())}
        for zone in ("DE_LU", "FR", "BE"):
            q = quality[(quality.timestamp >= start) & (quality.timestamp < end) & (quality.zone == zone)]
            row[zone + "_missing_label_hours"] = int(q.net_load_mw.isna().sum())
            row[zone + "_negative_generation_hours"] = int((q[["solar_mw", "wind_total_mw"]] < 0).any(axis=1).sum())
            row[zone + "_quality_flags"] = q.quality_flag.value_counts().to_dict()
            row[zone + "_night_reconstructed_hours"] = int(q.solar_night_reconstructed.fillna(False).sum())
            row[zone + "_archive_recovered_hours"] = int(q.solar_archive_recovered.fillna(False).sum())
        rows.append(row)
        for date in pd.date_range(start, end, freq="D", inclusive="left"):
            local_start = pd.Timestamp(date.date(), tz="Europe/Brussels")
            local_end = pd.Timestamp((date + pd.Timedelta(days=1)).date(), tz="Europe/Brussels")
            local_index = pd.date_range(local_start, local_end, freq="h", inclusive="left").tz_convert("UTC")
            local_rows.append({"audit_week": text, "local_date": str(date.date()),
                               "expected_local_day_hours": len(local_index),
                               "existing_unique_UTC_hours": int(panel.index.isin(local_index).sum()),
                               "complete_weather_hours": int(panel.reindex(local_index)[weather_columns].notna().all(axis=1).sum()),
                               "offsets_h": sorted(set(local_start.utcoffset().total_seconds()/3600 for local_start in
                                                        local_index.tz_convert("Europe/Brussels")))})
            init = (date - pd.Timedelta(days=1)).strftime("%Y%m%d")
            manifest = json.loads((CACHE / init / "manifest.json").read_text(encoding="utf-8"))
            for member in (1, 10, 20):
                for step in (24, 30, 36, 42, 48):
                    rp = OLD_RAW / init[:4] / init / f"gefs_{init}_m{member:02}_f{step:03}.receipt.json"
                    receipt = json.loads(rp.read_text(encoding="utf-8"))
                    modification = [x["source_last_modified"] for x in receipt["range_requests"]]
                    receipt_rows.append({"initialization_date": init, "member": member, "step": step,
                                         "member_count_in_cache": manifest["members"],
                                         "cache_status": manifest["status"],
                                         "receipt_retrieval_utc": receipt["retrieval_utc"],
                                         "source_last_modified_values": modification,
                                         "source_url": receipt["source_url"],
                                         "raw_bytes": receipt["bytes"]})
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "development_12_week_audit.csv", index=False)
    pd.DataFrame(local_rows).to_csv(OUT / "local_DST_day_audit.csv", index=False)
    pd.DataFrame(receipt_rows).to_parquet(OUT / "sampled_GEFS_object_receipts.parquet", index=False)
    upgrade = []
    for init in ("20200922", "20200923", "20200924"):
        m = json.loads((CACHE / init / "manifest.json").read_text(encoding="utf-8"))
        upgrade.append({k: m[k] for k in ("initialization_date", "native_grid_degrees", "feature_grid_degrees", "members", "steps")})
    label_offsets = ((availability.label_available_time - availability.index)/pd.Timedelta(hours=1)).unique().tolist()
    modification_offsets = []
    for record in receipt_rows:
        init = pd.to_datetime(record["initialization_date"], format="%Y%m%d", utc=True)
        for value in record["source_last_modified_values"]:
            modification_offsets.append(float((pd.to_datetime(value, utc=True)-init)/pd.Timedelta(hours=1)))
    all_dev = panel.loc["2022":"2023"]
    report = {"status": "completed", "created_utc": utcnow(), "selection": "Twelve fixed calendar weeks spanning both development years, seasons, and both DST changes; no outcomes used to select weeks",
              "weeks": rows, "aggregate": {"weeks": len(rows), "hours": int(frame.expected_hours.sum()),
               "common_complete_hours": int(frame.common_complete_hours.sum()),
               "common_complete_days": int(frame.common_complete_UTC_days.sum()),
               "sampled_object_receipts": len(receipt_rows), "DST_short_days": sum(x["expected_local_day_hours"] == 23 for x in local_rows),
               "DST_long_days": sum(x["expected_local_day_hours"] == 25 for x in local_rows),
               "full_development_index_unique": bool(all_dev.index.is_unique),
               "full_development_UTC_index_gap_count": len(pd.date_range("2022-01-01", "2024-01-01", tz="UTC", inclusive="left", freq="h").difference(all_dev.index))},
              "GEFS_upgrade_boundary": upgrade,
              "sampled_object_modification_offsets": {
                "n_range_records": len(modification_offsets),
                "min_hours_after_initialization": float(min(modification_offsets)),
                "max_hours_after_initialization": float(max(modification_offsets)),
                "median_hours_after_initialization": float(np.median(modification_offsets)),
                "n_later_than_issue_plus12h": sum(x > 12 for x in modification_offsets),
                "scope": "Current LastModified offsets only, not historical first-publication evidence"},
              "timestamps_not_historical_availability": {"weather_available_time": "Stored as initialization+12h by modelling convention. No sampled object provides historical first-publication evidence.",
               "object_LastModified": "Current HTTP object metadata; neither historical first arrival nor proof of all operational objects being public by issue time.",
               "retrieval_utc": "Actual acquisition of archival data in 2026, not historical operational availability.",
               "label_available_time_in_auxiliary": {"offset_hours_after_target_start": label_offsets, "meaning": "Generic completed-interval clock, not actual provider release. The frozen fitting code independently applies seven-day label cutoffs. This file alone is not a release-vintage ledger."},
               "electricity_vintage": "Retrospectively downloaded provider values plus explicitly flagged PV recovery/night reconstruction; no complete as-of revision history."},
              "asserted_scope": "Delayed historical replay, not certified real-time deployment. Structural UTC/missingness audit is independent of live-vintage recovery.",
              "existing_GEFS_raw_reuse": "Immutable raw contains original global GRIB messages for 2m temperature, 10m u/v, and surface shortwave. Existing decoded cache retains DE/FR/BE polygons; additional DK nodes can be decoded from the already-held raw without new weather download."}
    jwrite(OUT / "development_data_audit.json", report)
    return report


def weather_reuse_pilot():
    """Decode one prior development-date cycle at fixed Danish proxy rectangles.

    This proves existing raw geographical support. The main transfer experiment
    separately uses its frozen shared-weather basis; this is not an evaluation of
    a new Danish forecast model or an exact bidding-zone weather mask.
    """
    import eccodes
    init = "20220326"
    bounds = {"DK1_proxy": (54, 58, 8, 11), "DK2_proxy": (54, 57, 11, 13)}
    started = time.monotonic()
    rows = []
    for member in range(1, 21):
        for step in (24, 30, 36, 42, 48):
            path = OLD_RAW / init[:4] / init / f"gefs_{init}_m{member:02}_f{step:03}.grib2"
            arrays = {zone: {} for zone in bounds}
            with path.open("rb") as handle:
                while True:
                    gid = eccodes.codes_grib_new_from_file(handle)
                    if gid is None:
                        break
                    try:
                        name = eccodes.codes_get(gid, "shortName")
                        ni, nj = eccodes.codes_get(gid, "Ni"), eccodes.codes_get(gid, "Nj")
                        spacing = eccodes.codes_get(gid, "iDirectionIncrementInDegrees")
                        latitude = 90 - np.arange(nj)*spacing
                        longitude = np.arange(ni)*spacing
                        values = eccodes.codes_get_values(gid).reshape(nj, ni)
                        for zone, (lat0, lat1, lon0, lon1) in bounds.items():
                            keep_lat = (latitude >= lat0) & (latitude <= lat1) & np.isclose(latitude, np.round(latitude))
                            keep_lon = (longitude >= lon0) & (longitude <= lon1) & np.isclose(longitude, np.round(longitude))
                            arrays[zone][name] = values[np.ix_(keep_lat, keep_lon)].copy()
                    finally:
                        eccodes.codes_release(gid)
            for zone, fields in arrays.items():
                if set(fields) != {"2t", "10u", "10v", "sdswrf"}:
                    raise ValueError("Incomplete expected GRIB support")
                if not all(np.isfinite(value).all() for value in fields.values()):
                    raise ValueError("Nonfinite existing Danish proxy weather nodes")
                rows.append({"initialization": init, "member_id": member, "forecast_step_h": step,
                             "region": zone, "common_integer_degree_nodes": fields["2t"].size,
                             "temperature_K": float(fields["2t"].mean()),
                             "u10_ms": float(fields["10u"].mean()), "v10_ms": float(fields["10v"].mean()),
                             "wind_speed_ms": float(np.hypot(fields["10u"], fields["10v"]).mean()),
                             "dswrf_Wm2": float(fields["sdswrf"].mean())})
    data = pd.DataFrame(rows)
    data.to_parquet(OUT / "DK_existing_GRIB_one_cycle_members.parquet", index=False)
    data.groupby(["region", "forecast_step_h"])[["temperature_K", "wind_speed_ms", "dswrf_Wm2"]].agg(["mean", "std"]).to_csv(OUT / "DK_existing_GRIB_one_cycle_summary.csv")
    result = {"status": "completed", "initialization": init, "members": 20,
              "endpoints_h": [24, 30, 36, 42, 48], "global_objects_read": 100,
              "member_region_endpoint_rows": len(data), "proxy_rectangles_latlon": bounds,
              "common_integer_degree_nodes": data.groupby("region").common_integer_degree_nodes.first().to_dict(),
              "elapsed_seconds": time.monotonic()-started, "new_raw_weather_downloaded_bytes": 0,
              "interpretation": "Direct extraction from previously preserved original global forecast messages proves additional Danish forecast support is available. Fixed rectangular weather proxies include sea and land; they are not exact bidding-zone or installed-generation masks. No transfer forecast has been scored with these local weather variables in this pilot."}
    jwrite(OUT / "DK_existing_GRIB_weather_support_pilot.json", result)
    return result


class DanishWeatherWeights:
    """Natural Earth island-group proxies, with the original area-weight engine."""
    VERSION = "DK_LOCAL_NE_ISLAND_GROUPS_v1_20261002"

    def __init__(self):
        import shapefile
        import zipfile
        from shapely import make_valid, segmentize
        from shapely.geometry import box, shape
        from shapely.ops import unary_union
        from gefs_regional_features import DEFAULT_BOUNDARY, digest
        self.boundary_zip = DEFAULT_BOUNDARY
        self.boundary_sha = digest(DEFAULT_BOUNDARY.read_bytes())
        self.grid_mode = "common_1deg"
        self.memory = {}
        self.cache_dir = OUT / "DK_local_weather_weights"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        land, danish = [], None
        roi = box(4, 53, 17, 59)
        with zipfile.ZipFile(DEFAULT_BOUNDARY) as z:
            reader = shapefile.Reader(shp=io.BytesIO(z.read("ne_10m_admin_0_countries.shp")),
                                     shx=io.BytesIO(z.read("ne_10m_admin_0_countries.shx")),
                                     dbf=io.BytesIO(z.read("ne_10m_admin_0_countries.dbf")), encoding="utf8")
            for item in reader.iterShapeRecords():
                geom = make_valid(shape(item.shape.__geo_interface__))
                if geom.intersects(roi):
                    land.append(geom.intersection(roi))
                if item.record.as_dict().get("ADM0_A3") == "DNK":
                    danish = geom.intersection(roi)
        if danish is None:
            raise ValueError("Denmark missing in existing immutable boundary source")
        parts = list(danish.geoms) if hasattr(danish, "geoms") else [danish]
        # Land components are grouped west/east of the Great Belt. Læsø/Anholt
        # lie east of the Belt's longitude but belong to western Denmark: include
        # northern island components (latitude>56.6). This is a forecast-weather
        # geographical proxy, not a claimed recovered official electrical polygon.
        west = [p for p in parts if p.representative_point().x < 11.1 or p.representative_point().y > 56.6]
        east = [p for p in parts if p not in west]
        if not west or not east:
            raise ValueError("Incomplete Danish western/eastern geographic support")
        all_land = unary_union(land)
        self.geometries = {"DK1_land": unary_union(west), "DK2_land": unary_union(east),
                           "DK1_marine": box(5, 54, 8.8, 57.5).difference(all_land),
                           "DK2_marine": box(11, 54, 16, 56.5).difference(all_land)}
        self.geometries = {name: segmentize(geom, .05) for name, geom in self.geometries.items()}
        self.names = list(self.geometries)
        self.centroids = {zone: (self.geometries[zone+"_land"].centroid.y,
                                 self.geometries[zone+"_land"].centroid.x) for zone in ("DK1", "DK2")}
        self.geometry_sha = {name: digest(geom.wkb) for name, geom in self.geometries.items()}

    def for_grid(self, meta):
        from shapely.ops import transform
        from gefs_regional_features import TRANSFORM, cell_polygon, digest
        spacing = float(meta["iDirectionIncrementInDegrees"])
        key = (int(meta["Ni"]), int(meta["Nj"]), spacing)
        if key in self.memory:
            return self.memory[key]
        signature = digest(json.dumps({"version": self.VERSION, "grid": key,
                                      "boundary": self.boundary_sha, "geometry": self.geometry_sha}, sort_keys=True).encode())[:24]
        cache = self.cache_dir / (signature+".npz")
        if cache.exists():
            with np.load(cache, allow_pickle=False) as item:
                result = {name: item[name] for name in item.files}
            self.memory[key] = result
            return result
        latitude = 90-np.arange(int(meta["Nj"]))*spacing
        longitude = np.arange(int(meta["Ni"]))*spacing
        longitude[longitude > 180] -= 360
        lat = np.flatnonzero((latitude >= 53) & (latitude <= 59) & np.isclose(latitude, np.round(latitude)))
        lon = np.flatnonzero((longitude >= 4) & (longitude <= 17) & np.isclose(longitude, np.round(longitude)))
        locations, overlap_columns = [], []
        region_area = np.asarray([transform(TRANSFORM, self.geometries[name]).area for name in self.names])
        for j in lat:
            for i in lon:
                cell = cell_polygon(float(longitude[i]), float(latitude[j]), 1.)
                overlaps = np.asarray([transform(TRANSFORM, cell.intersection(self.geometries[name])).area
                                       if cell.intersects(self.geometries[name]) else 0 for name in self.names])
                if overlaps.sum() > 0:
                    locations.append((j, i))
                    overlap_columns.append(overlaps)
        matrix = np.asarray(overlap_columns).T
        coverage = matrix.sum(axis=1)/region_area
        if not np.allclose(coverage, 1, atol=2e-4):
            raise ValueError("Danish land/sea weather footprints not completely covered")
        locations = np.asarray(locations)
        result = {"weights": matrix/matrix.sum(axis=1)[:, None],
                  "indices": locations[:, 0]*int(meta["Ni"])+locations[:, 1],
                  "latitude": latitude[locations[:, 0]], "longitude": longitude[locations[:, 1]],
                  "area_km2": region_area/1e6, "polygon_coverage": coverage,
                  "native_grid_spacing": np.asarray(spacing), "effective_grid_spacing": np.asarray(1.)}
        np.savez_compressed(cache, **result)
        self.memory[key] = result
        return result


_DANISH_WEIGHTS = None


def local_weather_day(init_text, io_order="member_step"):
    """One immutable initialization, fully verified before a day cache is complete."""
    global _DANISH_WEIGHTS
    from gefs_regional_features import decode_object, disaggregate_solar, digest
    if _DANISH_WEIGHTS is None:
        _DANISH_WEIGHTS = DanishWeatherWeights()
    weights = _DANISH_WEIGHTS
    directory = OUT / "DK_local_weather_daily" / init_text
    manifest_path = directory / "manifest.json"
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text(encoding="utf8"))
        if old.get("status") == "completed" and old.get("version") == weights.VERSION:
            for filename in ("member_hourly.parquet", "ensemble_summary.parquet"):
                if not (directory/filename).exists():
                    raise ValueError("Completed daily weather cache missing outputs")
            return {**old, "resumed_cache": True}
        raise ValueError("Existing local weather cache has another version or incomplete manifest")
    init = pd.Timestamp(init_text, tz="UTC")
    steps = np.arange(5)*6+24
    target = init+pd.Timedelta(days=1)
    times = pd.date_range(target, periods=24, freq="h")
    midpoint_steps = np.arange(24)+24.5
    pos = np.clip(np.searchsorted(steps, midpoint_steps)-1, 0, 3)
    ratio = (midpoint_steps-steps[pos])/6
    rows, source_hashes = [], []
    started = time.monotonic()
    if io_order not in ("member_step", "write_time"):
        raise ValueError("Unsupported raw read ordering")
    decoded, ordered_hashes = {}, {}
    if io_order == "write_time":
        # Original acquisition was concurrent. Read the same immutable objects in
        # their write-time order to reduce HDD seek distance; assembly still uses
        # the fixed member/endpoint order, so this is purely an IO optimization.
        jobs = []
        for member in range(1, 21):
            for step in steps:
                stem = OLD_RAW/init_text[:4]/init_text/f"gefs_{init_text}_m{member:02}_f{step:03}"
                jobs.append((stem.with_suffix(".grib2").stat().st_mtime_ns, member, int(step), stem))
        for _, member, step, stem in sorted(jobs):
            receipt = json.loads(stem.with_suffix(".receipt.json").read_text(encoding="utf8"))
            if receipt["initialization_date"] != init_text or receipt["member_id"] != member or receipt["forecast_step_hours"] != step:
                raise ValueError("Raw receipt day/member/endpoint mismatch")
            decoded[(member, step)] = decode_object(stem.with_suffix(".grib2"), weights, init_text, member, step, receipt)
            ordered_hashes[(member, step)] = receipt["sha256"]
    for member in range(1, 21):
        objects = []
        for step in steps:
            if io_order == "write_time":
                objects.append(decoded[(member, int(step))])
                source_hashes.append(ordered_hashes[(member, int(step))])
                continue
            stem = OLD_RAW/init_text[:4]/init_text/f"gefs_{init_text}_m{member:02}_f{step:03}"
            receipt = json.loads(stem.with_suffix(".receipt.json").read_text(encoding="utf8"))
            if receipt["initialization_date"] != init_text or receipt["member_id"] != member or receipt["forecast_step_hours"] != int(step):
                raise ValueError("Raw receipt day/member/endpoint mismatch")
            objects.append(decode_object(stem.with_suffix(".grib2"), weights, init_text, member, int(step), receipt))
            source_hashes.append(receipt["sha256"])
        area = objects[0][2]
        if any(not np.array_equal(o[2]["indices"], area["indices"]) for o in objects[1:]):
            raise ValueError("Grid changed within cycle/member")
        node_hours = {}
        for variable in ("2t", "10u", "10v"):
            values = np.asarray([o[0][variable] for o in objects])
            node_hours[variable] = values[pos]*(1-ratio[:, None])+values[pos+1]*ratio[:, None]
        projected = {key: values@area["weights"].T for key, values in node_hours.items()}
        projected["wind_speed"] = np.hypot(node_hours["10u"], node_hours["10v"])@area["weights"].T
        radiation = np.asarray([o[0]["sdswrf"] for o in objects[1:]])@area["weights"].T
        out = pd.DataFrame({"target_time": times, "initialization_time": init,
                            "issue_time": target-pd.Timedelta(hours=12), "member_id": member,
                            "native_grid_degrees": float(area["native_grid_spacing"]), "feature_grid_degrees": 1.})
        for zone in ("DK1", "DK2"):
            land = weights.names.index(zone+"_land")
            marine = weights.names.index(zone+"_marine")
            for source, label in (("2t", "t2m"), ("10u", "u10"), ("10v", "v10"), ("wind_speed", "wind_speed")):
                out[f"{zone}_{label}"] = projected[source][:, land]
            lat, lon = weights.centroids[zone]
            out[f"{zone}_dswrf"] = disaggregate_solar(radiation[:, land], times, lat, lon)
            for source, label in (("10u", "marine_u10"), ("10v", "marine_v10"), ("wind_speed", "marine_wind_speed")):
                out[f"{zone}_{label}"] = projected[source][:, marine]
        rows.append(out)
    hourly = pd.concat(rows, ignore_index=True)
    features = [c for c in hourly if c.startswith(("DK1_", "DK2_"))]
    if len(hourly) != 480 or hourly.duplicated(["target_time", "member_id"]).any() or not np.isfinite(hourly[features].to_numpy()).all():
        raise ValueError("Incomplete or nonfinite Danish hourly forecast support")
    summary = hourly.groupby("target_time")[features].agg(["mean", "std"])
    summary.columns = ["_".join(c) for c in summary.columns]
    summary = summary.reset_index()
    for column in ("initialization_time", "issue_time", "native_grid_degrees", "feature_grid_degrees"):
        summary[column] = hourly[column].iloc[0]
    directory.mkdir(parents=True, exist_ok=True)
    hourly.to_parquet(directory/"member_hourly.parquet", index=False)
    summary.to_parquet(directory/"ensemble_summary.parquet", index=False)
    report = {"status": "completed", "version": weights.VERSION, "initialization_date": init_text,
              "target_date": str(target.date()), "members": 20, "steps": steps.tolist(),
              "source_receipts": 100, "source_sha256_fingerprint": digest("".join(source_hashes).encode()),
              "boundary_sha256": weights.boundary_sha, "geometry_sha": weights.geometry_sha,
              "region_names": weights.names, "polygon_coverage": area["polygon_coverage"].tolist(),
              "polygon_area_km2": area["area_km2"].tolist(), "native_grid_degrees": float(area["native_grid_spacing"]),
              "member_hourly_rows": len(hourly), "summary_rows": len(summary), "elapsed_seconds": time.monotonic()-started,
              "created_utc": utcnow(), "electricity_labels_read": False, "weather_download_bytes": 0,
              "io_order": io_order, "raw_full_reads_per_object": 1}
    jwrite(manifest_path, report)
    return report


def local_weather_batch(start, end, workers=4, stop_marker=None, progress_every=25, io_order="member_step"):
    if workers < 1 or workers > 4:
        raise ValueError("At most four worker processes permitted for this IO pilot")
    if shutil.disk_usage(_work_path()).free < 2_000_000_000:
        raise RuntimeError("Insufficient D work capacity")
    dates = [x.strftime("%Y%m%d") for x in pd.date_range(start, end, freq="D")]
    marker = Path(stop_marker) if stop_marker else OUT/"DK_local_weather_STOP_REQUESTED"
    if not marker.resolve().is_relative_to(OUT.resolve()):
        raise ValueError("Weather stop marker must be within owned work results")
    if progress_every < 1:
        raise ValueError("Progress interval must be positive")
    # Build both weights before child processes start to avoid concurrent writes.
    weights = DanishWeatherWeights()
    for ni, nj, spacing in ((360, 181, 1.), (720, 361, .5)):
        weights.for_grid({"Ni": ni, "Nj": nj, "iDirectionIncrementInDegrees": spacing})
    definition = {"status": "running", "version": weights.VERSION, "created_utc": utcnow(),
                  "start_initialization": start, "end_initialization": end, "requested_days": len(dates),
                  "workers": workers, "source": str(OLD_RAW), "new_weather_download_bytes": 0,
                  "io_order": io_order, "raw_full_reads_per_object": 1,
                  "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  "boundary_sha256": weights.boundary_sha, "geometry_sha": weights.geometry_sha,
                  "land_mask_rule": "Natural Earth Danish polygon components with representative point longitude<11.1 or latitude>56.6 assign western proxy; remaining Danish islands eastern proxy. This approximates electrical zone weather support, not official recovered bidding-zone polygons.",
                  "marine_rule": "Fixed North Sea [5,54,8.8,57.5] and Baltic [11,54,16,56.5] proxy boxes minus land; not EEZ or installed capacity weights.",
                  "numeric_rule": "Retain integer-degree nodes with area-weighted 1-degree cells; same-cycle midpoint linear temperature and u/v interpolation; node wind norm before regional mean; six-hour solar energy conserved by same existing disaggregate_solar.",
                  "temperature_unit": "Member/summary t2m mean K, std K. Margin feature consumer converts mean to degrees C by subtracting 273.15; std unchanged.",
                  "electricity_labels_read": False, "complete_days": 0, "failed_days": 0,
                  "stop_marker": str(marker), "scheduling": "At most workers outstanding; stop marker drains currently executing days, retains complete checkpoints, and submits no further day."}
    path = OUT/f"DK_local_weather_batch_{dates[0]}_{dates[-1]}.json"
    jwrite(path, definition)
    started = time.monotonic()
    completed, failed = [], []
    remaining = iter(dates)
    stopped = marker.exists()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {}
        while True:
            stopped = stopped or marker.exists()
            while not stopped and len(futures) < workers:
                day = next(remaining, None)
                if day is None:
                    break
                futures[pool.submit(local_weather_day, day, io_order)] = day
            if not futures:
                break
            ready, _ = wait_futures(futures, return_when=FIRST_COMPLETED)
            for future in ready:
                day = futures.pop(future)
                try:
                    result = future.result()
                    completed.append(result)
                except Exception as exc:
                    failed.append({"day": day, "error": str(exc)})
                if (len(completed)+len(failed)) % progress_every == 0 or len(completed)+len(failed) == len(dates):
                    definition.update(complete_days=len(completed), failed_days=len(failed),
                                      elapsed_seconds=time.monotonic()-started, failures=failed)
                    jwrite(path, definition)
                    print(json.dumps({"complete_days": len(completed), "failed_days": len(failed),
                                      "wall_seconds": time.monotonic()-started, "last_day": day}), flush=True)
    definition.update(status="stopped_at_day_checkpoints" if stopped and len(completed)+len(failed)<len(dates) else ("completed" if not failed else "incomplete"), complete_days=len(completed), failed_days=len(failed),
                      elapsed_seconds=time.monotonic()-started, failures=failed,
                      summed_new_worker_seconds=sum(x["elapsed_seconds"] for x in completed if not x.get("resumed_cache")),
                      resumed_days=sum(bool(x.get("resumed_cache")) for x in completed))
    jwrite(path, definition)
    if not failed and len(completed) == len(dates):
        summaries = [pd.read_parquet(OUT/"DK_local_weather_daily"/day/"ensemble_summary.parquet") for day in sorted(dates)]
        summary = pd.concat(summaries, ignore_index=True).sort_values("target_time").set_index("target_time")
        if len(summary) != len(dates)*24 or not summary.index.is_unique:
            raise ValueError("Combined Danish weather summary has missing/duplicate hours")
        target = OUT/f"DK_local_weather_summary_{dates[0]}_{dates[-1]}.parquet"
        summary.to_parquet(target)
        definition["summary_path"] = str(target)
        definition["summary_hours"] = len(summary)
        definition["summary_sha256"] = hashlib.sha256(target.read_bytes()).hexdigest()
        jwrite(path, definition)
    return definition


def normalize_settlement(pieces):
    d = pd.concat(pieces, ignore_index=True)
    d["timestamp"] = pd.to_datetime(d.HourUTC, utc=True)
    d = d[(d.timestamp >= "2019-01-01") & (d.timestamp < "2026-01-01")]
    if d.duplicated(["timestamp", "PriceArea"]).any():
        duplicates = d[d.duplicated(["timestamp", "PriceArea"], keep=False)]
        if any(len(g[SETTLEMENT].drop_duplicates()) != 1 for _, g in duplicates.groupby(["timestamp", "PriceArea"])):
            raise ValueError("Conflicting duplicate settlement observations")
        d = d.drop_duplicates(["timestamp", "PriceArea"])
    d["solar_total_mw"] = d[[c for c in SETTLEMENT if c.startswith("Solar")]].sum(axis=1, min_count=4)
    d["wind_total_mw"] = d[[c for c in SETTLEMENT if "Wind" in c]].sum(axis=1, min_count=4)
    # Follows current official warning to exclude commercial self-consumption
    # from total consumption because it overlaps estimated solar self-consumption.
    d["load_corrected_mw"] = d.GrossConsumptionMWh - d.LocalPowerSelfConMWh
    d["net_load_mw"] = d.load_corrected_mw - d.solar_total_mw - d.wind_total_mw
    d["historic_grouping_flag"] = np.where(d.timestamp.dt.year == 2019,
        "all PV reported in self-consumption field; zero other PV fields are classification, not missing-source imputation",
        "post-2019 PV capacity categories")
    d["measurement_warning"] = "Current settlement vintage; gross consumption and PV self-consumption are estimates; pre-2021 commercial/self-consumption split differs"
    d = d.sort_values(["timestamp", "PriceArea"])
    d.to_parquet(OUT / "DK_settlement_hourly_2019_2025.parquet", index=False)
    return d


def normalize_scada(pieces):
    d = pd.concat(pieces, ignore_index=True)
    d["timestamp"] = pd.to_datetime(d.HourUTC, utc=True)
    d = d[(d.timestamp >= "2019-01-01") & (d.timestamp < "2026-01-01")]
    if d.duplicated(["timestamp", "PriceArea"]).any():
        duplicate = d[d.duplicated(["timestamp", "PriceArea"], keep=False)]
        if any(len(group[SCADA].drop_duplicates()) > 1 for _, group in duplicate.groupby(["timestamp", "PriceArea"])):
            raise ValueError("Conflicting duplicate SCADA observations")
        d = d.drop_duplicates(["timestamp", "PriceArea"])
    year_start, year_end = int(d.timestamp.dt.year.min()), int(d.timestamp.dt.year.max())
    native = d.sort_values(["timestamp", "PriceArea"])
    native.to_parquet(OUT / f"DK_SCADA_native_{year_start}_{year_end}.parquet", index=False)
    hourly = []
    for zone, sub in native.groupby("PriceArea"):
        sub = sub.set_index("timestamp").sort_index()
        index = pd.date_range(f"{year_start}-01-01", f"{year_end+1}-01-01", tz="UTC", inclusive="left", freq="h")
        result = sub[SCADA].resample("h").mean().reindex(index)
        count = sub[SCADA].resample("h").count().reindex(index).fillna(0)
        # Metadata reports a 15-minute transition on 2025-04-23 12:45 CET.
        # Use observed quarter-hour presence at each hour, plus expected new-regime
        # density, so absent subintervals are not silently accepted as hourly means.
        expected = np.where(index >= pd.Timestamp("2025-04-23T11:00Z"), 4, 1)
        quarter_present = sub.index.minute.to_series(index=sub.index).ne(0).resample("h").max().reindex(index).fillna(False)
        expected = np.where(quarter_present, 4, expected)
        good = (count.to_numpy() == expected[:, None]).all(axis=1)
        result.loc[~good] = np.nan
        result["PriceArea"] = zone
        result["timestamp"] = index
        result["native_expected_intervals"] = expected
        result["native_observed_rows"] = sub.resample("h").size().reindex(index).fillna(0).to_numpy()
        result["complete_hour"] = good
        result["load_mw"] = result.TotalLoad
        result["solar_mw"] = result.SolarPower
        result["wind_total_mw"] = result.OnshoreWindPower + result.OffshoreWindPower
        result["net_load_mw"] = result.load_mw - result.solar_mw - result.wind_total_mw
        hourly.append(result.reset_index(drop=True))
    result = pd.concat(hourly, ignore_index=True).sort_values(["timestamp", "PriceArea"])
    result.to_parquet(OUT / f"DK_SCADA_hourly_{year_start}_{year_end}.parquet", index=False)
    return result


def external_download(resume_snapshot=None):
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    snapshot = Path(resume_snapshot) if resume_snapshot else RAW / ("upgrade_20261002_" + stamp)
    disk = {"work":shutil.disk_usage(_work_path()).free,"raw":shutil.disk_usage(RAW).free}
    if disk["raw"] < 1_000_000_000 or disk["work"] < 1_000_000_000:
        raise RuntimeError("Insufficient checked F or D capacity")
    metadata, receipts = {}, []
    for dataset in ("ProductionConsumptionSettlement", "ElectricityBalanceNonv"):
        r, receipt = request_snapshot(f"https://api.energidataservice.dk/meta/dataset/{dataset}", None,
                                      snapshot / f"{dataset}_metadata.json")
        metadata[dataset] = r.json()
        receipts.append(receipt)
    resources = [("terms_and_conditions.html", "https://www.energidataservice.dk/terms-and-conditions"),
                 ("api_guide.html", "https://www.energidataservice.dk/guides/api-guides"),
                 ("Gioia_accepted.pdf", "https://eprints.gla.ac.uk/342345/1/342345.pdf")]
    resource_errors = []
    for filename, url in resources:
        try:
            _, receipt = request_snapshot(url, None, snapshot / filename)
            receipts.append(receipt)
        except Exception as exc:
            resource_errors.append({"url": url, "error": str(exc)})
    jobs = [(dataset, year) for dataset in metadata for year in range(2019, 2026)]
    parsed = {key: [] for key in metadata}
    def acquire(job):
        dataset, year = job
        columns = SETTLEMENT if dataset == "ProductionConsumptionSettlement" else SCADA
        # API date filters are Danish local time, even though selected field is UTC.
        # Deliberately request padding and filter exactly in UTC after acquisition.
        params = {"start": f"{year}-01-01", "end": f"{year+1}-01-02", "limit": 100000,
                  "columns": ",".join(["HourUTC", "HourDK", "PriceArea", *columns]),
                  "filter": json.dumps({"PriceArea": ["DK1", "DK2"]}), "sort": "HourUTC ASC,PriceArea ASC"}
        response, receipt = request_snapshot(f"https://api.energidataservice.dk/dataset/{dataset}", params,
                                             snapshot / f"{dataset}_{year}.json")
        body = response.json()
        records = pd.DataFrame(body["records"])
        if len(records) >= 100000:
            raise ValueError("Possible API truncation")
        times = pd.to_datetime(records.HourUTC, utc=True)
        records = records[times.dt.year == year]
        return dataset, year, records, {**receipt, "returned_records": len(body["records"]), "UTC_year_records": len(records)}
    started = time.monotonic()
    # Sequential retrieval respects differentiated per-dataset limits. Completed
    # immutable snapshots are resumed rather than re-requested or overwritten.
    for job in jobs:
        dataset, year, records, receipt = acquire(job)
        parsed[dataset].append(records)
        receipts.append(receipt)
        print(f"available {dataset} {year}: {len(records)} records", flush=True)
        time.sleep(2)
    settle = normalize_settlement(parsed["ProductionConsumptionSettlement"])
    scada = normalize_scada(parsed["ElectricityBalanceNonv"])
    coverage = []
    for kind, d in (("settlement", settle), ("SCADA", scada)):
        for (zone, year), group in d.groupby(["PriceArea", d.timestamp.dt.year]):
            expected = (pd.Timestamp(f"{year+1}-01-01")-pd.Timestamp(f"{year}-01-01")).days*24
            coverage.append({"kind": kind, "zone": zone, "year": int(year), "expected_hours": expected,
                             "rows": len(group), "valid_net_load_hours": int(group.net_load_mw.notna().sum()),
                             "negative_net_load_hours": int((group.net_load_mw < 0).sum()),
                             "valid_complete_days": int(group.groupby(group.timestamp.dt.normalize()).net_load_mw.count().eq(24).sum())})
    a = settle[["timestamp", "PriceArea", "net_load_mw"]].rename(columns={"net_load_mw": "settlement_net"})
    b = scada[["timestamp", "PriceArea", "net_load_mw"]].rename(columns={"net_load_mw": "SCADA_net"})
    joined = a.merge(b, on=["timestamp", "PriceArea"]).dropna()
    comparisons = []
    for (zone, year), group in joined.groupby(["PriceArea", joined.timestamp.dt.year]):
        diff = group.settlement_net - group.SCADA_net
        comparisons.append({"zone": zone, "year": int(year), "matched_hours": len(group),
                            "mean_settlement_minus_SCADA_MW": float(diff.mean()),
                            "MAE_MW": float(diff.abs().mean()),
                            "p95_absolute_difference_MW": float(diff.abs().quantile(.95)),
                            "correlation": float(group.settlement_net.corr(group.SCADA_net))})
    pd.DataFrame(coverage).to_csv(OUT / "DK_coverage_by_year_zone.csv", index=False)
    pd.DataFrame(comparisons).to_csv(OUT / "DK_measurement_comparison_by_year_zone.csv", index=False)
    report = {"status": "completed", "created_utc": utcnow(), "raw_snapshot": str(snapshot),
              "checked_disk_free_bytes": disk, "download_elapsed_seconds": time.monotonic()-started,
              "raw_bytes_downloaded": sum(x["bytes"] for x in receipts), "source_receipts": receipts,
              "resource_errors": resource_errors, "coverage": coverage, "measurement_comparisons": comparisons,
              "metadata": {name: {key: m.get(key) for key in ["title", "description", "comment", "caution", "resolution", "dataFrom", "dataTo", "lastDataUpdate", "lastMetadataUpdate", "published"]} for name, m in metadata.items()},
              "licence": {"name": "CC BY 4.0", "verified_url": "https://www.energidataservice.dk/terms-and-conditions", "credit": "Source: Energinet (www.energidataservice.dk); transformations must be indicated; no endorsement implied"},
              "measurement_choices": {"SCADA": "TotalLoad minus SolarPower minus OnshoreWindPower minus OffshoreWindPower. All are mean MW (MWh/hour). Transition 15min observations are averaged only when all four intervals/core fields exist. Public metadata describes uncorrected SCADA errors and discontinuation in January 2026.",
               "settlement": "GrossConsumptionMWh minus LocalPowerSelfConMWh (as current official consumption warning requests), minus sum of all four solar fields and all four wind fields; hourly MWh divided by 1h equals mean MW. No null is imputed to zero. Official 2019 all-PV grouping is retained as a flag. Pre-2021 commercial/self-consumption classification differs.",
               "boundaries": "PriceArea values DK1 west and DK2 east of Great Belt are explicit bidding zones. These provider measurement boundaries differ from SMARD/RTE/Elia; no Europe-wide identical accounting assertion."},
              "availability": "Current settlement metadata says 9–15-day initial delay and revisions possible up to two years. New regional fitting should use >=15-day label lag if settlement is selected; current snapshots still are not recovered original vintages. SCADA public data are operational measurements but archival snapshots do not recover original publication times.",
              "evaluation_status": "New-region observations acquired and audited; no external prediction effectiveness is established by this report."}
    jwrite(OUT / "external_DK_source_audit.json", report)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--native", action="store_true")
    parser.add_argument("--external", action="store_true")
    parser.add_argument("--resume-snapshot")
    parser.add_argument("--local-weather", action="store_true")
    parser.add_argument("--start", default="2018-12-31")
    parser.add_argument("--end", default="2019-04-09")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--stop-marker")
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--io-order", choices=["member_step", "write_time"], default="member_step")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.native:
        r = native_audit()
        print(json.dumps(r["aggregate"], indent=2))
        print(json.dumps(weather_reuse_pilot(), indent=2))
    if args.external:
        r = external_download(args.resume_snapshot)
        print(json.dumps({key: r[key] for key in ["status", "raw_snapshot", "raw_bytes_downloaded", "download_elapsed_seconds"]}, indent=2))
    if args.local_weather:
        print(json.dumps(local_weather_batch(args.start, args.end, args.workers,
                                             args.stop_marker, args.progress_every, args.io_order), indent=2))


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
