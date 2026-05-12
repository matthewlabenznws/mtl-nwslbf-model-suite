# ============================================================
# REFS M01 | NWS LBF Viewer Product
# Reflectivity + UH + Sim IR + Theta Cold Pools + 4–6 km SR Winds
# Uses rrfs_a/rrfsens.YYYYMMDD/HH/m001 links
# ============================================================

import os
import re
import zipfile
import requests
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patheffects as pe

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader

import geopandas as gpd
from shapely.ops import unary_union
from shapely.prepared import prep

from scipy.ndimage import gaussian_filter
from scipy.interpolate import griddata
from mpl_toolkits.axes_grid1 import make_axes_locatable
from datetime import datetime, timedelta, timezone
from matplotlib.colors import ListedColormap, BoundaryNorm


# ============================================================
# PATHS / ASSETS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

ASSET_DIR = os.path.join(BASE_DIR, "assets")
COUNTY_SHP = os.path.join(ASSET_DIR, "cb_2018_us_county_500k.shp")
STATE_SHP = os.path.join(ASSET_DIR, "cb_2018_us_state_500k.shp")
LBF_CWA_SHP = os.path.join(ASSET_DIR, "c_18mr25.shp")
LOGO_PATH = os.path.join(ASSET_DIR, "NOAANWSLogos.png")

zip_path = os.path.join(ASSET_DIR, "c_18mr25.zip")
if os.path.exists(zip_path):
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(ASSET_DIR)

DATA_DIR = os.path.join(BASE_DIR, "refs_mem01_subsets")

OUTDIR_BASE = os.path.join(
    "site",
    "runs",
    "refs_mem01",
    "refl_uh"
)

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTDIR_BASE, exist_ok=True)


# ============================================================
# DOMAIN CONFIG
# ============================================================

DOMAINS = {
    "lbf": {
        "label": "LBF",
        "extent": [-103.8, -97.0, 40.0, 43.4],
        "title_size": 14,
        "subtitle_size": 11,
        "barb_skip": 11,
    },

    "regional": {
        "label": "Default",
        "extent": [-107.5, -93.0, 38.5, 44.2],
        "title_size": 13,
        "subtitle_size": 11,
        "barb_skip": 20,
    },

    "central_plains": {
        "label": "Central Plains",
        "extent": [-107.5, -91.0, 34.5, 45.2],
        "title_size": 13,
        "subtitle_size": 11,
        "barb_skip": 24,
    },
}


# ============================================================
# DYNAMIC SPC SEVERE DOMAIN
# ============================================================

SPC_DAY1_CAT_URL = (
    "https://mapservices.weather.noaa.gov/vector/rest/services/outlooks/"
    "SPC_wx_outlks/MapServer/1/query"
)

SPC_RISK_ORDER = {
    "TSTM": 1,
    "MRGL": 2,
    "SLGT": 3,
    "ENH": 4,
    "MDT": 5,
    "HIGH": 6,
}

MIN_SPC_RISK = "SLGT"
SEVERE_DOMAIN_WIDTH = 14.0
SEVERE_DOMAIN_HEIGHT = 10.0


def fetch_spc_day1_geojson():
    params = {
        "where": "1=1",
        "outFields": "*",
        "f": "geojson",
        "returnGeometry": "true",
        "outSR": "4326",
    }

    r = requests.get(SPC_DAY1_CAT_URL, params=params, timeout=30)
    r.raise_for_status()

    data = r.json()

    if "features" not in data or len(data["features"]) == 0:
        raise RuntimeError("SPC query returned no features.")

    return gpd.GeoDataFrame.from_features(data["features"], crs="EPSG:4326")


def add_spc_severe_domain():
    try:
        gdf = fetch_spc_day1_geojson().to_crs(epsg=4326)

        risk_col = None

        for col in gdf.columns:
            vals = gdf[col].astype(str).str.upper()
            if vals.isin(SPC_RISK_ORDER.keys()).any():
                risk_col = col
                break

        if risk_col is None:
            print("SPC severe domain skipped: could not find risk category column.")
            return

        gdf["risk"] = gdf[risk_col].astype(str).str.upper()
        gdf["risk_rank"] = gdf["risk"].map(SPC_RISK_ORDER)

        severe = gdf[gdf["risk_rank"] >= SPC_RISK_ORDER[MIN_SPC_RISK]].copy()

        if severe.empty:
            print("SPC severe domain skipped: no SLGT+ risk found.")
            return

        highest_rank = severe["risk_rank"].max()
        highest = severe[severe["risk_rank"] == highest_rank].copy()

        highest_proj = highest.to_crs(epsg=5070)
        highest["_area"] = highest_proj.geometry.area.values

        main_poly = highest.loc[highest["_area"].idxmax()]
        highest_label = main_poly["risk"]

        main_gdf = gpd.GeoDataFrame(
            [main_poly],
            geometry="geometry",
            crs="EPSG:4326"
        )

        centroid_proj = main_gdf.to_crs(epsg=5070).geometry.centroid

        centroid_ll = gpd.GeoSeries(
            centroid_proj,
            crs="EPSG:5070"
        ).to_crs(epsg=4326).iloc[0]

        center_lon = centroid_ll.x
        center_lat = centroid_ll.y

        extent = [
            center_lon - SEVERE_DOMAIN_WIDTH / 2,
            center_lon + SEVERE_DOMAIN_WIDTH / 2,
            center_lat - SEVERE_DOMAIN_HEIGHT / 2,
            center_lat + SEVERE_DOMAIN_HEIGHT / 2,
        ]

        DOMAINS["spc_severe"] = {
            "label": f"SPC {highest_label} Risk",
            "extent": extent,
            "title_size": 13,
            "subtitle_size": 11,
            "barb_skip": 22,
        }

        print(f"Added SPC severe domain: {highest_label}")
        print(f"SPC severe extent: {extent}")

    except Exception as e:
        print(f"SPC severe domain skipped due to error: {e}")


add_spc_severe_domain()


# ============================================================
# SETTINGS
# ============================================================

REFS_MEMBER = "m001"
REFS_LABEL = "REFS M01"

VALID_RRFS_CYCLES = [0, 6, 12, 18]

START_FHR = 1
CYCLE_DELAY_MINUTES = 45

MAX_FHR = 60

MANUAL_STORM_MOTION_FROM_DEG = 250
MANUAL_STORM_MOTION_SPEED_KT = 35

PLOT_SR_WIND_BARBS = True


# ============================================================
# REFLECTIVITY COLOR TABLE
# ============================================================

bounds = [
    0, 10, 12.5, 15, 17.5, 20, 22.5, 25, 27.5, 30,
    32.5, 35, 37.5, 40, 42.5, 45, 47.5, 50, 52.5,
    55, 57.5, 60, 62.5, 65, 67.5, 70, 72.5
]

colors = [
    "#ffffff", "#dae2f2", "#b4c4e5", "#8fa7d9", "#6a89cb", "#486cbf", "#2c4eb2",
    "#1e4f5e", "#48746d", "#799b7c", "#aac08b", "#fbf477", "#f1d461", "#e7b54c",
    "#dd9738", "#d37826", "#ca5917", "#c31d14", "#9a1511", "#710e10", "#9c3aae",
    "#7f27a0", "#601392", "#828282", "#b4b4b4", "#e6e6e6"
]

cmap = ListedColormap(colors, name="reflec_bins")
norm = BoundaryNorm(bounds, cmap.N, clip=True)

REF_LEVELS = [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75]


# ============================================================
# BASIC HELPERS
# ============================================================

def to_lon180(lon):
    return ((np.asarray(lon) + 180) % 360) - 180


def k_to_c(k):
    return np.asarray(k) - 273.15


def kt_to_ms(kt):
    return kt * 0.514444


def ms_to_kt(ms):
    return ms * 1.94384


def wind_from_dir_speed_to_uv(direction_from_deg, speed_ms):
    rad = np.deg2rad(direction_from_deg)
    u = -speed_ms * np.sin(rad)
    v = -speed_ms * np.cos(rad)
    return u, v


def get_lat_lon(da):
    if "latitude" in da.coords and "longitude" in da.coords:
        lat = np.asarray(da.latitude.values)
        lon = to_lon180(da.longitude.values)
    elif "lat" in da.coords and "lon" in da.coords:
        lat = np.asarray(da.lat.values)
        lon = to_lon180(da.lon.values)
    else:
        raise RuntimeError("Could not find latitude/longitude coordinates.")

    lat = np.squeeze(lat)
    lon = np.squeeze(lon)

    if lat.ndim != 2 or lon.ndim != 2:
        raise RuntimeError(f"Lat/lon not 2D. lat={lat.shape}, lon={lon.shape}")

    return lat, lon


def ensure_2d_field(da, label):
    arr = np.asarray(da.values, dtype=float)
    arr = np.squeeze(arr)

    if arr.ndim != 2:
        raise RuntimeError(
            f"{label} is not 2D after squeeze. "
            f"Shape={arr.shape}, dims={getattr(da, 'dims', None)}"
        )

    return arr


# ============================================================
# SHAPEFILE HELPERS
# ============================================================

def add_shapefile_outline(ax, shp_path, edgecolor="k", linewidth=1.2, zorder=6):
    if not os.path.exists(shp_path):
        print("Missing shapefile:", shp_path)
        return

    gdf = gpd.read_file(shp_path).to_crs(epsg=4326)
    gdf = gdf.cx[LON_MIN - 1:LON_MAX + 1, LAT_MIN - 1:LAT_MAX + 1]

    ax.add_geometries(
        gdf.geometry,
        crs=ccrs.PlateCarree(),
        facecolor="none",
        edgecolor=edgecolor,
        linewidth=linewidth,
        zorder=zorder,
    )


def get_lbf_cwa_geom(cwa_shp_path):
    if not os.path.exists(cwa_shp_path):
        print("Missing LBF CWA shapefile:", cwa_shp_path)
        return None

    reader = shpreader.Reader(cwa_shp_path)
    recs = list(reader.records())

    geoms = [
        r.geometry for r in recs
        if str(r.attributes.get("CWA", "")).upper() == "LBF"
        or str(r.attributes.get("WFO", "")).upper() == "LBF"
    ]

    if not geoms:
        geoms = [r.geometry for r in recs]

    return unary_union(geoms)


def add_counties_clipped_to_cwa(ax, counties_shp_path, cwa_geom, lw=1.0, color="black", zorder=6):
    if cwa_geom is None or not os.path.exists(counties_shp_path):
        return

    reader = shpreader.Reader(counties_shp_path)
    cwa_p = prep(cwa_geom)
    clipped = []

    for r in reader.records():
        g = r.geometry
        if cwa_p.intersects(g):
            inter = g.intersection(cwa_geom)
            if not inter.is_empty:
                clipped.append(inter)

    ax.add_geometries(
        clipped,
        crs=ccrs.PlateCarree(),
        facecolor="none",
        edgecolor=color,
        linewidth=lw,
        zorder=zorder,
    )


# ============================================================
# REFS URL / IDX BYTE-RANGE SUBSETTING
# ============================================================

def rrfs_grib_url(init_dt, fhr, product="2dfld"):
    ymd = init_dt.strftime("%Y%m%d")
    hh = init_dt.strftime("%H")

    if product == "2dfld":
        fname = f"rrfs.t{hh}z.{REFS_MEMBER}.2dfld.3km.f{fhr:03d}.conus.grib2"
    elif product == "prslev":
        fname = f"rrfs.t{hh}z.{REFS_MEMBER}.prslev.3km.f{fhr:03d}.conus.grib2"
    else:
        raise ValueError("product must be '2dfld' or 'prslev'")

    return (
        f"https://noaa-rrfs-pds.s3.amazonaws.com/"
        f"rrfs_a/rrfsens.{ymd}/{hh}/{REFS_MEMBER}/{fname}"
    )


def url_exists(url, timeout=10):
    try:
        r = requests.head(url, timeout=timeout, allow_redirects=True)
        return r.status_code == 200
    except Exception:
        return False


def find_latest_available_rrfs_cycle(max_back_hours=72):
    now = datetime.now(timezone.utc) - timedelta(minutes=CYCLE_DELAY_MINUTES)

    for back in range(max_back_hours + 1):
        dt = now - timedelta(hours=back)

        if dt.hour not in VALID_RRFS_CYCLES:
            continue

        dt = dt.replace(minute=0, second=0, microsecond=0, tzinfo=None)

        test_url = rrfs_grib_url(dt, 1, product="2dfld") + ".idx"

        if url_exists(test_url):
            print(f"Latest {REFS_LABEL} cycle found: {dt:%Y%m%d} {dt:%HZ}")
            print("Matched IDX:", test_url)
            return dt

    raise RuntimeError(f"Could not find recent {REFS_LABEL} cycle.")


def read_idx(idx_url):
    r = requests.get(idx_url, timeout=30)
    r.raise_for_status()
    return r.text.strip().splitlines()


def parse_idx_lines(lines):
    parsed = []

    for i, line in enumerate(lines):
        parts = line.split(":")

        if len(parts) < 5:
            continue

        try:
            msg_num = int(parts[0])
            start_byte = int(parts[1])
        except Exception:
            continue

        parsed.append({
            "i": i,
            "line": line,
            "msg_num": msg_num,
            "start": start_byte,
        })

    for j in range(len(parsed)):
        if j < len(parsed) - 1:
            parsed[j]["end"] = parsed[j + 1]["start"] - 1
        else:
            parsed[j]["end"] = None

    return parsed


def find_idx_match(parsed, all_terms, label):
    all_terms_lower = [t.lower() for t in all_terms]
    matches = []

    for item in parsed:
        line_lower = item["line"].lower()
        if all(term in line_lower for term in all_terms_lower):
            matches.append(item)

    if not matches:
        sample = "\n".join([p["line"] for p in parsed[:150]])
        raise RuntimeError(
            f"Could not find {label} in IDX using terms {all_terms}.\n"
            f"First 150 IDX lines:\n{sample}"
        )

    match = matches[0]

    print(f"Matched {label}:")
    print(match["line"])

    return match


def download_byte_range(grib_url, start, end, outpath):
    if os.path.exists(outpath) and os.path.getsize(outpath) > 0:
        print("Using cached subset:", outpath)
        return outpath

    headers = {}

    if end is None:
        headers["Range"] = f"bytes={start}-"
    else:
        headers["Range"] = f"bytes={start}-{end}"

    print("Downloading byte range:", headers["Range"])

    r = requests.get(grib_url, headers=headers, stream=True, timeout=120)
    r.raise_for_status()

    with open(outpath, "wb") as f:
        for chunk in r.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)

    return outpath


def open_subset_grib(path, label):
    ds = xr.open_dataset(
        path,
        engine="cfgrib",
        backend_kwargs={"indexpath": ""}
    )

    if len(ds.data_vars) == 0:
        raise RuntimeError(f"No variables found in subset for {label}")

    var = list(ds.data_vars)[0]
    da = ds[var]

    print(f"Opened {label}: var={var}, dims={da.dims}, shape={da.shape}")

    return da


def rrfs_idx_field(init_dt, fhr, term_sets, label, product="2dfld"):
    grib_url = rrfs_grib_url(init_dt, fhr, product=product)
    idx_url = grib_url + ".idx"

    lines = read_idx(idx_url)
    parsed = parse_idx_lines(lines)

    last_error = None

    for terms in term_sets:
        try:
            match = find_idx_match(parsed, terms, label)

            safe_label = re.sub(r"[^A-Za-z0-9]+", "_", label).strip("_")

            outname = (
                f"refs_{REFS_MEMBER}_{product}_{init_dt:%Y%m%d_%H}z_f{fhr:03d}_"
                f"{safe_label}_{match['msg_num']}.grib2"
            )

            outpath = os.path.join(DATA_DIR, outname)

            download_byte_range(
                grib_url,
                match["start"],
                match["end"],
                outpath
            )

            return open_subset_grib(outpath, label)

        except Exception as e:
            last_error = e

    raise RuntimeError(f"Could not open {label}. Last error: {last_error}")


# ============================================================
# SPATIAL HELPERS
# ============================================================

def subset_2d(lat, lon, *fields):
    mask = (
        np.isfinite(lat) &
        np.isfinite(lon) &
        (lon >= LON_MIN) &
        (lon <= LON_MAX) &
        (lat >= LAT_MIN) &
        (lat <= LAT_MAX)
    )

    if not np.any(mask):
        raise RuntimeError("No grid points found inside selected domain.")

    iy, ix = np.where(mask)

    iy0 = max(iy.min() - 2, 0)
    iy1 = min(iy.max() + 3, lat.shape[0])

    ix0 = max(ix.min() - 2, 0)
    ix1 = min(ix.max() + 3, lon.shape[1])

    return (
        lat[iy0:iy1, ix0:ix1],
        lon[iy0:iy1, ix0:ix1],
        [f[iy0:iy1, ix0:ix1] for f in fields]
    )


def interp_to_target_grid(src_lat, src_lon, src_field, tgt_lat, tgt_lon):
    src_points = np.column_stack((src_lon.ravel(), src_lat.ravel()))
    src_values = np.asarray(src_field).ravel()

    good = (
        np.isfinite(src_points[:, 0]) &
        np.isfinite(src_points[:, 1]) &
        np.isfinite(src_values)
    )

    out = griddata(
        src_points[good],
        src_values[good],
        (tgt_lon, tgt_lat),
        method="linear"
    )

    if np.isnan(out).any():
        out_nearest = griddata(
            src_points[good],
            src_values[good],
            (tgt_lon, tgt_lat),
            method="nearest"
        )
        out = np.where(np.isnan(out), out_nearest, out)

    return out


# ============================================================
# FIND REFS CYCLE
# ============================================================

init_dt = find_latest_available_rrfs_cycle()
cycle_str = init_dt.strftime("%Y%m%d_%Hz")

OUTDIR = os.path.join(OUTDIR_BASE, cycle_str)
os.makedirs(OUTDIR, exist_ok=True)

fhrs = range(START_FHR, MAX_FHR + 1)

print(f"Using {REFS_LABEL} init:", init_dt.strftime("%Y-%m-%d %HZ"))
print("Forecast hours:", list(fhrs))
print("Output directory:", OUTDIR)
print("Domains:", list(DOMAINS.keys()))

lbf_geom = get_lbf_cwa_geom(LBF_CWA_SHP)


# ============================================================
# LOAD REFS FIELDS
# ============================================================

def load_rrfs_fields_once(fhr):
    print("\n" + "=" * 70)
    print(f"Loading {REFS_LABEL} | Init {init_dt:%Y-%m-%d %HZ} | F{fhr:03d}")
    print("=" * 70)

    refl_da = rrfs_idx_field(
        init_dt,
        fhr,
        [
            ["REFD", "1000 m"],
            ["REFC"],
            ["REFD"],
        ],
        "reflectivity",
        product="2dfld"
    )

    lat, lon = get_lat_lon(refl_da)

    refl = ensure_2d_field(refl_da, "reflectivity")
    refl = np.where(refl >= REF_LEVELS[0], refl, np.nan)

    try:
        uh25_da = rrfs_idx_field(
            init_dt,
            fhr,
            [
                ["MXUPHL", "5000-2000"],
                ["MXUPHL", "5000 - 2000"],
                ["MXUPHL"],
            ],
            "2-5km UH",
            product="2dfld"
        )
        uh25 = ensure_2d_field(uh25_da, "2-5km UH")
    except Exception as e:
        print(f"2-5km UH not available for F{fhr:03d}. Using blank field.")
        print(e)
        uh25 = np.full_like(refl, np.nan)

    try:
        uh03_da = rrfs_idx_field(
            init_dt,
            fhr,
            [
                ["MXUPHL", "3000-0"],
                ["MXUPHL", "3000 - 0"],
            ],
            "0-3km UH",
            product="2dfld"
        )
        uh03 = ensure_2d_field(uh03_da, "0-3km UH")
    except Exception as e:
        print(f"0-3km UH not available for F{fhr:03d}. Using blank field.")
        print(e)
        uh03 = np.full_like(refl, np.nan)

    try:
        ir_da = rrfs_idx_field(
            init_dt,
            fhr,
            [
                ["SBT123"],
                ["SBT124"],
                ["SBTA"],
                ["SBT"],
                ["brightness"],
                ["satellite"],
            ],
            "simulated IR",
            product="2dfld"
        )
        ir_c = k_to_c(ensure_2d_field(ir_da, "simulated IR"))
        print("Loaded simulated IR.")
    except Exception as e:
        print(f"Simulated IR not available for F{fhr:03d}. Continuing without IR.")
        print(e)
        ir_c = np.full_like(refl, np.nan)

    t2_da = rrfs_idx_field(
        init_dt,
        fhr,
        [
            ["TMP", "2 m above ground"],
            ["TMP", "2 m"],
        ],
        "2m temperature",
        product="2dfld"
    )

    ps_da = rrfs_idx_field(
        init_dt,
        fhr,
        [
            ["PRES", "surface"],
        ],
        "surface pressure",
        product="2dfld"
    )

    t2_k = ensure_2d_field(t2_da, "2m temperature")
    ps_pa = ensure_2d_field(ps_da, "surface pressure")

    theta = t2_k * (100000.0 / ps_pa) ** 0.286
    theta_bg = gaussian_filter(theta, sigma=18)
    theta_prime = theta - theta_bg

    u700_da = rrfs_idx_field(init_dt, fhr, [["UGRD", "700 mb"], ["UGRD", "700"]], "700mb U wind", product="prslev")
    v700_da = rrfs_idx_field(init_dt, fhr, [["VGRD", "700 mb"], ["VGRD", "700"]], "700mb V wind", product="prslev")
    u600_da = rrfs_idx_field(init_dt, fhr, [["UGRD", "600 mb"], ["UGRD", "600"]], "600mb U wind", product="prslev")
    v600_da = rrfs_idx_field(init_dt, fhr, [["VGRD", "600 mb"], ["VGRD", "600"]], "600mb V wind", product="prslev")
    u500_da = rrfs_idx_field(init_dt, fhr, [["UGRD", "500 mb"], ["UGRD", "500"]], "500mb U wind", product="prslev")
    v500_da = rrfs_idx_field(init_dt, fhr, [["VGRD", "500 mb"], ["VGRD", "500"]], "500mb V wind", product="prslev")

    pr_lat, pr_lon = get_lat_lon(u700_da)

    u46_pr = np.nanmean(
        np.stack([
            ensure_2d_field(u700_da, "700mb U wind"),
            ensure_2d_field(u600_da, "600mb U wind"),
            ensure_2d_field(u500_da, "500mb U wind"),
        ]),
        axis=0
    )

    v46_pr = np.nanmean(
        np.stack([
            ensure_2d_field(v700_da, "700mb V wind"),
            ensure_2d_field(v600_da, "600mb V wind"),
            ensure_2d_field(v500_da, "500mb V wind"),
        ]),
        axis=0
    )

    u46_native = interp_to_target_grid(pr_lat, pr_lon, u46_pr, lat, lon)
    v46_native = interp_to_target_grid(pr_lat, pr_lon, v46_pr, lat, lon)

    try:
        u_stm_da = rrfs_idx_field(
            init_dt,
            fhr,
            [
                ["UEID"],
                ["USTM"],
                ["BUNK"],
            ],
            "Bunkers storm motion U",
            product="2dfld"
        )

        v_stm_da = rrfs_idx_field(
            init_dt,
            fhr,
            [
                ["VEID"],
                ["VSTM"],
                ["BUNK"],
            ],
            "Bunkers storm motion V",
            product="2dfld"
        )

        stm_lat, stm_lon = get_lat_lon(u_stm_da)

        u_stm_native = interp_to_target_grid(
            stm_lat,
            stm_lon,
            ensure_2d_field(u_stm_da, "Bunkers storm motion U"),
            lat,
            lon
        )

        v_stm_native = interp_to_target_grid(
            stm_lat,
            stm_lon,
            ensure_2d_field(v_stm_da, "Bunkers storm motion V"),
            lat,
            lon
        )

        sr_u46 = u46_native - u_stm_native
        sr_v46 = v46_native - v_stm_native

        storm_motion_source = f"{REFS_LABEL} UEID/VEID"
        print(f"Using {REFS_LABEL} UEID/VEID Bunkers storm motion for SR winds.")

    except Exception as e:
        print(f"Could not find {REFS_LABEL} UEID/VEID storm motion. Falling back to manual storm motion.")
        print(e)

        storm_u_scalar, storm_v_scalar = wind_from_dir_speed_to_uv(
            MANUAL_STORM_MOTION_FROM_DEG,
            kt_to_ms(MANUAL_STORM_MOTION_SPEED_KT)
        )

        sr_u46 = u46_native - np.full_like(refl, storm_u_scalar)
        sr_v46 = v46_native - np.full_like(refl, storm_v_scalar)

        storm_motion_source = "Manual storm motion"

    return {
        "lat": lat,
        "lon": lon,
        "refl": refl,
        "uh25": uh25,
        "uh03": uh03,
        "ir_c": ir_c,
        "theta_prime": theta_prime,
        "sr_u46": sr_u46,
        "sr_v46": sr_v46,
        "storm_motion_source": storm_motion_source,
    }


# ============================================================
# PLOT FUNCTION
# ============================================================

def plot_domain_from_fields(fields, domain_key, cfg, fhr):
    global LON_MIN, LON_MAX, LAT_MIN, LAT_MAX

    LON_MIN, LON_MAX, LAT_MIN, LAT_MAX = cfg["extent"]

    domain_outdir = os.path.join(OUTDIR, domain_key)
    os.makedirs(domain_outdir, exist_ok=True)

    print(f"Plotting {domain_key.upper()} | F{fhr:03d}")

    try:
        lat = fields["lat"]
        lon = fields["lon"]
        refl = fields["refl"]
        uh25 = fields["uh25"]
        uh03 = fields["uh03"]
        ir_c = fields["ir_c"]
        theta_prime = fields["theta_prime"]
        sr_u46 = fields["sr_u46"]
        sr_v46 = fields["sr_v46"]

        lat_sub, lon_sub, [
            refl_sub,
            uh25_sub,
            uh03_sub,
            ir_sub,
            theta_prime_sub,
            sr_u46_sub,
            sr_v46_sub,
        ] = subset_2d(
            lat,
            lon,
            refl,
            uh25,
            uh03,
            ir_c,
            theta_prime,
            sr_u46,
            sr_v46
        )

        refl_plot = gaussian_filter(np.nan_to_num(refl_sub, nan=0.0), sigma=0.5)
        refl_plot = np.where(refl_plot >= 5, refl_plot, np.nan)

        uh25_plot = gaussian_filter(np.nan_to_num(uh25_sub, nan=0.0), sigma=0.2)
        uh03_plot = gaussian_filter(np.nan_to_num(uh03_sub, nan=0.0), sigma=0.2)

        uh_combined = np.where((uh25_plot >= 75) | (uh03_plot >= 50), 1, np.nan)

        theta_prime_smooth = gaussian_filter(theta_prime_sub, sigma=2.5)
        theta_cp_mask = np.ma.masked_where(theta_prime_smooth > -2.0, theta_prime_smooth)

        ir_mask = None
        if np.isfinite(ir_sub).any():
            ir_smooth = gaussian_filter(ir_sub, sigma=4.0)
            ir_mask = np.ma.masked_where(ir_smooth > -40, ir_smooth)

        plt.close("all")
        plt.rcParams["hatch.color"] = "#b7d6ff"
        plt.rcParams["hatch.linewidth"] = 0.7
        plt.rcParams["contour.negative_linestyle"] = "solid"

        fig = plt.figure(figsize=(14, 10))
        ax = plt.axes(projection=ccrs.PlateCarree())

        ax.set_extent(cfg["extent"], crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.LAND, facecolor="white", zorder=0)

        if ir_mask is not None:
            ax.contourf(
                lon_sub,
                lat_sub,
                ir_mask,
                levels=[-130, -40],
                colors=["#d0d0d0"],
                alpha=0.35,
                transform=ccrs.PlateCarree(),
                zorder=2
            )

        ax.contourf(
            lon_sub,
            lat_sub,
            theta_cp_mask,
            levels=[-30, -2],
            colors="none",
            hatches=["///"],
            transform=ccrs.PlateCarree(),
            zorder=3
        )

        ax.contour(
            lon_sub,
            lat_sub,
            theta_prime_smooth,
            levels=[-2],
            colors="#b7d6ff",
            linewidths=1.2,
            transform=ccrs.PlateCarree(),
            zorder=4
        )

        pm = ax.contourf(
            lon_sub,
            lat_sub,
            refl_plot,
            levels=bounds,
            cmap=cmap,
            norm=norm,
            extend="neither",
            transform=ccrs.PlateCarree(),
            zorder=5
        )

        if np.isfinite(uh_combined).any():
            ax.contourf(
                lon_sub,
                lat_sub,
                uh_combined,
                levels=[0.5, 1.5],
                colors=["#8f8f8f"],
                alpha=0.55,
                transform=ccrs.PlateCarree(),
                zorder=8
            )

        if np.nanmax(uh25_plot) >= 75:
            ax.contour(
                lon_sub,
                lat_sub,
                uh25_plot,
                levels=[75],
                colors="#4a4a4a",
                linewidths=0.9,
                transform=ccrs.PlateCarree(),
                zorder=9
            )

        if np.nanmax(uh03_plot) >= 50:
            ax.contour(
                lon_sub,
                lat_sub,
                uh03_plot,
                levels=[50],
                colors="black",
                linewidths=0.9,
                transform=ccrs.PlateCarree(),
                zorder=10
            )

        if PLOT_SR_WIND_BARBS:
            barb_skip = cfg["barb_skip"]

            ax.barbs(
                lon_sub[::barb_skip, ::barb_skip],
                lat_sub[::barb_skip, ::barb_skip],
                ms_to_kt(sr_u46_sub[::barb_skip, ::barb_skip]),
                ms_to_kt(sr_v46_sub[::barb_skip, ::barb_skip]),
                length=5,
                linewidth=0.7,
                color="black",
                transform=ccrs.PlateCarree(),
                zorder=23
            )

        add_shapefile_outline(ax, STATE_SHP, edgecolor="black", linewidth=1.4, zorder=13)
        add_shapefile_outline(ax, COUNTY_SHP, edgecolor="lightgray", linewidth=0.35, zorder=12)

        if lbf_geom is not None:
            add_counties_clipped_to_cwa(ax, COUNTY_SHP, lbf_geom, lw=1.0, color="black", zorder=13)

            ax.add_geometries(
                [lbf_geom],
                crs=ccrs.PlateCarree(),
                facecolor="none",
                edgecolor="black",
                linewidth=3.5,
                zorder=14
            )

            ax.add_geometries(
                [lbf_geom],
                crs=ccrs.PlateCarree(),
                facecolor="none",
                edgecolor="white",
                linewidth=1.8,
                zorder=15
            )

        valid_dt = init_dt + timedelta(hours=fhr)
        valid_title = f"F{fhr:03d} Valid: {valid_dt:%a %Y-%m-%d %HZ}"
        init_title = f"Init: {init_dt:%a %Y-%m-%d %HZ} {REFS_LABEL}"

        main_title = (
            f"{REFS_LABEL} | Refl, 2-5km UH > 75, "
            "0-3km UH > 50, θ Cold Pools, 4-6 km SR Winds"
        )

        ax.text(
            0.0, 1.042,
            main_title,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=cfg["title_size"],
            fontweight="bold"
        )

        ax.text(
            0.0, 1.005,
            valid_title,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=cfg["subtitle_size"],
            fontweight="bold"
        )

        ax.text(
            1.0, 1.005,
            init_title,
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=cfg["subtitle_size"],
            fontweight="bold"
        )

        divider = make_axes_locatable(ax)

        cax = divider.append_axes(
            "bottom",
            size="3%",
            pad=0.25,
            axes_class=plt.Axes
        )

        cbar = plt.colorbar(
            pm,
            cax=cax,
            orientation="horizontal",
            ticks=REF_LEVELS,
            drawedges=True
        )

        cbar.set_label("Reflectivity (dBZ)", fontsize=10, weight="bold")
        cbar.ax.xaxis.set_label_position("top")
        cbar.ax.tick_params(axis="x", which="both", length=0)

        if os.path.exists(LOGO_PATH):
            logo = mpimg.imread(LOGO_PATH)

            logo_ax = ax.inset_axes(
                [0.82, 0.84, 0.165, 0.155],
                transform=ax.transAxes,
                zorder=50
            )

            logo_ax.imshow(logo)
            logo_ax.axis("off")

        ax.text(
            0.902,
            0.835,
            "NWS North Platte, NE",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=10,
            fontweight="bold",
            color="black",
            zorder=51,
            path_effects=[pe.withStroke(linewidth=2.5, foreground="white")]
        )

        ax.text(
            0.01,
            0.015,
            "Plot created by: Matthew Labenz",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=8,
            weight="bold",
            color="black",
            zorder=40,
            path_effects=[pe.withStroke(linewidth=2.5, foreground="white")]
        )

        outname = os.path.join(domain_outdir, f"refs_mem01_lbf_f{fhr:03d}.png")

        plt.savefig(outname, dpi=140, bbox_inches="tight")
        plt.close(fig)

        print("Saved:", outname)

    except Exception as e:
        print(f"Failed {domain_key.upper()} F{fhr:03d}: {e}")


# ============================================================
# MAIN LOOP
# ============================================================

for fhr in fhrs:
    try:
        fields = load_rrfs_fields_once(fhr)

        for domain_key, cfg in DOMAINS.items():
            plot_domain_from_fields(fields, domain_key, cfg, fhr)

    except Exception as e:
        print(f"FAILED F{fhr:03d}: {e}")

print("Done. Images saved to:", OUTDIR)
