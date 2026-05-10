# ============================================================
# RRFS | Maximum Surface Hail Swath | Site Version
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

DATA_DIR = os.path.join(BASE_DIR, "rrfs_hail_subsets")

OUTDIR_BASE = os.path.join(
    "site",
    "runs",
    "rrfs",
    "hail_swath"
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
    },

    "regional": {
        "label": "Default",
        "extent": [-107.5, -93.0, 38.5, 44.2],
        "title_size": 13,
        "subtitle_size": 11,
    },

    "central_plains": {
        "label": "Central Plains",
        "extent": [-107.5, -91.0, 34.5, 45.2],
        "title_size": 13,
        "subtitle_size": 11,
    },
}


# ============================================================
# SETTINGS
# ============================================================

VALID_RRFS_CYCLES = list(range(24))

START_FHR = 1
CYCLE_DELAY_MINUTES = 45


# ============================================================
# HAIL COLOR TABLE
# ============================================================

hail_bounds = [
    0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40,
    0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80,
    0.85, 0.90, 0.95, 1.0, 1.05, 1.10, 1.15, 1.20,
    1.25, 1.30, 1.35, 1.40, 1.45, 1.50, 1.55, 1.60, 1.65,
    1.70, 1.75, 1.80, 1.85, 1.90, 1.95, 2.00, 2.10,
    2.20, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0,
    3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4.0
]

hail_colors = [
    "#ffffff", "#f0f0f0", "#e1e1e1", "#d2d2d2", "#c3c3c3",
    "#a5a5a5", "#969696", "#878787", "#787878", "#696969",
    "#3b5269", "#475f74", "#546c7f", "#60798a", "#6d8695",
    "#7993a1", "#86a0ac", "#92adb7", "#9fbac2", "#abc7ce",
    "#e6de99", "#e4d289", "#e3c679", "#e1b96a", "#dfae5a",
    "#dfa24b", "#dd963c", "#dc8a2f", "#da7e24", "#d9731c",
    "#d3491f", "#cb4323", "#c23d27", "#b9362b", "#b13131",
    "#a82b37", "#9f253d", "#971f44", "#8e1a4a", "#861550",
    "#700e89", "#7b1c93", "#872b9e", "#923aa8", "#9e4ab2",
    "#a95bbd", "#b56ac7", "#c07ad1", "#cc8adc", "#d79ae6",
    "#e6bfc3", "#dfb1b7", "#d9a4ad", "#d297a1", "#cc8a95",
    "#c57c8a", "#be707e", "#b86272", "#b25667", "#ac485b"
]

hail_cmap = ListedColormap(hail_colors, name="hail_bins")
hail_norm = BoundaryNorm(hail_bounds, hail_cmap.N, clip=True)


# ============================================================
# BASIC HELPERS
# ============================================================

def to_lon180(lon):
    return ((np.asarray(lon) + 180) % 360) - 180


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

    return lat, lon


def ensure_2d_field(da, label):
    arr = np.asarray(da.values, dtype=float)
    arr = np.squeeze(arr)

    if arr.ndim != 2:
        raise RuntimeError(
            f"{label} is not 2D after squeeze. "
            f"Shape={arr.shape}"
        )

    return arr


# ============================================================
# RRFS URLS
# ============================================================

def rrfs_public_grib_url(init_dt, fhr, product="2dfld"):
    ymd = init_dt.strftime("%Y%m%d")
    hh = init_dt.strftime("%H")

    if product != "2dfld":
        raise ValueError("For hail swaths, product should be '2dfld'")

    fname = f"rrfs.t{hh}z.2dfld.3km.f{fhr:03d}.conus.grib2"

    return (
        f"https://noaa-rrfs-pds.s3.amazonaws.com/"
        f"rrfs_a/rrfs.{ymd}/{hh}/{fname}"
    )


def url_exists(url, timeout=10):
    try:
        r = requests.head(url, timeout=timeout, allow_redirects=True)
        return r.status_code == 200
    except Exception:
        return False


def find_latest_available_rrfs_cycle(max_back_hours=96):
    now = datetime.now(timezone.utc) - timedelta(minutes=CYCLE_DELAY_MINUTES)

    for back in range(max_back_hours + 1):
        dt = now - timedelta(hours=back)

        if dt.hour not in VALID_RRFS_CYCLES:
            continue

        dt = dt.replace(minute=0, second=0, microsecond=0, tzinfo=None)

        test_url = rrfs_public_grib_url(dt, 1, product="2dfld") + ".idx"

        if url_exists(test_url):
            print(f"Latest RRFS cycle found: {dt:%Y%m%d} {dt:%HZ}")
            return dt

    raise RuntimeError("Could not find recent RRFS cycle.")


# ============================================================
# IDX HELPERS
# ============================================================

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

        parsed.append({
            "line": line,
            "msg_num": int(parts[0]),
            "start": int(parts[1]),
        })

    for j in range(len(parsed)):
        if j < len(parsed) - 1:
            parsed[j]["end"] = parsed[j + 1]["start"] - 1
        else:
            parsed[j]["end"] = None

    return parsed


def find_idx_match(parsed, all_terms, label):
    all_terms_lower = [t.lower() for t in all_terms]

    for item in parsed:
        line_lower = item["line"].lower()

        if all(term in line_lower for term in all_terms_lower):
            print(f"Matched {label}:")
            print(item["line"])
            return item

    raise RuntimeError(f"Could not find {label}")


def download_byte_range(grib_url, start, end, outpath):
    if os.path.exists(outpath):
        return outpath

    headers = {}

    if end is None:
        headers["Range"] = f"bytes={start}-"
    else:
        headers["Range"] = f"bytes={start}-{end}"

    r = requests.get(grib_url, headers=headers, stream=True, timeout=120)
    r.raise_for_status()

    with open(outpath, "wb") as f:
        for chunk in r.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)

    return outpath


def open_subset_grib(path):
    ds = xr.open_dataset(
        path,
        engine="cfgrib",
        backend_kwargs={"indexpath": ""}
    )

    var = list(ds.data_vars)[0]

    return ds[var]


def rrfs_idx_field(init_dt, fhr, term_sets, label, product="2dfld"):
    grib_url = rrfs_public_grib_url(init_dt, fhr, product=product)
    idx_url = grib_url + ".idx"

    lines = read_idx(idx_url)
    parsed = parse_idx_lines(lines)

    for terms in term_sets:
        try:
            match = find_idx_match(parsed, terms, label)

            outname = (
                f"rrfs_{init_dt:%Y%m%d_%H}_f{fhr:03d}_{label}.grib2"
            )

            outpath = os.path.join(DATA_DIR, outname)

            download_byte_range(
                grib_url,
                match["start"],
                match["end"],
                outpath
            )

            return open_subset_grib(outpath)

        except Exception:
            pass

    raise RuntimeError(f"Could not open {label}")


# ============================================================
# FIND RRFS CYCLE
# ============================================================

init_dt = find_latest_available_rrfs_cycle()

cycle_str = init_dt.strftime("%Y%m%d_%Hz")

if init_dt.hour in [0, 6, 12, 18]:
    MAX_FHR = 60
else:
    MAX_FHR = 18

OUTDIR = os.path.join(OUTDIR_BASE, cycle_str)
os.makedirs(OUTDIR, exist_ok=True)

fhrs = range(START_FHR, MAX_FHR + 1)

print("Using RRFS init:", init_dt.strftime("%Y-%m-%d %HZ"))


# ============================================================
# LOAD HAIL FIELD
# ============================================================

def load_rrfs_hail_once(fhr):

    hail_da = rrfs_idx_field(
        init_dt,
        fhr,
        [
            ["HAIL", "surface"],
            ["HAIL"],
        ],
        "surface_hail",
        product="2dfld"
    )

    lat, lon = get_lat_lon(hail_da)

    hail = ensure_2d_field(hail_da, "surface hail")

    if np.nanmax(hail) < 0.25:
        hail = hail * 39.3701

    hail = np.where(hail >= 0.05, hail, np.nan)

    return {
        "lat": lat,
        "lon": lon,
        "hail": hail,
    }


# ============================================================
# SPATIAL HELPERS
# ============================================================

def subset_2d(lat, lon, extent, field):

    lon_min, lon_max, lat_min, lat_max = extent

    mask = (
        (lon >= lon_min) &
        (lon <= lon_max) &
        (lat >= lat_min) &
        (lat <= lat_max)
    )

    iy, ix = np.where(mask)

    iy0 = max(iy.min() - 2, 0)
    iy1 = min(iy.max() + 3, lat.shape[0])

    ix0 = max(ix.min() - 2, 0)
    ix1 = min(ix.max() + 3, lon.shape[1])

    return (
        lat[iy0:iy1, ix0:ix1],
        lon[iy0:iy1, ix0:ix1],
        field[iy0:iy1, ix0:ix1]
    )


# ============================================================
# PLOT
# ============================================================

def plot_hail_domain(fields, domain_key, cfg, fhr):

    extent = cfg["extent"]

    lat_sub, lon_sub, hail_sub = subset_2d(
        fields["lat"],
        fields["lon"],
        extent,
        fields["hail"]
    )

    hail_plot = gaussian_filter(
        np.nan_to_num(hail_sub, nan=0.0),
        sigma=1.0
    )

    hail_plot = np.where(hail_plot >= 0.05, hail_plot, np.nan)

    fig = plt.figure(figsize=(14, 10))

    ax = plt.axes(projection=ccrs.PlateCarree())

    ax.set_extent(extent)

    ax.add_feature(cfeature.LAND, facecolor="white")

    pm = ax.contourf(
        lon_sub,
        lat_sub,
        hail_plot,
        levels=hail_bounds,
        cmap=hail_cmap,
        norm=hail_norm,
        extend="max",
        transform=ccrs.PlateCarree()
    )

    valid_dt = init_dt + timedelta(hours=fhr)

    ax.set_title(
        f"RRFS Maximum Surface Hail Swath\n"
        f"Init: {init_dt:%Y-%m-%d %HZ} | "
        f"Valid: {valid_dt:%Y-%m-%d %HZ}",
        fontsize=14,
        weight="bold"
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
        orientation="horizontal"
    )

    cbar.set_label("Surface Hail Swath (inches)")

    outdir = os.path.join(OUTDIR, domain_key)
    os.makedirs(outdir, exist_ok=True)

    outname = os.path.join(
        outdir,
        f"rrfs_hail_f{fhr:03d}.png"
    )

    plt.savefig(outname, dpi=140, bbox_inches="tight")
    plt.close()

    print("Saved:", outname)


# ============================================================
# RUN LOOP
# ============================================================

running_hail = None
base_lat = None
base_lon = None

for fhr in fhrs:

    try:

        print(f"\nProcessing F{fhr:03d}")

        fields = load_rrfs_hail_once(fhr)

        if running_hail is None:
            running_hail = fields["hail"].copy()
            base_lat = fields["lat"]
            base_lon = fields["lon"]
        else:
            running_hail = np.fmax(running_hail, fields["hail"])

        swath_fields = {
            "lat": base_lat,
            "lon": base_lon,
            "hail": running_hail,
        }

        for domain_key, cfg in DOMAINS.items():
            plot_hail_domain(swath_fields, domain_key, cfg, fhr)

    except Exception as e:
        print(f"FAILED F{fhr:03d}: {e}")

print("Done.")
