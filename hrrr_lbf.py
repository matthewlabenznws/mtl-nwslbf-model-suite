# -*- coding: utf-8 -*-

# LATEST HRRR RUN FOR THE LBF CWA USING HERBIE
# ============================================================
# Latest HRRR Loop | LBF Domain
# 1 km Reflectivity + UH + Sim IR + Theta Cold Pools
# + 4–6 km Storm-Relative Winds using 700–500 mb proxy
# ============================================================
import os
import glob
import requests
import numpy as np
import xarray as xr
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
import matplotlib.image as mpimg

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader

from scipy.ndimage import gaussian_filter
from scipy.interpolate import griddata
from mpl_toolkits.axes_grid1 import make_axes_locatable
from shapely.ops import unary_union
from shapely.prepared import prep
from datetime import datetime, timedelta
import geopandas as gpd

from matplotlib.colors import ListedColormap, BoundaryNorm
from herbie import Herbie


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

import zipfile

zip_path = os.path.join(BASE_DIR, "assets", "c_18mr25.zip")
extract_path = os.path.join(BASE_DIR, "assets")

if os.path.exists(zip_path):
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_path)

DOMAINS = {
    "lbf": {
        "label": "LBF",
        "extent": [-103.8, -97.0, 40.0, 43.4],
        "title_size": 14,
        "subtitle_size": 11,
        "logo_ax": [0.78, 0.70, 0.10, 0.10],
        "office_text_xy": [0.83, 0.71],
        "credit_xy": [0.13, 0.25],
        "barb_skip": 11,
    },

    "regional": {
        "label": "Default",
        "extent": [-107.5, -93.0, 38.5, 44.2],
        "title_size": 13,
        "subtitle_size": 11,
        "logo_ax": [0.78, 0.63, 0.10, 0.10],
        "office_text_xy": [0.83, 0.64],
        "credit_xy": [0.13, 0.31],
        "barb_skip": 20,
    },

    "central_plains": {
        "label": "Central Plains",
        "extent": [-107.5, -91.0, 34.5, 45.2],
        "title_size": 13,
        "subtitle_size": 11,
        "logo_ax": [0.78, 0.77, 0.10, 0.10],
        "office_text_xy": [0.83, 0.78],
        "credit_xy": [0.13, 0.175],
        "barb_skip": 24,
    },
}
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
            "logo_ax": [0.78, 0.70, 0.10, 0.10],
            "office_text_xy": [0.83, 0.71],
            "credit_xy": [0.13, 0.25],
            "barb_skip": 22,
        }

        print(f"Added SPC severe domain: {highest_label}")
        print(f"SPC severe extent: {extent}")

    except Exception as e:
        print(f"SPC severe domain skipped due to error: {e}")


add_spc_severe_domain()
COUNTY_SHP = os.path.join(BASE_DIR, "assets", "cb_2018_us_county_500k.shp")
STATE_SHP = os.path.join(BASE_DIR, "assets", "cb_2018_us_state_500k.shp")
LBF_CWA_SHP = os.path.join(BASE_DIR, "assets", "c_18mr25.shp")
#INTERSTATE_SHP = os.path.join(BASE_DIR, "assets", "tl_2023_us_primaryroads.shp")
LOGO_PATH = os.path.join(BASE_DIR, "assets", "NOAANWSLogos.png")

# Manual storm motion fallback
MANUAL_STORM_MOTION_FROM_DEG = 250
MANUAL_STORM_MOTION_SPEED_KT = 35

# Loop settings
MAX_FHR = 1
START_FHR = 0
PLOT_SR_WIND_BARBS = True
BARB_SKIP = 11

# Optional city labels
PLOT_CITY_LABELS = False

STATIONS = {
    "Gordon":       (-102.2038, 42.8061),
    "Ellsworth":    (-102.3172, 42.0628),
    "Oshkosh":      (-102.3465, 41.4047),
    "Ogallala":     (-101.7205, 41.1275),
    "Mullen":       (-101.0427, 42.0425),
    "Valentine":    (-100.5514, 42.8586),
    "Ainsworth":    (-99.8516, 42.5467),
    "Burwell":      (-99.1766, 41.7666),
    "North Platte": (-100.6689, 41.1220),
    "Broken Bow":   (-99.6385, 41.4365),
    "Imperial":     (-101.6243, 40.5106),
    "Curtis":       (-100.5219, 40.6344),
    "O'Neill":      (-98.6470, 42.4578),
    "Butte":        (-98.8511, 42.9130),
}


# ----------------------------
# COLOR TABLES
# ----------------------------
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


# ----------------------------
# HELPERS
# ----------------------------
def url_exists(url, timeout=12):
    try:
        r = requests.head(url, timeout=timeout, allow_redirects=True)
        return r.status_code == 200
    except Exception:
        return False


def find_latest_hrrr_cycle(max_back_hours=36):
    now = datetime.utcnow().replace(minute=0, second=0, microsecond=0)

    for back in range(max_back_hours + 1):
        dt = now - timedelta(hours=back)
        cycle_date = dt.strftime("%Y%m%d")
        cycle_hour = dt.hour

        test_url = (
            f"https://noaa-hrrr-bdp-pds.s3.amazonaws.com/"
            f"hrrr.{cycle_date}/conus/hrrr.t{cycle_hour:02d}z.wrfsfcf00.grib2"
        )

        if url_exists(test_url):
            print(f"Latest HRRR cycle found: {cycle_date} {cycle_hour:02d}Z")
            return cycle_date, cycle_hour

    raise RuntimeError("Could not find a recent HRRR cycle.")


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
    return lat, lon


def hrrr_field(cycle_date, cycle_hour, fhr, product, search, label):
    """
    Opens a subset HRRR field using Herbie.

    product:
      nat = wrfnat
      sfc = wrfsfc
      prs = wrfprs
    """
    init_dt = datetime.strptime(f"{cycle_date}{cycle_hour:02d}", "%Y%m%d%H")

    H = Herbie(
        init_dt,
        model="hrrr",
        product=product,
        fxx=fhr,
        priority=["aws", "google", "azure", "nomads"],
        verbose=False
    )

    ds = H.xarray(search, remove_grib=False)

    if isinstance(ds, list):
        ds = ds[0]

    if len(ds.data_vars) == 0:
        raise RuntimeError(f"Could not open {label} with Herbie search: {search}")

    var = list(ds.data_vars)[0]
    return ds[var].squeeze()


def subset_2d(lat, lon, *fields):
    mask = (
        np.isfinite(lat) & np.isfinite(lon) &
        (lon >= LON_MIN) & (lon <= LON_MAX) &
        (lat >= LAT_MIN) & (lat <= LAT_MAX)
    )

    if not np.any(mask):
        raise RuntimeError("No grid points found inside LBF domain.")

    iy, ix = np.where(mask)
    iy0 = max(iy.min() - 2, 0)
    iy1 = min(iy.max() + 3, lat.shape[0])
    ix0 = max(ix.min() - 2, 0)
    ix1 = min(ix.max() + 3, lat.shape[1])

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


def plot_city_labels(ax, cities, zorder=40, fontsize=9):
    for name, (lon, lat) in cities.items():
        ax.text(
            lon,
            lat,
            name,
            transform=ccrs.PlateCarree(),
            fontsize=fontsize,
            color="black",
            ha="center",
            va="center",
            zorder=zorder,
            path_effects=[pe.withStroke(linewidth=3, foreground="white")]
        )


# ----------------------------
# GET LATEST HRRR CYCLE
# ----------------------------
cycle_date, cycle_hour = find_latest_hrrr_cycle()
cycle_str = f"{cycle_date}_{cycle_hour:02d}z"

OUTDIR = os.path.join(
    "site",
    "runs",
    "hrrr",
    "refl_uh",
    cycle_str
)

os.makedirs(OUTDIR, exist_ok=True)
os.makedirs("site", exist_ok=True)
if cycle_hour in [0, 6, 12, 18]:
    fhrs = range(START_FHR, MAX_FHR + 1)
else:
    fhrs = range(START_FHR, min(MAX_FHR, 18) + 1)

lbf_geom = get_lbf_cwa_geom(LBF_CWA_SHP)

print("Forecast hours:", list(fhrs))


# ----------------------------
# MAIN LOOP - LOAD ONCE PER FORECAST HOUR
# ----------------------------
def load_hrrr_fields_once(fhr):
    print("\n" + "=" * 70)
    print(f"Loading HRRR once | HRRR {cycle_date} {cycle_hour:02d}Z F{fhr:03d}")
    print("=" * 70)

    refl_da = hrrr_field(cycle_date, cycle_hour, fhr, "nat", ":REFD:1000 m", "1 km reflectivity")
    lat, lon = get_lat_lon(refl_da)

    refl = np.asarray(refl_da.values, dtype=float)
    refl = np.where(refl >= REF_LEVELS[0], refl, np.nan)

    uh25_da = hrrr_field(cycle_date, cycle_hour, fhr, "sfc", ":MXUPHL:5000-2000 m", "2–5 km UH")
    uh03_da = hrrr_field(cycle_date, cycle_hour, fhr, "sfc", ":MXUPHL:3000-0 m", "0–3 km UH")

    uh25 = np.asarray(uh25_da.values, dtype=float)
    uh03 = np.asarray(uh03_da.values, dtype=float)

    ir_da = hrrr_field(cycle_date, cycle_hour, fhr, "sfc", ":SBT123:", "simulated IR brightness temperature")
    ir_c = k_to_c(ir_da.values)

    t2_da = hrrr_field(cycle_date, cycle_hour, fhr, "sfc", ":TMP:2 m", "2m temperature")
    ps_da = hrrr_field(cycle_date, cycle_hour, fhr, "sfc", ":PRES:surface", "surface pressure")

    t2_k = np.asarray(t2_da.values, dtype=float)
    ps_pa = np.asarray(ps_da.values, dtype=float)

    theta = t2_k * (100000.0 / ps_pa) ** 0.286
    theta_bg = gaussian_filter(theta, sigma=18)
    theta_prime = theta - theta_bg

    u700_da = hrrr_field(cycle_date, cycle_hour, fhr, "prs", ":UGRD:700 mb", "700 mb u wind")
    v700_da = hrrr_field(cycle_date, cycle_hour, fhr, "prs", ":VGRD:700 mb", "700 mb v wind")
    u600_da = hrrr_field(cycle_date, cycle_hour, fhr, "prs", ":UGRD:600 mb", "600 mb u wind")
    v600_da = hrrr_field(cycle_date, cycle_hour, fhr, "prs", ":VGRD:600 mb", "600 mb v wind")
    u500_da = hrrr_field(cycle_date, cycle_hour, fhr, "prs", ":UGRD:500 mb", "500 mb u wind")
    v500_da = hrrr_field(cycle_date, cycle_hour, fhr, "prs", ":VGRD:500 mb", "500 mb v wind")

    pr_lat, pr_lon = get_lat_lon(u700_da)

    u46_pr = np.nanmean(np.stack([
        np.asarray(u700_da.values, dtype=float),
        np.asarray(u600_da.values, dtype=float),
        np.asarray(u500_da.values, dtype=float)
    ]), axis=0)

    v46_pr = np.nanmean(np.stack([
        np.asarray(v700_da.values, dtype=float),
        np.asarray(v600_da.values, dtype=float),
        np.asarray(v500_da.values, dtype=float)
    ]), axis=0)

    try:
        storm_u_da = hrrr_field(cycle_date, cycle_hour, fhr, "sfc", ":USTM:", "storm motion u")
        storm_v_da = hrrr_field(cycle_date, cycle_hour, fhr, "sfc", ":VSTM:", "storm motion v")

        storm_u_native = np.asarray(storm_u_da.values, dtype=float)
        storm_v_native = np.asarray(storm_v_da.values, dtype=float)

    except Exception:
        storm_u_scalar, storm_v_scalar = wind_from_dir_speed_to_uv(
            MANUAL_STORM_MOTION_FROM_DEG,
            kt_to_ms(MANUAL_STORM_MOTION_SPEED_KT)
        )
        storm_u_native = np.full_like(refl, storm_u_scalar)
        storm_v_native = np.full_like(refl, storm_v_scalar)

    u46_native = interp_to_target_grid(pr_lat, pr_lon, u46_pr, lat, lon)
    v46_native = interp_to_target_grid(pr_lat, pr_lon, v46_pr, lat, lon)

    sr_u46 = u46_native - storm_u_native
    sr_v46 = v46_native - storm_v_native
    sr46_kt = ms_to_kt(np.sqrt(sr_u46**2 + sr_v46**2))

    return {
        "lat": lat,
        "lon": lon,
        "refl": refl,
        "uh25": uh25,
        "uh03": uh03,
        "ir_c": ir_c,
        "theta_prime": theta_prime,
        "sr46_kt": sr46_kt,
        "sr_u46": sr_u46,
        "sr_v46": sr_v46,
    }


def plot_domain_from_fields(fields, domain_key, cfg, fhr):
    global LON_MIN, LON_MAX, LAT_MIN, LAT_MAX

    LON_MIN, LON_MAX, LAT_MIN, LAT_MAX = cfg["extent"]

    domain_outdir = os.path.join(
    OUTDIR,
    domain_key)
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
        sr46_kt = fields["sr46_kt"]
        sr_u46 = fields["sr_u46"]
        sr_v46 = fields["sr_v46"]

        lat_sub, lon_sub, [
            refl_sub, uh25_sub, uh03_sub, ir_sub,
            theta_prime_sub, sr46_sub, sr_u46_sub, sr_v46_sub
        ] = subset_2d(
            lat, lon, refl, uh25, uh03, ir_c,
            theta_prime, sr46_kt, sr_u46, sr_v46
        )

        refl_plot = gaussian_filter(np.nan_to_num(refl_sub, nan=0.0), sigma=0.5)
        refl_plot = np.where(refl_plot >= 5, refl_plot, np.nan)

        uh25_plot = gaussian_filter(uh25_sub, sigma=0.2)
        uh03_plot = gaussian_filter(uh03_sub, sigma=0.2)

        uh_combined = np.where((uh25_plot >= 75) | (uh03_plot >= 50), 1, np.nan)

        theta_prime_smooth = gaussian_filter(theta_prime_sub, sigma=2.5)
        theta_cp_mask = np.ma.masked_where(theta_prime_smooth > -2.0, theta_prime_smooth)

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

        ax.contourf(
            lon_sub, lat_sub, ir_mask,
            levels=[-130, -40],
            colors=["#d0d0d0"],
            alpha=0.35,
            transform=ccrs.PlateCarree(),
            zorder=2
        )

        ax.contourf(
            lon_sub, lat_sub, theta_cp_mask,
            levels=[-30, -2],
            colors="none",
            hatches=["///"],
            transform=ccrs.PlateCarree(),
            zorder=3
        )

        ax.contour(
            lon_sub, lat_sub, theta_prime_smooth,
            levels=[-2],
            colors="#b7d6ff",
            linewidths=1.2,
            transform=ccrs.PlateCarree(),
            zorder=4
        )

        pm = ax.contourf(
            lon_sub, lat_sub, refl_plot,
            levels=bounds,
            cmap=cmap,
            norm=norm,
            extend="neither",
            transform=ccrs.PlateCarree(),
            zorder=5
        )

        ax.contourf(
            lon_sub, lat_sub, uh_combined,
            levels=[0.5, 1.5],
            colors=["#8f8f8f"],
            alpha=0.55,
            transform=ccrs.PlateCarree(),
            zorder=8
        )

        ax.contour(
            lon_sub, lat_sub, uh25_plot,
            levels=[75],
            colors="#4a4a4a",
            linewidths=0.9,
            transform=ccrs.PlateCarree(),
            zorder=9
        )

        ax.contour(
            lon_sub, lat_sub, uh03_plot,
            levels=[50],
            colors="black",
            linewidths=0.9,
            transform=ccrs.PlateCarree(),
            zorder=10
        )

        if PLOT_SR_WIND_BARBS:
            barb_skip = cfg.get("barb_skip", BARB_SKIP)

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

        if PLOT_CITY_LABELS:
            plot_city_labels(ax, STATIONS, zorder=40, fontsize=9)

        init_dt = datetime.strptime(f"{cycle_date}{cycle_hour:02d}", "%Y%m%d%H")
        valid_dt = init_dt + timedelta(hours=fhr)

        main_title = (
       "HRRR | 1 km Refl, 2-5km UH > 75, "
       "0-3km UH > 50, Sim. IR, θ Cold Pools, 4-6 km SR Winds"
       )
        valid_title = f"F{fhr:03d} Valid: {valid_dt:%a %Y-%m-%d %Hz}"
        init_title = f"Init: {init_dt:%a %Y-%m-%d %Hz} HRRR"

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
        cax = divider.append_axes("bottom", size="3%", pad=0.25, axes_class=plt.Axes)

        cbar = plt.colorbar(
            pm,
            cax=cax,
            orientation="horizontal",
            ticks=REF_LEVELS,
            drawedges=True
        )

        cbar.set_label("1 km Reflectivity (dBZ)", fontsize=10, weight="bold")
        cbar.ax.xaxis.set_label_position("top")
        cbar.ax.tick_params(axis="x", which="both", length=0)

        # ============================================================
# NOAA/NWS LOGO
# ============================================================
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
          fontsize=9,
          weight="bold",
          color="black",
          zorder=40,
          path_effects=[pe.withStroke(linewidth=2.5, foreground="white")]
          )

        outname = os.path.join(domain_outdir, f"hrrr_lbf_f{fhr:03d}.png")

        plt.savefig(outname, dpi=140, bbox_inches="tight")
        plt.close(fig)

        print("Saved:", outname)

    except Exception as e:
        print(f"Failed {domain_key.upper()} F{fhr:03d}: {e}")


for fhr in fhrs:
    fields = load_hrrr_fields_once(fhr)

    for domain_key, cfg in DOMAINS.items():
        plot_domain_from_fields(fields, domain_key, cfg, fhr)
    








