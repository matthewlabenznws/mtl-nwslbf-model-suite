# ============================================================
# Google Colab | Latest HRRR Surface Hail Swath
# LBF / Regional / Central Plains Domains
# ============================================================
import os
import zipfile
import requests
import numpy as np
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
from datetime import datetime, timedelta

from matplotlib.colors import ListedColormap, BoundaryNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from herbie import Herbie


# ============================================================
# USER SETTINGS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ASSET_DIR = os.path.join(BASE_DIR, "assets")

OUTDIR = os.path.join(
    "site",
    "runs",
    "hrrr",
    "hail_swath"
)

os.makedirs(OUTDIR, exist_ok=True)

COUNTY_SHP = os.path.join(ASSET_DIR, "cb_2018_us_county_500k.shp")
STATE_SHP = os.path.join(ASSET_DIR, "cb_2018_us_state_500k.shp")
LBF_CWA_SHP = os.path.join(ASSET_DIR, "c_18mr25.shp")
LOGO_PATH = os.path.join(ASSET_DIR, "NOAANWSLogos.png")

# Optional: unzip CWA shapefile if uploaded as c_18mr25.zip
zip_path = os.path.join(ASSET_DIR, "c_18mr25.zip")
if os.path.exists(zip_path):
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(ASSET_DIR)

MAX_FHR = 48
START_FHR = 0

PLOT_CITY_LABELS = False


DOMAINS = {
    "lbf": {
        "label": "LBF",
        "extent": [-103.8, -97.0, 40.0, 43.4],
        "title_size": 14,
        "subtitle_size": 11,
        "logo_ax": [0.78, 0.70, 0.10, 0.10],
        "office_text_xy": [0.83, 0.71],
        "credit_xy": [0.13, 0.25],
    },

    "regional": {
        "label": "Default",
        "extent": [-107.5, -93.0, 38.5, 44.2],
        "title_size": 13,
        "subtitle_size": 11,
        "logo_ax": [0.78, 0.63, 0.10, 0.10],
        "office_text_xy": [0.83, 0.64],
        "credit_xy": [0.13, 0.31],
    },

    "central_plains": {
        "label": "Central Plains",
        "extent": [-107.5, -91.0, 34.5, 45.2],
        "title_size": 13,
        "subtitle_size": 11,
        "logo_ax": [0.78, 0.77, 0.10, 0.10],
        "office_text_xy": [0.83, 0.78],
        "credit_xy": [0.13, 0.175],
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


# ============================================================
# HAIL COLORMAP
# ============================================================

hail_bounds = [
    0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40,
    0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80,
    0.85, 0.90, 0.95, 1.0, 1.05, 1.10, 1.15, 1.20,
    1.25, 1.30, 1.35, 1.40, 1.45, 1.50, 1.55, 1.60, 1.65,
    1.70, 1.75, 1.80, 1.85, 1.90, 1.95, 2.00, 2.10,
    2.20, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0,
    3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4.0]

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
# HELPERS
# ============================================================

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

        # Safer to require f01 instead of only f00
        test_url = (
            f"https://noaa-hrrr-bdp-pds.s3.amazonaws.com/"
            f"hrrr.{cycle_date}/conus/hrrr.t{cycle_hour:02d}z.wrfsfcf01.grib2"
        )

        if url_exists(test_url):
            print(f"Latest HRRR cycle found: {cycle_date} {cycle_hour:02d}Z")
            return cycle_date, cycle_hour

    raise RuntimeError("Could not find a recent HRRR cycle.")


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

    return lat, lon


def hrrr_field(cycle_date, cycle_hour, fhr, product, search, label):
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

    print(f"Opened {label}: var={var}, units={ds[var].attrs.get('GRIB_units', ds[var].attrs.get('units', 'unknown'))}")

    return ds[var].squeeze()


def subset_2d(lat, lon, extent, *fields):
    lon_min, lon_max, lat_min, lat_max = extent

    mask = (
        np.isfinite(lat) & np.isfinite(lon) &
        (lon >= lon_min) & (lon <= lon_max) &
        (lat >= lat_min) & (lat <= lat_max)
    )

    if not np.any(mask):
        raise RuntimeError("No grid points found inside domain.")

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


def add_shapefile_outline(ax, shp_path, extent, edgecolor="k", linewidth=1.2, zorder=6):
    if not os.path.exists(shp_path):
        print("Missing shapefile:", shp_path)
        return

    lon_min, lon_max, lat_min, lat_max = extent

    gdf = gpd.read_file(shp_path).to_crs(epsg=4326)
    gdf = gdf.cx[lon_min - 1:lon_max + 1, lat_min - 1:lat_max + 1]

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


def plot_city_labels(ax, cities, fontsize=9):
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
            zorder=40,
            path_effects=[pe.withStroke(linewidth=3, foreground="white")]
        )


# ============================================================
# GET LATEST HRRR CYCLE
# ============================================================

cycle_date, cycle_hour = find_latest_hrrr_cycle()
cycle_str = f"{cycle_date}_{cycle_hour:02d}z"

if cycle_hour in [0, 6, 12, 18]:
    fhrs = range(START_FHR, MAX_FHR + 1)
else:
    fhrs = range(START_FHR, min(MAX_FHR, 18) + 1)

print("Forecast hours:", list(fhrs))

lbf_geom = get_lbf_cwa_geom(LBF_CWA_SHP)


# ============================================================
# LOAD ONCE PER FORECAST HOUR
# ============================================================

def load_hail_once(fhr):
    print("\n" + "=" * 70)
    print(f"Loading HRRR hail swath | {cycle_date} {cycle_hour:02d}Z F{fhr:03d}")
    print("=" * 70)

    # Try common HRRR HAIL search strings.
    # If one fails, the next one is attempted.
    search_options = [
    ":HAIL:surface:",
    ":HAIL:surface",]

    last_err = None

    for search in search_options:
        try:
            hail_da = hrrr_field(
                cycle_date,
                cycle_hour,
                fhr,
                product="sfc",
                search=search,
                label=f"surface hail swath {search}"
            )

            lat, lon = get_lat_lon(hail_da)
            hail = np.asarray(hail_da.values, dtype=float)

            # Many HRRR hail fields are meters. If max is small, convert m -> inches.
            finite_max = np.nanmax(hail)

            print(f"Raw hail max: {finite_max:.4f}")

            if finite_max < 0.25:
                print("Assuming hail units are meters. Converting to inches.")
                hail = hail * 39.3701
            else:
                print("Assuming hail units are already inches or inch-like.")

            hail = np.where(hail >= 0.05, hail, np.nan)

            return {
                "lat": lat,
                "lon": lon,
                "hail": hail,
                "search": search,
            }

        except Exception as e:
            print(f"Failed search {search}: {e}")
            last_err = e

    raise RuntimeError(f"Could not open HRRR hail field. Last error: {last_err}")


# ============================================================
# PLOT DOMAIN
# ============================================================

def plot_hail_domain(fields, domain_key, cfg, fhr):
    extent = cfg["extent"]

    domain_outdir = os.path.join(OUTDIR, cycle_str, domain_key)
    os.makedirs(domain_outdir, exist_ok=True)

    lat = fields["lat"]
    lon = fields["lon"]
    hail = fields["hail"]

    lat_sub, lon_sub, [hail_sub] = subset_2d(
        lat,
        lon,
        extent,
        hail
    )

    hail_plot = gaussian_filter(
        np.nan_to_num(hail_sub, nan=0.0),
        sigma=1.0
    )

    hail_plot = np.where(hail_plot >= 0.05, hail_plot, np.nan)

    plt.close("all")

    fig = plt.figure(figsize=(14, 10))
    ax = plt.axes(projection=ccrs.PlateCarree())

    ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor="white", zorder=0)

    pm = ax.contourf(
        lon_sub,
        lat_sub,
        hail_plot,
        levels=hail_bounds,
        cmap=hail_cmap,
        norm=hail_norm,
        extend="max",
        transform=ccrs.PlateCarree(),
        zorder=5
    )

    # Map features
    add_shapefile_outline(ax, STATE_SHP, extent, edgecolor="black", linewidth=1.4, zorder=13)
    add_shapefile_outline(ax, COUNTY_SHP, extent, edgecolor="lightgray", linewidth=0.35, zorder=12)

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

    if PLOT_CITY_LABELS:
        plot_city_labels(ax, STATIONS, fontsize=9)

    init_dt = datetime.strptime(f"{cycle_date}{cycle_hour:02d}", "%Y%m%d%H")
    valid_dt = init_dt + timedelta(hours=fhr)
    main_title = "HRRR | Maximum Surface Hail Swath"
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

    # Colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("bottom", size="3%", pad=0.25, axes_class=plt.Axes)

    cbar = plt.colorbar(
        pm,
        cax=cax,
        orientation="horizontal",
        ticks=[0, 0.50, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
        drawedges=True
    )

    cbar.set_label("Surface Hail Swath (inches)", fontsize=10, weight="bold")
    cbar.ax.xaxis.set_label_position("top")
    cbar.ax.tick_params(axis="x", which="both", length=0)

    # Logo
    if os.path.exists(LOGO_PATH):
        logo = mpimg.imread(LOGO_PATH)
        logo_ax = fig.add_axes(cfg["logo_ax"], zorder=30)
        logo_ax.imshow(logo)
        logo_ax.axis("off")

    fig.text(
        cfg["office_text_xy"][0],
        cfg["office_text_xy"][1],
        "NWS North Platte, NE",
        ha="center",
        va="top",
        fontsize=10,
        fontweight="bold",
        color="black",
        zorder=31,
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

    outname = os.path.join(
        domain_outdir,
        f"hrrr_hail_f{fhr:03d}.png"
    )

    plt.savefig(outname, dpi=140, bbox_inches="tight")
    plt.show()
    plt.close(fig)

    print("Saved:", outname)


# ============================================================
# RUN
# ============================================================

running_hail = None
base_lat = None
base_lon = None
hail_search_used = None

for fhr in fhrs:
    fields = load_hail_once(fhr)

    if running_hail is None:
        running_hail = fields["hail"].copy()
        base_lat = fields["lat"]
        base_lon = fields["lon"]
        hail_search_used = fields.get("search", "")
    else:
        running_hail = np.fmax(running_hail, fields["hail"])

    swath_fields = {
        "lat": base_lat,
        "lon": base_lon,
        "hail": running_hail,
        "search": hail_search_used,
    }

    for domain_key, cfg in DOMAINS.items():
        plot_hail_domain(swath_fields, domain_key, cfg, fhr)

print("Done. Images saved to:", OUTDIR)
