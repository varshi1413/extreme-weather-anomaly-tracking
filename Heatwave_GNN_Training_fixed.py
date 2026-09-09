r"""
Heatwave_GNN_Training.py

WeatherGPT - Heatwave-specific spatio-temporal GNN training (GPU-optimized).

Pipeline:
    ERA5 NetCDF (Single-Level + Pressure-Level)
        -> daily atmospheric features
        -> pressure-level channels (e.g. t_850, u_850, v_850, z_850, r_850)
        -> 0.1 degree (~12 km) grid
        -> icosphere graph
        -> MeshGraphNet-style spatial message passing
        -> Temporal Transformer
        -> future Tmax regression + heatwave classification

Heatwave label:
    A future day is labelled as a heatwave when:
        future Tmax >= local training-period 95th percentile
        AND
        future Tmax - local training-period mean >= 2 degC

Important:
    This is a model-training pipeline. It does not by itself provide
    official IMD heatwave certification.

Recommended first test on CPU:
    python Heatwave_GNN_Training.py --era5 "C:\path\to\ERA5_INDIA" --epochs 2 --max-samples 30

Full training:
    python Heatwave_GNN_Training.py --era5 "C:\path\to\ERA5_INDIA" --epochs 50

Dependencies:
    numpy
    xarray
    torch
    scipy
    netCDF4 (or another xarray NetCDF backend)
"""

import argparse
import copy
import glob
import math
import os
import random
from dataclasses import dataclass, asdict

import numpy as np
import xarray as xr
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from scipy.ndimage import map_coordinates
from weather_utils import mesh_map


# ============================================================
# 1. REPRODUCIBILITY
# ============================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# 2. CONFIGURATION
# ============================================================

@dataclass
class Config:
    era5_path: str = "ERA5"

    # ERA5 data types
    use_single_level: bool = True
    use_pressure_level: bool = True
    pressure_levels: list = None

    # Temporal setup
    window: int = 5
    horizon: int = 5

    # Spatial setup
    target_resolution_deg: float = 0.1
    mesh_subdivisions: int = 3

    # Training
    epochs: int = 50
    batch_size: int = 8
    learning_rate: float = 2e-4
    weight_decay: float = 1e-4

    # Model
    hidden_dim: int = 128
    edge_dim: int = 64
    processor_steps: int = 6
    temporal_dim: int = 256
    temporal_heads: int = 8
    temporal_layers: int = 4
    dropout: float = 0.15

    # Runtime
    spatial_chunk_size: int = 5
    use_amp: bool = True
    fast_mesh_sampling: bool = True
    max_samples: int = 0
    num_workers: int = 0
    seed: int = 42

    # Output
    checkpoint_path: str = "heatwave_gnn.pt"

    def __post_init__(self):
        if self.pressure_levels is None:
            self.pressure_levels = DEFAULT_PRESSURE_LEVELS.copy()


# ============================================================
# 3. UTILITIES
# ============================================================

def resolve_nc_files(path_pattern):
    """Resolve a file, directory, or recursive glob into NetCDF files."""
    if os.path.isfile(path_pattern):
        return [path_pattern]

    if os.path.isdir(path_pattern):
        files = sorted(glob.glob(os.path.join(path_pattern, "*.nc")))
        if not files:
            files = sorted(glob.glob(os.path.join(path_pattern, "**", "*.nc"),
                                     recursive=True))
        return files

    files = sorted(glob.glob(path_pattern, recursive=True))
    return files


def normalize_longitudes(lon):
    """Convert longitude to [-180, 180)."""
    lon = np.asarray(lon, dtype=np.float64)
    return ((lon + 180.0) % 360.0) - 180.0


def find_coord(ds, candidates):
    for name in candidates:
        if name in ds.coords:
            return name
        if name in ds.variables:
            return name
    return None


def find_var(ds, candidates):
    for name in candidates:
        if name in ds.data_vars:
            return name
    return None


def pressure_to_hpa(x):
    x = np.asarray(x, dtype=np.float32)
    # ERA5 surface pressure / MSLP are commonly Pa.
    if np.nanmedian(x) > 2000.0:
        return x / 100.0
    return x


def precipitation_to_mm(x, attrs=None):
    """
    ERA5 total precipitation is normally stored in metres.
    Convert to mm/day for daily accumulated precipitation.
    """
    x = np.asarray(x, dtype=np.float32)

    units = ""
    if attrs:
        units = str(attrs.get("units", "")).lower()

    if "mm" in units:
        return x

    # ERA5 tp in metres: 1 m = 1000 mm.
    return x * 1000.0


# ============================================================
# 4. ICAOSPHERE MESH
# ============================================================

def create_icosphere(subdivisions=3):
    """
    Create a unit-sphere triangular mesh.

    subdivisions=3 -> 642 vertices
    subdivisions=4 -> 2562 vertices
    """
    t = (1.0 + math.sqrt(5.0)) / 2.0

    verts = [
        (-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0),
        (0, -1, t), (0, 1, t), (0, -1, -t), (0, 1, -t),
        (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1)
    ]

    faces = [
        (0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11),
        (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
        (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
        (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1)
    ]

    verts = [np.asarray(v, dtype=np.float64) for v in verts]

    def midpoint(i, j, cache):
        key = tuple(sorted((i, j)))
        if key in cache:
            return cache[key]
        v = verts[i] + verts[j]
        v = v / np.linalg.norm(v)
        verts.append(v)
        idx = len(verts) - 1
        cache[key] = idx
        return idx

    for _ in range(subdivisions):
        cache = {}
        new_faces = []

        for a, b, c in faces:
            ab = midpoint(a, b, cache)
            bc = midpoint(b, c, cache)
            ca = midpoint(c, a, cache)

            new_faces.extend([
                (a, ab, ca),
                (b, bc, ab),
                (c, ca, bc),
                (ab, bc, ca)
            ])

        faces = new_faces

    vertices = np.asarray(verts, dtype=np.float32)
    vertices /= np.linalg.norm(vertices, axis=1, keepdims=True)

    return vertices, np.asarray(faces, dtype=np.int64)


def faces_to_edges(faces):
    edges = set()

    for a, b, c in faces:
        for u, v in ((a, b), (b, c), (c, a)):
            edges.add((u, v))
            edges.add((v, u))

    edge_index = np.asarray(sorted(edges), dtype=np.int64).T
    return edge_index


def xyz_to_latlon(xyz):
    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]

    lat = np.degrees(np.arcsin(np.clip(z, -1, 1)))
    lon = np.degrees(np.arctan2(y, x))

    return lat.astype(np.float32), lon.astype(np.float32)


# ============================================================
# 5. ERA5 PREPROCESSING
# ============================================================

# ERA5 Single-Level parameters requested in Dataset.docx.
# The code uses common ERA5 CDS short names plus readable aliases.
SINGLE_LEVEL_CANDIDATES = {
    "t2m": ["t2m", "2m_temperature"],
    "d2m": ["d2m", "2m_dewpoint_temperature"],
    "u10": ["u10", "10m_u_component_of_wind"],
    "v10": ["v10", "10m_v_component_of_wind"],
    "msl": ["msl", "mean_sea_level_pressure"],
    "sp": ["sp", "surface_pressure"],
    "tp": ["tp", "total_precipitation"],
    "skt": ["skt", "skin_temperature"],
    "sst": ["sst", "sea_surface_temperature"],
    "tcc": ["tcc", "total_cloud_cover"],
    # ERA5 soil moisture is normally supplied as swvl1-swvl4.
    "swvl1": ["swvl1", "volumetric_soil_water_layer_1"],
    "swvl2": ["swvl2", "volumetric_soil_water_layer_2"],
    "swvl3": ["swvl3", "volumetric_soil_water_layer_3"],
    "swvl4": ["swvl4", "volumetric_soil_water_layer_4"],
    # ERA5 soil temperature is normally supplied as stl1-stl4.
    "stl1": ["stl1", "soil_temperature_level_1"],
    "stl2": ["stl2", "soil_temperature_level_2"],
    "stl3": ["stl3", "soil_temperature_level_3"],
    "stl4": ["stl4", "soil_temperature_level_4"],
    # Common ERA5 surface-radiation/energy variables.
    "ssrd": ["ssrd", "surface_solar_radiation_downwards"],
    "strd": ["strd", "surface_thermal_radiation_downwards"],
    "sshf": ["sshf", "surface_sensible_heat_flux"],
    "slhf": ["slhf", "surface_latent_heat_flux"],
    "blh": ["blh", "boundary_layer_height"],
    "tcwv": ["tcwv", "total_column_water_vapour"],
}

# ERA5 Pressure-Level parameters requested in Dataset.docx.
PRESSURE_LEVEL_CANDIDATES = {
    "t": ["t", "temperature"],
    "u": ["u", "u_component_of_wind"],
    "v": ["v", "v_component_of_wind"],
    "z": ["z", "geopotential"],
    "q": ["q", "specific_humidity"],
    "r": ["r", "relative_humidity"],
    "w": ["w", "vertical_velocity", "omega"],
}

# Requested ERA5 pressure levels from the document.
DEFAULT_PRESSURE_LEVELS = [1000, 925, 850, 700, 500, 300, 250, 200]


def find_pressure_dim(da):
    """Find the pressure-level dimension/coordinate used by an ERA5 file."""
    candidates = [
        "pressure_level",
        "level",
        "isobaricInhPa",
        "isobaricInPa",
    ]

    for name in candidates:
        if name in da.dims or name in da.coords:
            return name

    # Some files use a generic vertical coordinate with a standard_name.
    for name in da.dims:
        coord = da.coords.get(name)
        if coord is not None:
            standard_name = str(coord.attrs.get("standard_name", "")).lower()
            if "air_pressure" in standard_name or "pressure" in standard_name:
                return name

    return None


def normalize_pressure_coordinate(da, pressure_dim):
    """Convert an ERA5 pressure coordinate to hPa."""
    values = np.asarray(da[pressure_dim].values, dtype=np.float32)

    # If the coordinate is in Pa, convert it to hPa.
    if np.nanmedian(values) > 2000.0:
        values = values / 100.0

    return da.assign_coords({pressure_dim: values})


def find_first_var(ds, candidates):
    """Return the first matching variable name from an ERA5 candidate list."""
    return find_var(ds, candidates)


def _extract_year_month(path):
    """Extract YYYY-MM from either filename or parent directories."""
    import re
    text = os.path.normpath(path)
    patterns = [
        r"(?<!\d)((?:19|20)\d{2})[_-](0[1-9]|1[0-2])(?!\d)",
        r"[\\/]((?:19|20)\d{2})[\\/](0[1-9]|1[0-2])[\\/]",
    ]
    for pattern in patterns:
        m = re.search(pattern, text)
        if m:
            return f"{m.group(1)}-{m.group(2)}"
    return None


def _group_era5_files_by_month(files):
    """Group READY-layout ERA5 files by YYYY-MM and remove exact duplicates."""
    groups = {}
    for f in sorted(set(os.path.abspath(x) for x in files)):
        ym = _extract_year_month(f)
        if ym is not None:
            groups.setdefault(ym, []).append(f)

    # Keep the preferred copy when ERA5_READY contains both a nested monthly
    # copy and a top-level monthly pressure file. Nested files are preferred
    # because they belong to the prepared YEAR/MONTH layout.
    for ym, group in groups.items():
        pressure = []
        instant = []
        accum = []
        other = []
        for f in group:
            name = os.path.basename(f).lower()
            if name.startswith("era5_india_pressure_") or "pressure" in name:
                pressure.append(f)
            elif "steptype-instant" in name:
                instant.append(f)
            elif "steptype-accum" in name:
                accum.append(f)
            else:
                other.append(f)

        def prefer(items):
            if not items:
                return []
            nested = [x for x in items if os.sep + ym.replace("-", os.sep) + os.sep in x]
            return [sorted(nested or items)[0]]

        groups[ym] = prefer(pressure) + prefer(instant) + prefer(accum) + prefer(other)

    return dict(sorted(groups.items()))


def _load_month_group(month, files, cfg):
    """Load one month, merge its ERA5 sources, and return daily fields."""
    parts = []
    skipped = 0

    print(f"\n--- Processing {month} ({len(files)} source file(s)) ---", flush=True)

    for f in files:
        ds = None
        try:
            print(f"  Loading {os.path.basename(f)} ...", flush=True)
            ds = xr.open_dataset(f, cache=False)

            time_name = find_coord(ds, ["time", "valid_time", "date"])
            lat_name = find_coord(ds, ["latitude", "lat"])
            lon_name = find_coord(ds, ["longitude", "lon"])
            if time_name is None or lat_name is None or lon_name is None:
                raise ValueError("time/latitude/longitude coordinate not found")

            rename = {}
            if time_name != "time":
                rename[time_name] = "time"
            if lat_name != "latitude":
                rename[lat_name] = "latitude"
            if lon_name != "longitude":
                rename[lon_name] = "longitude"
            if rename:
                ds = ds.rename(rename)

            selected = {}

            if cfg.use_single_level:
                for key, candidates in SINGLE_LEVEL_CANDIDATES.items():
                    var = find_first_var(ds, candidates)
                    if var is not None:
                        selected[key] = ds[var]

            if cfg.use_pressure_level:
                for key, candidates in PRESSURE_LEVEL_CANDIDATES.items():
                    var = find_first_var(ds, candidates)
                    if var is not None:
                        da = ds[var]
                        p_dim = find_pressure_dim(da)
                        if p_dim is not None:
                            da = normalize_pressure_coordinate(da, p_dim)
                            selected[key] = da

            if not selected:
                raise ValueError("no requested variables found")

            # Load only this source file. The handle is closed immediately.
            part = xr.Dataset(selected).load()
            ds.close()
            ds = None

            lon = normalize_longitudes(part["longitude"].values)
            part = part.assign_coords(longitude=lon).sortby("longitude")
            if part["latitude"].values[0] > part["latitude"].values[-1]:
                part = part.sortby("latitude")

            parts.append(part)
            print(f"    [OK] {len(selected)} variable group(s)", flush=True)

        except Exception as exc:
            skipped += 1
            print(f"    [SKIP] {os.path.basename(f)}: {exc}", flush=True)
        finally:
            if ds is not None:
                try:
                    ds.close()
                except Exception:
                    pass

    if not parts:
        return None, skipped

    # Merge only the files belonging to this month. This avoids the fatal
    # all-years combine that the previous loader performed.
    try:
        month_ds = xr.merge(parts, compat="override", join="outer", combine_attrs="override")
    except Exception:
        month_ds = xr.combine_by_coords(
            parts,
            compat="override",
            combine_attrs="override",
            join="outer",
        )

    month_ds = month_ds.sortby("time")

    # If duplicate timestamps exist inside this month, retain the first copy.
    if "time" in month_ds.dims:
        _, unique_idx = np.unique(month_ds.time.values, return_index=True)
        unique_idx = np.sort(unique_idx)
        if len(unique_idx) != month_ds.sizes["time"]:
            month_ds = month_ds.isel(time=unique_idx)

    # Pressure-only 2021 files cannot provide the heatwave Tmax target.
    if "t2m" not in month_ds.data_vars:
        print(f"  [SKIP MONTH] {month}: t2m not available", flush=True)
        month_ds.close()
        return None, skipped

    try:
        daily = daily_aggregate(month_ds, cfg).load()
    finally:
        month_ds.close()
        for part in parts:
            try:
                part.close()
            except Exception:
                pass
        del parts

    return daily, skipped


def open_era5(files, cfg):
    """Memory-safe ERA5 loader that processes one month at a time.

    The previous implementation accumulated every batch and then called
    combine_by_coords() over the complete multi-year collection. That created
    a large temporary allocation and could be killed by the OS. This version
    combines only the source files belonging to one month, converts that month
    to daily fields, releases the raw hourly/sub-daily data, and only then
    concatenates the compact daily datasets.
    """
    if not files:
        raise FileNotFoundError("No .nc files were found.")

    groups = _group_era5_files_by_month(files)
    if not groups:
        raise ValueError("Could not determine YYYY-MM for any ERA5 NetCDF file.")

    print(f"Found {len(files)} NetCDF file(s) in {len(groups)} month group(s).")
    print("Memory-safe loader: one month at a time.")

    daily_parts = []
    skipped = 0
    usable_months = 0

    for month, month_files in groups.items():
        daily, month_skipped = _load_month_group(month, month_files, cfg)
        skipped += month_skipped
        if daily is None:
            continue
        daily_parts.append(daily)
        usable_months += 1
        print(
            f"  [MONTH OK] {month}: {daily.sizes.get('time', 0)} daily step(s) | "
            f"usable months={usable_months}",
            flush=True,
        )

    if not daily_parts:
        raise ValueError("No usable ERA5 months were found.")

    print("\nCombining DAILY monthly datasets...", flush=True)
    try:
        ds_all = xr.concat(
            daily_parts,
            dim="time",
            data_vars="minimal",
            coords="minimal",
            compat="override",
            join="outer",
        )
    except Exception:
        ds_all = xr.combine_by_coords(
            daily_parts,
            compat="override",
            combine_attrs="override",
            join="outer",
        )

    ds_all = ds_all.sortby("time")

    if "time" in ds_all.dims:
        _, unique_idx = np.unique(ds_all.time.values, return_index=True)
        unique_idx = np.sort(unique_idx)
        if len(unique_idx) != ds_all.sizes["time"]:
            ds_all = ds_all.isel(time=unique_idx)
            print("Removed duplicate daily timestamps.")

    print(
        f"ERA5 loading complete: usable_months={usable_months}, "
        f"skipped_files={skipped}, time_steps={ds_all.sizes.get('time', 0)}",
        flush=True,
    )

    if "tmax" not in ds_all.data_vars:
        raise ValueError("Daily Tmax could not be created because ERA5 t2m was unavailable.")

    return ds_all

def convert_temperature(da):
    """Convert temperature from Kelvin to Celsius when necessary."""
    units = str(da.attrs.get("units", "")).lower()
    if units in ["k", "kelvin"]:
        return da - 273.15

    try:
        if float(da.mean(skipna=True)) > 150.0:
            return da - 273.15
    except Exception:
        pass

    return da


def aggregate_pressure_level(da, variable_name, requested_levels):
    """
    Daily-aggregate one pressure-level variable and return a dictionary
    of [time, latitude, longitude] DataArrays keyed by feature name.
    """
    p_dim = find_pressure_dim(da)
    if p_dim is None:
        return {}

    da = normalize_pressure_coordinate(da, p_dim)

    available = np.asarray(da[p_dim].values, dtype=np.float32)
    result = {}

    for level in requested_levels:
        matches = np.where(np.isclose(available, float(level), atol=0.5))[0]

        if len(matches) == 0:
            continue

        selected = da.isel({p_dim: int(matches[0])}).drop_vars(p_dim, errors="ignore")

        # Pressure-level temperature is converted to Celsius.
        if variable_name == "t":
            selected = convert_temperature(selected)

        # All requested pressure-level fields are instantaneous/rate-like
        # atmospheric state variables, so daily mean is the appropriate
        # aggregation for this training pipeline.
        daily = selected.resample(time="1D").mean(skipna=True)

        result[f"{variable_name}_{int(level)}"] = daily

    return result


def daily_aggregate(ds, cfg):
    """
    Create daily training fields from both ERA5 data types.

    Single-Level examples:
        tmax, d2m, u10, v10, msl, sp, tp, skt, sst, tcc,
        swvl1-swvl4, stl1-stl4, ssrd, strd, sshf, slhf, blh, tcwv.

    Pressure-Level examples for each requested level:
        t, u, v, z, q, r, w.

    The exact feature list is dynamic: only parameters actually present in
    the downloaded NetCDF files are included.
    """
    print("\nAggregating ERA5 to daily fields...")

    daily_vars = {}

    # ------------------------------------------------------------
    # Single-Level variables
    # ------------------------------------------------------------
    single_level_found = []

    for name in SINGLE_LEVEL_CANDIDATES:
        if not cfg.use_single_level or name not in ds.data_vars:
            continue

        da = ds[name]

        if name in ["t2m", "d2m", "skt", "sst"]:
            da = convert_temperature(da)

        elif name in ["msl", "sp"]:
            values = pressure_to_hpa(da.values)
            da = xr.DataArray(
                values,
                dims=da.dims,
                coords=da.coords,
                attrs=da.attrs,
                name=da.name,
            )

        elif name == "tp":
            values = precipitation_to_mm(da.values, da.attrs)
            da = xr.DataArray(
                values,
                dims=da.dims,
                coords=da.coords,
                attrs=da.attrs,
                name=da.name,
            )

        # Tmax is calculated from hourly/sub-daily 2-m temperature.
        if name == "t2m":
            daily_vars["tmax"] = da.resample(
                time="1D"
            ).max(skipna=True)
        elif name == "tp":
            # ERA5 total precipitation is accumulated; sum daily values.
            daily_vars["tp"] = da.resample(
                time="1D"
            ).sum(skipna=True)
        else:
            daily_vars[name] = da.resample(
                time="1D"
            ).mean(skipna=True)

        single_level_found.append(name)

    # ------------------------------------------------------------
    # Pressure-Level variables
    # ------------------------------------------------------------
    pressure_level_found = []

    if cfg.use_pressure_level:
        for name in PRESSURE_LEVEL_CANDIDATES:
            if name not in ds.data_vars:
                continue

            expanded = aggregate_pressure_level(
                ds[name],
                name,
                cfg.pressure_levels,
            )

            for feature_name, daily_da in expanded.items():
                daily_vars[feature_name] = daily_da
                pressure_level_found.append(feature_name)

    if "tmax" not in daily_vars:
        raise ValueError(
            "Daily Tmax could not be created because ERA5 t2m was not available."
        )

    daily = xr.Dataset(daily_vars)

    # Remove dates for which the required Tmax target is unavailable.
    # Optional ERA5 variables may still contain NaNs and will be handled
    # later by the existing preprocessing/masking pipeline.
    before_days = daily.sizes.get("time", 0)
    daily = daily.dropna("time", subset=["tmax"])
    after_days = daily.sizes.get("time", 0)

    if before_days != after_days:
        print(
            f"Removed {before_days - after_days} day(s) with no usable "
            "2-m temperature/Tmax target."
        )

    # Seasonal encoding.
    doy = daily.time.dt.dayofyear.values.astype(np.float32)
    angle = 2.0 * np.pi * doy / 365.25

    daily["doy_sin"] = xr.DataArray(
        np.sin(angle),
        dims=["time"],
        coords={"time": daily.time},
    )
    daily["doy_cos"] = xr.DataArray(
        np.cos(angle),
        dims=["time"],
        coords={"time": daily.time},
    )

    print(f"\nSingle-Level parameters used: {single_level_found}")
    print(f"Pressure-Level features used: {pressure_level_found}")

    return daily


# ============================================================
# 6. HEATWAVE TARGETS
# ============================================================

FEATURE_NAMES = [
    "tmax",
    "d2m",
    "u10",
    "v10",
    "msl",
    "sp",
    "tp",
    "doy_sin",
    "doy_cos",
]


def build_mesh_geometry(daily, cfg):
    """Create the icosphere geometry once from the native ERA5 grid."""
    grid_lat = daily.latitude.values.astype(np.float32)
    grid_lon = daily.longitude.values.astype(np.float32)

    xyz, faces = create_icosphere(cfg.mesh_subdivisions)
    edge_index = faces_to_edges(faces)
    mesh_lat, mesh_lon = xyz_to_latlon(xyz)

    valid = (
        (mesh_lat >= grid_lat.min()) &
        (mesh_lat <= grid_lat.max()) &
        (mesh_lon >= grid_lon.min()) &
        (mesh_lon <= grid_lon.max())
    )

    if not np.any(valid):
        raise ValueError("No icosphere nodes overlap the ERA5 spatial domain.")

    # np.interp requires an increasing coordinate array. ERA5 latitude is
    # commonly descending, so reverse both coordinates and their indices when
    # needed. This preserves the original array-index convention.
    if grid_lat[0] > grid_lat[-1]:
        lat_fraction = np.interp(
            mesh_lat[valid],
            grid_lat[::-1],
            np.arange(len(grid_lat), dtype=np.float32)[::-1],
        )
    else:
        lat_fraction = np.interp(
            mesh_lat[valid],
            grid_lat,
            np.arange(len(grid_lat), dtype=np.float32),
        )

    if grid_lon[0] > grid_lon[-1]:
        lon_fraction = np.interp(
            mesh_lon[valid],
            grid_lon[::-1],
            np.arange(len(grid_lon), dtype=np.float32)[::-1],
        )
    else:
        lon_fraction = np.interp(
            mesh_lon[valid],
            grid_lon,
            np.arange(len(grid_lon), dtype=np.float32),
        )

    return {
        "mesh_lat": mesh_lat.astype(np.float32),
        "mesh_lon": mesh_lon.astype(np.float32),
        "edge_index": edge_index,
        "valid_mask": valid.astype(bool),
        "lat_fraction": lat_fraction.astype(np.float32),
        "lon_fraction": lon_fraction.astype(np.float32),
    }


def sample_daily_to_mesh(daily, geometry, cfg):
    """Sample one already-daily ERA5 month directly onto mesh nodes.

    This is intentionally month-by-month. It avoids ever constructing the
    enormous [4230, ~hundreds/thousands of 0.1-degree cells, features]
    intermediate array that caused the OS to kill the previous process.
    """
    mesh_lat = geometry["mesh_lat"]
    mesh_lon = geometry["mesh_lon"]
    valid = geometry["valid_mask"]
    lat_fraction = geometry["lat_fraction"]
    lon_fraction = geometry["lon_fraction"]

    ntime = int(daily.sizes["time"])
    nvalid = int(valid.sum())

    feature_names = [
        name for name in daily.data_vars
        if name not in ["tmax", "doy_sin", "doy_cos"]
    ]
    feature_names = ["tmax"] + feature_names + ["doy_sin", "doy_cos"]

    # Construct interpolation coordinates only for this month.
    time_coords = np.broadcast_to(
        np.arange(ntime, dtype=np.float32)[:, None],
        (ntime, nvalid),
    )
    lat_coords = np.broadcast_to(
        lat_fraction[None, :],
        (ntime, nvalid),
    )
    lon_coords = np.broadcast_to(
        lon_fraction[None, :],
        (ntime, nvalid),
    )
    coords = np.stack([time_coords, lat_coords, lon_coords], axis=0)

    values = []
    for name in feature_names:
        if name in ["doy_sin", "doy_cos"]:
            arr = np.asarray(daily[name].values, dtype=np.float32)
            node_values = np.repeat(arr[:, None], nvalid, axis=1)
        else:
            arr = np.asarray(daily[name].values, dtype=np.float32)
            if arr.ndim != 3:
                raise ValueError(
                    f"Expected [time, latitude, longitude] for {name}, got {arr.shape}"
                )

            # Fill only this variable/month. Do not make a second full-dataset
            # copy. NaNs are uncommon in the usable ERA5 fields.
            if not np.isfinite(arr).all():
                finite_mean = np.nanmean(arr)
                if not np.isfinite(finite_mean):
                    finite_mean = 0.0
                arr = np.where(np.isfinite(arr), arr, finite_mean).astype(np.float32)

            node_values = map_coordinates(
                arr,
                coords,
                order=1,
                mode="nearest",
                prefilter=False,
            ).astype(np.float32)

        values.append(node_values)

    features_valid = np.stack(values, axis=-1).astype(np.float32)

    features = np.full(
        (ntime, len(mesh_lat), len(feature_names)),
        np.nan,
        dtype=np.float32,
    )
    features[:, valid, :] = features_valid

    return {
        "features": features,
        "mesh_lat": mesh_lat,
        "mesh_lon": mesh_lon,
        "edge_index": geometry["edge_index"],
        "valid_mask": valid,
        "feature_names": feature_names,
    }


def build_mesh_arrays_fast(daily, cfg):
    """Compatibility wrapper for one daily dataset."""
    geometry = build_mesh_geometry(daily, cfg)
    print(
        f"\nFast mesh sampling: {daily.sizes['time']} days -> "
        f"{int(geometry['valid_mask'].sum())}/{len(geometry['mesh_lat'])} "
        "valid icosphere nodes",
        flush=True,
    )
    return sample_daily_to_mesh(daily, geometry, cfg)

def build_mesh_arrays(daily, cfg):
    """Compatibility wrapper selecting the fast graph sampling path."""
    if getattr(cfg, "fast_mesh_sampling", True):
        return build_mesh_arrays_fast(daily, cfg)
    return build_mesh_arrays_original(daily, cfg)


def build_mesh_arrays_original(daily, cfg):
    """Original full-grid regridding implementation retained as fallback."""
    grid_lat = daily.latitude.values.astype(np.float32)
    grid_lon = daily.longitude.values.astype(np.float32)

    xyz, faces = create_icosphere(cfg.mesh_subdivisions)
    edge_index = faces_to_edges(faces)
    mesh_lat, mesh_lon = xyz_to_latlon(xyz)

    valid = (
        (mesh_lat >= grid_lat.min()) &
        (mesh_lat <= grid_lat.max()) &
        (mesh_lon >= grid_lon.min()) &
        (mesh_lon <= grid_lon.max())
    )

    lat_idx, lon_idx = mesh_map(
        mesh_lat, mesh_lon, grid_lat, grid_lon
    )

    feature_names = [
        name for name in daily.data_vars
        if name not in ["tmax", "doy_sin", "doy_cos"]
    ]
    feature_names = ["tmax"] + feature_names + ["doy_sin", "doy_cos"]

    values = []
    for name in feature_names:
        arr = daily[name].values
        if name in ["doy_sin", "doy_cos"]:
            node_values = np.repeat(arr[:, None], len(mesh_lat), axis=1)
        else:
            node_values = arr[:, lat_idx, lon_idx]
        values.append(node_values.astype(np.float32))

    features = np.stack(values, axis=-1)
    features[:, ~valid, :] = np.nan

    return {
        "features": features,
        "mesh_lat": mesh_lat,
        "mesh_lon": mesh_lon,
        "edge_index": edge_index,
        "valid_mask": valid.astype(bool),
        "feature_names": feature_names,
    }


def compute_heatwave_climatology(tmax, train_end):
    """
    Compute local statistics using only the chronological training period.

    tmax: [time, node]
    train_end: exclusive index
    """
    train_tmax = tmax[:train_end]

    mean = np.nanmean(train_tmax, axis=0)
    p95 = np.nanpercentile(train_tmax, 95, axis=0)

    return mean.astype(np.float32), p95.astype(np.float32)


# ============================================================
# 7. NORMALIZATION
# ============================================================

class StandardScaler:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, x):
        self.mean = np.nanmean(x, axis=(0, 1)).astype(np.float32)
        self.std = np.nanstd(x, axis=(0, 1)).astype(np.float32)

        self.std[self.std < 1e-6] = 1.0
        return self

    def transform(self, x):
        return (x - self.mean) / self.std

    def inverse_transform(self, x):
        return x * self.std + self.mean


# ============================================================
# 8. SEQUENCE DATASET
# ============================================================

class HeatwaveSequenceDataset(Dataset):
    def __init__(
        self,
        features,
        tmax,
        heat_labels,
        start,
        end,
        window,
        horizon,
    ):
        self.features = features
        self.tmax = tmax
        self.heat_labels = heat_labels
        self.window = window
        self.horizon = horizon

        # Last valid starting index is:
        # end - window - horizon.
        max_start = end - window - horizon

        if max_start >= start:
            self.indices = list(
                range(start, max_start + 1)
            )
        else:
            self.indices = []

        self.indices = [
            i for i in self.indices
            if i + window + horizon <= end
        ]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]

        x = self.features[i:i + self.window]
        y_tmax = self.tmax[
            i + self.window:i + self.window + self.horizon
        ]
        y_heat = self.heat_labels[
            i + self.window:i + self.window + self.horizon
        ]

        return (
            torch.from_numpy(x).float(),
            torch.from_numpy(y_tmax).float(),
            torch.from_numpy(y_heat).float(),
        )


# ============================================================
# 9. GRAPH LAYERS
# ============================================================

class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.0):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class EdgeBlock(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim, dropout):
        super().__init__()

        self.mlp = MLP(
            node_dim * 2 + edge_dim,
            hidden_dim,
            edge_dim,
            dropout
        )

    def forward(self, src, dst, edge_attr):
        x = torch.cat([src, dst, edge_attr], dim=-1)
        return edge_attr + self.mlp(x)


class NodeBlock(nn.Module):
    """
    MeshGraphNet node-update block.

    Edge features live in edge_dim space (64 by default), while node
    features live in node_dim space (128 by default).  Therefore edge
    messages are projected to node_dim before aggregation.
    """
    def __init__(self, node_dim, edge_dim, hidden_dim, dropout):
        super().__init__()

        # edge_dim -> node_dim so index_add_ can aggregate into a
        # [num_nodes, node_dim] message tensor.
        self.edge_to_node = nn.Linear(
            edge_dim,
            node_dim
        )

        self.mlp = MLP(
            node_dim * 2,
            hidden_dim,
            node_dim,
            dropout
        )

    def forward(self, node_attr, edge_attr, src_index, dst_index):
        # [E, edge_dim] -> [E, node_dim]
        edge_messages = self.edge_to_node(edge_attr)

        # Aggregate messages at destination nodes.
        messages = torch.zeros_like(node_attr)

        messages.index_add_(
            0,
            dst_index,
            edge_messages
        )

        degree = torch.bincount(
            dst_index,
            minlength=node_attr.shape[0]
        ).to(
            node_attr.device
        ).float().unsqueeze(-1)

        messages = messages / degree.clamp_min(1.0)

        # Node state + aggregated edge message.
        update = self.mlp(
            torch.cat(
                [node_attr, messages],
                dim=-1
            )
        )

        return node_attr + update


class MeshGraphNetBlock(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim, dropout):
        super().__init__()

        self.edge_block = EdgeBlock(
            node_dim,
            edge_dim,
            hidden_dim,
            dropout
        )

        self.node_block = NodeBlock(
            node_dim,
            edge_dim,
            hidden_dim,
            dropout
        )

    def forward(self, node_attr, edge_attr, edge_index):
        src, dst = edge_index

        edge_attr = self.edge_block(
            node_attr[src],
            node_attr[dst],
            edge_attr
        )

        node_attr = self.node_block(
            node_attr,
            edge_attr,
            src,
            dst
        )

        return node_attr, edge_attr


# ============================================================
# 10. HEATWAVE MGN + TEMPORAL TRANSFORMER
# ============================================================

class HeatwaveGNN(nn.Module):
    def __init__(
        self,
        input_dim,
        num_nodes,
        edge_input_dim=5,
        hidden_dim=128,
        edge_dim=64,
        processor_steps=6,
        temporal_dim=256,
        temporal_heads=8,
        temporal_layers=4,
        dropout=0.15,
        horizon=5,
    ):
        super().__init__()

        self.num_nodes = num_nodes
        self.horizon = horizon
        self.hidden_dim = hidden_dim
        self.temporal_dim = temporal_dim

        self.node_encoder = nn.Linear(input_dim, hidden_dim)

        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_input_dim, edge_dim),
            nn.GELU(),
            nn.Linear(edge_dim, edge_dim),
        )

        self.processors = nn.ModuleList([
            MeshGraphNetBlock(
                hidden_dim,
                edge_dim,
                hidden_dim,
                dropout
            )
            for _ in range(processor_steps)
        ])

        self.temporal_projection = nn.Linear(
            hidden_dim,
            temporal_dim
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=temporal_dim,
            nhead=temporal_heads,
            dim_feedforward=temporal_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.temporal_transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=temporal_layers
        )

        self.tmax_head = nn.Sequential(
            nn.Linear(temporal_dim, temporal_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(temporal_dim // 2, horizon),
        )

        self.heatwave_head = nn.Sequential(
            nn.Linear(temporal_dim, temporal_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(temporal_dim // 2, horizon),
        )

    def make_edge_features(self, edge_index, mesh_xyz):
        src, dst = edge_index
        src_xyz = mesh_xyz[src]
        dst_xyz = mesh_xyz[dst]
        delta = dst_xyz - src_xyz
        distance = torch.linalg.norm(delta, dim=-1, keepdim=True)

        return torch.cat([
            src_xyz,
            dst_xyz,
            distance,
        ], dim=-1)

    def spatial_forward_batched(
        self,
        x,
        edge_index,
        edge_features,
    ):
        """
        Batched MeshGraphNet.

        x: [B, N, F]
        edge_index: [2, E]
        edge_features: [E, 5]
        returns: [B, N, H]

        This avoids the original Python loop over every batch element and
        allows the L40S to process the whole batch in parallel.
        """
        B, N, _ = x.shape
        src, dst = edge_index

        node = self.node_encoder(x)
        edge0 = self.edge_encoder(edge_features)

        # Static edge features are copied across the batch only once per
        # spatial timestep.
        edge = edge0.unsqueeze(0).expand(B, -1, -1)

        # Flattened destination indices for batched index_add.
        batch_offsets = (
            torch.arange(B, device=x.device, dtype=dst.dtype) * N
        )
        flat_dst = (dst.unsqueeze(0) + batch_offsets[:, None]).reshape(-1)

        degree = torch.bincount(
            dst,
            minlength=N
        ).to(x.device).float().clamp_min(1.0).view(1, N, 1)

        for block in self.processors:
            src_attr = node[:, src, :]
            dst_attr = node[:, dst, :]

            edge_update = block.edge_block(
                src_attr,
                dst_attr,
                edge,
            )

            edge_messages = block.node_block.edge_to_node(edge_update)

            messages = torch.zeros_like(node)
            messages_flat = messages.reshape(B * N, self.hidden_dim)
            messages_flat.index_add_(
                0,
                flat_dst,
                edge_messages.reshape(-1, self.hidden_dim),
            )
            messages = messages_flat.reshape(B, N, self.hidden_dim)
            messages = messages / degree

            update = block.node_block.mlp(
                torch.cat([node, messages], dim=-1)
            )
            node = node + update
            edge = edge_update

        return node

    def spatial_forward(self, x, edge_index, edge_features):
        """Single-sample compatibility wrapper."""
        return self.spatial_forward_batched(
            x.unsqueeze(0),
            edge_index,
            edge_features,
        )[0]

    def forward(
        self,
        x,
        edge_index,
        edge_features,
        valid_mask=None,
    ):
        """
        x: [B, T, N, F]
        Returns:
            tmax_pred: [B, horizon, N]
            heat_logits: [B, horizon, N]
        """
        B, T, N, _ = x.shape

        spatial_outputs = []
        for t in range(T):
            spatial_outputs.append(
                self.spatial_forward_batched(
                    x[:, t, :, :],
                    edge_index,
                    edge_features,
                )
            )

        # [B, T, N, H]
        spatial = torch.stack(spatial_outputs, dim=1)
        spatial = self.temporal_projection(spatial)

        # [B*N, T, D]
        temporal = spatial.permute(0, 2, 1, 3).reshape(
            B * N,
            T,
            self.temporal_dim,
        )

        temporal = self.temporal_transformer(temporal)
        context = temporal[:, -1, :].reshape(
            B, N, self.temporal_dim
        )

        tmax_pred = self.tmax_head(context).permute(0, 2, 1)
        heat_logits = self.heatwave_head(context).permute(0, 2, 1)

        return tmax_pred, heat_logits


# ============================================================
# 11. EDGE FEATURES
# ============================================================

def build_edge_features(edge_index, mesh_lat, mesh_lon):
    """
    Build static graph edge features.

    Features:
        source latitude
        source longitude
        destination latitude
        destination longitude
        great-circle distance approximation
    """
    lat = np.radians(mesh_lat)
    lon = np.radians(mesh_lon)

    src = edge_index[0]
    dst = edge_index[1]

    src_lat = lat[src]
    src_lon = lon[src]
    dst_lat = lat[dst]
    dst_lon = lon[dst]

    dlat = dst_lat - src_lat
    dlon = dst_lon - src_lon

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(src_lat)
        * np.cos(dst_lat)
        * np.sin(dlon / 2) ** 2
    )

    distance = 2.0 * np.arctan2(
        np.sqrt(np.clip(a, 0, 1)),
        np.sqrt(np.clip(1 - a, 0, 1))
    )

    edge_features = np.stack([
        mesh_lat[src] / 90.0,
        mesh_lon[src] / 180.0,
        mesh_lat[dst] / 90.0,
        mesh_lon[dst] / 180.0,
        distance.astype(np.float32),
    ], axis=-1)

    return edge_features.astype(np.float32)


# ============================================================
# 12. MASKED LOSSES
# ============================================================

def masked_mse(pred, target, valid_mask):
    """
    pred/target: [B, H, N]
    valid_mask: [N]
    """
    mask = valid_mask.view(1, 1, -1).expand_as(pred)

    diff = (pred - target) ** 2

    selected = diff[mask]
    if selected.numel() == 0:
        return torch.zeros(
            (),
            device=pred.device,
            dtype=pred.dtype
        )
    return selected.mean()


def masked_bce(logits, target, valid_mask, pos_weight=None):
    mask = valid_mask.view(1, 1, -1).expand_as(logits)

    if pos_weight is None:
        loss = F.binary_cross_entropy_with_logits(
            logits,
            target,
            reduction="none"
        )
    else:
        loss = F.binary_cross_entropy_with_logits(
            logits,
            target,
            reduction="none",
            pos_weight=pos_weight
        )

    selected = loss[mask]
    if selected.numel() == 0:
        return torch.zeros(
            (),
            device=logits.device,
            dtype=logits.dtype
        )
    return selected.mean()


# ============================================================
# 13. METRICS
# ============================================================

def regression_metrics(pred, target, valid_mask, y_std):
    mask = valid_mask.view(1, 1, -1).expand_as(pred)

    diff = (pred - target)[mask]

    mse_scaled = torch.mean(diff ** 2).item()
    mae_scaled = torch.mean(torch.abs(diff)).item()

    rmse_original = math.sqrt(mse_scaled) * float(y_std)
    mae_original = mae_scaled * float(y_std)

    return mse_scaled, mae_scaled, rmse_original, mae_original


def classification_metrics(logits, target, valid_mask):
    probs = torch.sigmoid(logits)
    pred = (probs >= 0.5).float()

    mask = valid_mask.view(1, 1, -1).expand_as(logits)

    pred = pred[mask].detach().cpu().numpy()
    target = target[mask].detach().cpu().numpy()

    if pred.size == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    tp = float(np.sum((pred == 1) & (target == 1)))
    tn = float(np.sum((pred == 0) & (target == 0)))
    fp = float(np.sum((pred == 1) & (target == 0)))
    fn = float(np.sum((pred == 0) & (target == 1)))

    precision = tp / max(tp + fp, 1.0)
    recall = tp / max(tp + fn, 1.0)
    f1 = (
        2 * precision * recall /
        max(precision + recall, 1e-8)
    )

    return precision, recall, f1, tp, tn, fp, fn


# ============================================================
# 14. TRAIN / EVALUATE
# ============================================================

def run_epoch(
    model,
    loader,
    optimizer,
    edge_index,
    edge_features,
    valid_mask,
    device,
    y_std,
    train=True,
    scaler=None,
    pos_weight=None,
):
    model.train(train)

    total_loss = 0.0
    total_count = 0

    mse_sum = 0.0
    bce_sum = 0.0

    all_tmax_pred = []
    all_tmax_true = []
    all_heat_logits = []
    all_heat_true = []

    use_amp = device.type == "cuda" and scaler is not None

    for x, y_tmax, y_heat in loader:
        x = x.to(device, non_blocking=True)
        y_tmax = y_tmax.to(device, non_blocking=True)
        y_heat = y_heat.to(device, non_blocking=True)

        if train:
            optimizer.zero_grad(set_to_none=True)

        if use_amp:
            with torch.cuda.amp.autocast(dtype=torch.float16):
                tmax_pred, heat_logits = model(
                    x, edge_index, edge_features, valid_mask
                )
                loss_reg = masked_mse(tmax_pred, y_tmax, valid_mask)
                loss_cls = masked_bce(
                    heat_logits, y_heat, valid_mask, pos_weight=pos_weight
                )
                loss = loss_reg + 0.5 * loss_cls
        else:
            tmax_pred, heat_logits = model(
                x, edge_index, edge_features, valid_mask
            )
            loss_reg = masked_mse(tmax_pred, y_tmax, valid_mask)
            loss_cls = masked_bce(
                heat_logits, y_heat, valid_mask, pos_weight=pos_weight
            )
            loss = loss_reg + 0.5 * loss_cls

        if train:
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=1.0
                )
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=1.0
                )
                optimizer.step()

        batch_size = x.shape[0]
        total_loss += loss.item() * batch_size
        mse_sum += loss_reg.item() * batch_size
        bce_sum += loss_cls.item() * batch_size
        total_count += batch_size

        all_tmax_pred.append(tmax_pred.detach())
        all_tmax_true.append(y_tmax.detach())
        all_heat_logits.append(heat_logits.detach())
        all_heat_true.append(y_heat.detach())

    if total_count == 0:
        return None

    tmax_pred = torch.cat(all_tmax_pred, dim=0)
    tmax_true = torch.cat(all_tmax_true, dim=0)
    heat_logits = torch.cat(all_heat_logits, dim=0)
    heat_true = torch.cat(all_heat_true, dim=0)

    mse, mae, rmse_c, mae_c = regression_metrics(
        tmax_pred, tmax_true, valid_mask, y_std
    )

    precision, recall, f1, tp, tn, fp, fn = classification_metrics(
        heat_logits, heat_true, valid_mask
    )

    return {
        "loss": total_loss / total_count,
        "reg_loss": mse_sum / total_count,
        "cls_loss": bce_sum / total_count,
        "mse_scaled": mse,
        "mae_scaled": mae,
        "rmse_c": rmse_c,
        "mae_c": mae_c,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


# ============================================================
# 15. DATA PREPARATION
# ============================================================

def prepare_data(cfg):
    """Prepare the full ERA5 dataset without ever combining raw daily grids.

    The critical optimization is to process one month at a time:
        raw ERA5 -> daily month -> direct mesh sampling -> release month

    Only the compact [time, mesh_node, feature] arrays are retained. This
    prevents the multi-GB xarray dataset from coexisting with the mesh feature
    array and avoids the OS-level `Killed` failure seen in the previous run.
    """
    files = resolve_nc_files(cfg.era5_path)
    if not files:
        raise FileNotFoundError(
            f"No NetCDF (.nc) files were found under: {cfg.era5_path}"
        )

    print("\n" + "=" * 70)
    print("ERA5 FILE DISCOVERY")
    print("=" * 70)

    groups = _group_era5_files_by_month(files)
    if not groups:
        raise ValueError("Could not determine YYYY-MM for any ERA5 NetCDF file.")

    print(f"Found {len(files)} NetCDF file(s) in {len(groups)} month group(s).")
    print("Streaming month -> daily -> mesh preprocessing: no full daily grid concat.", flush=True)

    # Load only the first usable month to establish the fixed graph geometry.
    geometry = None
    feature_parts = []
    usable_months = 0
    skipped_files = 0
    feature_names = None

    for month, month_files in groups.items():
        print(f"\n--- Processing {month} ({len(month_files)} source file(s)) ---", flush=True)
        daily, month_skipped = _load_month_group(month, month_files, cfg)
        skipped_files += month_skipped
        if daily is None:
            continue

        # Every usable month must have exactly the same feature schema.
        # One damaged/incomplete pressure-level month (for example a month
        # where only the Single-Level file is usable) must not be concatenated
        # with the normal 49-feature months.
        expected_features = [
            "tmax", "d2m", "u10", "v10", "msl", "sp", "tp",
            "doy_sin", "doy_cos",
        ]
        if cfg.use_pressure_level:
            # Fixed pressure-level schema matching the actual ERA5 files used
            # in this project.  These files contain t/u/v/z/r, but not q/w.
            # Do NOT fabricate missing q or w values.
            for p_name in ("t", "u", "v", "z", "r"):
                for level in cfg.pressure_levels:
                    expected_features.append(f"{p_name}_{int(level)}")

        missing_features = [
            name for name in expected_features
            if name not in daily.data_vars
        ]
        if missing_features:
            print(
                f"  [MONTH SKIP] {month}: missing "
                f"{len(missing_features)} required feature(s): "
                f"{missing_features[:8]}" + (" ..." if len(missing_features) > 8 else ""),
                flush=True,
            )
            try:
                daily.close()
            except Exception:
                pass
            continue

        if geometry is None:
            geometry = build_mesh_geometry(daily, cfg)
            print(
                f"Fast mesh sampling: {daily.sizes['time']} days -> "
                f"{int(geometry['valid_mask'].sum())}/{len(geometry['mesh_lat'])} "
                "valid icosphere nodes",
                flush=True,
            )

        part = sample_daily_to_mesh(daily, geometry, cfg)

        # Force a fixed ordering so every month has identical feature columns.
        fixed_names = expected_features
        index_by_name = {name: i for i, name in enumerate(part["feature_names"])}
        reorder = [index_by_name[name] for name in fixed_names]
        part_features = part["features"][:, :, reorder]

        feature_names = fixed_names
        feature_parts.append(part_features)
        usable_months += 1

        print(
            f"  [MONTH OK] {month}: {part['features'].shape[0]} daily step(s) "
            f"| usable months={usable_months}",
            flush=True,
        )

        # Explicitly release the large monthly xarray dataset immediately.
        try:
            daily.close()
        except Exception:
            pass
        del daily, part

    if not feature_parts:
        raise ValueError("No usable ERA5 months were found.")

    print("\nCombining compact mesh-month arrays...", flush=True)
    features = np.concatenate(feature_parts, axis=0).astype(np.float32, copy=False)
    del feature_parts

    print(
        f"ERA5 preprocessing complete: usable_months={usable_months}, "
        f"skipped_files={skipped_files}, time_steps={features.shape[0]}, "
        f"nodes={features.shape[1]}, features={features.shape[2]}",
        flush=True,
    )

    valid_mask = geometry["valid_mask"]
    tmax_index = feature_names.index("tmax")
    tmax = features[:, :, tmax_index].copy()

    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    tmax = np.nan_to_num(tmax, nan=0.0, posinf=0.0, neginf=0.0)

    total_time = features.shape[0]
    if total_time < cfg.window + cfg.horizon + 10:
        raise ValueError(
            f"Not enough daily samples ({total_time}). "
            f"Need at least {cfg.window + cfg.horizon + 10}."
        )

    # Chronological split.
    train_end = int(total_time * 0.70)
    val_end = int(total_time * 0.85)

    clim_mean, clim_p95 = compute_heatwave_climatology(tmax, train_end)

    heat_labels = (
        (tmax >= clim_p95[None, :]) &
        ((tmax - clim_mean[None, :]) >= 2.0)
    ).astype(np.float32)

    scaler_x = StandardScaler()
    train_x = features[:train_end]
    valid_train = train_x[:, valid_mask, :]
    scaler_x.fit(valid_train)
    del train_x, valid_train

    features_scaled = scaler_x.transform(features).astype(np.float32)

    scaler_y = StandardScaler()
    train_y = tmax[:train_end, valid_mask]
    scaler_y.fit(train_y[:, :, None])
    del train_y

    tmax_scaled = scaler_y.transform(tmax[:, :, None])[:, :, 0].astype(np.float32)

    features_scaled[:, ~valid_mask, :] = 0.0
    tmax_scaled[:, ~valid_mask] = 0.0
    heat_labels[:, ~valid_mask] = 0.0

    if cfg.max_samples < 0:
        raise ValueError("--max-samples must be >= 0.")

    if cfg.max_samples > 0:
        max_needed = min(
            total_time,
            cfg.max_samples + cfg.window + cfg.horizon,
        )
        features_scaled = features_scaled[:max_needed]
        tmax_scaled = tmax_scaled[:max_needed]
        heat_labels = heat_labels[:max_needed]
        total_time = max_needed
        train_end = int(total_time * 0.70)
        val_end = int(total_time * 0.85)

    # Free the unscaled copies before training.
    del features, tmax

    return {
        "features": features_scaled,
        "tmax": tmax_scaled,
        "heat_labels": heat_labels.astype(np.float32),
        "train_end": train_end,
        "val_end": val_end,
        "mesh_lat": geometry["mesh_lat"],
        "mesh_lon": geometry["mesh_lon"],
        "edge_index": geometry["edge_index"],
        "valid_mask": valid_mask,
        "scaler_x": scaler_x,
        "scaler_y": scaler_y,
        "clim_mean": clim_mean,
        "clim_p95": clim_p95,
        "feature_names": feature_names,
    }


# ============================================================
# 16. CHECKPOINT
# ============================================================

def save_checkpoint(
    path,
    model,
    cfg,
    data,
):
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "config": asdict(cfg),
        "model_type": "HeatwaveGNN",
        "mesh_lat": data["mesh_lat"],
        "mesh_lon": data["mesh_lon"],
        "edge_index": data["edge_index"],
        "valid_mask": data["valid_mask"],
        "input_dim": len(data["feature_names"]),
        "edge_input_dim": 5,
        "feature_names": data["feature_names"],
        "scaler_x_mean": data["scaler_x"].mean,
        "scaler_x_std": data["scaler_x"].std,
        "scaler_y_mean": data["scaler_y"].mean,
        "scaler_y_std": data["scaler_y"].std,
        "heatwave_climatology_mean": data["clim_mean"],
        "heatwave_climatology_p95": data["clim_p95"],
    }

    torch.save(checkpoint, path)


# ============================================================
# 17. MAIN TRAINING
# ============================================================

def train_heatwave(cfg):
    set_seed(cfg.seed)

    print("=" * 70)
    print("WEATHERGPT - HEATWAVE GNN TRAINING")
    print("=" * 70)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"\nDevice: {device}")

    if device.type == "cuda":
        print(
            f"GPU: {torch.cuda.get_device_name(0)}"
        )
    else:
        print("CUDA is not available. Training will use CPU.")

    print("\n" + "=" * 70)
    print("[1/7] LOADING AND PREPROCESSING ERA5")
    print("=" * 70)

    data = prepare_data(cfg)

    if device.type == "cuda":
        print(
            f"GPU optimization: AMP={cfg.use_amp} | batch_size={cfg.batch_size} | "
            f"batched graph message passing=ON",
            flush=True,
        )

    features = data["features"]
    tmax = data["tmax"]
    heat_labels = data["heat_labels"]

    train_end = data["train_end"]
    val_end = data["val_end"]

    print(
        f"\nPrepared feature tensor: {features.shape}"
    )

    print(
        f"Train time range: 0 -> {train_end}"
    )
    print(
        f"Validation time range: {train_end} -> {val_end}"
    )
    print(
        f"Test time range: {val_end} -> {features.shape[0]}"
    )

    # --------------------------------------------------------
    # Datasets
    # --------------------------------------------------------

    train_ds = HeatwaveSequenceDataset(
        features,
        tmax,
        heat_labels,
        0,
        train_end,
        cfg.window,
        cfg.horizon,
    )

    val_ds = HeatwaveSequenceDataset(
        features,
        tmax,
        heat_labels,
        train_end,
        val_end,
        cfg.window,
        cfg.horizon,
    )

    test_ds = HeatwaveSequenceDataset(
        features,
        tmax,
        heat_labels,
        val_end,
        features.shape[0],
        cfg.window,
        cfg.horizon,
    )

    print("\nSequence counts:")
    print(f"  Train: {len(train_ds)}")
    print(f"  Val:   {len(val_ds)}")
    print(f"  Test:  {len(test_ds)}")

    if len(train_ds) == 0:
        raise ValueError(
            "Training dataset contains zero sequences. "
            "Increase the amount of ERA5 data or reduce window/horizon."
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(cfg.num_workers > 0),
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(cfg.num_workers > 0),
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(cfg.num_workers > 0),
    )

    # --------------------------------------------------------
    # Graph
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("[2/7] BUILDING GRAPH")
    print("=" * 70)

    edge_index_np = data["edge_index"]

    edge_features_np = build_edge_features(
        edge_index_np,
        data["mesh_lat"],
        data["mesh_lon"],
    )

    edge_index = torch.from_numpy(
        edge_index_np
    ).long().to(device)

    edge_features = torch.from_numpy(
        edge_features_np
    ).float().to(device)

    valid_mask = torch.from_numpy(
        data["valid_mask"]
    ).bool().to(device)

    print(
        f"Nodes: {len(data['mesh_lat'])}"
    )
    print(
        f"Directed edges: {edge_index.shape[1]}"
    )
    print(
        f"Valid nodes: {int(data['valid_mask'].sum())}"
    )

    # --------------------------------------------------------
    # Static dimension checks
    # --------------------------------------------------------
    if edge_features.ndim != 2 or edge_features.shape[1] != 5:
        raise ValueError(
            "Edge feature tensor must have shape [E, 5], "
            f"but got {tuple(edge_features.shape)}."
        )

    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError(
            "edge_index must have shape [2, E], "
            f"but got {tuple(edge_index.shape)}."
        )

    if edge_index.shape[1] != edge_features.shape[0]:
        raise ValueError(
            "Number of edges in edge_index and edge_features does not match: "
            f"{edge_index.shape[1]} vs {edge_features.shape[0]}."
        )

    if not torch.isfinite(edge_features).all():
        raise ValueError("Non-finite values detected in graph edge features.")

    if int(data["valid_mask"].sum()) == 0:
        raise ValueError(
            "No valid icosphere nodes overlap the ERA5 spatial domain."
        )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("[3/7] CREATING HEATWAVE GNN")
    print("=" * 70)

    model = HeatwaveGNN(
        input_dim=len(data["feature_names"]),
        num_nodes=len(data["mesh_lat"]),
        edge_input_dim=5,
        hidden_dim=cfg.hidden_dim,
        edge_dim=cfg.edge_dim,
        processor_steps=cfg.processor_steps,
        temporal_dim=cfg.temporal_dim,
        temporal_heads=cfg.temporal_heads,
        temporal_layers=cfg.temporal_layers,
        dropout=cfg.dropout,
        horizon=cfg.horizon,
    ).to(device)

    parameter_count = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        f"Trainable parameters: {parameter_count:,}"
    )

    # --------------------------------------------------------
    # Model shape self-check
    # --------------------------------------------------------
    with torch.no_grad():
        sample_x = torch.from_numpy(
            data["features"][0]
        ).float().to(device)

        encoded_nodes = model.node_encoder(sample_x)
        encoded_edges = model.edge_encoder(edge_features)

        expected_nodes = (
            len(data["mesh_lat"]),
            cfg.hidden_dim
        )
        expected_edges = (
            edge_features.shape[0],
            cfg.edge_dim
        )

        if tuple(encoded_nodes.shape) != expected_nodes:
            raise RuntimeError(
                "Node encoder shape mismatch: "
                f"expected {expected_nodes}, got {tuple(encoded_nodes.shape)}."
            )

        if tuple(encoded_edges.shape) != expected_edges:
            raise RuntimeError(
                "Edge encoder shape mismatch: "
                f"expected {expected_edges}, got {tuple(encoded_edges.shape)}."
            )

        print(
            f"Shape check: nodes {tuple(encoded_nodes.shape)} | "
            f"edges {tuple(encoded_edges.shape)} | OK"
        )

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("[4/7] OPTIMIZER")
    print("=" * 70)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=4,
        min_lr=1e-6,
    )

    # Counter class imbalance so the heatwave head is not rewarded for
    # predicting "no heatwave" everywhere.
    train_heat = data["heat_labels"][:train_end, data["valid_mask"]]
    positive = float(train_heat.sum())
    negative = float(train_heat.size - positive)
    pos_weight_value = negative / max(positive, 1.0)
    pos_weight_value = float(np.clip(pos_weight_value, 1.0, 20.0))
    pos_weight = torch.tensor(
        pos_weight_value, device=device, dtype=torch.float32
    )
    print(
        f"Heatwave training labels: positive={int(positive):,} | "
        f"negative={int(negative):,} | pos_weight={pos_weight_value:.3f}"
    )

    amp_scaler = (
        torch.cuda.amp.GradScaler(enabled=(device.type == "cuda" and cfg.use_amp))
        if device.type == "cuda" else None
    )

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    best_val = float("inf")
    best_state = None

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("[5/7] TRAINING")
    print("=" * 70)

    for epoch in range(1, cfg.epochs + 1):

        train_metrics = run_epoch(
            model,
            train_loader,
            optimizer,
            edge_index,
            edge_features,
            valid_mask,
            device,
            float(data["scaler_y"].std[0]),
            train=True,
            scaler=amp_scaler,
            pos_weight=pos_weight,
        )

        with torch.no_grad():
            if len(val_loader) > 0:
                val_metrics = run_epoch(
                    model,
                    val_loader,
                    None,
                    edge_index,
                    edge_features,
                    valid_mask,
                    device,
                    float(data["scaler_y"].std[0]),
                    train=False,
                    scaler=amp_scaler,
                    pos_weight=pos_weight,
                )
            else:
                val_metrics = train_metrics

        scheduler.step(
            val_metrics["loss"]
        )

        lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:03d}/{cfg.epochs} | "
            f"Train Loss: {train_metrics['loss']:.5f} | "
            f"Val Loss: {val_metrics['loss']:.5f} | "
            f"Val MAE: {val_metrics['mae_c']:.3f} °C | "
            f"Val RMSE: {val_metrics['rmse_c']:.3f} °C | "
            f"Heat F1: {val_metrics['f1']:.3f} | "
            f"LR: {lr:.2e}"
        )

        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            best_state = copy.deepcopy(
                model.state_dict()
            )

            save_checkpoint(
                cfg.checkpoint_path,
                model,
                cfg,
                data,
            )

    # --------------------------------------------------------
    # Test
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("[6/7] TESTING BEST MODEL")
    print("=" * 70)

    if best_state is not None:
        model.load_state_dict(best_state)

    with torch.no_grad():
        test_metrics = run_epoch(
            model,
            test_loader,
            None,
            edge_index,
            edge_features,
            valid_mask,
            device,
            float(data["scaler_y"].std[0]),
            train=False,
            scaler=amp_scaler,
            pos_weight=pos_weight,
        )

    if test_metrics is not None:
        print("\nTest results:")
        print(
            f"  Tmax MAE : {test_metrics['mae_c']:.4f} °C"
        )
        print(
            f"  Tmax RMSE: {test_metrics['rmse_c']:.4f} °C"
        )
        print(
            f"  Heat Precision: {test_metrics['precision']:.4f}"
        )
        print(
            f"  Heat Recall   : {test_metrics['recall']:.4f}"
        )
        print(
            f"  Heat F1       : {test_metrics['f1']:.4f}"
        )

    # --------------------------------------------------------
    # Save final best checkpoint again.
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("[7/7] SAVING")
    print("=" * 70)

    save_checkpoint(
        cfg.checkpoint_path,
        model,
        cfg,
        data,
    )

    print(
        f"\nBest heatwave model saved to:\n"
        f"  {os.path.abspath(cfg.checkpoint_path)}"
    )

    print("\nTraining complete.")


# ============================================================
# 18. ARGUMENTS
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Train a heatwave-specific "
            "ERA5 -> 12 km grid -> MGN -> "
            "Temporal Transformer model."
        )
    )

    parser.add_argument(
        "--era5",
        type=str,
        required=True,
        help=(
            "ERA5 NetCDF file, directory, "
            "or recursive glob."
        ),
    )

    parser.add_argument(
        "--single-level",
        dest="use_single_level",
        action="store_true",
        default=True,
        help="Use ERA5 Single-Level parameters (default: enabled).",
    )

    parser.add_argument(
        "--no-single-level",
        dest="use_single_level",
        action="store_false",
        help="Disable ERA5 Single-Level parameters.",
    )

    parser.add_argument(
        "--pressure-level",
        dest="use_pressure_level",
        action="store_true",
        default=True,
        help="Use ERA5 Pressure-Level parameters (default: enabled).",
    )

    parser.add_argument(
        "--no-pressure-level",
        dest="use_pressure_level",
        action="store_false",
        help="Disable ERA5 Pressure-Level parameters.",
    )

    parser.add_argument(
        "--pressure-levels",
        type=int,
        nargs="+",
        default=DEFAULT_PRESSURE_LEVELS,
        choices=DEFAULT_PRESSURE_LEVELS,
        help=(
            "ERA5 pressure levels in hPa. Default: "
            "1000 925 850 700 500 300 250 200."
        ),
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Training batch size. 8 is a good starting point for the L40S.",
    )

    parser.add_argument(
        "--window",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--horizon",
        type=int,
        default=5,
        help="Number of future daily Tmax predictions.",
    )

    parser.add_argument(
        "--mesh-subdivisions",
        type=int,
        default=3,
        help="3=642 nodes, 4=2562 nodes.",
    )

    parser.add_argument(
        "--target-resolution",
        type=float,
        default=0.1,
    )

    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help=(
            "Debug mode. 0 = use all available daily samples. "
            "For example, 40."
        ),
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default="heatwave_gnn.pt",
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=2e-4,
    )

    return parser.parse_args()


# ============================================================
# 19. ENTRY POINT
# ============================================================

if __name__ == "__main__":
    args = parse_args()

    cfg = Config(
        era5_path=args.era5,
        use_single_level=args.use_single_level,
        use_pressure_level=args.use_pressure_level,
        pressure_levels=args.pressure_levels,
        epochs=args.epochs,
        batch_size=args.batch_size,
        window=args.window,
        horizon=args.horizon,
        mesh_subdivisions=args.mesh_subdivisions,
        target_resolution_deg=args.target_resolution,
        max_samples=args.max_samples,
        checkpoint_path=args.checkpoint,
        learning_rate=args.lr,
    )

    train_heatwave(cfg)