"""
weather_utils.py

Shared spatial utilities for the Heatwave and ColdWave GNN pipelines.

Functions
---------
mesh_map(mesh_lat, mesh_lon, grid_lat, grid_lon)
    Map mesh coordinates to the nearest ERA5 latitude/longitude grid points.

regrid_daily(ds, target_resolution_deg)
    Regrid an xarray daily dataset to a regular latitude/longitude grid
    using xarray interpolation.
"""

import numpy as np
import xarray as xr


def _normalise_longitude(lon):
    """Convert longitude values to the -180..180 convention."""
    lon = np.asarray(lon, dtype=float)
    return ((lon + 180.0) % 360.0) - 180.0


def mesh_map(mesh_lat, mesh_lon, grid_lat, grid_lon):
    """
    Map each mesh point to the nearest point in an ERA5 grid.

    Parameters
    ----------
    mesh_lat : array-like
        Latitude of GNN mesh points in degrees.
    mesh_lon : array-like
        Longitude of GNN mesh points in degrees.
    grid_lat : array-like
        ERA5 latitude coordinates.
    grid_lon : array-like
        ERA5 longitude coordinates.

    Returns
    -------
    lat_idx : np.ndarray
        Index of the nearest latitude grid point for every mesh point.
    lon_idx : np.ndarray
        Index of the nearest longitude grid point for every mesh point.

    Notes
    -----
    Longitude is handled cyclically so that points close to ±180 degrees
    are mapped correctly.
    """
    mesh_lat = np.asarray(mesh_lat, dtype=float).reshape(-1)
    mesh_lon = np.asarray(mesh_lon, dtype=float).reshape(-1)
    grid_lat = np.asarray(grid_lat, dtype=float).reshape(-1)
    grid_lon = np.asarray(grid_lon, dtype=float).reshape(-1)

    if mesh_lat.size != mesh_lon.size:
        raise ValueError("mesh_lat and mesh_lon must have the same length.")

    if grid_lat.size == 0 or grid_lon.size == 0:
        raise ValueError("grid_lat and grid_lon cannot be empty.")

    if not np.all(np.isfinite(mesh_lat)) or not np.all(np.isfinite(mesh_lon)):
        raise ValueError("Mesh coordinates contain non-finite values.")

    if not np.all(np.isfinite(grid_lat)) or not np.all(np.isfinite(grid_lon)):
        raise ValueError("Grid coordinates contain non-finite values.")

    # Latitude: nearest-neighbour mapping.
    lat_idx = np.abs(
        grid_lat[:, None] - mesh_lat[None, :]
    ).argmin(axis=0)

    # Longitude: cyclic nearest-neighbour mapping.
    grid_lon_norm = _normalise_longitude(grid_lon)
    mesh_lon_norm = _normalise_longitude(mesh_lon)

    lon_difference = np.abs(
        grid_lon_norm[:, None] - mesh_lon_norm[None, :]
    )

    # Account for the wrap-around at ±180°.
    lon_difference = np.minimum(lon_difference, 360.0 - lon_difference)

    lon_idx = lon_difference.argmin(axis=0)

    return lat_idx.astype(np.int64), lon_idx.astype(np.int64)


def _regular_axis(start, stop, resolution):
    """Create a regular coordinate axis including the requested endpoints."""
    if resolution <= 0:
        raise ValueError("target_resolution_deg must be greater than zero.")

    n = int(round((stop - start) / resolution))

    axis = start + np.arange(n + 1, dtype=float) * resolution

    # Numerical protection for the final coordinate.
    if axis.size == 0 or not np.isclose(axis[-1], stop):
        axis = np.append(axis, stop)

    return axis


def regrid_daily(ds, target_resolution_deg):
    """
    Regrid a daily xarray Dataset to a regular latitude/longitude grid.

    Parameters
    ----------
    ds : xarray.Dataset
        Daily dataset containing ``latitude`` and ``longitude`` coordinates.
    target_resolution_deg : float
        Target horizontal resolution in degrees.

    Returns
    -------
    xarray.Dataset
        Dataset interpolated onto the new regular grid.

    Notes
    -----
    Linear interpolation is used for continuous meteorological variables.
    Variables that cannot be interpolated are retained when possible.
    """
    if not isinstance(ds, xr.Dataset):
        raise TypeError("ds must be an xarray.Dataset.")

    if "latitude" not in ds.coords:
        raise ValueError("Dataset must contain a 'latitude' coordinate.")

    if "longitude" not in ds.coords:
        raise ValueError("Dataset must contain a 'longitude' coordinate.")

    resolution = float(target_resolution_deg)

    if resolution <= 0:
        raise ValueError("target_resolution_deg must be greater than zero.")

    # Sort coordinates before interpolation.
    ds = ds.sortby("latitude").sortby("longitude")

    lat = np.asarray(ds["latitude"].values, dtype=float)
    lon = np.asarray(ds["longitude"].values, dtype=float)

    if lat.size < 2 or lon.size < 2:
        raise ValueError(
            "At least two latitude and longitude grid points are required."
        )

    # Build target grid from the actual spatial extent of the dataset.
    lat_min = float(np.nanmin(lat))
    lat_max = float(np.nanmax(lat))
    lon_min = float(np.nanmin(lon))
    lon_max = float(np.nanmax(lon))

    target_lat = _regular_axis(lat_min, lat_max, resolution)
    target_lon = _regular_axis(lon_min, lon_max, resolution)

    # Interpolate only variables that have latitude and longitude dimensions.
    spatial_vars = []
    for name, variable in ds.data_vars.items():
        if "latitude" in variable.dims and "longitude" in variable.dims:
            spatial_vars.append(name)

    if not spatial_vars:
        # Nothing spatial to regrid.
        return ds

    spatial_ds = ds[spatial_vars]

    regridded = spatial_ds.interp(
        latitude=target_lat,
        longitude=target_lon,
        method="linear",
    )

    # Preserve non-spatial variables, such as time metadata.
    non_spatial_vars = [
        name for name in ds.data_vars
        if name not in spatial_vars
    ]

    if non_spatial_vars:
        regridded = xr.merge(
            [regridded, ds[non_spatial_vars]],
            compat="override",
        )

    # Preserve dataset-level attributes.
    regridded.attrs = ds.attrs.copy()

    return regridded


__all__ = ["mesh_map", "regrid_daily"]
