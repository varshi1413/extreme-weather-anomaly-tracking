from __future__ import annotations

import json
from urllib.request import Request, urlopen

import numpy as np
import torch
import xarray as xr

from .data import DEFAULT_VARIABLES


def fetch_imd_dataset(url: str, timeout: int = 30) -> xr.Dataset:
    """Fetch a live IMD JSON payload using the project's stable interchange schema.

    Expected payload: {"valid_time": ..., "latitude": [...], "longitude": [...],
    "variables": {"u10": [lat][lon], ...}}. The endpoint is configurable because
    IMD deployments differ and no public endpoint is guaranteed here.
    """
    request = Request(url, headers={"Accept": "application/json", "User-Agent": "extreme-weather-intelligence/0.1"})
    with urlopen(request, timeout=timeout) as response:
        payload = json.load(response)
    required = {"valid_time", "latitude", "longitude", "variables"}
    missing = required.difference(payload)
    if missing:
        raise ValueError(f"IMD payload is missing fields: {sorted(missing)}")
    latitudes = np.asarray(payload["latitude"], dtype=np.float32)
    longitudes = np.asarray(payload["longitude"], dtype=np.float32)
    data_vars = {}
    for name, values in payload["variables"].items():
        array = np.asarray(values, dtype=np.float32)
        if array.shape != (latitudes.size, longitudes.size):
            raise ValueError(f"IMD variable {name!r} has shape {array.shape}, expected {(latitudes.size, longitudes.size)}")
        data_vars[name] = (("latitude", "longitude"), array)
    return xr.Dataset(data_vars, coords={"valid_time": payload["valid_time"], "latitude": latitudes, "longitude": longitudes})


def prepare_imd_condition(
    dataset: xr.Dataset,
    climatology_path: str,
    target_lat: np.ndarray,
    target_lon: np.ndarray,
    variables: tuple[str, ...] = DEFAULT_VARIABLES,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Prepare a live IMD grid as `(coarse, anomaly)` Stage 2 conditions."""
    missing = [name for name in variables if name not in dataset]
    if missing:
        raise ValueError(f"IMD dataset is missing variables: {missing}")
    interpolated = dataset[list(variables)].interp(
        latitude=np.asarray(target_lat), longitude=np.asarray(target_lon), method="linear"
    )
    values = np.stack([interpolated[name].values for name in variables], axis=0).astype(np.float32)
    if values.ndim != 3:
        raise ValueError("IMD variables must be two-dimensional latitude/longitude fields")
    timestamp = dataset.valid_time.values
    month = int(str(timestamp)[:7].split("-")[1])
    with xr.open_dataset(climatology_path) as climatology:
        mean = np.stack([climatology[name].sel(month=month).interp(latitude=target_lat, longitude=target_lon).values for name in variables], axis=0)
    std_path = climatology_path.replace("mean.nc", "std.nc")
    with xr.open_dataset(std_path) as climatology_std:
        std = np.stack([climatology_std[name].sel(month=month).interp(latitude=target_lat, longitude=target_lon).values for name in variables], axis=0)
    normalized = (values - mean) / np.maximum(std, 1e-6)
    target = torch.from_numpy(normalized)
    coarse = torch.nn.functional.avg_pool2d(target.unsqueeze(0), kernel_size=4, stride=4)
    anomaly = (coarse.abs().amax(dim=1, keepdim=True) >= 2.0).float()
    return coarse, anomaly