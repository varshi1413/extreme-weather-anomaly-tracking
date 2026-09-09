from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch

from .efi import compute_efi
from .mesh import Mesh
from .model import SphericalAnomalyGNN


@dataclass(frozen=True)
class TemporalBoundingBox:
    start: object
    end: object
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float


class AnomalyTracker:
    def __init__(self, mesh: Mesh, model: SphericalAnomalyGNN, threshold: float = 2.0) -> None:
        self.mesh = mesh
        self.model = model
        self.threshold = threshold

    def track(
        self,
        forecast: np.ndarray,
        baseline: np.ndarray,
        times: Sequence[object] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, TemporalBoundingBox | None]:
        efi = compute_efi(forecast, baseline)
        features = torch.from_numpy(efi)
        edge_index = torch.from_numpy(self.mesh.edge_index)
        with torch.no_grad():
            scores = self.model(features, edge_index).cpu().numpy()
        mask = scores >= self.threshold
        active = np.flatnonzero(mask.any(axis=1))
        if active.size == 0:
            return scores, mask, None
        active_nodes = np.flatnonzero(mask[active].any(axis=0))
        box = TemporalBoundingBox(
            start=times[int(active[0])] if times is not None else int(active[0]),
            end=times[int(active[-1])] if times is not None else int(active[-1]),
            lat_min=float(self.mesh.lat[active_nodes].min()),
            lat_max=float(self.mesh.lat[active_nodes].max()),
            lon_min=float(self.mesh.lon[active_nodes].min()),
            lon_max=float(self.mesh.lon[active_nodes].max()),
        )
        return scores, mask, box


def rasterize_mesh_scores(
    scores: np.ndarray,
    mesh: Mesh,
    target_lat: np.ndarray,
    target_lon: np.ndarray,
) -> np.ndarray:
    """Map node scores onto a regular NEPS-G latitude/longitude crop.

    Uses nearest spherical node assignment and returns [time, latitude, longitude].
    This preserves the Stage 1 signal without assuming that mesh node ordering
    matches the forecast grid ordering.
    """
    scores = np.asarray(scores, dtype=np.float32)
    target_lat = np.asarray(target_lat, dtype=np.float32)
    target_lon = np.asarray(target_lon, dtype=np.float32)
    if scores.ndim != 2 or scores.shape[1] != mesh.num_nodes:
        raise ValueError("scores must have shape [time, mesh_nodes]")
    if target_lat.ndim != 1 or target_lon.ndim != 1:
        raise ValueError("target_lat and target_lon must be one-dimensional")
    lat_grid, lon_grid = np.meshgrid(target_lat, target_lon, indexing="ij")
    lat_grid_rad = np.deg2rad(lat_grid)[..., None]
    lon_grid_rad = np.deg2rad(lon_grid)[..., None]
    mesh_lat_rad = np.deg2rad(mesh.lat)[None, None, :]
    mesh_lon_rad = np.deg2rad(mesh.lon)[None, None, :]
    cosine_distance = (
        np.sin(lat_grid_rad) * np.sin(mesh_lat_rad)
        + np.cos(lat_grid_rad) * np.cos(mesh_lat_rad) * np.cos(lon_grid_rad - mesh_lon_rad)
    )
    nearest = np.argmax(cosine_distance, axis=-1)
    return scores[:, nearest]


def load_netcdf_variables(path: str, variables: Sequence[str]) -> np.ndarray:
    """Load variables from a NetCDF file as [time, nodes, channels]."""
    try:
        import xarray as xr
    except ImportError as error:
        raise ImportError("install the 'netcdf' extra to load NetCDF files") from error
    with xr.open_dataset(path) as dataset:
        arrays = [dataset[name].transpose("time", ...).values for name in variables]
    flattened = [array.reshape(array.shape[0], -1) for array in arrays]
    return np.stack(flattened, axis=-1).astype(np.float32)
