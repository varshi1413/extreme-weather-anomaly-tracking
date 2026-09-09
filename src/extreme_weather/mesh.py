from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class Mesh:
    lat: np.ndarray
    lon: np.ndarray
    edge_index: np.ndarray

    @property
    def num_nodes(self) -> int:
        return int(self.lat.shape[0])


def _unit_to_latlon(vertex: np.ndarray) -> tuple[float, float]:
    lat = math.degrees(math.asin(float(vertex[2])))
    lon = math.degrees(math.atan2(float(vertex[1]), float(vertex[0])))
    return lat, lon


def build_indian_icosahedral_mesh(
    subdivisions: int = 4,
    lat_min: float = 6.0,
    lat_max: float = 38.0,
    lon_min: float = 68.0,
    lon_max: float = 98.0,
) -> Mesh:
    """Build a triangulated spherical mesh clipped to the Indian boundary box."""
    if subdivisions < 1:
        raise ValueError("subdivisions must be at least 1")

    phi = (1.0 + math.sqrt(5.0)) / 2.0
    raw = np.array([
        (-1, phi, 0), (1, phi, 0), (-1, -phi, 0), (1, -phi, 0),
        (0, -1, phi), (0, 1, phi), (0, -1, -phi), (0, 1, -phi),
        (phi, 0, -1), (phi, 0, 1), (-phi, 0, -1), (-phi, 0, 1),
    ], dtype=np.float64)
    vertices = [vertex / np.linalg.norm(vertex) for vertex in raw]
    faces = [
        (0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11),
        (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
        (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
        (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1),
    ]

    for _ in range(subdivisions - 1):
        midpoint_cache: dict[tuple[int, int], int] = {}

        def midpoint(first: int, second: int) -> int:
            key = tuple(sorted((first, second)))
            if key not in midpoint_cache:
                point = (vertices[first] + vertices[second]) / 2.0
                vertices.append(point / np.linalg.norm(point))
                midpoint_cache[key] = len(vertices) - 1
            return midpoint_cache[key]

        refined = []
        for first, second, third in faces:
            ab = midpoint(first, second)
            bc = midpoint(second, third)
            ca = midpoint(third, first)
            refined.extend([(first, ab, ca), (second, bc, ab), (third, ca, bc), (ab, bc, ca)])
        faces = refined

    coordinates = np.array([_unit_to_latlon(vertex) for vertex in vertices])
    selected = (
        (coordinates[:, 0] >= lat_min) & (coordinates[:, 0] <= lat_max)
        & (coordinates[:, 1] >= lon_min) & (coordinates[:, 1] <= lon_max)
    )
    selected_ids = np.flatnonzero(selected)
    if selected_ids.size == 0:
        raise ValueError("the requested geographic bounds contain no mesh nodes")
    remap = {old: new for new, old in enumerate(selected_ids.tolist())}
    edges: set[tuple[int, int]] = set()
    for first, second, third in faces:
        for source, target in ((first, second), (second, third), (third, first)):
            if source in remap and target in remap:
                edges.add((remap[source], remap[target]))
                edges.add((remap[target], remap[source]))
    if not edges:
        raise ValueError("the requested geographic bounds contain no connected mesh edges")
    return Mesh(
        lat=coordinates[selected_ids, 0].astype(np.float32),
        lon=coordinates[selected_ids, 1].astype(np.float32),
        edge_index=np.array(sorted(edges), dtype=np.int64).T,
    )
