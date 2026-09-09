import numpy as np


def mesh_to_grid(
    mesh_values,
    mesh_lat,
    mesh_lon,
    target_lat,
    target_lon,
    valid_mask=None,
):
    """
    Rasterize spherical-mesh values onto a regular lat/lon grid.

    Parameters
    ----------
    mesh_values : [N] or [T, N]
        Heatwave probabilities from Stage 1.

    mesh_lat, mesh_lon : [N]
        Coordinates of the spherical mesh nodes.

    target_lat, target_lon : [H], [W]
        Target regular latitude/longitude grid.

    valid_mask : [N], optional
        Valid Stage-1 mesh nodes.

    Returns
    -------
    grid : [T, H, W]
        Rasterized Stage-1 probabilities.
    """

    values = np.asarray(mesh_values, dtype=np.float32)

    if values.ndim == 1:
        values = values[None, :]

    mesh_lat = np.asarray(mesh_lat)
    mesh_lon = np.asarray(mesh_lon)

    if valid_mask is None:
        valid_mask = np.ones(len(mesh_lat), dtype=bool)
    else:
        valid_mask = np.asarray(valid_mask, dtype=bool)

    valid_indices = np.where(valid_mask)[0]

    if len(valid_indices) == 0:
        raise ValueError("No valid spherical mesh nodes.")

    valid_lat = mesh_lat[valid_indices]
    valid_lon = mesh_lon[valid_indices]

    lat_grid, lon_grid = np.meshgrid(
        target_lat,
        target_lon,
        indexing="ij",
    )

    # Convert target grid points to unit-sphere XYZ
    lat_rad = np.radians(lat_grid.ravel())
    lon_rad = np.radians(lon_grid.ravel())

    target_xyz = np.stack(
        [
            np.cos(lat_rad) * np.cos(lon_rad),
            np.cos(lat_rad) * np.sin(lon_rad),
            np.sin(lat_rad),
        ],
        axis=1,
    )

    # Convert valid mesh nodes to unit-sphere XYZ
    mesh_lat_rad = np.radians(valid_lat)
    mesh_lon_rad = np.radians(valid_lon)

    mesh_xyz = np.stack(
        [
            np.cos(mesh_lat_rad) * np.cos(mesh_lon_rad),
            np.cos(mesh_lat_rad) * np.sin(mesh_lon_rad),
            np.sin(mesh_lat_rad),
        ],
        axis=1,
    )

    # Nearest spherical mesh node
    similarities = target_xyz @ mesh_xyz.T
    nearest = np.argmax(similarities, axis=1)

    output = []

    for t in range(values.shape[0]):
        valid_values = values[t, valid_indices]

        raster = valid_values[nearest].reshape(
            len(target_lat),
            len(target_lon),
        )

        output.append(raster)

    return np.stack(output, axis=0)


def probability_to_mask(probability_grid, threshold=0.5):
    """
    Convert Stage-1 heatwave probabilities to the binary anomaly
    condition expected by the current diffusion prototype.
    """

    probability_grid = np.asarray(
        probability_grid,
        dtype=np.float32,
    )

    return (
        probability_grid >= threshold
    ).astype(np.float32)


def extract_roi(
    probability_grid,
    anomaly_mask,
    padding=4,
):
    """
    Extract the geographic anomaly ROI.

    Returns None when Stage 1 detects no anomaly.
    """

    probability_grid = np.asarray(probability_grid)
    anomaly_mask = np.asarray(anomaly_mask)

    active = np.argwhere(anomaly_mask > 0)

    if len(active) == 0:
        return None

    # active columns = [time, lat, lon]
    y_min = max(int(active[:, 1].min()) - padding, 0)
    y_max = min(
        int(active[:, 1].max()) + padding + 1,
        probability_grid.shape[1],
    )

    x_min = max(int(active[:, 2].min()) - padding, 0)
    x_max = min(
        int(active[:, 2].max()) + padding + 1,
        probability_grid.shape[2],
    )

    return {
        "y_min": y_min,
        "y_max": y_max,
        "x_min": x_min,
        "x_max": x_max,

        "probability": probability_grid[
            :, y_min:y_max, x_min:x_max
        ],

        "mask": anomaly_mask[
            :, y_min:y_max, x_min:x_max
        ],
    }


def prepare_stage2_condition(
    heatwave_probability,
    mesh_lat,
    mesh_lon,
    valid_mask,
    target_lat,
    target_lon,
    threshold=0.5,
    padding=4,
):
    """
    Complete Stage-1 -> Stage-2 anomaly bridge.
    """

    probability_grid = mesh_to_grid(
        heatwave_probability,
        mesh_lat,
        mesh_lon,
        target_lat,
        target_lon,
        valid_mask,
    )

    anomaly_mask = probability_to_mask(
        probability_grid,
        threshold,
    )

    roi = extract_roi(
        probability_grid,
        anomaly_mask,
        padding,
    )

    return {
        "probability_grid": probability_grid,
        "anomaly_mask": anomaly_mask,
        "roi": roi,
    }


if __name__ == "__main__":

    # Tiny independent bridge test.
    mesh_lat = np.array([10.0, 20.0, 30.0])
    mesh_lon = np.array([70.0, 80.0, 90.0])

    probability = np.array(
        [
            [0.1, 0.9, 0.2],
            [0.2, 0.8, 0.1],
        ],
        dtype=np.float32,
    )

    target_lat = np.linspace(8, 32, 20)
    target_lon = np.linspace(68, 92, 20)

    result = prepare_stage2_condition(
        probability,
        mesh_lat,
        mesh_lon,
        np.array([True, True, True]),
        target_lat,
        target_lon,
    )

    print("\nSTAGE 1 -> STAGE 2 BRIDGE TEST")
    print("--------------------------------")

    print(
        "Probability grid:",
        result["probability_grid"].shape,
    )

    print(
        "Anomaly mask:",
        result["anomaly_mask"].shape,
    )

    if result["roi"] is None:
        print("ROI: none")
    else:
        print(
            "ROI:",
            result["roi"]["mask"].shape,
        )

    print("\nBridge test PASSED.")