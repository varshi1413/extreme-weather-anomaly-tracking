import os
import sys
import numpy as np

# ------------------------------------------------------------
# Project root
# ------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Exact preprocessing functions used by training
from Heatwave_GNN_Training_fixed import (
    Config,
    resolve_nc_files,
    open_era5,
    build_mesh_arrays,
)

from src.extreme_weather.heatwave_inference import HeatwavePredictor
from src.extreme_weather.heatwave_tracker import HeatwaveTracker


# ============================================================
# PATHS
# ============================================================

ERA5_PATH = r"C:\Users\Varshitha S B\Documents\ERA5_DOWNLOAD\ERA5_INDIA"

CHECKPOINT_PATH = os.path.join(
    PROJECT_ROOT,
    "checkpoints",
    "heatwave_schema_test.pt",
)


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n==========================================")
    print(" REAL ERA5 -> HEATWAVE GNN -> TRACKER")
    print("==========================================")

    # --------------------------------------------------------
    # 1. Load checkpoint first
    # --------------------------------------------------------

    predictor = HeatwavePredictor(CHECKPOINT_PATH)

    print("\nCheckpoint ready.")

    # --------------------------------------------------------
    # 2. Create preprocessing config matching checkpoint
    # --------------------------------------------------------

    cfg = Config()

    cfg.era5_path = ERA5_PATH
    cfg.window = predictor.window
    cfg.horizon = predictor.horizon

    # These must match checkpoint training configuration
    ckpt_cfg = predictor.checkpoint.get("config", {})

    cfg.target_resolution_deg = ckpt_cfg.get(
        "target_resolution_deg",
        0.1
    )

    cfg.mesh_subdivisions = ckpt_cfg.get(
        "mesh_subdivisions",
        3
    )

    cfg.pressure_levels = ckpt_cfg.get(
        "pressure_levels",
        [1000, 925, 850, 700, 500, 300, 250, 200]
    )

    cfg.use_single_level = ckpt_cfg.get(
        "use_single_level",
        True
    )

    cfg.use_pressure_level = ckpt_cfg.get(
        "use_pressure_level",
        True
    )

    # --------------------------------------------------------
    # 3. Find ERA5 files
    # --------------------------------------------------------

    print("\nSearching ERA5 files...")

    files = resolve_nc_files(ERA5_PATH)

    print("NetCDF files found:", len(files))

    if not files:
        raise FileNotFoundError(
            f"No ERA5 .nc files found under:\n{ERA5_PATH}"
        )

    # --------------------------------------------------------
    # 4. Exact daily preprocessing from training code
    # --------------------------------------------------------

    print("\nLoading + preprocessing ERA5...")

    daily = open_era5(
        files,
        cfg
    )

    print("\nDaily ERA5 prepared.")
    print(
        "Days available:",
        daily.sizes.get("time", 0)
    )

    # --------------------------------------------------------
    # 5. Build spherical mesh features
    # --------------------------------------------------------

    print("\nMapping ERA5 onto spherical mesh...")

    mesh_data = build_mesh_arrays(
        daily,
        cfg
    )

    features = mesh_data["features"]
    feature_names = mesh_data["feature_names"]

    print("Mesh feature shape:", features.shape)

    print(
        "Valid mesh nodes:",
        int(mesh_data["valid_mask"].sum())
    )

    print(
        "Features generated:",
        len(feature_names)
    )

    # --------------------------------------------------------
    # 6. IMPORTANT: match checkpoint feature order
    # --------------------------------------------------------

    checkpoint_features = predictor.feature_names

    print("\nChecking feature compatibility...")

    missing = [
        name
        for name in checkpoint_features
        if name not in feature_names
    ]

    if missing:
        raise ValueError(
            "ERA5 is missing checkpoint features:\n"
            + "\n".join(missing)
        )

    indices = [
        feature_names.index(name)
        for name in checkpoint_features
    ]

    features = features[:, :, indices]

    print(
        "Feature order matched checkpoint:",
        len(checkpoint_features),
        "features"
    )

    # --------------------------------------------------------
    # 7. Check mesh compatibility
    # --------------------------------------------------------

    if features.shape[1] != len(predictor.mesh_lat):
        raise ValueError(
            "Mesh node mismatch. "
            f"ERA5 preprocessing produced {features.shape[1]} nodes "
            f"but checkpoint expects {len(predictor.mesh_lat)}."
        )

    # --------------------------------------------------------
    # 8. Need one 5-day input window
    # --------------------------------------------------------

    if features.shape[0] < predictor.window:
        raise ValueError(
            f"Need at least {predictor.window} ERA5 days."
        )

    # For integration testing use latest available 5 days
    x = features[-predictor.window:]

    dates = daily.time.values[-predictor.window:]

    print("\nInput ERA5 dates:")

    for d in dates:
        print(" ", str(d)[:10])

    print("\nGNN input shape:", x.shape)

    # --------------------------------------------------------
    # 9. Run REAL GNN inference
    # --------------------------------------------------------

    print("\nRunning HeatwaveGNN inference...")

    prediction = predictor.predict(x)

    heatwave_probability = prediction[
        "heatwave_probability"
    ]

    predicted_tmax = prediction[
        "tmax_celsius"
    ]

    print(
        "Heatwave probability shape:",
        heatwave_probability.shape
    )

    print(
        "Predicted Tmax shape:",
        predicted_tmax.shape
    )

    # --------------------------------------------------------
    # 10. Track predicted heatwave footprints
    # --------------------------------------------------------

    print("\nRunning spherical heatwave tracker...")

    tracker = HeatwaveTracker(
        mesh_lat=predictor.mesh_lat,
        mesh_lon=predictor.mesh_lon,
        edge_index=predictor.edge_index_np,
        valid_mask=predictor.valid_mask_np,
        probability_threshold=0.5,
    )

    tracking = tracker.track(
        heatwave_probability
    )

    # --------------------------------------------------------
    # 11. Display results
    # --------------------------------------------------------

    print("\n==========================================")
    print(" HEATWAVE GNN RESULT")
    print("==========================================")

    print(
        "Input window:",
        predictor.window,
        "days"
    )

    print(
        "Prediction horizon:",
        predictor.horizon,
        "days"
    )

    print(
        "Heatwave tracks detected:",
        len(tracking["tracks"])
    )

    if not tracking["tracks"]:

        print(
            "\nNo heatwave footprint exceeded "
            "the current 0.5 classification threshold."
        )

    else:

        for track in tracking["tracks"]:

            print("\n--------------------------------")
            print("Event:", track["event_id"])

            print(
                "Day:",
                track["start_day"],
                "->",
                track["end_day"]
            )

            print(
                "Duration:",
                track["duration_days"],
                "days"
            )

            print(
                "Confidence:",
                round(
                    track["confidence"],
                    4
                )
            )

            print(
                "Bounding box:",
                track["bounding_box"]
            )

            print("Trajectory:")

            for point in track["trajectory"]:

                print(
                    " Day",
                    point["day"],
                    "| Lat:",
                    round(point["lat"], 3),
                    "| Lon:",
                    round(point["lon"], 3),
                    "| Probability:",
                    round(
                        point["probability"],
                        4
                    )
                )

    print("\n==========================================")
    print(" REAL ERA5 PIPELINE COMPLETED")
    print("==========================================")


if __name__ == "__main__":
    main()