import json
from pathlib import Path
from src.extreme_weather.coldwave_tracker import ColdwaveTracker

import numpy as np

from coldwave_gnn_training import (
    Config,
    resolve_nc_files,
    open_era5,
    build_mesh_arrays,
)

from src.extreme_weather.coldwave_inference import ColdwavePredictor


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

ERA5_PATH = Path(
    r"C:\Users\Varshitha S B\Documents\ERA5_DOWNLOAD\ERA5_INDIA"
)

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "checkpoints"
    / "coldwave_final.pt"
)


# ============================================================
# FEATURE ALIGNMENT
# ============================================================

def align_features_to_checkpoint(
    features,
    generated_feature_names,
    checkpoint_feature_names,
):
    """
    Reorder ERA5-generated features so that they exactly match
    the feature order used while training the checkpoint.

    features:
        [time, node, feature]
    """

    generated_feature_names = list(generated_feature_names)
    checkpoint_feature_names = list(checkpoint_feature_names)

    print("\nChecking feature compatibility...")

    missing = [
        name
        for name in checkpoint_feature_names
        if name not in generated_feature_names
    ]

    if missing:
        raise ValueError(
            "ERA5 preprocessing is missing checkpoint features:\n"
            + "\n".join(missing)
        )

    indices = [
        generated_feature_names.index(name)
        for name in checkpoint_feature_names
    ]

    aligned = features[:, :, indices]

    print(
        f"Feature alignment successful: "
        f"{aligned.shape[-1]} checkpoint features"
    )

    return aligned


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():

    print("=" * 65)
    print("REAL ERA5 -> COLDWAVE GNN PIPELINE")
    print("=" * 65)

    # --------------------------------------------------------
    # 1. Load trained model first
    # --------------------------------------------------------

    predictor = ColdwavePredictor(
        CHECKPOINT_PATH
    )

    # --------------------------------------------------------
    # 2. Check ERA5 directory
    # --------------------------------------------------------

    if not ERA5_PATH.exists():
        raise FileNotFoundError(
            f"ERA5 directory does not exist:\n{ERA5_PATH}"
        )

    print("\nERA5 directory:")
    print(ERA5_PATH)

    files = resolve_nc_files(
        str(ERA5_PATH)
    )

    print(
        f"\nFound {len(files)} NetCDF file(s)."
    )

    if not files:
        raise FileNotFoundError(
            "No .nc ERA5 files found."
        )

    # --------------------------------------------------------
    # 3. Build preprocessing configuration
    # --------------------------------------------------------

    # Start with the training defaults, then override the
    # parameters that MUST match this checkpoint.
    cfg = Config()

    cfg.window = predictor.window
    cfg.horizon = predictor.horizon

    cfg.mesh_subdivisions = int(
        predictor.config["mesh_subdivisions"]
    )

    cfg.target_resolution_deg = float(
        predictor.config["target_resolution_deg"]
    )

    cfg.pressure_levels = list(
        predictor.config["pressure_levels"]
    )

    cfg.use_single_level = bool(
        predictor.config["use_single_level"]
    )

    cfg.use_pressure_level = bool(
        predictor.config["use_pressure_level"]
    )

    cfg.fast_mesh_sampling = bool(
        predictor.config.get(
            "fast_mesh_sampling",
            True,
        )
    )

    # --------------------------------------------------------
    # 4. ERA5 -> daily fields
    # --------------------------------------------------------

    print("\nLoading and aggregating real ERA5...")

    daily = open_era5(
        files,
        cfg,
    )

    print("\nERA5 DAILY DATA READY")
    print(
        "Daily time steps:",
        daily.sizes.get("time", 0),
    )

    if daily.sizes.get("time", 0) < predictor.window:
        raise ValueError(
            f"Need at least {predictor.window} daily steps."
        )

    # --------------------------------------------------------
    # 5. Daily fields -> spherical mesh
    # --------------------------------------------------------

    print("\nSampling ERA5 onto spherical mesh...")

    mesh_data = build_mesh_arrays(
        daily,
        cfg,
    )

    features = np.asarray(
        mesh_data["features"],
        dtype=np.float32,
    )

    generated_feature_names = list(
        mesh_data["feature_names"]
    )

    print("\nMesh feature array:")
    print(features.shape)

    print(
        "Generated features:",
        len(generated_feature_names),
    )

    print(
        "Valid mesh nodes:",
        int(
            np.asarray(
                mesh_data["valid_mask"]
            ).sum()
        ),
    )

    # --------------------------------------------------------
    # 6. Verify geometry matches checkpoint
    # --------------------------------------------------------

    if features.shape[1] != len(predictor.mesh_lat):
        raise ValueError(
            "Mesh node mismatch: "
            f"ERA5 preprocessing produced {features.shape[1]}, "
            f"checkpoint expects {len(predictor.mesh_lat)}."
        )

    generated_valid = np.asarray(
        mesh_data["valid_mask"],
        dtype=bool,
    )

    if generated_valid.shape != predictor.valid_mask_np.shape:
        raise ValueError(
            "Valid-mask shape does not match checkpoint."
        )

    print("\nMesh geometry compatible with checkpoint.")

    # --------------------------------------------------------
    # 7. Reorder features exactly as checkpoint expects
    # --------------------------------------------------------

    features = align_features_to_checkpoint(
        features,
        generated_feature_names,
        predictor.feature_names,
    )

    if features.shape[-1] != predictor.input_dim:
        raise ValueError(
            f"Expected {predictor.input_dim} features, "
            f"received {features.shape[-1]}."
        )

    # --------------------------------------------------------
    # 8. Select last 5 REAL ERA5 days
    # --------------------------------------------------------

    input_features = features[
        -predictor.window:
    ]

    input_dates = daily.time.values[
        -predictor.window:
    ]

    print("\n===== MODEL INPUT =====")

    print("Input shape:")
    print(input_features.shape)

    print("\nERA5 input dates:")

    for date in input_dates:
        print(
            np.datetime_as_string(
                date,
                unit="D",
            )
        )

    # --------------------------------------------------------
    # 9. REAL MODEL INFERENCE
    # --------------------------------------------------------

    print("\nRunning ColdwaveGNN inference...")

    result = predictor.predict(
        input_features
    )

    # ============================================================
    # SPATIO-TEMPORAL TRACKING
    # ============================================================

    tracker = ColdwaveTracker(
        mesh_lat=predictor.mesh_lat,
        mesh_lon=predictor.mesh_lon,
        edge_index=predictor.edge_index_np,
        valid_mask=predictor.valid_mask_np,
        probability_threshold=0.5,
    )

    tracking_result = tracker.track(
        result["coldwave_probability"]
    )

    probabilities = result[
        "coldwave_probability"
    ]

    tmin = result[
        "tmin_celsius"
    ]

    print("\n===== MODEL OUTPUT =====")

    print(
        "Coldwave probability shape:",
        probabilities.shape,
    )

    print(
        "Predicted Tmin shape:",
        tmin.shape,
    )

    # --------------------------------------------------------
    # 10. Examine only valid India nodes
    # --------------------------------------------------------

    valid = predictor.valid_mask_np

    valid_probabilities = probabilities[
        :, :, valid
    ]

    valid_tmin = tmin[
        :, :, valid
    ]

    print("\n===== VALID INDIA NODES =====")

    print(
        "Probability minimum:",
        float(
            np.nanmin(valid_probabilities)
        ),
    )

    print(
        "Probability maximum:",
        float(
            np.nanmax(valid_probabilities)
        ),
    )

    print(
        "Probability mean:",
        float(
            np.nanmean(valid_probabilities)
        ),
    )

    print(
        "Predicted Tmin minimum:",
        float(
            np.nanmin(valid_tmin)
        ),
        "C",
    )

    print(
        "Predicted Tmin maximum:",
        float(
            np.nanmax(valid_tmin)
        ),
        "C",
    )

    # --------------------------------------------------------
    # 11. Apply checkpoint's classification threshold
    # --------------------------------------------------------

    threshold = 0.5

    detected = (
        probabilities >= threshold
    )

    detected[:, :, ~valid] = False

    print("\n===== COLDWAVE DETECTION =====")

    for day in range(predictor.horizon):

        count = int(
            detected[0, day].sum()
        )

        max_probability = float(
            probabilities[
                0,
                day,
                valid,
            ].max()
        )

        min_temperature = float(
            tmin[
                0,
                day,
                valid,
            ].min()
        )

        print(
            f"Forecast Day {day + 1}: "
            f"{count} coldwave node(s) | "
            f"max P={max_probability:.4f} | "
            f"min Tmin={min_temperature:.2f} C"
        )

    # --------------------------------------------------------
    # 12. Final verification
    # --------------------------------------------------------

    if not np.isfinite(
        valid_probabilities
    ).all():
        raise ValueError(
            "Non-finite probabilities detected."
        )

    if not np.isfinite(
        valid_tmin
    ).all():
        raise ValueError(
            "Non-finite Tmin predictions detected."
        )

    print("\n" + "=" * 65)

    print("\n===== REAL SPATIO-TEMPORAL COLDWAVE TRACKS =====")

    tracks = tracking_result["tracks"]

    print("Extreme Type : COLD WAVE")
    print("Tracks Found :", len(tracks))

    if len(tracks) == 0:
        print("No persistent coldwave tracks detected.")

    for track in tracks:

        print("\n" + "-" * 55)

        print("Event ID   :", track["event_id"])

        print(
            "Forecast   : Day",
            track["start_day"],
            "-> Day",
            track["end_day"],
        )

        print(
            "Duration   :",
            track["duration_days"],
            "day(s)",
        )

        print(
            "Confidence :",
            f"{track['confidence']:.4f}",
        )

        print("\nMovement / trajectory:")

        for point in track["trajectory"]:

            print(
                f"  Day {point['day']} -> "
                f"Lat {point['lat']:.2f}, "
                f"Lon {point['lon']:.2f}, "
                f"P={point['probability']:.4f}"
            )

        bbox = track["bounding_box"]

        print("\nOverall geographic footprint:")

        print(
            f"  Latitude  : "
            f"{bbox['lat_min']:.2f} "
            f"to {bbox['lat_max']:.2f}"
        )

        print(
            f"  Longitude : "
            f"{bbox['lon_min']:.2f} "
            f"to {bbox['lon_max']:.2f}"
        )

        # ============================================================
        # SAVE REAL TRACKS FOR BACKEND / UI
        # ============================================================

        output_dir = PROJECT_ROOT / "outputs"
        output_dir.mkdir(exist_ok=True)

        output_file = output_dir / "coldwave_tracks.json"

        ui_events = []

        for track in tracking_result["tracks"]:

            latest = track["trajectory"][-1]

            ui_events.append({
                "id": track["event_id"],
                "name": f"Cold Wave {track['event_id']}",
                "type": "Cold Wave",

                "probability": float(track["confidence"]),
                "confidence": float(track["confidence"]),

                # Keep aliases temporarily because the current React UI
                # still uses these old field names in a few places.
                "efi": float(track["confidence"]),
                "sev": (
                    "Severe"
                    if track["confidence"] >= 0.8
                    else "Moderate"
                    if track["confidence"] >= 0.6
                    else "Low"
                ),

                "severity": (
                    "Severe"
                    if track["confidence"] >= 0.8
                    else "Moderate"
                    if track["confidence"] >= 0.6
                    else "Low"
                ),

                "lat": float(latest["lat"]),
                "lon": float(latest["lon"]),

                "startDay": int(track["start_day"]),
                "endDay": int(track["end_day"]),
                "durationDays": int(track["duration_days"]),

                "track": [
                    {
                        "day": int(point["day"]),
                        "lat": float(point["lat"]),
                        "lon": float(point["lon"]),
                        "probability": float(point["probability"]),
                    }
                    for point in track["trajectory"]
                ],

                "boundingBox": track["bounding_box"],
                "footprints": track["footprints"],

                "source": "ColdwaveGNN",
                "isModelOutput": True,
            })


        payload = {
            "mode": "model",
            "model": "ColdwaveGNN",
            "region": "India",
            "forecastDays": predictor.horizon,
            "count": len(ui_events),
            "anomalies": ui_events,
        }


        with open(
            output_file,
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                payload,
                f,
                indent=2,
            )


        print("\n===== UI OUTPUT =====")
        print("Saved:", output_file)
        print("Events:", len(ui_events))
                
        print(
            "REAL ERA5 -> COLDWAVE GNN PIPELINE COMPLETED"
        )

        print("=" * 65)


if __name__ == "__main__":
    main()