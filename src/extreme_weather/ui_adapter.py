from __future__ import annotations


def _severity(probability: float) -> str:
    """
    UI display category based on model probability.
    These are presentation bands, not meteorological severity thresholds.
    """
    if probability >= 0.80:
        return "severe"
    if probability >= 0.60:
        return "moderate"
    return "low"


def heatwave_tracks_to_ui(
    tracks: list[dict],
    predicted_tmax=None,
) -> list[dict]:
    """
    Convert HeatwaveTracker output into the JSON structure
    expected by the React dashboard.

    predicted_tmax:
        Optional array [H, N] or [1, H, N].
        It is intentionally not used yet because a track footprint
        must first be associated correctly with the corresponding
        mesh-node Tmax values.
    """

    anomalies = []

    for track in tracks:
        trajectory_raw = track.get("trajectory", [])

        trajectory = []

        for point in trajectory_raw:
            lat = point.get("lat")
            lon = point.get("lon")

            if lat is not None and lon is not None:
                trajectory.append(
                    [float(lat), float(lon)]
                )

        # A map marker requires a geographic position.
        if not trajectory:
            continue

        latest_lat, latest_lon = trajectory[-1]

        probability = float(
            track.get("confidence", 0.0)
        )

        duration = int(
            track.get(
                "duration",
                len(trajectory),
            )
        )

        event_id = str(
            track.get(
                "event_id",
                f"HW{len(anomalies) + 1:03d}",
            )
        )

        anomaly = {
            # -------------------------
            # Event identity
            # -------------------------
            "id": event_id,
            "name": f"Tracked Heatwave {event_id}",
            "type": "Heatwave",

            # -------------------------
            # Model result
            # -------------------------
            "probability": probability,
            "confidence": round(probability * 100),
            "severity": _severity(probability),

            # Legacy fields retained only so the
            # existing React components do not break.
            "efi": probability,
            "sev": _severity(probability),

            # -------------------------
            # Temporal information
            # -------------------------
            "startDay": int(
                track.get("start_day", 1)
            ),
            "endDay": int(
                track.get(
                    "end_day",
                    duration,
                )
            ),
            "durationDays": duration,

            # -------------------------
            # Geographic information
            # -------------------------
            "lat": latest_lat,
            "lon": latest_lon,
            "track": trajectory,

            # -------------------------
            # Raw footprint information
            # -------------------------
            "footprints": track.get(
                "footprints",
                [],
            ),

            "boundingBox": track.get(
                "bounding_box",
                None,
            ),

            # -------------------------
            # Provenance
            # -------------------------
            "source": "HeatwaveGNN",
            "isModelOutput": True,
        }

        anomalies.append(anomaly)

    return anomalies


def dashboard_response(
    tracks: list[dict],
) -> dict:
    """
    Final API response consumed by App.jsx.
    """

    anomalies = heatwave_tracks_to_ui(tracks)

    return {
        "mode": "model",
        "model": "HeatwaveGNN",
        "anomalies": anomalies,
        "count": len(anomalies),
    }


if __name__ == "__main__":
    # Synthetic TRACKER output only for testing the adapter.
    synthetic_tracks = [
        {
            "event_id": "HW001",
            "event_type": "heatwave",
            "start_day": 1,
            "end_day": 4,
            "duration": 4,
            "confidence": 0.87,
            "trajectory": [
                {"day": 1, "lat": 26.8, "lon": 74.9},
                {"day": 2, "lat": 27.0, "lon": 75.2},
                {"day": 3, "lat": 27.2, "lon": 75.5},
                {"day": 4, "lat": 27.4, "lon": 75.8},
            ],
            "footprints": [],
            "bounding_box": {
                "lat_min": 26.5,
                "lat_max": 27.8,
                "lon_min": 74.5,
                "lon_max": 76.1,
            },
        }
    ]

    result = dashboard_response(synthetic_tracks)

    print("\n================================")
    print(" TRACKER -> UI ADAPTER TEST")
    print("================================")

    print("Mode:", result["mode"])
    print("Events:", result["count"])

    for event in result["anomalies"]:
        print("\nID          :", event["id"])
        print("Type        :", event["type"])
        print("Probability :", event["probability"])
        print("Confidence  :", event["confidence"])
        print("Duration    :", event["durationDays"])
        print("Location    :", event["lat"], event["lon"])
        print("Track       :", event["track"])

    print("\nUI adapter test PASSED.")