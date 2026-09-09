import numpy as np


class HeatwaveTracker:
    """
    Post-processing tracker for HeatwaveGNN predictions.

    Input:
        heatwave_probability : [H, N]
        H = forecast days
        N = mesh nodes

    Produces:
        - daily heatwave footprints
        - centroid for each footprint
        - same-event association across days
        - trajectory
        - temporal/spatial bounding box
    """

    def __init__(
        self,
        mesh_lat,
        mesh_lon,
        edge_index,
        valid_mask,
        probability_threshold=0.5,
        min_nodes=1,
    ):
        self.mesh_lat = np.asarray(mesh_lat)
        self.mesh_lon = np.asarray(mesh_lon)
        self.edge_index = np.asarray(edge_index)
        self.valid_mask = np.asarray(valid_mask, dtype=bool)

        # 0.5 matches the current model's classification threshold.
        # Later replace with a validation-selected threshold.
        self.threshold = probability_threshold
        self.min_nodes = min_nodes

        self.num_nodes = len(self.mesh_lat)

        self.neighbours = self._build_adjacency()


    # ============================================================
    # GRAPH ADJACENCY
    # ============================================================

    def _build_adjacency(self):

        adjacency = [
            set() for _ in range(self.num_nodes)
        ]

        src = self.edge_index[0]
        dst = self.edge_index[1]

        for u, v in zip(src, dst):
            u = int(u)
            v = int(v)

            adjacency[u].add(v)
            adjacency[v].add(u)

        return adjacency


    # ============================================================
    # CONNECTED COMPONENTS
    # ============================================================

    def _connected_components(self, active):

        visited = np.zeros(
            self.num_nodes,
            dtype=bool
        )

        components = []

        active_set = set(
            np.where(active)[0].tolist()
        )

        for start in active_set:

            if visited[start]:
                continue

            stack = [start]
            visited[start] = True

            component = []

            while stack:

                node = stack.pop()
                component.append(node)

                for neighbour in self.neighbours[node]:

                    if (
                        neighbour in active_set
                        and not visited[neighbour]
                    ):
                        visited[neighbour] = True
                        stack.append(neighbour)

            if len(component) >= self.min_nodes:
                components.append(
                    np.asarray(component, dtype=int)
                )

        return components


    # ============================================================
    # SPHERICAL CENTROID
    # ============================================================

    def _centroid(self, nodes, probabilities):

        lat = np.radians(
            self.mesh_lat[nodes]
        )

        lon = np.radians(
            self.mesh_lon[nodes]
        )

        weights = probabilities[nodes]

        if np.sum(weights) <= 0:
            weights = np.ones_like(weights)

        x = np.cos(lat) * np.cos(lon)
        y = np.cos(lat) * np.sin(lon)
        z = np.sin(lat)

        x = np.average(x, weights=weights)
        y = np.average(y, weights=weights)
        z = np.average(z, weights=weights)

        norm = np.sqrt(
            x * x + y * y + z * z
        )

        if norm > 0:
            x /= norm
            y /= norm
            z /= norm

        centroid_lat = np.degrees(
            np.arctan2(
                z,
                np.sqrt(x * x + y * y)
            )
        )

        centroid_lon = np.degrees(
            np.arctan2(y, x)
        )

        return (
            float(centroid_lat),
            float(centroid_lon)
        )


    # ============================================================
    # DAILY FOOTPRINT
    # ============================================================

    def _make_footprint(
        self,
        day,
        nodes,
        probabilities,
    ):

        centroid_lat, centroid_lon = (
            self._centroid(
                nodes,
                probabilities
            )
        )

        return {
            "day": int(day),

            "nodes": nodes.tolist(),

            "centroid": {
                "lat": centroid_lat,
                "lon": centroid_lon,
            },

            "bbox": {
                "lat_min": float(
                    np.min(self.mesh_lat[nodes])
                ),
                "lat_max": float(
                    np.max(self.mesh_lat[nodes])
                ),
                "lon_min": float(
                    np.min(self.mesh_lon[nodes])
                ),
                "lon_max": float(
                    np.max(self.mesh_lon[nodes])
                ),
            },

            "mean_probability": float(
                np.mean(probabilities[nodes])
            ),

            "max_probability": float(
                np.max(probabilities[nodes])
            ),

            "node_count": int(len(nodes)),
        }


    # ============================================================
    # OVERLAP BETWEEN TWO FOOTPRINTS
    # ============================================================

    @staticmethod
    def _jaccard(a, b):

        a = set(a["nodes"])
        b = set(b["nodes"])

        union = len(a | b)

        if union == 0:
            return 0.0

        return len(a & b) / union


    # ============================================================
    # MATCH FOOTPRINTS BETWEEN DAYS
    # ============================================================

    def _associate(
        self,
        daily_footprints,
    ):

        tracks = []
        next_track_id = 1

        previous = []

        for day_footprints in daily_footprints:

            current = []

            used_tracks = set()

            for footprint in day_footprints:

                best_track = None
                best_overlap = 0.0

                for previous_item in previous:

                    track_index = previous_item[
                        "track_index"
                    ]

                    if track_index in used_tracks:
                        continue

                    overlap = self._jaccard(
                        previous_item["footprint"],
                        footprint,
                    )

                    if overlap > best_overlap:
                        best_overlap = overlap
                        best_track = track_index

                # Same event if there is actual mesh overlap.
                if (
                    best_track is not None
                    and best_overlap > 0.0
                ):

                    tracks[best_track][
                        "footprints"
                    ].append(footprint)

                    track_index = best_track

                else:

                    tracks.append({
                        "event_id":
                            f"HW{next_track_id:03d}",

                        "event_type":
                            "heatwave",

                        "footprints": [
                            footprint
                        ],
                    })

                    track_index = (
                        len(tracks) - 1
                    )

                    next_track_id += 1

                used_tracks.add(track_index)

                current.append({
                    "track_index": track_index,
                    "footprint": footprint,
                })

            previous = current

        return tracks


    # ============================================================
    # FINAL TRACK SUMMARY
    # ============================================================

    def _summarise_track(self, track):

        footprints = track["footprints"]

        start_day = footprints[0]["day"]
        end_day = footprints[-1]["day"]

        trajectory = []

        all_lat_min = []
        all_lat_max = []
        all_lon_min = []
        all_lon_max = []

        probabilities = []

        for fp in footprints:

            trajectory.append({
                "day": fp["day"],
                "lat": fp["centroid"]["lat"],
                "lon": fp["centroid"]["lon"],
                "probability":
                    fp["mean_probability"],
            })

            all_lat_min.append(
                fp["bbox"]["lat_min"]
            )
            all_lat_max.append(
                fp["bbox"]["lat_max"]
            )
            all_lon_min.append(
                fp["bbox"]["lon_min"]
            )
            all_lon_max.append(
                fp["bbox"]["lon_max"]
            )

            probabilities.append(
                fp["mean_probability"]
            )

        return {
            "event_id": track["event_id"],
            "event_type": "heatwave",

            "start_day": int(start_day),
            "end_day": int(end_day),

            "duration_days":
                int(end_day - start_day + 1),

            "trajectory": trajectory,

            "bounding_box": {
                "lat_min":
                    float(min(all_lat_min)),
                "lat_max":
                    float(max(all_lat_max)),
                "lon_min":
                    float(min(all_lon_min)),
                "lon_max":
                    float(max(all_lon_max)),
            },

            "confidence": float(
                np.mean(probabilities)
            ),

            "footprints": footprints,
        }


    # ============================================================
    # PUBLIC TRACK FUNCTION
    # ============================================================

    def track(self, heatwave_probability):

        probs = np.asarray(
            heatwave_probability,
            dtype=np.float32,
        )

        # Allow [1,H,N]
        if probs.ndim == 3:

            if probs.shape[0] != 1:
                raise ValueError(
                    "Tracker currently expects "
                    "one forecast sequence at a time."
                )

            probs = probs[0]

        if probs.ndim != 2:
            raise ValueError(
                "Expected heatwave probability "
                "[horizon,nodes]."
            )

        H, N = probs.shape

        if N != self.num_nodes:
            raise ValueError(
                f"Expected {self.num_nodes} nodes, "
                f"received {N}."
            )

        daily_footprints = []

        for day_index in range(H):

            day_probs = probs[day_index]

            active = (
                (day_probs >= self.threshold)
                & self.valid_mask
            )

            components = (
                self._connected_components(active)
            )

            footprints = []

            for component in components:

                footprints.append(
                    self._make_footprint(
                        day=day_index + 1,
                        nodes=component,
                        probabilities=day_probs,
                    )
                )

            daily_footprints.append(
                footprints
            )

        raw_tracks = self._associate(
            daily_footprints
        )

        tracks = [
            self._summarise_track(track)
            for track in raw_tracks
        ]

        return {
            "threshold": self.threshold,
            "forecast_days": H,
            "daily_footprints":
                daily_footprints,
            "tracks": tracks,
        }

if __name__ == "__main__":

    from .heatwave_inference import (
        HeatwavePredictor
    )

    import os

    project_root = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
        )
    )

    checkpoint_path = os.path.join(
        project_root,
        "checkpoints",
        "heatwave_schema_test.pt",
    )

    predictor = HeatwavePredictor(
        checkpoint_path
    )

    tracker = HeatwaveTracker(
        mesh_lat=predictor.mesh_lat,
        mesh_lon=predictor.mesh_lon,
        edge_index=predictor.edge_index_np,
        valid_mask=predictor.valid_mask_np,
    )

    # ----------------------------------------
    # Deterministic synthetic probabilities
    # ONLY to test tracker wiring.
    # ----------------------------------------

    H = predictor.horizon
    N = len(predictor.mesh_lat)

    probs = np.zeros(
        (H, N),
        dtype=np.float32
    )

    valid_nodes = np.where(
        predictor.valid_mask_np
    )[0]

    # Activate a few valid nodes for every day.
    # This is NOT model inference.
    selected = valid_nodes[:min(5, len(valid_nodes))]

    for day in range(H):
        probs[day, selected] = 0.90

    result = tracker.track(probs)

    print("\n----------------------------")
    print("HEATWAVE TRACKER TEST")
    print("----------------------------")

    print(
        "Forecast days:",
        result["forecast_days"]
    )

    print(
        "Tracks detected:",
        len(result["tracks"])
    )

    for track in result["tracks"]:
        print(
            track["event_id"],
            "Day",
            track["start_day"],
            "->",
            track["end_day"],
            "| duration:",
            track["duration_days"],
            "| confidence:",
            round(track["confidence"], 3),
        )

    print(
        "\nHeatwave tracker integration test PASSED."
    )