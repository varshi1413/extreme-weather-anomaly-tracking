import os
import sys
import numpy as np
import torch


# ------------------------------------------------------------
# Make project root importable
# ------------------------------------------------------------

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(
    os.path.join(CURRENT_DIR, "..", "..")
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# Architecture already written by your teammate
from Heatwave_GNN_Training_fixed import (
    HeatwaveGNN,
    build_edge_features,
)


class HeatwavePredictor:
    """
    Loads a trained HeatwaveGNN checkpoint and performs inference.

    Input:
        ERA5 features already sampled onto the same mesh used during training.

        Shape:
            [window, nodes, features]

        or

            [batch, window, nodes, features]

    Output:
        Future Tmax and heatwave probability for every mesh node.
    """

    def __init__(self, checkpoint_path, device=None):

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint_path}"
            )

        # --------------------------------------------------------
        # Device
        # --------------------------------------------------------

        if device is None:
            device = (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )

        self.device = torch.device(device)

        print(f"Loading Heatwave GNN on: {self.device}")

        # --------------------------------------------------------
        # Load checkpoint
        # --------------------------------------------------------

        self.checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device,
            weights_only=False,
        )

        checkpoint = self.checkpoint
        cfg = checkpoint["config"]

        # --------------------------------------------------------
        # Metadata
        # --------------------------------------------------------

        self.feature_names = list(
            checkpoint["feature_names"]
        )

        self.mesh_lat = np.asarray(
            checkpoint["mesh_lat"],
            dtype=np.float32,
        )

        self.mesh_lon = np.asarray(
            checkpoint["mesh_lon"],
            dtype=np.float32,
        )

        self.valid_mask_np = np.asarray(
            checkpoint["valid_mask"],
            dtype=bool,
        )

        self.edge_index_np = np.asarray(
            checkpoint["edge_index"],
            dtype=np.int64,
        )

        self.window = int(cfg["window"])
        self.horizon = int(cfg["horizon"])

        self.input_dim = int(
            checkpoint["input_dim"]
        )

        # --------------------------------------------------------
        # Normalization information
        # --------------------------------------------------------

        self.x_mean = np.asarray(
            checkpoint["scaler_x_mean"],
            dtype=np.float32,
        )

        self.x_std = np.asarray(
            checkpoint["scaler_x_std"],
            dtype=np.float32,
        )

        self.y_mean = np.asarray(
            checkpoint["scaler_y_mean"],
            dtype=np.float32,
        )

        self.y_std = np.asarray(
            checkpoint["scaler_y_std"],
            dtype=np.float32,
        )

        # --------------------------------------------------------
        # Graph tensors
        # --------------------------------------------------------

        self.edge_index = torch.from_numpy(
            self.edge_index_np
        ).long().to(self.device)

        edge_features_np = build_edge_features(
            self.edge_index_np,
            self.mesh_lat,
            self.mesh_lon,
        )

        self.edge_features = torch.from_numpy(
            edge_features_np
        ).float().to(self.device)

        self.valid_mask = torch.from_numpy(
            self.valid_mask_np
        ).bool().to(self.device)

        # --------------------------------------------------------
        # Rebuild exact architecture used during training
        # --------------------------------------------------------

        self.model = HeatwaveGNN(
            input_dim=self.input_dim,
            num_nodes=len(self.mesh_lat),
            edge_input_dim=int(
                checkpoint.get(
                    "edge_input_dim",
                    5,
                )
            ),
            hidden_dim=int(cfg["hidden_dim"]),
            edge_dim=int(cfg["edge_dim"]),
            processor_steps=int(
                cfg["processor_steps"]
            ),
            temporal_dim=int(
                cfg["temporal_dim"]
            ),
            temporal_heads=int(
                cfg["temporal_heads"]
            ),
            temporal_layers=int(
                cfg["temporal_layers"]
            ),
            dropout=float(cfg["dropout"]),
            horizon=self.horizon,
        ).to(self.device)

        # --------------------------------------------------------
        # Load trained weights
        # --------------------------------------------------------

        self.model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        self.model.eval()

        print("Heatwave checkpoint loaded successfully.")
        print(f"Nodes       : {len(self.mesh_lat)}")
        print(f"Features    : {self.input_dim}")
        print(f"Window      : {self.window} days")
        print(f"Horizon     : {self.horizon} days")
        print(f"Valid nodes : {self.valid_mask_np.sum()}")

    # ============================================================
    # NORMALIZATION
    # ============================================================

    def _normalise(self, x):

        return (
            x - self.x_mean
        ) / self.x_std

    # ============================================================
    # PREDICTION
    # ============================================================

    def predict(self, features):
        """
        features:

        [window, nodes, features]

        OR

        [batch, window, nodes, features]
        """

        features = np.asarray(
            features,
            dtype=np.float32,
        )

        # Add batch dimension if necessary
        if features.ndim == 3:
            features = features[None, ...]

        if features.ndim != 4:
            raise ValueError(
                "Expected input shape "
                "[window,nodes,features] or "
                "[batch,window,nodes,features]. "
                f"Received {features.shape}"
            )

        B, T, N, F = features.shape

        # --------------------------------------------------------
        # Shape verification
        # --------------------------------------------------------

        if T != self.window:
            raise ValueError(
                f"Model expects {self.window} input days, "
                f"but received {T}."
            )

        if N != len(self.mesh_lat):
            raise ValueError(
                f"Model expects {len(self.mesh_lat)} mesh nodes, "
                f"but received {N}."
            )

        if F != self.input_dim:
            raise ValueError(
                f"Model expects {self.input_dim} features, "
                f"but received {F}."
            )

        # --------------------------------------------------------
        # Normalize exactly like training
        # --------------------------------------------------------

        x = self._normalise(features)

        x = np.nan_to_num(
            x,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        # Invalid nodes were zeroed during training
        x[:, :, ~self.valid_mask_np, :] = 0.0

        x = torch.from_numpy(
            x
        ).float().to(self.device)

        # --------------------------------------------------------
        # GNN inference
        # --------------------------------------------------------

        with torch.no_grad():

            tmax_scaled, heat_logits = self.model(
                x,
                self.edge_index,
                self.edge_features,
                self.valid_mask,
            )

            heat_probs = torch.sigmoid(
                heat_logits
            )

        # --------------------------------------------------------
        # Back to numpy
        # --------------------------------------------------------

        tmax_scaled = (
            tmax_scaled
            .detach()
            .cpu()
            .numpy()
        )

        heat_logits = (
            heat_logits
            .detach()
            .cpu()
            .numpy()
        )

        heat_probs = (
            heat_probs
            .detach()
            .cpu()
            .numpy()
        )

        # --------------------------------------------------------
        # Restore Tmax to degrees Celsius
        # --------------------------------------------------------

        tmax_celsius = (
            tmax_scaled * float(self.y_std[0])
            + float(self.y_mean[0])
        )

        # Invalid mesh nodes must never become detections
        heat_probs[:, :, ~self.valid_mask_np] = 0.0

        tmax_celsius[:, :, ~self.valid_mask_np] = np.nan

        return {
            "tmax_celsius": tmax_celsius,
            "heatwave_logits": heat_logits,
            "heatwave_probability": heat_probs,

            "mesh_lat": self.mesh_lat,
            "mesh_lon": self.mesh_lon,
            "valid_mask": self.valid_mask_np,

            "feature_names": self.feature_names,

            "window": self.window,
            "horizon": self.horizon,
        }


# ================================================================
# SIMPLE CHECKPOINT TEST
# ================================================================

if __name__ == "__main__":

    checkpoint_path = os.path.join(
        PROJECT_ROOT,
        "checkpoints",
        "heatwave_schema_test.pt",
    )

    predictor = HeatwavePredictor(
        checkpoint_path
    )

    print("\n--------------------------------")
    print("CHECKPOINT INFORMATION")
    print("--------------------------------")

    print(
        "Model type:",
        predictor.checkpoint.get(
            "model_type"
        )
    )

    print(
        "Number of mesh nodes:",
        len(predictor.mesh_lat)
    )

    print(
        "Input features:",
        len(predictor.feature_names)
    )

    print(
        "Feature names:"
    )

    for i, name in enumerate(
        predictor.feature_names,
        start=1,
    ):
        print(f"{i:02d}. {name}")

    print("\nCheckpoint integration test PASSED.")