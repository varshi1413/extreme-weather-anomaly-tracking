from pathlib import Path

import numpy as np
import torch

from coldwave_gnn_training import ColdwaveGNN, build_edge_features


class ColdwavePredictor:
    """
    Inference wrapper for the trained ColdwaveGNN checkpoint.

    Input:
        features -> [window, nodes, features]
                 or [batch, window, nodes, features]

    Output:
        future Tmin and coldwave probabilities for each forecast day/node.
    """

    def __init__(self, checkpoint_path, device=None):
        self.checkpoint_path = Path(checkpoint_path)

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"Coldwave checkpoint not found: {self.checkpoint_path}"
            )

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = torch.device(device)

        print(f"Loading ColdwaveGNN on {self.device}...")
        print(f"Checkpoint: {self.checkpoint_path}")

        checkpoint = torch.load(
            self.checkpoint_path,
            map_location=self.device,
            weights_only=False,
        )

        # ---------------------------------------------------------
        # Checkpoint metadata
        # ---------------------------------------------------------
        self.config = checkpoint["config"]

        self.mesh_lat = np.asarray(
            checkpoint["mesh_lat"], dtype=np.float32
        )

        self.mesh_lon = np.asarray(
            checkpoint["mesh_lon"], dtype=np.float32
        )

        self.edge_index_np = np.asarray(
            checkpoint["edge_index"], dtype=np.int64
        )

        self.valid_mask_np = np.asarray(
            checkpoint["valid_mask"], dtype=bool
        )

        self.feature_names = list(checkpoint["feature_names"])

        self.input_dim = int(checkpoint["input_dim"])
        self.edge_input_dim = int(checkpoint["edge_input_dim"])

        self.window = int(self.config["window"])
        self.horizon = int(self.config["horizon"])

        # ---------------------------------------------------------
        # Saved normalization parameters
        # ---------------------------------------------------------
        self.scaler_x_mean = np.asarray(
            checkpoint["scaler_x_mean"], dtype=np.float32
        )

        self.scaler_x_std = np.asarray(
            checkpoint["scaler_x_std"], dtype=np.float32
        )

        self.scaler_y_mean = np.asarray(
            checkpoint["scaler_y_mean"], dtype=np.float32
        )

        self.scaler_y_std = np.asarray(
            checkpoint["scaler_y_std"], dtype=np.float32
        )

        # Coldwave climatology
        self.climatology_mean = np.asarray(
            checkpoint["coldwave_climatology_mean"],
            dtype=np.float32,
        )

        self.climatology_p05 = np.asarray(
            checkpoint["coldwave_climatology_p05"],
            dtype=np.float32,
        )

        # ---------------------------------------------------------
        # Reconstruct exact model
        # ---------------------------------------------------------
        self.model = ColdwaveGNN(
            input_dim=self.input_dim,
            num_nodes=len(self.mesh_lat),
            edge_input_dim=self.edge_input_dim,
            hidden_dim=int(self.config["hidden_dim"]),
            edge_dim=int(self.config["edge_dim"]),
            processor_steps=int(self.config["processor_steps"]),
            temporal_dim=int(self.config["temporal_dim"]),
            temporal_heads=int(self.config["temporal_heads"]),
            temporal_layers=int(self.config["temporal_layers"]),
            dropout=float(self.config["dropout"]),
            horizon=self.horizon,
        ).to(self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        # ---------------------------------------------------------
        # Graph tensors
        # ---------------------------------------------------------
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

        print("ColdwaveGNN loaded successfully.")
        print(f"Mesh nodes       : {len(self.mesh_lat)}")
        print(f"Valid nodes      : {int(self.valid_mask_np.sum())}")
        print(f"Input features   : {self.input_dim}")
        print(f"Input window     : {self.window} days")
        print(f"Forecast horizon : {self.horizon} days")

    # -------------------------------------------------------------
    # Normalization
    # -------------------------------------------------------------
    def normalize_features(self, features):
        features = np.asarray(features, dtype=np.float32)

        return (
            features - self.scaler_x_mean
        ) / self.scaler_x_std

    def denormalize_tmin(self, prediction):
        return (
            prediction * self.scaler_y_std
            + self.scaler_y_mean
        )

    # -------------------------------------------------------------
    # Prediction
    # -------------------------------------------------------------
    @torch.no_grad()
    def predict(self, features, already_normalized=False):

        x = np.asarray(features, dtype=np.float32)

        # Accept [T,N,F]
        if x.ndim == 3:
            x = x[None, ...]

        if x.ndim != 4:
            raise ValueError(
                "Expected features with shape "
                "[T,N,F] or [B,T,N,F]. "
                f"Received {x.shape}"
            )

        B, T, N, F = x.shape

        if T != self.window:
            raise ValueError(
                f"Expected {self.window} input days, got {T}"
            )

        if N != len(self.mesh_lat):
            raise ValueError(
                f"Expected {len(self.mesh_lat)} mesh nodes, got {N}"
            )

        if F != self.input_dim:
            raise ValueError(
                f"Expected {self.input_dim} features, got {F}"
            )

        # Invalid global mesh nodes do not contain ERA5 observations.
        # Set them to zero after normalization so NaNs never enter model.
        if not already_normalized:
            x = self.normalize_features(x)

        x = np.nan_to_num(
            x,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).astype(np.float32)

        tensor = torch.from_numpy(x).float().to(self.device)

        tmin_scaled, cold_logits = self.model(
            tensor,
            self.edge_index,
            self.edge_features,
            self.valid_mask,
        )

        cold_probability = torch.sigmoid(cold_logits)

        tmin_scaled = tmin_scaled.cpu().numpy()
        cold_logits = cold_logits.cpu().numpy()
        cold_probability = cold_probability.cpu().numpy()

        tmin_celsius = self.denormalize_tmin(tmin_scaled)

        return {
            "tmin_celsius": tmin_celsius,
            "coldwave_logits": cold_logits,
            "coldwave_probability": cold_probability,
            "mesh_lat": self.mesh_lat,
            "mesh_lon": self.mesh_lon,
            "valid_mask": self.valid_mask_np,
            "feature_names": self.feature_names,
            "window": self.window,
            "horizon": self.horizon,
        }


# -----------------------------------------------------------------
# Checkpoint integration test
# -----------------------------------------------------------------

def main():

    project_root = Path(__file__).resolve().parents[2]

    checkpoint = (
        project_root
        / "checkpoints"
        / "coldwave_final.pt"
    )

    predictor = ColdwavePredictor(checkpoint)

    print("\nCreating synthetic input for checkpoint test...")

    # IMPORTANT:
    # Create values in ORIGINAL feature scale around the saved training mean.
    # This allows predict() to exercise the real normalization path.
    x = np.broadcast_to(
        predictor.scaler_x_mean,
        (
            predictor.window,
            len(predictor.mesh_lat),
            predictor.input_dim,
        ),
    ).copy().astype(np.float32)

    # Invalid nodes are NaN in the real preprocessing pipeline.
    x[:, ~predictor.valid_mask_np, :] = np.nan

    print("Input shape:", x.shape)

    result = predictor.predict(x)

    print("\n===== COLDWAVE CHECKPOINT TEST =====")

    print(
        "Tmin output:",
        result["tmin_celsius"].shape
    )

    print(
        "Probability output:",
        result["coldwave_probability"].shape
    )

    valid_probs = result[
        "coldwave_probability"
    ][:, :, predictor.valid_mask_np]

    print(
        "Probability range:",
        float(valid_probs.min()),
        "to",
        float(valid_probs.max()),
    )

    print(
        "All probabilities finite:",
        bool(np.isfinite(valid_probs).all()),
    )

    print("\nCOLDWAVE CHECKPOINT INTEGRATION PASSED")


if __name__ == "__main__":
    main()