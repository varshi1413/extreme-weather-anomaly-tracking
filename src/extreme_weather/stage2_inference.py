import torch
import numpy as np

from .diffusion import (
    ConditionalDiffusionDownscaler,
    DiffusionSchedule,
    sample_ensemble,
    sample_ensemble_posterior,
)

class Stage2DiffusionInference:

    def __init__(
        self,
        checkpoint_path="checkpoints/stage2_era5_256.pt",
        device=None,
    ):

        if device is None:
            device = (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )

        self.device = torch.device(device)

        print(f"Loading Stage-2 diffusion on: {self.device}")

        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device,
            weights_only=False,
        )

        self.variables = list(checkpoint["variables"])
        self.scale_factor = int(checkpoint["scale_factor"])

        self.model = ConditionalDiffusionDownscaler(
            weather_channels=len(self.variables),
            scale_factor=self.scale_factor,
        ).to(self.device)

        self.model.load_state_dict(checkpoint["model"])
        self.model.eval()

        self.schedule = DiffusionSchedule()

        print("Stage-2 checkpoint loaded successfully.")
        print("Weather channels :", len(self.variables))
        print("Variables        :", self.variables)
        print("Scale factor     :", self.scale_factor)

    @torch.no_grad()
    def predict(
        self,
        coarse_weather,
        anomaly_mask,
        ensemble_size=1,
    ):
        """
        coarse_weather:
            numpy [C,H,W] or [B,C,H,W]

        anomaly_mask:
            numpy [H,W], [1,H,W] or [B,1,H,W]
        """

        coarse = np.asarray(
            coarse_weather,
            dtype=np.float32,
        )

        mask = np.asarray(
            anomaly_mask,
            dtype=np.float32,
        )

        if coarse.ndim == 3:
            coarse = coarse[None, ...]

        if mask.ndim == 2:
            mask = mask[None, None, ...]

        elif mask.ndim == 3:
            mask = mask[:, None, ...]

        if coarse.ndim != 4:
            raise ValueError(
                "coarse_weather must have shape "
                "[C,H,W] or [B,C,H,W]"
            )

        if mask.ndim != 4:
            raise ValueError(
                "anomaly_mask must become [B,1,H,W]"
            )

        if coarse.shape[1] != len(self.variables):
            raise ValueError(
                f"Diffusion expects {len(self.variables)} "
                f"weather channels {self.variables}, "
                f"but received {coarse.shape[1]}."
            )

        if coarse.shape[0] != mask.shape[0]:
            raise ValueError(
                "Weather batch and anomaly-mask batch differ."
            )

        if coarse.shape[-2:] != mask.shape[-2:]:
            raise ValueError(
                "Weather grid and anomaly-mask grid differ."
            )

        coarse_tensor = torch.from_numpy(
            coarse
        ).to(self.device)

        mask_tensor = torch.from_numpy(
            mask
        ).to(self.device)

        samples = sample_ensemble(
            model=self.model,
            schedule=self.schedule,
            coarse_condition=coarse_tensor,
            anomaly_condition=mask_tensor,
            samples=ensemble_size,
        )

        return samples.detach().cpu().numpy()


if __name__ == "__main__":

    print("\n================================")
    print(" STAGE-2 DIFFUSION INFERENCE TEST")
    print("================================")

    inference = Stage2DiffusionInference()

    # Small synthetic 7-channel coarse ROI.
    #
    # Stage-2 model is 4x, therefore:
    # 8x8 coarse -> 32x32 output.

    coarse = np.random.randn(
        len(inference.variables),
        8,
        8,
    ).astype(np.float32)

    # Binary anomaly condition.
    mask = np.zeros(
        (8, 8),
        dtype=np.float32,
    )

    mask[2:6, 2:6] = 1.0

    print("\nCoarse input shape:", coarse.shape)
    print("Mask shape        :", mask.shape)

    output = inference.predict(
        coarse,
        mask,
        ensemble_size=1,
    )

    print("\nDiffusion output shape:", output.shape)

    expected_h = coarse.shape[-2] * inference.scale_factor
    expected_w = coarse.shape[-1] * inference.scale_factor

    if output.shape[-2:] != (
        expected_h,
        expected_w,
    ):
        raise RuntimeError(
            "Unexpected Stage-2 output resolution."
        )

    print(
        f"Expected spatial resolution: "
        f"{expected_h} x {expected_w}"
    )

    print("\nStage-2 inference test PASSED.")