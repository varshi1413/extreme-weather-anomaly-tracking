from pathlib import Path
import numpy as np
import xarray as xr

from src.extreme_weather.heatwave_inference import HeatwavePredictor
from src.extreme_weather.heatwave_tracker import HeatwaveTracker
from src.extreme_weather.stage1_to_stage2 import prepare_stage2_condition
from src.extreme_weather.stage2_inference import Stage2DiffusionInference


PROJECT_ROOT = Path(__file__).resolve().parents[2]

HEATWAVE_CHECKPOINT = (
    PROJECT_ROOT / "checkpoints" / "heatwave_schema_test.pt"
)

DIFFUSION_CHECKPOINT = (
    PROJECT_ROOT / "checkpoints" / "stage2_era5_256.pt"
)


class IntegratedWeatherPipeline:

    def __init__(self):

        print("\n==========================================")
        print(" INTEGRATED EXTREME WEATHER PIPELINE")
        print("==========================================")

        print("\n[1/2] Loading Heatwave GNN...")
        self.heatwave_model = HeatwavePredictor(
            HEATWAVE_CHECKPOINT
        )

        print("\n[2/2] Loading Stage-2 Diffusion...")
        self.diffusion_model = Stage2DiffusionInference(
            DIFFUSION_CHECKPOINT
        )

        print("\nModels loaded successfully.")

    @staticmethod
    def load_stage2_climatology(
        climatology_path,
        std_path
    ):
        """
        Load the exact monthly mean/std climatology
        used during Stage-2 diffusion training.

        These files MUST come from the training setup
        used for stage2_era5_256.pt.
        """

        climatology_path = Path(climatology_path)
        std_path = Path(std_path)

        if not climatology_path.exists():
            raise FileNotFoundError(
                f"Stage-2 climatology not found: "
                f"{climatology_path}"
            )

        if not std_path.exists():
            raise FileNotFoundError(
                f"Stage-2 std file not found: "
                f"{std_path}"
            )

        return (
            xr.open_dataset(climatology_path),
            xr.open_dataset(std_path),
        )

    @staticmethod
    def normalize_stage2_weather(
        weather,
        month,
        mean_dataset,
        std_dataset,
        variables,
    ):
        """
        Normalize Stage-2 ERA5 weather using the same
        monthly climatology convention as training.

        weather shape:
            [C, H, W]

        month:
            1 ... 12
        """

        weather = np.asarray(weather, dtype=np.float32)

        if weather.ndim != 3:
            raise ValueError(
                "weather must have shape [C,H,W]"
            )

        if weather.shape[0] != len(variables):
            raise ValueError(
                f"Expected {len(variables)} channels, "
                f"got {weather.shape[0]}"
            )

        month_index = month - 1

        means = []
        stds = []

        for variable in variables:

            if variable not in mean_dataset:
                raise KeyError(
                    f"{variable} missing from climatology"
                )

            if variable not in std_dataset:
                raise KeyError(
                    f"{variable} missing from std.nc"
                )

            means.append(
                mean_dataset[variable]
                .isel(month=month_index)
                .values
            )

            stds.append(
                std_dataset[variable]
                .isel(month=month_index)
                .values
            )

        mean = np.stack(means).astype(np.float32)
        std = np.stack(stds).astype(np.float32)

        std = np.maximum(std, 1e-6)

        if mean.shape != weather.shape:
            raise ValueError(
                "Stage-2 climatology spatial shape does "
                "not match weather input.\n"
                f"Weather: {weather.shape}\n"
                f"Mean   : {mean.shape}\n"
                "Do not resize blindly; use the same "
                "training grid/crop."
            )

        normalized = (weather - mean) / std

        return normalized


def main():

    pipeline = IntegratedWeatherPipeline()

    print("\n==========================================")
    print(" INTEGRATION STATUS")
    print("==========================================")

    print("Heatwave GNN       : READY")
    print("Heatwave Tracker   : READY")
    print("Stage1 -> Stage2   : READY")
    print("Diffusion Model    : READY")

    print(
        "\nReal Stage-2 ERA5 normalization:"
        " WAITING FOR TRAINING CLIMATOLOGY"
    )

    print("\nIntegrated pipeline initialization PASSED.")


if __name__ == "__main__":
    main()