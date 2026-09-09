import numpy as np
import torch
import xarray as xr

from src.extreme_weather.stage2_inference import Stage2DiffusionInference


ERA5_ROOT = r"C:\Users\Varshitha S B\Documents\ERA5_DOWNLOAD\ERA5_INDIA"

INSTANT_FILE = (
    ERA5_ROOT
    + r"\ERA5_INDIA_SINGLE_2015_05_stepType-instant.nc"
)

ACCUM_FILE = (
    ERA5_ROOT
    + r"\ERA5_INDIA_SINGLE_2015_05_stepType-accum.nc"
)

MEAN_FILE = r"normalization\mean.nc"
STD_FILE = r"normalization\std.nc"

VARIABLES = [
    "u10",
    "v10",
    "t2m",
    "d2m",
    "tp",
    "msl",
    "wind_speed",
]

MONTH = 5
CROP_SIZE = 120


print("\n========================================")
print(" STAGE-2 REAL ERA5 DIFFUSION TEST")
print("========================================")


# --------------------------------------------------
# 1. LOAD ERA5
# --------------------------------------------------

instant = xr.open_dataset(INSTANT_FILE)
accum = xr.open_dataset(ACCUM_FILE)

print("\nInstant grid :", instant.sizes)
print("Accum grid   :", accum.sizes)


# --------------------------------------------------
# 2. SELECT COMMON GRID WITH CLIMATOLOGY
# --------------------------------------------------

instant = instant.sel(
    latitude=slice(37.5, 6.0),
    longitude=slice(68.0, 98.0),
)

accum = accum.sel(
    latitude=slice(37.5, 6.0),
    longitude=slice(68.0, 98.0),
)

print(
    "Common ERA5 grid:",
    instant.sizes["latitude"],
    "x",
    instant.sizes["longitude"],
)


# --------------------------------------------------
# 3. SELECT ONE 6-HOURLY SAMPLE
# --------------------------------------------------

time_id = 0

timestamp = instant.valid_time.isel(
    valid_time=time_id
).values

print("Timestamp:", timestamp)


# --------------------------------------------------
# 4. CREATE 7 WEATHER CHANNELS
# --------------------------------------------------

u10 = instant["u10"].isel(valid_time=time_id).values.astype(np.float32)
v10 = instant["v10"].isel(valid_time=time_id).values.astype(np.float32)

wind_speed = np.sqrt(u10 ** 2 + v10 ** 2).astype(np.float32)

weather = {
    "u10": u10,
    "v10": v10,

    # Kelvin -> Celsius
    "t2m": (
        instant["t2m"].isel(valid_time=time_id).values.astype(np.float32)
        - 273.15
    ),

    # Kelvin -> Celsius
    "d2m": (
        instant["d2m"].isel(valid_time=time_id).values.astype(np.float32)
        - 273.15
    ),

    # metres -> millimetres
    "tp": (
        accum["tp"].isel(valid_time=time_id).values.astype(np.float32)
        * 1000.0
    ),

    # Pa -> hPa
    "msl": (
        instant["msl"].isel(valid_time=time_id).values.astype(np.float32)
        / 100.0
    ),

    "wind_speed": wind_speed,
}

values = np.stack(
    [weather[name] for name in VARIABLES],
    axis=0,
).astype(np.float32)

print("7-channel field:", values.shape)


# --------------------------------------------------
# 5. LOAD TRAINING NORMALIZATION
# --------------------------------------------------

mean_ds = xr.open_dataset(MEAN_FILE)
std_ds = xr.open_dataset(STD_FILE)

mean_ds = mean_ds.sel(
    latitude=slice(37.5, 6.0),
    longitude=slice(68.0, 98.0),
)

std_ds = std_ds.sel(
    latitude=slice(37.5, 6.0),
    longitude=slice(68.0, 98.0),
)

mean = np.stack(
    [
        mean_ds[name]
        .sel(month=MONTH)
        .values
        for name in VARIABLES
    ],
    axis=0,
).astype(np.float32)

std = np.stack(
    [
        std_ds[name]
        .sel(month=MONTH)
        .values
        for name in VARIABLES
    ],
    axis=0,
).astype(np.float32)


# Protect precipitation cells with zero std.
std = np.maximum(std, 1e-6)


# --------------------------------------------------
# 6. EXACT TRAINING-STYLE CENTER CROP
# --------------------------------------------------

height, width = values.shape[-2:]

row_start = (
    height - CROP_SIZE
) // 2

col_start = (
    width - CROP_SIZE
) // 2

values = values[
    :,
    row_start:row_start + CROP_SIZE,
    col_start:col_start + CROP_SIZE,
]

mean = mean[
    :,
    row_start:row_start + CROP_SIZE,
    col_start:col_start + CROP_SIZE,
]

std = std[
    :,
    row_start:row_start + CROP_SIZE,
    col_start:col_start + CROP_SIZE,
]

print("Fine target crop:", values.shape)


# --------------------------------------------------
# 7. NORMALIZE EXACTLY LIKE TRAINING
# --------------------------------------------------

target = (
    values - mean
) / std

print(
    "Normalized range:",
    float(np.nanmin(target)),
    "to",
    float(np.nanmax(target)),
)

print(
    "Finite:",
    np.isfinite(target).all(),
)


# --------------------------------------------------
# 8. CREATE 31x31 COARSE CONDITION
# --------------------------------------------------

target_tensor = torch.from_numpy(target)

coarse = torch.nn.functional.avg_pool2d(
    target_tensor.unsqueeze(0),
    kernel_size=4,
    stride=4,
).squeeze(0)

print(
    "Coarse condition:",
    tuple(coarse.shape),
)


# --------------------------------------------------
# 9. CREATE TRAINING-STYLE ANOMALY MASK
# --------------------------------------------------

anomaly = (
    coarse.abs()
    .amax(dim=0, keepdim=True)
    >= 2.0
).float()

print(
    "Anomaly mask:",
    tuple(anomaly.shape),
)

print(
    "Anomaly cells:",
    int(anomaly.sum()),
)


# --------------------------------------------------
# 10. RUN TRAINED DIFFUSION
# --------------------------------------------------

inference = Stage2DiffusionInference(
    checkpoint_path=
    "checkpoints/stage2_era5_256.pt"
)

output = inference.predict(
    coarse.numpy(),
    anomaly.numpy(),
    ensemble_size=1,
)

print(
    "\nDiffusion normalized output:",
    output.shape,
)


# output:
# [ensemble, batch, channel, H, W]

generated_normalized = output[0, 0]


# --------------------------------------------------
# 11. DE-NORMALIZE
# --------------------------------------------------

generated_physical = (
    generated_normalized * std + mean
)

print(
    "Physical output:",
    generated_physical.shape,
)

print(
    "Physical finite:",
    np.isfinite(
        generated_physical
    ).all(),
)


# --------------------------------------------------
# 12. PRINT VARIABLE RANGES
# --------------------------------------------------

print("\nGenerated physical ranges:")

for index, name in enumerate(VARIABLES):

    field = generated_physical[index]

    print(
        f"{name:12s}",
        "min=",
        float(np.nanmin(field)),
        "max=",
        float(np.nanmax(field)),
    )


instant.close()
accum.close()
mean_ds.close()
std_ds.close()


print("\n========================================")
print(" REAL STAGE-2 TEST COMPLETED")
print("========================================")