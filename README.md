# Extreme Weather Intelligence

Stage 1 tracks macro-scale extreme-weather anomalies over the Indian boundary region on a spherical icosahedral mesh. It computes an Extreme Forecast Index (EFI)-style standardized anomaly against a 30-year ERA5 baseline, then applies a message-passing GNN to produce node anomaly scores and a temporal geographic bounding box for Stage 2 conditioning.

## Region and variables

The default Indian boundary is 6-38 degrees north and 68-98 degrees east. The pipeline accepts NEPS-G forecast variables and matching ERA5 baseline variables with shape `[time, nodes, channels]`, using the channel order supplied by the caller.

The baseline should contain the same mesh nodes and comparable lead-time/seasonal slices. The implementation uses `z = (forecast - baseline_mean) / baseline_std` as the practical EFI proxy; production calibration can replace this with a percentile/CDF EFI without changing the GNN interface.

## Install

```powershell
python -m pip install -e ".[test]"
```

For NetCDF loading:

```powershell
python -m pip install -e ".[netcdf]"
```

## Run the synthetic check

```powershell
pytest -q
```

## Run the dashboard integration

Start the model API from the repository root:

```powershell
python server.py
```

In a second terminal, start the React dashboard:

```powershell
cd reactfinal\react2
npm install
npm run dev
```

The dashboard reads `GET /api/anomalies` through the Vite proxy. For external
inputs, send `POST /api/predict` with `forecast` and `baseline` arrays shaped
`[time, mesh_nodes, channels]`.

To enable live location weather, configure one provider before starting the API:

```powershell
$env:IMD_API_URL = "https://your-imd-adapter.example/weather?lat={lat}&lon={lon}"
# or: $env:WINDY_API_KEY = "your-windy-api-key"
python server.py
```

`/api/weather` returns all configured IMD variables, or Windy's wind, dewpoint,
relative humidity, pressure, temperature, precipitation, and cloud fields.

## Data contract

- Forecast: `[T, N, C]` tensor or NumPy array on the mesh nodes.
- ERA5 baseline: `[S, N, C]` samples, where `S` covers the 30-year historical baseline.
- Mesh: `Mesh(lat, lon, edge_index)` with `[N]` coordinates and `[2, E]` directed edges.
- Stage 1 output: node scores `[T, N]`, anomaly mask `[T, N]`, and `TemporalBoundingBox`.

Stage 2 accepts a cropped coarse field `[B, C, H, W]` and a rasterized Stage 1 anomaly channel `[B, 1, H, W]`. Use `rasterize_mesh_scores(scores, mesh, target_lat, target_lon)` to map node scores onto a NEPS-G latitude/longitude crop. `ConditionalDiffusionDownscaler` predicts noise for `[B, C, 4H, 4W]`; `sample_ensemble` returns `[samples, B, C, 4H, 4W]` scenarios.

For training, convert EFI values to node labels such as `(efi.max(axis=-1) >= 2).astype(float)`, then call `train_epoch` with batches shaped `[batch, nodes, channels]`. Pass `times=` to `AnomalyTracker.track` to return the original timestamps instead of integer indices in the bounding box.

## Train on the available data

The processed 6-hourly files under `D:\sih2026\data\processed\era5_6hourly` are loaded lazily and normalized with the month-matched `D:\sih2026\data\climatology\mean.nc` and `std.nc` files. Run a bounded smoke training job with:

```powershell
python scripts/train_diffusion.py --data-root D:\sih2026\data --max-samples 16 --diffusion-steps 20 --output checkpoints\stage2_smoke.pt
```

For a full run, omit `--max-samples` and use 1000 diffusion steps. Live IMD integration is provided by `fetch_imd_dataset` and `prepare_imd_condition`; configure an IMD adapter endpoint that returns the documented JSON schema, prepare the coarse/anomaly tensors, then call `sample_ensemble` with the loaded checkpoint.
