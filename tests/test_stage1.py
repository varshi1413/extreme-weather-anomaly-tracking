import numpy as np
import torch

from extreme_weather import (
    AnomalyTracker,
    ConditionalDiffusionDownscaler,
    DiffusionSchedule,
    SphericalAnomalyGNN,
    build_indian_icosahedral_mesh,
    diffusion_loss,
    rasterize_mesh_scores,
    sample_ensemble,
)


def test_indian_mesh_and_tracker_produce_bounding_box():
    mesh = build_indian_icosahedral_mesh(subdivisions=4)
    assert mesh.num_nodes > 10
    assert np.all((mesh.lat >= 6) & (mesh.lat <= 38))
    assert np.all((mesh.lon >= 68) & (mesh.lon <= 98))

    rng = np.random.default_rng(7)
    baseline = rng.normal(size=(30 * 4, mesh.num_nodes, 2)).astype(np.float32)
    forecast = rng.normal(size=(4, mesh.num_nodes, 2)).astype(np.float32)
    forecast[2, 0, :] += 8.0

    torch.manual_seed(7)
    tracker = AnomalyTracker(mesh, SphericalAnomalyGNN(input_channels=2, hidden_channels=16), threshold=-1.0)
    scores, mask, box = tracker.track(forecast, baseline)

    assert scores.shape == (4, mesh.num_nodes)
    assert mask.shape == scores.shape
    assert box is not None
    assert box.start == 0
    assert box.end == 3


def test_tracker_returns_none_when_no_node_crosses_threshold():
    mesh = build_indian_icosahedral_mesh(subdivisions=3)
    baseline = np.zeros((120, mesh.num_nodes, 1), dtype=np.float32)
    forecast = np.zeros((2, mesh.num_nodes, 1), dtype=np.float32)
    tracker = AnomalyTracker(mesh, SphericalAnomalyGNN(input_channels=1, hidden_channels=8), threshold=1e9)
    _, mask, box = tracker.track(forecast, baseline)
    assert not mask.any()
    assert box is None


def test_tracker_preserves_supplied_time_labels():
    mesh = build_indian_icosahedral_mesh(subdivisions=3)
    baseline = np.zeros((120, mesh.num_nodes, 1), dtype=np.float32)
    forecast = np.zeros((2, mesh.num_nodes, 1), dtype=np.float32)
    tracker = AnomalyTracker(mesh, SphericalAnomalyGNN(input_channels=1, hidden_channels=8), threshold=-1.0)
    _, _, box = tracker.track(forecast, baseline, times=["2020-01-01T00", "2020-01-01T06"])
    assert box is not None
    assert box.start == "2020-01-01T00"
    assert box.end == "2020-01-01T06"


def test_conditional_diffusion_preserves_amplitude_shape_and_samples():
    torch.manual_seed(4)
    model = ConditionalDiffusionDownscaler(weather_channels=2, base_channels=16)
    schedule = DiffusionSchedule(steps=4)
    coarse = torch.randn(1, 2, 4, 5)
    anomaly = torch.randn(1, 1, 4, 5)
    clean = torch.randn(1, 2, 16, 20)
    loss = diffusion_loss(model, schedule, clean, coarse, anomaly)
    assert loss.ndim == 0
    ensemble = sample_ensemble(model, schedule, coarse, anomaly, samples=2)
    assert ensemble.shape == (2, 1, 2, 16, 20)


def test_mesh_scores_rasterize_to_neps_grid():
    mesh = build_indian_icosahedral_mesh(subdivisions=3)
    scores = np.arange(mesh.num_nodes, dtype=np.float32)[None, :]
    raster = rasterize_mesh_scores(scores, mesh, mesh.lat[:2], mesh.lon[:3])
    assert raster.shape == (1, 2, 3)
    assert np.all(np.isin(raster, scores))
