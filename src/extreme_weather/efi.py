from __future__ import annotations

import numpy as np


def compute_efi(
    forecast: np.ndarray,
    baseline: np.ndarray,
    min_std: float = 1e-6,
) -> np.ndarray:
    """Compute a stable standardized EFI proxy.

    forecast is [T, N, C], baseline is [S, N, C]. The baseline mean/std are
    estimated over the historical sample axis S.
    """
    forecast = np.asarray(forecast, dtype=np.float32)
    baseline = np.asarray(baseline, dtype=np.float32)
    if forecast.ndim != 3 or baseline.ndim != 3:
        raise ValueError("forecast and baseline must have shape [samples, nodes, channels]")
    if forecast.shape[1:] != baseline.shape[1:]:
        raise ValueError("forecast and baseline node/channel dimensions must match")
    mean = baseline.mean(axis=0)
    std = np.maximum(baseline.std(axis=0), min_std)
    return (forecast - mean) / std
