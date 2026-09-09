from __future__ import annotations

from typing import Iterable

import numpy as np
import torch
from torch import nn

from .mesh import Mesh
from .model import SphericalAnomalyGNN


def train_epoch(
    model: SphericalAnomalyGNN,
    optimizer: torch.optim.Optimizer,
    forecast_batches: Iterable[np.ndarray],
    target_batches: Iterable[np.ndarray],
    mesh: Mesh,
) -> float:
    """Train node-level anomaly scores against EFI-derived targets."""
    model.train()
    edge_index = torch.from_numpy(mesh.edge_index)
    loss_function = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    batch_count = 0
    for forecast, targets in zip(forecast_batches, target_batches):
        features = torch.as_tensor(forecast, dtype=torch.float32)
        labels = torch.as_tensor(targets, dtype=torch.float32)
        optimizer.zero_grad(set_to_none=True)
        logits = model(features, edge_index)
        loss = loss_function(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.detach())
        batch_count += 1
    if batch_count == 0:
        raise ValueError("at least one forecast/target batch is required")
    return total_loss / batch_count