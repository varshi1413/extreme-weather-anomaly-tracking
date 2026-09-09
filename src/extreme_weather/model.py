from __future__ import annotations

import torch
from torch import nn


class GraphBlock(nn.Module):
    def __init__(self, channels: int, hidden_channels: int) -> None:
        super().__init__()
        self.self_projection = nn.Linear(channels, hidden_channels)
        self.neighbor_projection = nn.Linear(channels, hidden_channels)
        self.output = nn.Sequential(nn.SiLU(), nn.Linear(hidden_channels, hidden_channels))
        self.skip = nn.Linear(channels, hidden_channels) if channels != hidden_channels else nn.Identity()

    def forward(self, features: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        source, target = edge_index
        messages = self.neighbor_projection(features[:, source])
        aggregated = torch.zeros(
            features.shape[0], features.shape[1], messages.shape[-1],
            device=features.device, dtype=features.dtype,
        )
        aggregated.index_add_(1, target, messages)
        degree = torch.bincount(target, minlength=features.shape[1]).clamp_min(1)
        aggregated = aggregated / degree.to(features.device, features.dtype).view(1, -1, 1)
        return self.output(self.self_projection(features) + aggregated) + self.skip(features)


class SphericalAnomalyGNN(nn.Module):
    """Message-passing network operating directly on spherical mesh nodes."""

    def __init__(self, input_channels: int, hidden_channels: int = 64, layers: int = 3) -> None:
        super().__init__()
        if layers < 1:
            raise ValueError("layers must be at least 1")
        blocks = []
        channels = input_channels
        for _ in range(layers):
            blocks.append(GraphBlock(channels, hidden_channels))
            channels = hidden_channels
        self.blocks = nn.ModuleList(blocks)
        self.score_head = nn.Linear(hidden_channels, 1)

    def forward(self, features: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        if features.ndim != 3:
            raise ValueError("features must have shape [batch, nodes, channels]")
        for block in self.blocks:
            features = block(features, edge_index)
        return self.score_head(features).squeeze(-1)
