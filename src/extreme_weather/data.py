from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import xarray as xr
from torch.utils.data import Dataset


DEFAULT_VARIABLES = ("u10", "v10", "t2m", "d2m", "tp", "msl", "wind_speed")


class Era5DiffusionDataset(Dataset):
    """Lazy 6-hourly ERA5 dataset producing 4x diffusion training pairs."""

    def __init__(
        self,
        files: Sequence[str | Path],
        climatology_path: str | Path,
        variables: Sequence[str] = DEFAULT_VARIABLES,
        crop_size: int = 124,
        max_samples: int | None = None,
    ) -> None:
        self.files = [str(path) for path in files]
        self.variables = tuple(variables)
        self.crop_size = crop_size
        if crop_size % 4 != 0:
            raise ValueError("crop_size must be divisible by 4")
        self.index: list[tuple[int, int]] = []
        for file_id, path in enumerate(self.files):
            with xr.open_dataset(path) as dataset:
                self.index.extend((file_id, time_id) for time_id in range(dataset.sizes["valid_time"]))
        if max_samples is not None:
            self.index = self.index[:max_samples]
        if not self.index:
            raise ValueError("no training samples found")
        with xr.open_dataset(climatology_path) as climatology:
            missing = [name for name in self.variables if name not in climatology]
            if missing:
                raise ValueError(f"climatology is missing variables: {missing}")
            self.mean = np.stack([climatology[name].values for name in self.variables], axis=1).astype(np.float32)
        std_path = Path(climatology_path).with_name("std.nc")
        with xr.open_dataset(std_path) as climatology_std:
            self.std = np.stack([climatology_std[name].values for name in self.variables], axis=1).astype(np.float32)
        self._open_file_id: int | None = None
        self._open_dataset: xr.Dataset | None = None

    def __len__(self) -> int:
        return len(self.index)

    def _dataset_for(self, file_id: int) -> xr.Dataset:
        if self._open_file_id != file_id:
            if self._open_dataset is not None:
                self._open_dataset.close()
            self._open_dataset = xr.open_dataset(self.files[file_id])
            self._open_file_id = file_id
        return self._open_dataset

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        file_id, time_id = self.index[item]
        dataset = self._dataset_for(file_id)
        timestamp = dataset.valid_time.isel(valid_time=time_id).values
        month = int(str(timestamp)[5:7])
        values = np.stack(
            [dataset[name].isel(valid_time=time_id).values for name in self.variables], axis=0
        ).astype(np.float32)
        height, width = values.shape[-2:]
        row_start = (height - self.crop_size) // 2
        col_start = (width - self.crop_size) // 2
        values = values[:, row_start:row_start + self.crop_size, col_start:col_start + self.crop_size]
        month_id = month - 1
        mean = self.mean[month_id, :, row_start:row_start + self.crop_size, col_start:col_start + self.crop_size]
        std = np.maximum(self.std[month_id, :, row_start:row_start + self.crop_size, col_start:col_start + self.crop_size], 1e-6)
        target = (values - mean) / std
        target_tensor = torch.from_numpy(target)
        coarse = torch.nn.functional.avg_pool2d(target_tensor.unsqueeze(0), kernel_size=4, stride=4).squeeze(0)
        anomaly = (coarse.abs().amax(dim=0, keepdim=True) >= 2.0).float()
        return target_tensor, coarse, anomaly