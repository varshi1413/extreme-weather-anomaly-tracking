from __future__ import annotations

import math

import torch
import torch.nn.functional as functional
from torch import nn


class DiffusionSchedule:
    def __init__(self, steps: int = 1000, beta_start: float = 1e-4, beta_end: float = 2e-2) -> None:
        if steps < 2:
            raise ValueError("steps must be at least 2")
        self.steps = steps
        self.betas = torch.linspace(beta_start, beta_end, steps)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def to(self, device: torch.device) -> "DiffusionSchedule":
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alpha_bars = self.alpha_bars.to(device)
        return self

    def add_noise(self, clean: torch.Tensor, noise: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        alpha_bar = self.alpha_bars[timestep].view(-1, 1, 1, 1)
        return alpha_bar.sqrt() * clean + (1.0 - alpha_bar).sqrt() * noise


def _time_embedding(timestep: torch.Tensor, channels: int) -> torch.Tensor:
    half = channels // 2
    scale = math.log(10000) / max(half - 1, 1)
    frequencies = torch.exp(torch.arange(half, device=timestep.device) * -scale)
    angles = timestep.float().view(-1, 1) * frequencies.view(1, -1)
    embedding = torch.cat((angles.sin(), angles.cos()), dim=1)
    return functional.pad(embedding, (0, channels - embedding.shape[1]))


class ConditionalDiffusionDownscaler(nn.Module):
    """Compact conditional DDPM denoiser for 4x coarse-to-high-resolution fields."""

    def __init__(self, weather_channels: int, base_channels: int = 64, scale_factor: int = 4) -> None:
        super().__init__()
        if scale_factor != 4:
            raise ValueError("the initial Stage 2 contract supports 4x upscaling")
        self.scale_factor = scale_factor
        condition_channels = weather_channels + 1
        self.time_projection = nn.Sequential(
            nn.Linear(base_channels, base_channels), nn.SiLU(), nn.Linear(base_channels, base_channels)
        )
        self.input_projection = nn.Conv2d(weather_channels + condition_channels, base_channels, 3, padding=1)
        self.block_one = nn.Sequential(nn.GroupNorm(8, base_channels), nn.SiLU(), nn.Conv2d(base_channels, base_channels, 3, padding=1))
        self.block_two = nn.Sequential(nn.GroupNorm(8, base_channels), nn.SiLU(), nn.Conv2d(base_channels, base_channels, 3, padding=1))
        self.output_projection = nn.Conv2d(base_channels, weather_channels, 3, padding=1)

    def forward(
        self,
        noisy_high: torch.Tensor,
        coarse_condition: torch.Tensor,
        timestep: torch.Tensor,
        anomaly_condition: torch.Tensor,
    ) -> torch.Tensor:
        if coarse_condition.ndim != 4 or anomaly_condition.ndim != 4:
            raise ValueError("conditions must have shape [batch, channels, height, width]")
        target_size = noisy_high.shape[-2:]
        coarse_high = functional.interpolate(coarse_condition, size=target_size, mode="bilinear", align_corners=False)
        anomaly_high = functional.interpolate(anomaly_condition, size=target_size, mode="bilinear", align_corners=False)
        features = self.input_projection(torch.cat((noisy_high, coarse_high, anomaly_high), dim=1))
        time_bias = self.time_projection(_time_embedding(timestep, self.time_projection[0].in_features))
        features = features + time_bias[:, :, None, None]
        residual = features
        features = self.block_one(features)
        features = self.block_two(features) + residual
        return self.output_projection(features)


def diffusion_loss(
    model: ConditionalDiffusionDownscaler,
    schedule: DiffusionSchedule,
    clean_high: torch.Tensor,
    coarse_condition: torch.Tensor,
    anomaly_condition: torch.Tensor,
) -> torch.Tensor:
    schedule.to(clean_high.device)
    timestep = torch.randint(0, schedule.steps, (clean_high.shape[0],), device=clean_high.device)
    noise = torch.randn_like(clean_high)
    noisy = schedule.add_noise(clean_high, noise, timestep)
    predicted = model(noisy, coarse_condition, timestep, anomaly_condition)
    return functional.mse_loss(predicted, noise)


@torch.no_grad()
def sample_ensemble(
    model: ConditionalDiffusionDownscaler,
    schedule: DiffusionSchedule,
    coarse_condition: torch.Tensor,
    anomaly_condition: torch.Tensor,
    samples: int = 4,
) -> torch.Tensor:
    if samples < 1:
        raise ValueError("samples must be at least 1")
    model.eval()
    schedule.to(coarse_condition.device)
    batch, channels, height, width = coarse_condition.shape
    output_shape = (samples * batch, channels, height * model.scale_factor, width * model.scale_factor)
    coarse = coarse_condition.repeat(samples, 1, 1, 1)
    anomaly = anomaly_condition.repeat(samples, 1, 1, 1)
    result = torch.randn(output_shape, device=coarse.device, dtype=coarse.dtype)
    for step in reversed(range(schedule.steps)):
        timestep = torch.full((result.shape[0],), step, device=result.device, dtype=torch.long)
        predicted_noise = model(result, coarse, timestep, anomaly)
        alpha = schedule.alphas[step]
        alpha_bar = schedule.alpha_bars[step]
        beta = schedule.betas[step]
        result = (result - beta / (1.0 - alpha_bar).sqrt() * predicted_noise) / alpha.sqrt()
        if step > 0:
            result = result + beta.sqrt() * torch.randn_like(result)
    return result.view(samples, batch, channels, height * model.scale_factor, width * model.scale_factor)

@torch.no_grad()
def sample_ensemble_posterior(
    model: ConditionalDiffusionDownscaler,
    schedule: DiffusionSchedule,
    coarse_condition: torch.Tensor,
    anomaly_condition: torch.Tensor,
    samples: int = 1,
) -> torch.Tensor:

    if samples < 1:
        raise ValueError("samples must be at least 1")

    model.eval()
    schedule.to(coarse_condition.device)

    batch, channels, height, width = coarse_condition.shape

    output_shape = (
        samples * batch,
        channels,
        height * model.scale_factor,
        width * model.scale_factor,
    )

    coarse = coarse_condition.repeat(samples, 1, 1, 1)
    anomaly = anomaly_condition.repeat(samples, 1, 1, 1)

    result = torch.randn(
        output_shape,
        device=coarse.device,
        dtype=coarse.dtype,
    )

    for step in reversed(range(schedule.steps)):

        timestep = torch.full(
            (result.shape[0],),
            step,
            device=result.device,
            dtype=torch.long,
        )

        predicted_noise = model(
            result,
            coarse,
            timestep,
            anomaly,
        )

        alpha = schedule.alphas[step]
        alpha_bar = schedule.alpha_bars[step]
        beta = schedule.betas[step]

        # DDPM reverse mean
        result = (
            result
            - beta
            / torch.sqrt(1.0 - alpha_bar)
            * predicted_noise
        ) / torch.sqrt(alpha)

        if step > 0:

            previous_alpha_bar = schedule.alpha_bars[step - 1]

            # Posterior variance:
            # beta_tilde =
            # beta_t * (1-alpha_bar_(t-1))
            #          / (1-alpha_bar_t)

            posterior_variance = (
                beta
                * (1.0 - previous_alpha_bar)
                / (1.0 - alpha_bar)
            )

            noise = torch.randn_like(result)

            result = (
                result
                + torch.sqrt(
                    torch.clamp(
                        posterior_variance,
                        min=1e-20,
                    )
                )
                * noise
            )

    return result.view(
        samples,
        batch,
        channels,
        height * model.scale_factor,
        width * model.scale_factor,
    )