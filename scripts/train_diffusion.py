from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from extreme_weather import ConditionalDiffusionDownscaler, DiffusionSchedule, diffusion_loss
from extreme_weather.data import Era5DiffusionDataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Stage 2 on processed 6-hourly ERA5 data.")
    parser.add_argument("--data-root", type=Path, default=Path(r"D:\sih2026\data"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--diffusion-steps", type=int, default=1000)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--val-year", type=str, default="2024")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("checkpoints/stage2_diffusion.pt"))
    args = parser.parse_args()
    all_files = sorted(args.data_root.glob("processed/era5_6hourly/era5_india_*/surface/*_6hourly.nc"))
    if not all_files:
        raise FileNotFoundError("No processed 6-hourly ERA5 files found")
    train_files = [path for path in all_files if f"era5_india_{args.val_year}" not in str(path)]
    val_files = [path for path in all_files if f"era5_india_{args.val_year}" in str(path)]
    if not train_files or not val_files:
        raise ValueError(f"year holdout produced train={len(train_files)} and validation={len(val_files)} files")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset = Era5DiffusionDataset(train_files, args.data_root / "climatology" / "mean.nc", max_samples=args.max_samples)
    val_dataset = Era5DiffusionDataset(val_files, args.data_root / "climatology" / "mean.nc")
    loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=device.type == "cuda")
    model = ConditionalDiffusionDownscaler(weather_channels=7).to(device)
    schedule = DiffusionSchedule(steps=args.diffusion_steps)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    start_epoch = 0
    best_validation = float("inf")
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"] + 1
        best_validation = checkpoint.get("validation_loss", best_validation)
    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        for target, coarse, anomaly in loader:
            target, coarse, anomaly = target.to(device), coarse.to(device), anomaly.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                loss = diffusion_loss(model, schedule, target, coarse, anomaly)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            total += float(loss.detach())
        model.eval()
        validation_total = 0.0
        with torch.no_grad():
            for target, coarse, anomaly in val_loader:
                target, coarse, anomaly = target.to(device), coarse.to(device), anomaly.to(device)
                validation_total += float(diffusion_loss(model, schedule, target, coarse, anomaly))
        train_loss = total / len(loader)
        validation_loss = validation_total / len(val_loader)
        print(f"epoch={epoch + 1}/{args.epochs} train_loss={train_loss:.6f} validation_loss={validation_loss:.6f} device={device}")
        if validation_loss < best_validation:
            best_validation = validation_loss
            args.output.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "epoch": epoch, "validation_loss": validation_loss,
                "variables": list(train_dataset.variables), "scale_factor": 4,
                "train_years": sorted({path.parent.parent.parent.name for path in train_files}),
                "validation_year": args.val_year, "diffusion_steps": args.diffusion_steps,
            }, args.output)
            print(f"saved_best={args.output}")


if __name__ == "__main__":
    main()