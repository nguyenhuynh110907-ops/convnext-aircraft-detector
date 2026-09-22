#!/usr/bin/env python3
"""Train ConvNeXt-FPN Faster R-CNN on Pascal VOC aircraft."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from aircraft_detector.checkpoint import load_checkpoint, save_checkpoint
from aircraft_detector.data import VOCAircraftDataset, collate_fn
from aircraft_detector.engine import evaluate, train_one_epoch
from aircraft_detector.model import build_aircraft_detector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/convnext_tiny"))
    parser.add_argument("--year", choices=["2007", "2012"], default="2007")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--trainable-stages", type=int, choices=range(5), default=2)
    parser.add_argument("--min-size", type=int, default=512)
    parser.add_argument("--max-size", type=int, default=1024)
    parser.add_argument("--positive-only", action="store_true")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def choose_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = choose_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_set = VOCAircraftDataset(
        args.data_dir,
        year=args.year,
        image_set="trainval",
        download=args.download,
        train=True,
        positive_only=args.positive_only,
    )
    validation_set_name = "test" if args.year == "2007" else "val"
    validation_set = VOCAircraftDataset(
        args.data_dir,
        year=args.year,
        image_set=validation_set_name,
        download=args.download,
        train=False,
    )
    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )
    validation_loader = DataLoader(
        validation_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )

    model = build_aircraft_detector(
        pretrained_backbone=args.resume is None,
        trainable_stages=args.trainable_stages,
        min_size=args.min_size,
        max_size=args.max_size,
    ).to(device)
    trainable_parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_parameters, lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.epochs, 1)
    )
    scaler = torch.cuda.amp.GradScaler(
        enabled=args.amp and device.type == "cuda"
    )

    start_epoch = 0
    best_map50 = -1.0
    if args.resume:
        checkpoint = load_checkpoint(args.resume, device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if "scheduler" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = checkpoint.get("epoch", -1) + 1
        best_map50 = checkpoint.get("best_map50", best_map50)

    history_path = args.output_dir / "history.jsonl"
    print(
        f"device={device} train_images={len(train_set)} "
        f"validation_images={len(validation_set)}"
    )
    for epoch in range(start_epoch, args.epochs):
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, device, scaler=scaler
        )
        validation_metrics = evaluate(model, validation_loader, device)
        scheduler.step()

        record = {
            "epoch": epoch + 1,
            "lr": optimizer.param_groups[0]["lr"],
            "train": train_metrics,
            "validation": validation_metrics,
        }
        print(json.dumps(record, indent=2))
        with history_path.open("a", encoding="utf-8") as history_file:
            history_file.write(json.dumps(record) + "\n")

        state = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_map50": max(best_map50, validation_metrics["map_50"]),
            "args": vars(args),
        }
        save_checkpoint(state, args.output_dir / "last.pt")
        if validation_metrics["map_50"] > best_map50:
            best_map50 = validation_metrics["map_50"]
            save_checkpoint(state, args.output_dir / "best.pt")


if __name__ == "__main__":
    main()

