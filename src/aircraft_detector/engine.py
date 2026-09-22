"""Training and evaluation loops for torchvision detection models."""

from __future__ import annotations

from collections import defaultdict
from contextlib import nullcontext
from typing import Dict, Iterable, List

import torch
from torch import Tensor, nn
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from tqdm import tqdm


def move_target(target: Dict[str, Tensor], device: torch.device):
    return {key: value.to(device) for key, value in target.items()}


def train_one_epoch(
    model: nn.Module,
    loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    scaler=None,
) -> Dict[str, float]:
    model.train()
    totals = defaultdict(float)
    steps = 0

    for images, targets in tqdm(loader, desc="train", leave=False):
        images = [image.to(device) for image in images]
        targets = [move_target(target, device) for target in targets]

        optimizer.zero_grad(set_to_none=True)
        amp_enabled = scaler is not None and scaler.is_enabled()
        amp_context = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if amp_enabled
            else nullcontext()
        )
        with amp_context:
            losses = model(images, targets)
            total_loss = sum(loss for loss in losses.values())

        if not torch.isfinite(total_loss):
            raise RuntimeError(f"Non-finite loss encountered: {total_loss.item()}")

        if amp_enabled:
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()

        totals["loss"] += total_loss.item()
        for name, value in losses.items():
            totals[name] += value.item()
        steps += 1

    return {name: value / max(steps, 1) for name, value in totals.items()}


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: Iterable,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox")

    for images, targets in tqdm(loader, desc="evaluate", leave=False):
        images = [image.to(device) for image in images]
        predictions = model(images)

        predictions_cpu: List[Dict[str, Tensor]] = [
            {key: value.detach().cpu() for key, value in prediction.items()}
            for prediction in predictions
        ]
        targets_cpu: List[Dict[str, Tensor]] = [
            {key: value.detach().cpu() for key, value in target.items()}
            for target in targets
        ]
        metric.update(predictions_cpu, targets_cpu)

    values = metric.compute()
    selected = ("map", "map_50", "map_75", "mar_100")
    return {name: float(values[name]) for name in selected}

