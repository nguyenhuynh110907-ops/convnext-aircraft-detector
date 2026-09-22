"""Pascal VOC dataset adapter for single-class aircraft detection."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.datasets import VOCDetection
from torchvision.transforms import functional as F


def _as_list(value: Any) -> List[Dict[str, Any]]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def parse_aircraft_annotation(
    annotation: Dict[str, Any],
    *,
    class_name: str = "aeroplane",
    keep_difficult: bool = False,
) -> Tuple[Tensor, Tensor]:
    """Convert one Pascal VOC annotation into aircraft boxes and labels.

    VOC coordinates are one-based. Subtracting one from xmin/ymin follows the
    convention used by common VOC evaluation code while xmax/ymax stay as-is.
    Label 0 is reserved for background; aircraft use label 1.
    """

    root = annotation.get("annotation", annotation)
    boxes: List[List[float]] = []

    for obj in _as_list(root.get("object")):
        if obj.get("name") != class_name:
            continue
        if not keep_difficult and int(obj.get("difficult", 0)) == 1:
            continue

        box = obj["bndbox"]
        xmin = max(0.0, float(box["xmin"]) - 1.0)
        ymin = max(0.0, float(box["ymin"]) - 1.0)
        xmax = float(box["xmax"])
        ymax = float(box["ymax"])
        if xmax > xmin and ymax > ymin:
            boxes.append([xmin, ymin, xmax, ymax])

    box_tensor = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)
    labels = torch.ones((len(boxes),), dtype=torch.int64)
    return box_tensor, labels


class DetectionTransform:
    """Small detection-safe augmentation pipeline."""

    def __init__(self, train: bool, horizontal_flip_probability: float = 0.5):
        self.train = train
        self.horizontal_flip_probability = horizontal_flip_probability

    def __call__(self, image: Image.Image, target: Dict[str, Tensor]):
        image_tensor = F.convert_image_dtype(F.pil_to_tensor(image), torch.float32)

        if self.train and random.random() < self.horizontal_flip_probability:
            image_tensor = F.hflip(image_tensor)
            width = image_tensor.shape[-1]
            boxes = target["boxes"].clone()
            if boxes.numel() > 0:
                boxes[:, [0, 2]] = width - boxes[:, [2, 0]]
                target["boxes"] = boxes

        return image_tensor, target


class VOCAircraftDataset(Dataset):
    """Pascal VOC wrapper that exposes only the ``aeroplane`` class."""

    def __init__(
        self,
        root: str | Path,
        *,
        year: str = "2007",
        image_set: str = "trainval",
        download: bool = False,
        train: bool = False,
        positive_only: bool = False,
        keep_difficult: bool = False,
    ) -> None:
        self.voc = VOCDetection(
            root=str(root),
            year=year,
            image_set=image_set,
            download=download,
        )
        self.transform = DetectionTransform(train=train)
        self.keep_difficult = keep_difficult
        self.indices = list(range(len(self.voc)))

        if positive_only:
            self.indices = [
                index
                for index in self.indices
                if self._has_aircraft(self.voc[index][1])
            ]

    def _has_aircraft(self, annotation: Dict[str, Any]) -> bool:
        boxes, _ = parse_aircraft_annotation(
            annotation, keep_difficult=self.keep_difficult
        )
        return bool(boxes.shape[0])

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int):
        voc_index = self.indices[item]
        image, annotation = self.voc[voc_index]
        boxes, labels = parse_aircraft_annotation(
            annotation, keep_difficult=self.keep_difficult
        )

        area = (
            (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            if boxes.numel()
            else torch.zeros((0,), dtype=torch.float32)
        )
        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([voc_index], dtype=torch.int64),
            "area": area,
            "iscrowd": torch.zeros((len(boxes),), dtype=torch.int64),
        }
        return self.transform(image.convert("RGB"), target)


def collate_fn(batch):
    """Keep differently sized images as a list for torchvision detectors."""

    return tuple(zip(*batch))

