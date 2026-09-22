#!/usr/bin/env python3
"""Render aircraft ground-truth boxes from Pascal VOC XML annotations."""

from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--voc-root", type=Path, default=Path("data/VOCdevkit/VOC2007"))
    parser.add_argument("--output-dir", type=Path, default=Path("assets/ground_truth"))
    parser.add_argument("image_ids", nargs="+", help="VOC image IDs, such as 000045")
    return parser.parse_args()


def aircraft_boxes(annotation_path: Path) -> list[tuple[int, int, int, int]]:
    root = ET.parse(annotation_path).getroot()
    boxes = []
    for obj in root.findall("object"):
        if obj.findtext("name") != "aeroplane":
            continue
        if obj.findtext("difficult", "0") == "1":
            continue
        bbox = obj.find("bndbox")
        if bbox is None:
            continue
        xmin = max(0, int(bbox.findtext("xmin", "0")) - 1)
        ymin = max(0, int(bbox.findtext("ymin", "0")) - 1)
        xmax = int(bbox.findtext("xmax", "0"))
        ymax = int(bbox.findtext("ymax", "0"))
        if xmax > xmin and ymax > ymin:
            boxes.append((xmin, ymin, xmax, ymax))
    return boxes


def render(image_path: Path, boxes: list[tuple[int, int, int, int]], output: Path) -> None:
    with Image.open(image_path) as source:
        image = source.convert("RGB")

    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    width = max(3, min(image.size) // 120)

    for box in boxes:
        x1, y1, x2, y2 = box
        draw.rectangle(box, outline="#ff375f", width=width)
        label = "GT: aircraft"
        left, top, right, bottom = draw.textbbox((0, 0), label, font=font)
        label_width = right - left + 12
        label_height = bottom - top + 8
        label_y = max(0, y1 - label_height)
        draw.rectangle(
            (x1, label_y, x1 + label_width, label_y + label_height),
            fill="#ff375f",
        )
        draw.text((x1 + 6, label_y + 4), label, fill="white", font=font)

    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output, quality=92, optimize=True)


def main() -> None:
    args = parse_args()
    for image_id in args.image_ids:
        if not image_id.isdigit():
            raise ValueError(f"VOC image ID must contain digits only: {image_id}")
        annotation_path = args.voc_root / "Annotations" / f"{image_id}.xml"
        image_path = args.voc_root / "JPEGImages" / f"{image_id}.jpg"
        if not annotation_path.is_file() or not image_path.is_file():
            raise FileNotFoundError(f"Missing image or annotation for {image_id}")
        boxes = aircraft_boxes(annotation_path)
        if not boxes:
            raise ValueError(f"No non-difficult aircraft boxes for {image_id}")
        output = args.output_dir / f"{image_id}_gt.jpg"
        render(image_path, boxes, output)
        print(f"{image_id}: {len(boxes)} aircraft box(es) -> {output}")


if __name__ == "__main__":
    main()
