#!/usr/bin/env python3
"""Run aircraft detection and draw bounding boxes on images."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List

import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision.transforms import functional as F

from aircraft_detector.checkpoint import load_checkpoint
from aircraft_detector.model import build_aircraft_detector


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--min-size", type=int, default=512)
    parser.add_argument("--max-size", type=int, default=1024)
    return parser.parse_args()


def choose_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def iter_images(path: Path) -> Iterable[Path]:
    if path.is_file():
        yield path
        return
    yield from sorted(
        item for item in path.iterdir() if item.suffix.lower() in IMAGE_SUFFIXES
    )


def draw_predictions(
    image: Image.Image,
    boxes: torch.Tensor,
    scores: torch.Tensor,
    threshold: float,
) -> Image.Image:
    result = image.copy()
    draw = ImageDraw.Draw(result)
    font = ImageFont.load_default()
    line_width = max(2, round(min(result.size) / 250))

    for box, score in zip(boxes.tolist(), scores.tolist()):
        if score < threshold:
            continue
        x1, y1, x2, y2 = box
        label = f"aircraft {score:.2f}"
        draw.rectangle((x1, y1, x2, y2), outline="#ff3b30", width=line_width)
        text_box = draw.textbbox((x1, y1), label, font=font)
        text_height = text_box[3] - text_box[1] + 6
        label_top = max(0, y1 - text_height)
        draw.rectangle(
            (x1, label_top, x1 + text_box[2] - text_box[0] + 8, y1),
            fill="#ff3b30",
        )
        draw.text((x1 + 4, label_top + 2), label, fill="white", font=font)
    return result


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    model = build_aircraft_detector(
        pretrained_backbone=False,
        min_size=args.min_size,
        max_size=args.max_size,
    ).to(device)
    checkpoint = load_checkpoint(args.checkpoint, device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    image_paths: List[Path] = list(iter_images(args.input))
    if not image_paths:
        raise FileNotFoundError(f"No supported images found in {args.input}")

    for image_path in image_paths:
        image = Image.open(image_path).convert("RGB")
        tensor = F.convert_image_dtype(F.pil_to_tensor(image), torch.float32).to(device)
        with torch.inference_mode():
            prediction = model([tensor])[0]
        result = draw_predictions(
            image,
            prediction["boxes"].cpu(),
            prediction["scores"].cpu(),
            args.threshold,
        )
        destination = args.output_dir / f"{image_path.stem}_detected.jpg"
        result.save(destination, quality=95)
        kept = int((prediction["scores"] >= args.threshold).sum().item())
        print(f"{image_path.name}: {kept} aircraft -> {destination}")


if __name__ == "__main__":
    main()

