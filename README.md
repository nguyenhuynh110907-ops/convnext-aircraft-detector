# ConvNeXt Aircraft Detector

An end-to-end, portfolio-ready aircraft object detector built with **Faster
R-CNN**, an ImageNet-pretrained **ConvNeXt-Tiny** backbone, and a feature
pyramid network (FPN). The project trains on the `aeroplane` class from Pascal
VOC and predicts aircraft bounding boxes in new images.

## Why this architecture?

ConvNeXt provides strong hierarchical visual features, while FPN exposes four
feature resolutions for aircraft at different scales. Faster R-CNN adds the
region proposal and box-classification heads needed for true object detection
rather than image-level classification.

```text
image -> ConvNeXt stages -> FPN (P2-P6) -> RPN -> RoI heads -> aircraft boxes
```

## Features

- ConvNeXt-Tiny backbone with configurable fine-tuning depth
- Five-level FPN and anchors for small-to-large aircraft
- Automatic Pascal VOC 2007 download
- Mixed-precision CUDA training and Apple Silicon MPS support
- mAP, mAP@50, mAP@75, and recall evaluation
- Resume training and atomic best/last checkpoints
- Batch inference with annotated image output
- Unit tests for annotation parsing, box augmentation, and FPN outputs

## Installation

Python 3.9+ is supported. A GPU is recommended for training.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

For CUDA, install the matching PyTorch build from the official PyTorch selector
before running the final command.

## Train

The first run downloads Pascal VOC 2007 into `data/`:

```bash
python train.py --download --epochs 12 --batch-size 2
```

On an NVIDIA GPU, mixed precision is enabled by default. On Apple Silicon the
script selects MPS automatically. To reduce memory usage:

```bash
python train.py --download --batch-size 1 --min-size 384 --max-size 768
```

Important outputs:

```text
runs/convnext_tiny/best.pt       best validation mAP@50 checkpoint
runs/convnext_tiny/last.pt       latest checkpoint
runs/convnext_tiny/history.jsonl per-epoch metrics
```

Resume an interrupted run:

```bash
python train.py --resume runs/convnext_tiny/last.pt --epochs 20
```

## Predict

Run inference on one image:

```bash
python predict.py \
  --checkpoint runs/convnext_tiny/best.pt \
  --input path/to/aircraft.jpg \
  --threshold 0.5
```

Or pass a directory to process every supported image inside it:

```bash
python predict.py \
  --checkpoint runs/convnext_tiny/best.pt \
  --input path/to/images \
  --output-dir outputs
```

## Test

```bash
pytest -q
```

## Dataset notes

Pascal VOC uses the British class name `aeroplane`. This project maps it to the
single foreground label `aircraft` and ignores the other 19 VOC classes. By
default, both positive images and aircraft-free negative images are used.
`--positive-only` is useful for a quick experiment but can increase false
positives because the model sees less background diversity.

## Suggested portfolio experiments

1. Compare 1, 2, and 4 trainable ConvNeXt stages.
2. Plot mAP@50 against image resolution and inference latency.
3. Fine-tune on a domain-specific aerial dataset and compare with Pascal VOC.
4. Add Grad-CAM or proposal visualizations to explain model behavior.

## Limitations

- Pascal VOC contains relatively few aircraft examples and mostly natural
  ground-level images.
- Tiny aircraft in satellite imagery need larger inputs, tuned anchors, and a
  domain-specific dataset.
- The repository provides the training pipeline, not pretrained detector
  weights; train `best.pt` before inference.

