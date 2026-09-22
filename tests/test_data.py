import random

import torch
from PIL import Image

from aircraft_detector.data import DetectionTransform, parse_aircraft_annotation


def test_parse_aircraft_annotation_filters_other_classes_and_difficult():
    annotation = {
        "annotation": {
            "object": [
                {
                    "name": "aeroplane",
                    "difficult": "0",
                    "bndbox": {"xmin": "11", "ymin": "21", "xmax": "50", "ymax": "80"},
                },
                {
                    "name": "person",
                    "difficult": "0",
                    "bndbox": {"xmin": "1", "ymin": "2", "xmax": "3", "ymax": "4"},
                },
                {
                    "name": "aeroplane",
                    "difficult": "1",
                    "bndbox": {"xmin": "5", "ymin": "6", "xmax": "7", "ymax": "8"},
                },
            ]
        }
    }

    boxes, labels = parse_aircraft_annotation(annotation)

    assert torch.equal(boxes, torch.tensor([[10.0, 20.0, 50.0, 80.0]]))
    assert torch.equal(labels, torch.tensor([1]))


def test_horizontal_flip_updates_boxes(monkeypatch):
    monkeypatch.setattr(random, "random", lambda: 0.0)
    transform = DetectionTransform(train=True, horizontal_flip_probability=0.5)
    image = Image.new("RGB", (100, 50))
    target = {"boxes": torch.tensor([[10.0, 5.0, 30.0, 25.0]])}

    _, transformed = transform(image, target)

    assert torch.equal(
        transformed["boxes"], torch.tensor([[70.0, 5.0, 90.0, 25.0]])
    )

