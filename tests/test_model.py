import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from aircraft_detector.model import build_aircraft_detector


def test_model_backbone_emits_five_fpn_levels():
    model = build_aircraft_detector(
        pretrained_backbone=False,
        trainable_stages=1,
        min_size=128,
        max_size=128,
    )
    features = model.backbone(torch.rand(1, 3, 128, 128))

    assert list(features) == ["0", "1", "2", "3", "pool"]
    assert all(feature.shape[1] == 256 for feature in features.values())

