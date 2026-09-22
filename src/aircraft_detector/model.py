"""ConvNeXt-Tiny FPN backbone wired into Faster R-CNN."""

from __future__ import annotations

from typing import Dict

from torch import nn
from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.backbone_utils import BackboneWithFPN
from torchvision.ops import MultiScaleRoIAlign


CONVNEXT_TINY_CHANNELS = [96, 192, 384, 768]
CONVNEXT_TINY_STAGE_INDICES = [1, 3, 5, 7]


def _set_trainable_stages(features: nn.Sequential, trainable_stages: int) -> None:
    if not 0 <= trainable_stages <= 4:
        raise ValueError("trainable_stages must be between 0 and 4")

    for parameter in features.parameters():
        parameter.requires_grad = False

    if trainable_stages == 0:
        return

    first_stage = CONVNEXT_TINY_STAGE_INDICES[-trainable_stages]
    # Include the downsampling block immediately before each trainable stage.
    first_layer = max(0, first_stage - 1)
    for layer_index in range(first_layer, len(features)):
        for parameter in features[layer_index].parameters():
            parameter.requires_grad = True


def build_aircraft_detector(
    *,
    num_classes: int = 2,
    pretrained_backbone: bool = True,
    trainable_stages: int = 2,
    min_size: int = 512,
    max_size: int = 1024,
) -> FasterRCNN:
    """Build Faster R-CNN with ImageNet-pretrained ConvNeXt-Tiny + FPN.

    ``num_classes`` includes the background class, so aircraft-only detection
    uses the default value of 2.
    """

    weights = (
        ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained_backbone else None
    )
    convnext = convnext_tiny(weights=weights)
    features = convnext.features
    _set_trainable_stages(features, trainable_stages)

    return_layers: Dict[str, str] = {
        str(stage): str(level)
        for level, stage in enumerate(CONVNEXT_TINY_STAGE_INDICES)
    }
    backbone = BackboneWithFPN(
        features,
        return_layers=return_layers,
        in_channels_list=CONVNEXT_TINY_CHANNELS,
        out_channels=256,
    )

    anchor_sizes = ((16,), (32,), (64,), (128,), (256,))
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
    anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)
    roi_pooler = MultiScaleRoIAlign(
        featmap_names=["0", "1", "2", "3"],
        output_size=7,
        sampling_ratio=2,
    )

    return FasterRCNN(
        backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler,
        min_size=min_size,
        max_size=max_size,
        box_score_thresh=0.05,
    )

