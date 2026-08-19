from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torchvision.models.detection.roi_heads import paste_masks_in_image

from satlaspretrain_models.models.backbones import SwinBackbone
from satlaspretrain_models.models.fpn import FPN
from satlaspretrain_models.models.heads import FRCNNHead

from .data import IMAGENET_MEAN, IMAGENET_STD


def _component(
    state: dict[str, torch.Tensor], prefix: str
) -> dict[str, torch.Tensor]:
    return {
        key[len(prefix) :]: value
        for key, value in state.items()
        if key.startswith(prefix)
    }


class SatlasBuildingPrior(nn.Module):
    """Frozen SatlasPretrain aerial building-instance branch."""

    def __init__(self, checkpoint: Path, detection_floor: float = 0.2) -> None:
        super().__init__()
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.backbone = SwinBackbone(3, arch="swinb")
        self.backbone.load_state_dict(_component(state, "backbone."), strict=True)
        self.fpn = FPN(self.backbone.out_channels)
        self.fpn.load_state_dict(
            _component(state, "intermediates.0."), strict=True
        )
        # Head 14 is the official two-class aerial building instance head.
        self.head = FRCNNHead("instance", self.fpn.out_channels, num_categories=2)
        self.head.load_state_dict(_component(state, "heads.14."), strict=True)
        self.head.rpn._pre_nms_top_n["testing"] = 600
        self.head.rpn._post_nms_top_n["testing"] = 200
        self.head.roi_heads.detections_per_img = 50
        self.detection_floor = float(detection_floor)
        self.register_buffer("mean", IMAGENET_MEAN.unsqueeze(0))
        self.register_buffer("std", IMAGENET_STD.unsqueeze(0))
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        self.eval()

    def train(self, mode: bool = True) -> "SatlasBuildingPrior":
        super().train(False)
        return self

    @torch.inference_mode()
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        raw = (image * self.std + self.mean).clamp(0.0, 1.0)
        features = self.fpn(self.backbone(raw))
        detections, _ = self.head(raw, features)
        output: list[torch.Tensor] = []
        size = tuple(image.shape[-2:])
        for detection in detections:
            selected = (detection["labels"] == 1) & (
                detection["scores"] >= self.detection_floor
            )
            if not selected.any():
                output.append(torch.zeros((1, *size), device=image.device))
                continue
            masks = paste_masks_in_image(
                detection["masks"][selected], detection["boxes"][selected], size
            )
            confidence = detection["scores"][selected].view(-1, 1, 1, 1)
            output.append((masks * confidence).amax(dim=0))
        return torch.stack(output)
