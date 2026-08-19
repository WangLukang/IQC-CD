from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class FrozenDINOv3(nn.Module):
    """Frozen multi-scale DINOv3 ConvNeXt encoder."""

    def __init__(
        self,
        model_name: str,
        out_indices: Sequence[int],
        weights: Path | None = None,
    ) -> None:
        super().__init__()
        options: dict[str, object] = {
            "features_only": True,
            "out_indices": tuple(out_indices),
            "pretrained": True,
        }
        if weights is not None:
            if not weights.is_file():
                raise FileNotFoundError(weights)
            options["pretrained_cfg_overlay"] = {"file": str(weights)}
        self.encoder = timm.create_model(model_name, **options)
        self.channels = tuple(int(c) for c in self.encoder.feature_info.channels())
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        self.eval()

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return tuple(self.encoder(image))


class MultiScaleQueryHead(nn.Module):
    """Time-swap-invariant query head over four DINOv3 feature scales."""

    def __init__(self, channels: Sequence[int], width: int = 24) -> None:
        super().__init__()
        self.projections = nn.ModuleList(
            nn.Sequential(nn.Conv2d(channel + 1, width, 1), nn.GELU())
            for channel in channels
        )
        fused = width * len(channels)
        self.fusion = nn.Sequential(
            nn.Conv2d(fused, 2 * width, 1),
            nn.GELU(),
            nn.Conv2d(2 * width, 2 * width, 3, padding=1, groups=2 * width),
            nn.GELU(),
            nn.Conv2d(2 * width, width, 1),
            nn.GELU(),
        )
        self.mask = nn.Conv2d(width, 1, 1)

    def forward(
        self,
        first: Sequence[torch.Tensor],
        second: Sequence[torch.Tensor],
    ) -> torch.Tensor:
        output_size = first[0].shape[-2:]
        values: list[torch.Tensor] = []
        for f1, f2, projection in zip(
            first, second, self.projections, strict=True
        ):
            u1, u2 = F.normalize(f1, dim=1), F.normalize(f2, dim=1)
            absolute = (u1 - u2).abs()
            cosine = (1.0 - (u1 * u2).sum(dim=1, keepdim=True)).clamp_min(0.0)
            value = projection(torch.cat((absolute, cosine), dim=1))
            if value.shape[-2:] != output_size:
                value = F.interpolate(
                    value, size=output_size, mode="bilinear", align_corners=False
                )
            values.append(value)
        return self.mask(self.fusion(torch.cat(values, dim=1)))


class ChangeQuery(nn.Module):
    """Frozen DINOv3 encoder with a lightweight trainable dense query head."""

    def __init__(
        self,
        *,
        model_name: str,
        out_indices: Sequence[int] = (0, 1, 2, 3),
        weights: Path | None = None,
        decoder_width: int = 24,
        top_fraction: float = 0.05,
    ) -> None:
        super().__init__()
        if not 0.0 < top_fraction <= 1.0:
            raise ValueError("top_fraction must be in (0, 1]")
        self.encoder = FrozenDINOv3(model_name, out_indices, weights)
        self.head = MultiScaleQueryHead(self.encoder.channels, decoder_width)
        self.top_fraction = float(top_fraction)

    def train(self, mode: bool = True) -> "ChangeQuery":
        super().train(mode)
        self.encoder.eval()
        return self

    def encode_pair(
        self, first: torch.Tensor, second: torch.Tensor
    ) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
        use_amp = first.device.type == "cuda"
        with torch.no_grad(), torch.autocast(
            device_type=first.device.type,
            dtype=torch.float16 if use_amp else torch.bfloat16,
            enabled=use_amp,
        ):
            combined = self.encoder(torch.cat((first, second), dim=0))
        split = [feature.float().chunk(2, dim=0) for feature in combined]
        return tuple(x[0] for x in split), tuple(x[1] for x in split)

    def decode(
        self,
        first: Sequence[torch.Tensor],
        second: Sequence[torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        logits = self.head(first, second)
        flat = logits.flatten(1)
        count = max(1, round(flat.shape[1] * self.top_fraction))
        return {
            "logits": logits,
            "image_logits": flat.topk(count, dim=1).values.mean(dim=1),
        }

    def forward(
        self, first: torch.Tensor, second: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        return self.decode(*self.encode_pair(first, second))

    @property
    def trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def trainable_state_dict(self) -> dict[str, torch.Tensor]:
        return self.head.state_dict()

    def load_trainable_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        self.head.load_state_dict(state, strict=True)
