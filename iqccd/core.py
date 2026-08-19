from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .gate import describe_change
from .model import ChangeQuery


def harmonic_consensus(
    first: torch.Tensor, second: torch.Tensor, epsilon: float = 1e-6
) -> torch.Tensor:
    """Parameter-free consensus that requires both evidence sources."""
    return 2.0 * first * second / (first + second).clamp_min(epsilon)


class IQCCD(nn.Module):
    """Interventional query and building-semantic consensus."""

    def __init__(self, query: ChangeQuery, building_prior: nn.Module) -> None:
        super().__init__()
        self.query = query
        self.building_prior = building_prior
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        self.eval()

    def train(self, mode: bool = True) -> "IQCCD":
        super().train(False)
        return self

    @torch.inference_mode()
    def forward(
        self, first: torch.Tensor, second: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        if first.shape != second.shape:
            raise ValueError(f"paired shapes differ: {first.shape} vs {second.shape}")
        size = tuple(first.shape[-2:])
        occupancy_t1 = self.building_prior(first)
        occupancy_t2 = self.building_prior(second)
        query_logits = F.interpolate(
            self.query(first, second)["logits"],
            size=size,
            mode="bilinear",
            align_corners=False,
        )
        query_probability = query_logits.sigmoid()
        building_support = torch.maximum(occupancy_t1, occupancy_t2)
        change_probability = harmonic_consensus(
            query_probability, building_support
        )
        return {
            "occupancy_t1": occupancy_t1,
            "occupancy_t2": occupancy_t2,
            "building_support": building_support,
            "query_probability": query_probability,
            "change_probability": change_probability,
            "descriptors": describe_change(change_probability),
        }
