from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

import torch
import torch.nn.functional as F

from .model import ChangeQuery


def irregular_support(
    batch_size: int,
    size: tuple[int, int],
    device: torch.device,
    *,
    quantiles: tuple[float, float] = (0.65, 0.85),
    field_size: int = 7,
) -> torch.Tensor:
    """Sample smooth irregular regions with no learned shape parameters."""
    low, high = quantiles
    field = F.interpolate(
        torch.rand((batch_size, 1, field_size, field_size), device=device),
        size=size,
        mode="bicubic",
        align_corners=False,
    )
    flat = field.flatten(1)
    q = torch.empty(batch_size, device=device).uniform_(low, high)
    threshold = torch.quantile(flat, q, dim=1).diagonal().view(-1, 1, 1, 1)
    return field >= threshold


def exchange_negative_features(
    features: Sequence[torch.Tensor], support: torch.Tensor
) -> tuple[torch.Tensor, ...]:
    """Replace F2 features only inside a known support in unchanged pairs."""
    exchanged: list[torch.Tensor] = []
    for feature in features:
        mask = F.interpolate(
            support.float(), size=feature.shape[-2:], mode="nearest"
        )
        if feature.shape[0] > 1:
            donor = feature.roll(1, dims=0)
        else:
            donor = feature.roll(
                shifts=(
                    max(1, feature.shape[-2] // 3),
                    max(1, feature.shape[-1] // 3),
                ),
                dims=(-2, -1),
            )
        exchanged.append(feature * (1.0 - mask) + donor * mask)
    return tuple(exchanged)


def intervention_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    positive = target.sum().clamp_min(1.0)
    negative = (1.0 - target).sum().clamp_min(1.0)
    binary = F.binary_cross_entropy_with_logits(
        logits, target, pos_weight=(negative / positive).detach()
    )
    probability = logits.sigmoid()
    intersection = (probability * target).sum()
    dice = 1.0 - (2.0 * intersection + 1.0) / (
        probability.sum() + target.sum() + 1.0
    )
    return binary + dice


def smoothness_loss(probability: torch.Tensor) -> torch.Tensor:
    dx = (probability[..., 1:] - probability[..., :-1]).abs().mean()
    dy = (probability[..., 1:, :] - probability[..., :-1, :]).abs().mean()
    return dx + dy


def train_epoch(
    model: ChangeQuery,
    loader: Iterable[dict[str, object]],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    loss_weights: Mapping[str, float] | None = None,
) -> dict[str, float]:
    weights = {"negative": 1.0, "intervention": 1.0, "smoothness": 0.05}
    if loss_weights:
        weights.update({key: float(value) for key, value in loss_weights.items()})
    model.train()
    totals = {"loss": 0.0, "mil": 0.0, "intervention": 0.0}
    samples = 0
    for batch in loader:
        labels = batch["label"].float().to(device)
        first, second = model.encode_pair(
            batch["t1"].to(device, non_blocking=True),
            batch["t2"].to(device, non_blocking=True),
        )
        output = model.decode(first, second)
        logits = output["logits"]
        mil = F.binary_cross_entropy_with_logits(output["image_logits"], labels)

        negative_index = labels < 0.5
        zero = logits.sum() * 0.0
        negative = (
            F.softplus(logits[negative_index]).mean()
            if negative_index.any()
            else zero
        )
        synthetic = zero
        if negative_index.any():
            f1 = tuple(value[negative_index] for value in first)
            f2 = tuple(value[negative_index] for value in second)
            support = irregular_support(
                int(negative_index.sum()), tuple(logits.shape[-2:]), device
            )
            exchanged = exchange_negative_features(f2, support)
            synthetic_logits = model.head(f1, exchanged)
            synthetic = intervention_loss(synthetic_logits, support.float())
        smooth = smoothness_loss(logits.sigmoid())
        loss = (
            mil
            + weights["negative"] * negative
            + weights["intervention"] * synthetic
            + weights["smoothness"] * smooth
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        count = labels.numel()
        totals["loss"] += float(loss.detach()) * count
        totals["mil"] += float(mil.detach()) * count
        totals["intervention"] += float(synthetic.detach()) * count
        samples += count
    return {name: value / max(samples, 1) for name, value in totals.items()}


@torch.inference_mode()
def image_predictions(
    model: ChangeQuery,
    loader: Iterable[dict[str, object]],
    device: torch.device,
) -> tuple[list[int], list[float]]:
    model.eval()
    labels: list[int] = []
    scores: list[float] = []
    for batch in loader:
        output = model(
            batch["t1"].to(device, non_blocking=True),
            batch["t2"].to(device, non_blocking=True),
        )
        labels.extend(int(value) for value in batch["label"])
        scores.extend(float(value) for value in output["image_logits"].sigmoid())
    return labels, scores
