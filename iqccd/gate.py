from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def describe_change(change: torch.Tensor) -> torch.Tensor:
    """Peak, top-mean, RMS and active-area descriptors."""
    flat = change.flatten(1)
    top_count = max(1, flat.shape[1] // 256)
    return torch.stack(
        (
            flat.amax(dim=1),
            flat.topk(top_count, dim=1).values.mean(dim=1),
            flat.square().mean(dim=1).sqrt(),
            (flat >= 0.5).float().mean(dim=1),
        ),
        dim=1,
    )


@dataclass(frozen=True)
class ImageGate:
    mean: np.ndarray
    scale: np.ndarray
    weight: np.ndarray
    bias: float

    @classmethod
    def fit(cls, descriptors: np.ndarray, labels: np.ndarray) -> "ImageGate":
        scaler = StandardScaler().fit(descriptors)
        classifier = LogisticRegression(
            class_weight="balanced",
            solver="liblinear",
            max_iter=200,
            random_state=0,
        ).fit(scaler.transform(descriptors), labels)
        return cls(
            mean=scaler.mean_,
            scale=scaler.scale_,
            weight=classifier.coef_[0],
            bias=float(classifier.intercept_[0]),
        )

    def probability(self, descriptors: np.ndarray) -> np.ndarray:
        standardized = (descriptors - self.mean) / self.scale
        logits = standardized @ self.weight + self.bias
        return 1.0 / (1.0 + np.exp(-np.clip(logits, -40.0, 40.0)))

    def to_dict(self) -> dict[str, object]:
        return {
            "descriptor_names": ["peak", "top_mean", "rms", "active_area"],
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
            "weight": self.weight.tolist(),
            "bias": self.bias,
        }

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "ImageGate":
        return cls(
            mean=np.asarray(value["mean"], dtype=np.float64),
            scale=np.asarray(value["scale"], dtype=np.float64),
            weight=np.asarray(value["weight"], dtype=np.float64),
            bias=float(value["bias"]),
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> "ImageGate":
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))
