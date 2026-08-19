import numpy as np
import torch
import torch.nn as nn

from iqccd.core import IQCCD, harmonic_consensus
from iqccd.gate import ImageGate, describe_change
from iqccd.intervention import exchange_negative_features, irregular_support


def test_harmonic_consensus() -> None:
    first = torch.tensor([0.2, 0.8, 0.0])
    second = torch.tensor([0.8, 0.8, 1.0])
    result = harmonic_consensus(first, second)
    assert torch.allclose(result, torch.tensor([0.32, 0.8, 0.0]))


def test_descriptors_and_gate_roundtrip() -> None:
    change = torch.zeros(2, 1, 16, 16)
    change[1, :, 4:12, 4:12] = 1.0
    descriptors = describe_change(change).numpy()
    gate = ImageGate.fit(descriptors, np.array([0, 1], dtype=np.uint8))
    probability = gate.probability(descriptors)
    restored = ImageGate.from_dict(gate.to_dict())
    assert descriptors.shape == (2, 4)
    assert np.allclose(probability, restored.probability(descriptors))


def test_feature_intervention() -> None:
    torch.manual_seed(7)
    support = irregular_support(2, (16, 16), torch.device("cpu"))
    feature = torch.stack((torch.zeros(1, 8, 8), torch.ones(1, 8, 8)))
    exchanged = exchange_negative_features((feature,), support)[0]
    assert support.shape == (2, 1, 16, 16)
    assert 0 < int(support.sum()) < support.numel()
    assert not torch.equal(exchanged, feature)


class DummyQuery(nn.Module):
    def forward(self, first: torch.Tensor, second: torch.Tensor):
        return {"logits": torch.zeros(first.shape[0], 1, 4, 4)}


class DummyPrior(nn.Module):
    def forward(self, image: torch.Tensor):
        return torch.full(
            (image.shape[0], 1, *image.shape[-2:]), 0.5, device=image.device
        )


def test_iqccd_output_contract() -> None:
    model = IQCCD(DummyQuery(), DummyPrior())
    image = torch.zeros(2, 3, 16, 16)
    output = model(image, image)
    assert output["change_probability"].shape == (2, 1, 16, 16)
    assert output["descriptors"].shape == (2, 4)
