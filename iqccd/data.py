from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


IMAGENET_MEAN = torch.tensor((0.485, 0.456, 0.406))[:, None, None]
IMAGENET_STD = torch.tensor((0.229, 0.224, 0.225))[:, None, None]


@dataclass(frozen=True)
class PairRecord:
    sample_id: str
    first: Path
    second: Path
    label: int
    mask: Path | None


def read_manifest(root: Path, manifest: Path) -> list[PairRecord]:
    records: list[PairRecord] = []
    with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            label = int(row["label"])
            if label not in (0, 1):
                raise ValueError(f"image label must be 0 or 1, got {label}")
            records.append(
                PairRecord(
                    sample_id=row["id"],
                    first=(root / row["t1"]).resolve(),
                    second=(root / row["t2"]).resolve(),
                    label=label,
                    mask=(root / row["mask"]).resolve() if row.get("mask") else None,
                )
            )
    if not records:
        raise ValueError(f"empty manifest: {manifest}")
    return records


def deterministic_subset(
    records: Sequence[PairRecord], limit: int, seed: int
) -> list[PairRecord]:
    if limit <= 0 or limit >= len(records):
        return list(records)
    indices = list(range(len(records)))
    random.Random(seed).shuffle(indices)
    selected = indices[:limit]
    if limit >= 2:
        present = {records[index].label for index in selected}
        for missing in {0, 1} - present:
            replacement = next(
                index for index in indices[limit:] if records[index].label == missing
            )
            replace_at = next(
                position
                for position, index in enumerate(selected)
                if sum(
                    records[item].label == records[index].label for item in selected
                )
                > 1
            )
            selected[replace_at] = replacement
    return [records[index] for index in selected]


def image_tensor(path: Path, image_size: int) -> torch.Tensor:
    with Image.open(path) as image:
        image = image.convert("RGB").resize(
            (image_size, image_size), Image.Resampling.BICUBIC
        )
        array = np.asarray(image, dtype=np.float32).copy() / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    return (tensor - IMAGENET_MEAN) / IMAGENET_STD


class PairDataset(Dataset):
    def __init__(
        self,
        records: Sequence[PairRecord],
        *,
        image_size: int,
        augment: bool,
        load_mask: bool = False,
    ) -> None:
        self.records = list(records)
        self.image_size = int(image_size)
        self.augment = bool(augment)
        self.load_mask = bool(load_mask)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, object]:
        record = self.records[index]
        first = image_tensor(record.first, self.image_size)
        second = image_tensor(record.second, self.image_size)
        mask: torch.Tensor | None = None
        if self.load_mask:
            if record.mask is None:
                raise ValueError(f"mask path is missing for {record.sample_id}")
            with Image.open(record.mask) as image:
                image = image.convert("L").resize(
                    (self.image_size, self.image_size), Image.Resampling.NEAREST
                )
                mask = torch.from_numpy(
                    (np.asarray(image) > 0).astype(np.uint8)
                )
        if self.augment:
            first, second, mask = self._augment(first, second, mask)
        sample: dict[str, object] = {
            "id": record.sample_id,
            "t1": first,
            "t2": second,
            "label": record.label,
        }
        if mask is not None:
            sample["mask"] = mask
        return sample

    @staticmethod
    def _augment(
        first: torch.Tensor,
        second: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        if torch.rand(()) < 0.5:
            first, second = first.flip(-1), second.flip(-1)
            mask = None if mask is None else mask.flip(-1)
        if torch.rand(()) < 0.5:
            first, second = first.flip(-2), second.flip(-2)
            mask = None if mask is None else mask.flip(-2)
        turns = int(torch.randint(0, 4, ()).item())
        if turns:
            first = torch.rot90(first, turns, (-2, -1))
            second = torch.rot90(second, turns, (-2, -1))
            mask = None if mask is None else torch.rot90(mask, turns, (-2, -1))
        return first, second, mask


def make_loader(
    records: Sequence[PairRecord],
    *,
    image_size: int,
    batch_size: int,
    workers: int,
    seed: int,
    training: bool,
    load_mask: bool = False,
) -> DataLoader:
    if training and load_mask:
        raise ValueError("training must not load pixel masks")
    generator = torch.Generator().manual_seed(seed)
    sampler = None
    if training:
        counts = {
            label: sum(record.label == label for record in records)
            for label in (0, 1)
        }
        if not counts[0] or not counts[1]:
            raise ValueError("weak training requires both image-level classes")
        sampler = WeightedRandomSampler(
            [1.0 / counts[record.label] for record in records],
            num_samples=len(records),
            replacement=True,
            generator=generator,
        )
    return DataLoader(
        PairDataset(
            records,
            image_size=image_size,
            augment=training,
            load_mask=load_mask,
        ),
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
        generator=generator,
    )
