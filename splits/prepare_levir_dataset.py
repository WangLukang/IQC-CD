from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


TILE_FIELDS = [
    "id",
    "scene",
    "source_split",
    "source_id",
    "row",
    "col",
    "x",
    "y",
    "tile_size",
    "change_pixels",
    "change_ratio",
    "t1",
    "t2",
    "mask",
]
MANIFEST_FIELDS = ["id", "t1", "t2", "label", "mask", "change_ratio"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Crop LEVIR-CD into 224x224 paired tiles while retaining the official "
            "train/validation/test scene partition."
        )
    )
    parser.add_argument("--source-root", default="data/LEVIR-CD-raw")
    parser.add_argument("--output-root", default="data/LEVIR-CD")
    parser.add_argument("--split-root", default="splits/LEVIR-CD")
    parser.add_argument("--tile-size", type=int, default=224)
    parser.add_argument("--min-change-pixels", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve(repo_root: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (repo_root / path).resolve()


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def axis_positions(length: int, tile_size: int) -> list[int]:
    if length < tile_size:
        raise ValueError(f"Image side {length} is smaller than tile size {tile_size}")
    positions = list(range(0, max(length - tile_size + 1, 1), tile_size))
    last = length - tile_size
    if positions[-1] != last:
        positions.append(last)
    return positions


def numeric_key(path: Path) -> tuple[str, int]:
    stem = path.stem
    try:
        prefix, number = stem.rsplit("_", 1)
        return prefix, int(number)
    except (ValueError, IndexError):
        return stem, 0


def prepare_split(
    source_root: Path,
    output_root: Path,
    split: str,
    tile_size: int,
    min_change_pixels: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    split_root = source_root / split
    for subdirectory in ("A", "B", "label"):
        if not (split_root / subdirectory).is_dir():
            raise FileNotFoundError(split_root / subdirectory)

    tile_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    for t1_path in sorted((split_root / "A").glob("*.png"), key=numeric_key):
        t2_path = split_root / "B" / t1_path.name
        mask_path = split_root / "label" / t1_path.name
        if not t2_path.is_file() or not mask_path.is_file():
            raise FileNotFoundError(f"Missing pair for {t1_path.name}")

        t1 = Image.open(t1_path).convert("RGB")
        t2 = Image.open(t2_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        if t1.size != t2.size or t1.size != mask.size:
            raise ValueError(
                f"Size mismatch for {t1_path.name}: {t1.size}, {t2.size}, {mask.size}"
            )

        width, height = t1.size
        source_id = t1_path.stem
        for row, y in enumerate(axis_positions(height, tile_size)):
            for col, x in enumerate(axis_positions(width, tile_size)):
                box = (x, y, x + tile_size, y + tile_size)
                sample_id = f"LEVIR_{split}_{source_id}_r{row:04d}_c{col:04d}"
                t1_rel = f"T1/{sample_id}.png"
                t2_rel = f"T2/{sample_id}.png"
                mask_rel = f"masks/{sample_id}.png"

                for image, relative_path in ((t1, t1_rel), (t2, t2_rel)):
                    destination = output_root / relative_path
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    image.crop(box).save(destination)

                mask_array = (np.asarray(mask.crop(box)) > 0).astype(np.uint8) * 255
                mask_destination = output_root / mask_rel
                mask_destination.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(mask_array, mode="L").save(mask_destination)

                change_pixels = int((mask_array > 0).sum())
                change_ratio = change_pixels / float(tile_size * tile_size)
                tile_row = {
                    "id": sample_id,
                    "scene": f"LEVIR_{split}_{source_id}",
                    "source_split": split,
                    "source_id": source_id,
                    "row": row,
                    "col": col,
                    "x": x,
                    "y": y,
                    "tile_size": tile_size,
                    "change_pixels": change_pixels,
                    "change_ratio": f"{change_ratio:.8f}",
                    "t1": t1_rel,
                    "t2": t2_rel,
                    "mask": mask_rel,
                }
                tile_rows.append(tile_row)
                manifest_rows.append(
                    {
                        "id": sample_id,
                        "t1": t1_rel,
                        "t2": t2_rel,
                        "label": int(change_pixels >= min_change_pixels),
                        "mask": mask_rel,
                        "change_ratio": f"{change_ratio:.8f}",
                    }
                )
    return tile_rows, manifest_rows


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    source_root = resolve(repo_root, args.source_root)
    output_root = resolve(repo_root, args.output_root)
    split_root = resolve(repo_root, args.split_root)

    if output_root.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Output already exists: {output_root}. Pass --overwrite to rebuild it."
            )
        shutil.rmtree(output_root)

    all_tiles: list[dict[str, Any]] = []
    manifests: dict[str, list[dict[str, Any]]] = {}
    for split in ("train", "val", "test"):
        tile_rows, manifest_rows = prepare_split(
            source_root,
            output_root,
            split,
            tile_size=args.tile_size,
            min_change_pixels=args.min_change_pixels,
        )
        all_tiles.extend(tile_rows)
        manifests[split] = manifest_rows

    write_csv(output_root / "tiles.csv", all_tiles, TILE_FIELDS)
    for split, rows in manifests.items():
        write_csv(split_root / f"{split}.csv", rows, MANIFEST_FIELDS)

    print(f"Prepared LEVIR-CD tiles: {output_root}")
    print(f"Total: {len(all_tiles)} samples")
    for split in ("train", "val", "test"):
        rows = manifests[split]
        positives = sum(int(row["label"]) for row in rows)
        print(
            f"{split:>7}: {len(rows):5d} samples, "
            f"{positives:5d} changed, {len(rows) - positives:5d} unchanged"
        )


if __name__ == "__main__":
    main()
