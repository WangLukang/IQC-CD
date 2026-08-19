from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


Image.MAX_IMAGE_PIXELS = None

TILE_FIELDS = [
    "id",
    "scene",
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
        description="Crop the WHU-CD mosaic and reproduce the IQC-CD spatial split."
    )
    parser.add_argument("--source-root", default="data/WHU-CD-raw")
    parser.add_argument("--output-root", default="data/WHU-CD")
    parser.add_argument("--split-root", default="splits/WHU-CD")
    parser.add_argument("--tile-size", type=int, default=224)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--gap-tiles", type=int, default=1)
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


def spatial_split(
    rows: list[dict[str, Any]],
    train_ratio: float,
    val_ratio: float,
    gap_tiles: int,
) -> dict[str, list[dict[str, Any]]]:
    if not 0 < train_ratio < 1 or not 0 < val_ratio < 1:
        raise ValueError("train_ratio and val_ratio must be in (0, 1)")
    if train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio + val_ratio must be less than 1")
    if gap_tiles < 0:
        raise ValueError("gap_tiles must be non-negative")

    coordinates = sorted({int(row["row"]) for row in rows})
    train_cut = round(len(coordinates) * train_ratio)
    val_cut = train_cut + round(len(coordinates) * val_ratio)
    coordinate_splits = {
        "train": set(coordinates[: max(train_cut - gap_tiles, 0)]),
        "val": set(
            coordinates[
                min(train_cut + gap_tiles, len(coordinates)) : max(
                    val_cut - gap_tiles, 0
                )
            ]
        ),
        "test": set(coordinates[min(val_cut + gap_tiles, len(coordinates)) :]),
    }

    splits = {"train": [], "val": [], "test": [], "ignored": []}
    for row in rows:
        destination = "ignored"
        for name, selected in coordinate_splits.items():
            if int(row["row"]) in selected:
                destination = name
                break
        splits[destination].append(row)
    return splits


def crop_whu(
    source_root: Path,
    output_root: Path,
    tile_size: int,
    min_change_pixels: int,
) -> list[dict[str, Any]]:
    source_paths = {
        "t1": source_root / "before" / "before.tif",
        "t2": source_root / "after" / "after.tif",
        "mask": source_root / "change label" / "change_label.tif",
    }
    for path in source_paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)

    t1 = Image.open(source_paths["t1"]).convert("RGB")
    t2 = Image.open(source_paths["t2"]).convert("RGB")
    mask = Image.open(source_paths["mask"]).convert("L")
    if t1.size != t2.size or t1.size != mask.size:
        raise ValueError(f"Source size mismatch: {t1.size}, {t2.size}, {mask.size}")

    width, height = t1.size
    xs = list(range(0, width - tile_size + 1, tile_size))
    ys = list(range(0, height - tile_size + 1, tile_size))
    if not xs or not ys:
        raise ValueError(f"Source image {t1.size} is smaller than tile size {tile_size}")

    rows: list[dict[str, Any]] = []
    for row, y in enumerate(ys):
        for col, x in enumerate(xs):
            box = (x, y, x + tile_size, y + tile_size)
            sample_id = f"BCD_r{row:04d}_c{col:04d}"
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
            rows.append(
                {
                    "id": sample_id,
                    "scene": "BCD",
                    "row": row,
                    "col": col,
                    "x": x,
                    "y": y,
                    "tile_size": tile_size,
                    "change_pixels": change_pixels,
                    "change_ratio": f"{change_ratio:.8f}",
                    "t1": t1_rel,
                    "t2": t2_rel,
                    "label": int(change_pixels >= min_change_pixels),
                    "mask": mask_rel,
                }
            )
    return rows


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

    rows = crop_whu(
        source_root,
        output_root,
        tile_size=args.tile_size,
        min_change_pixels=args.min_change_pixels,
    )
    write_csv(output_root / "tiles.csv", rows, TILE_FIELDS)

    splits = spatial_split(
        rows,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        gap_tiles=args.gap_tiles,
    )
    for name, split_rows in splits.items():
        write_csv(split_root / f"{name}.csv", split_rows, MANIFEST_FIELDS)

    print(f"Prepared WHU-CD tiles: {output_root}")
    print(f"Total: {len(rows)} samples")
    for name in ("train", "val", "test", "ignored"):
        split_rows = splits[name]
        positives = sum(int(row["label"]) for row in split_rows)
        print(
            f"{name:>7}: {len(split_rows):5d} samples, "
            f"{positives:5d} changed, {len(split_rows) - positives:5d} unchanged"
        )


if __name__ == "__main__":
    main()
