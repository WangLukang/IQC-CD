from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from iqccd.core import IQCCD
from iqccd.data import make_loader, read_manifest
from iqccd.gate import ImageGate
from iqccd.model import ChangeQuery
from iqccd.prior import SatlasBuildingPrior
from iqccd.utils import load_config, project_path, resolve_device


ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate IQC-CD on a test split.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/whu.json")
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config.resolve())
    model_config = config["model"]
    data = config["data"]
    device = resolve_device(args.device)
    checkpoint_dir = args.checkpoint_dir or project_path(ROOT, config["output"])
    data_root = project_path(ROOT, data["root"])
    split_root = project_path(ROOT, data["split_root"])
    if checkpoint_dir is None or data_root is None or split_root is None:
        raise ValueError("checkpoint and dataset paths are required")

    query = ChangeQuery(
        model_name=str(model_config["dinov3_name"]),
        weights=project_path(ROOT, model_config.get("dinov3_weights")),
        out_indices=tuple(int(x) for x in model_config["out_indices"]),
        decoder_width=int(model_config["decoder_width"]),
        top_fraction=float(model_config["top_fraction"]),
    )
    checkpoint = torch.load(
        checkpoint_dir / "query_head.pt",
        map_location="cpu",
        weights_only=False,
    )
    query.load_trainable_state_dict(checkpoint["trainable_state"])
    prior = SatlasBuildingPrior(
        project_path(ROOT, model_config["satlas_weights"]),
        detection_floor=float(model_config["detection_floor"]),
    )
    model = IQCCD(query, prior).to(device).eval()
    gate = ImageGate.load(checkpoint_dir / "image_gate.json")
    records = read_manifest(
        data_root, split_root / str(data["test_manifest"])
    )
    loader = make_loader(
        records,
        image_size=int(data["image_size"]),
        batch_size=1,
        workers=int(config["training"]["num_workers"]),
        seed=int(config["seed"]),
        training=False,
        load_mask=True,
    )

    counts = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    labels: list[np.ndarray] = []
    image_scores: list[np.ndarray] = []
    pixel_threshold = float(model_config["pixel_threshold"])
    with torch.inference_mode():
        for batch in loader:
            output = model(
                batch["t1"].to(device, non_blocking=True),
                batch["t2"].to(device, non_blocking=True),
            )
            probability = gate.probability(output["descriptors"].cpu().numpy())
            prediction = (
                output["change_probability"][:, 0].cpu().numpy()
                >= pixel_threshold
            ) & (probability >= 0.5)[:, None, None]
            target = batch["mask"].numpy().astype(bool)
            counts["tp"] += int((prediction & target).sum())
            counts["fp"] += int((prediction & ~target).sum())
            counts["fn"] += int((~prediction & target).sum())
            counts["tn"] += int((~prediction & ~target).sum())
            labels.append(batch["label"].numpy())
            image_scores.append(probability)

    tp, fp, fn, tn = (counts[key] for key in ("tp", "fp", "fn", "tn"))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    result = {
        "samples": len(records),
        "image_auc": float(
            roc_auc_score(np.concatenate(labels), np.concatenate(image_scores))
        ),
        "precision": precision,
        "recall": recall,
        "f1": 2.0 * precision * recall / max(precision + recall, 1e-12),
        "iou": tp / max(tp + fp + fn, 1),
        "oa": (tp + tn) / max(tp + fp + fn + tn, 1),
        **counts,
    }
    destination = checkpoint_dir / "evaluation.json"
    destination.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
