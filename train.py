from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from iqccd.core import IQCCD
from iqccd.data import make_loader, read_manifest
from iqccd.gate import ImageGate
from iqccd.intervention import image_predictions, train_epoch
from iqccd.model import ChangeQuery
from iqccd.prior import SatlasBuildingPrior
from iqccd.utils import load_config, project_path, resolve_device, seed_everything


ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train IQC-CD with image labels.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/example.json")
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def build_query(config: dict[str, object]) -> ChangeQuery:
    model = config["model"]
    return ChangeQuery(
        model_name=str(model["dinov3_name"]),
        weights=project_path(ROOT, model.get("dinov3_weights")),
        out_indices=tuple(int(x) for x in model["out_indices"]),
        decoder_width=int(model["decoder_width"]),
        top_fraction=float(model["top_fraction"]),
    )


@torch.inference_mode()
def fit_gate(
    model: IQCCD, loader: object, device: torch.device
) -> ImageGate:
    descriptors: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for batch in loader:
        output = model(
            batch["t1"].to(device, non_blocking=True),
            batch["t2"].to(device, non_blocking=True),
        )
        descriptors.append(output["descriptors"].cpu().numpy())
        labels.append(batch["label"].numpy().astype(np.uint8))
    return ImageGate.fit(np.concatenate(descriptors), np.concatenate(labels))


def main() -> None:
    args = parse_args()
    config = load_config(args.config.resolve())
    seed = int(config["seed"])
    seed_everything(seed)
    device = resolve_device(args.device)
    data = config["data"]
    training = config["training"]
    data_root = project_path(ROOT, data["root"])
    if data_root is None:
        raise ValueError("data.root is required")
    train_records = read_manifest(
        data_root, data_root / str(data["train_manifest"])
    )
    val_records = read_manifest(
        data_root, data_root / str(data["validation_manifest"])
    )
    calibration_records = read_manifest(
        data_root, data_root / str(data["calibration_manifest"])
    )
    loader_options = {
        "image_size": int(data["image_size"]),
        "batch_size": int(training["batch_size"]),
        "workers": int(training["num_workers"]),
    }
    train_loader = make_loader(
        train_records, **loader_options, seed=seed, training=True
    )
    val_loader = make_loader(
        val_records, **loader_options, seed=seed + 1, training=False
    )
    calibration_loader = make_loader(
        calibration_records, **loader_options, seed=seed + 2, training=False
    )

    query = build_query(config).to(device)
    optimizer = torch.optim.AdamW(
        (p for p in query.parameters() if p.requires_grad),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    output = project_path(ROOT, config["output"])
    if output is None:
        raise ValueError("output is required")
    output.mkdir(parents=True, exist_ok=True)
    query_path = output / "query_head.pt"
    best_auc, stale = -math.inf, 0
    for epoch in range(1, int(training["epochs"]) + 1):
        losses = train_epoch(query, train_loader, optimizer, device)
        labels, scores = image_predictions(query, val_loader, device)
        auc = float(roc_auc_score(labels, scores))
        print(
            f"epoch={epoch:02d} loss={losses['loss']:.4f} "
            f"mil={losses['mil']:.4f} val_auc={auc:.4f}"
        )
        if auc > best_auc:
            best_auc, stale = auc, 0
            torch.save(
                {
                    "method": "IQC-CD",
                    "trainable_state": query.trainable_state_dict(),
                    "validation_image_auc": auc,
                },
                query_path,
            )
        else:
            stale += 1
        if stale >= int(training["patience"]):
            break

    checkpoint = torch.load(query_path, map_location="cpu", weights_only=False)
    query.load_trainable_state_dict(checkpoint["trainable_state"])
    prior = SatlasBuildingPrior(
        project_path(ROOT, config["model"]["satlas_weights"]),
        detection_floor=float(config["model"]["detection_floor"]),
    ).to(device)
    gate = fit_gate(IQCCD(query, prior).to(device), calibration_loader, device)
    gate.save(output / "image_gate.json")
    print(f"saved query and image gate to {output}")


if __name__ == "__main__":
    main()
