from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from iqccd.core import IQCCD
from iqccd.data import image_tensor
from iqccd.gate import ImageGate
from iqccd.model import ChangeQuery
from iqccd.prior import SatlasBuildingPrior
from iqccd.utils import load_config, project_path, resolve_device


ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run IQC-CD on one image pair.")
    parser.add_argument("t1", type=Path)
    parser.add_argument("t2", type=Path)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/whu.json")
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/prediction")
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def save_map(path: Path, value: torch.Tensor) -> None:
    array = (value.detach().cpu().numpy().clip(0.0, 1.0) * 255).astype(np.uint8)
    Image.fromarray(array).save(path)


def main() -> None:
    args = parse_args()
    config = load_config(args.config.resolve())
    model_config = config["model"]
    device = resolve_device(args.device)
    checkpoint_dir = args.checkpoint_dir or project_path(ROOT, config["output"])
    if checkpoint_dir is None:
        raise ValueError("checkpoint directory is required")
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

    size = int(config["data"]["image_size"])
    first = image_tensor(args.t1, size).unsqueeze(0).to(device)
    second = image_tensor(args.t2, size).unsqueeze(0).to(device)
    output = model(first, second)
    image_probability = float(
        gate.probability(output["descriptors"].cpu().numpy())[0]
    )
    change = output["change_probability"][0, 0]
    threshold = float(model_config["pixel_threshold"])
    mask = ((change >= threshold) & (image_probability >= 0.5)).float()

    args.output.mkdir(parents=True, exist_ok=True)
    for name in (
        "occupancy_t1",
        "occupancy_t2",
        "building_support",
        "query_probability",
        "change_probability",
    ):
        save_map(args.output / f"{name}.png", output[name][0, 0])
    save_map(args.output / "change_mask.png", mask)
    (args.output / "prediction.json").write_text(
        json.dumps(
            {
                "image_probability": image_probability,
                "pixel_threshold": threshold,
                "changed": image_probability >= 0.5,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"saved prediction to {args.output.resolve()}")


if __name__ == "__main__":
    main()
