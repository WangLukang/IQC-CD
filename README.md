# IQC-CD

Core implementation of **Image-level weakly supervised building change detection based on feature intervention and semantic consensus**.

IQC-CD contains four components:

1. a frozen DINOv3 ConvNeXt-Tiny encoder and a 41,089-parameter multi-scale change-query head;
2. feature intervention on unchanged pairs, which provides spatially known synthetic inconsistency without pixel annotations;
3. a frozen SatlasPretrain aerial building-instance branch;
4. parameter-free harmonic consensus plus a five-parameter image-presence gate.

Only image-level binary labels are read during training and validation. Pixel masks are not required by this repository.

## Installation

```bash
conda create -n iqccd python=3.11 -y
conda activate iqccd
pip install -r requirements.txt
```

## Pretrained weights

Weights are intentionally not stored in this repository.

- **DINOv3 ConvNeXt-Tiny**: the default `null` value for `dinov3_weights` lets `timm` download [`timm/convnext_tiny.dinov3_lvd1689m`](https://huggingface.co/timm/convnext_tiny.dinov3_lvd1689m). To work offline, download `model.safetensors`, place it under `weights/`, and set its path in the configuration. DINOv3 weights are governed by the DINOv3 license.
- **SatlasPretrain aerial SwinB single-image checkpoint**: download [`aerial_swinb_si.pth`](https://huggingface.co/allenai/satlas-pretrain/resolve/main/aerial_swinb_si.pth?download=true), place it at `weights/aerial_swinb_si.pth`, and retain the upstream ODC-BY terms.
- **IQC-CD query head and image gate**: run `train.py`; it creates `query_head.pt` and `image_gate.json`. These task-specific files are not bundled.

The upstream implementations are available from [DINOv3](https://github.com/facebookresearch/dinov3) and [SatlasPretrain Models](https://github.com/allenai/satlaspretrain_models).

## Data

Each CSV manifest contains only four columns:

```csv
id,t1,t2,label
000001,images/t1/000001.png,images/t2/000001.png,1
000002,images/t1/000002.png,images/t2/000002.png,0
```

Paths are relative to `data.root`. `label=1` means that at least one building change is present; `label=0` means unchanged. The calibration manifest also uses image-level labels only and may point to the training manifest.

## Training

Edit `configs/example.json`, then run:

```bash
python train.py --config configs/example.json --device cuda
```

Model selection uses validation image-level ROC-AUC. The frozen DINOv3 and SatlasPretrain parameters are never optimized.

## Inference

```bash
python predict.py path/to/T1.png path/to/T2.png \
  --config configs/example.json \
  --checkpoint-dir runs/iqccd \
  --output outputs/example \
  --device cuda
```

The output directory contains the query map, building-support maps, consensus probability, binary change mask, and image-level probability.

## Repository scope

This release contains only the final IQC-CD training and inference path. Datasets, pretrained weights, paper files, experiment outputs, comparison methods, and ablation code are excluded.
