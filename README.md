# SEADP-Net

**Shared-Encoder Asymmetric Dual-Pyramid Network** for RGB-D semantic segmentation.

English | [中文](README_zh-CN.md)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23294255.svg)](https://zenodo.org/doi/10.5281/zenodo.23294255)

## Introduction

<p align="center">
  <img src="docs/figures/architecture_overview.png" width="88%" alt="SEADP-Net architecture overview"><br>
  <em>The shared RGB-D context encoder (SCE) and high-resolution stem (HRS) feed two
  asymmetric local pyramids (ALP_A on RGB, ALP_B on depth) and two decoders whose
  probabilities fuse with a fixed 0.5/0.5 mean.</em>
</p>

SEADP-Net keeps a whole RF-DETR SegSmall ViT-S encoder (12 blocks, width 384,
patch 12) as a shared RGB-D context encoder, adds a stride-4 high-resolution
stem, and decodes with two *asymmetric* local pyramids — an RGB branch (A) and
a deeper depth branch (B, fed `[filled depth, raw sensor validity,
depth-gradient magnitude]`). Training supervises both heads with an
equal-weight cross entropy; inference fuses the two heads' probabilities with
a fixed 0.5/0.5 mean. S / M / L share the same encoder and differ only by the
encoder input scale (0.5 / 0.75 / 1.0), local widths, repeat depths and
decoder width.

## Main results

| Model | Backbone | Params | NYUv2 In | NYUv2 MACs | NYUv2 mIoU | SUN-RGBD In | SUN-RGBD MACs | SUN-RGBD mIoU |
|---|---|---|---|---:|---:|---|---:|---:|
| Omnivore | Swin-T | 28.9M | 480×640 | 28.6G | 49.7 | 530×730 | – | – |
| TokenFusion | MiT-B2 | 26.0M | 480×640 | 54.9G | 53.3 | 530×730 | – | – |
| DFormer-T | DFormer-T | 6.0M | 480×640 | 14.7G | 51.8 | 530×730 | 18.9G | 48.8 |
| DFormer-S | DFormer-S | 18.7M | 480×640 | 28.7G | 53.6 | 530×730 | 37.0G | 50.0 |
| DFormer-B | DFormer-B | 29.5M | 480×640 | 45.3G | 55.6 | 530×730 | 58.4G | 51.2 |
| DFormerv2-S | DFormerv2-S | 26.7M | 480×640 | 33.9G | 56.0 | 530×730 | 43.7G | 51.5 |
| DFormer++-Tiny | DFormer++-T | 17.3M | 480×640 | 29.5G | 57.0 | 530×730 | 39.3G | 51.5 |
| **SEADP-Net-S** | RF-DETR ViT-S | 26.3M | 480×640 | 28.4G | **60.2** | 530×730 | 38.8G | **53.7** |
| ESANet | ResNet-34 | 47.0M | 480×640 | 45.1G | 50.3 | 480×640 | 34.9G | 45.0 |
| EMSANet | ResNet-34 | 46.9M | 480×640 | 45.1G | 51.0 | 530×730 | 58.7G | 48.4 |
| TokenFusion | MiT-B3 | 45.9M | 480×640 | 94.0G | 54.2 | 530×730 | – | – |
| Omnivore | Swin-S | 50.6M | 480×640 | 56.0G | 52.7 | 530×730 | – | – |
| DFormer-L | DFormer-L | 39.0M | 480×640 | 69.3G | 57.2 | 530×730 | 89.2G | 52.5 |
| AsymFormer | B0+Conv-T | 33.0M | 480×640 | 39.4G | 55.3 | 530×730 | 52.6G | 49.1 |
| Sigma | VMamba-T | 48.3M | 480×640 | 89.5G | 53.9 | 480×640 | 89.5G | 50.0 |
| DFormerv2-B | DFormerv2-B | 53.9M | 480×640 | 67.2G | 57.7 | 530×730 | 86.9G | 52.8 |
| DFormer++-Small | DFormer++-S | 37.2M | 480×640 | 56.4G | 58.1 | 530×730 | 74.6G | 52.7 |
| **SEADP-Net-M** | RF-DETR ViT-S | 29.0M | 480×640 | 56.8G | **61.2** | 530×730 | 76.7G | **54.7** |
| ACNet | ResNet-50 | 115.6M | 480×640 | 115.6G | 48.3 | 530×730 | 150.6G | 48.1 |
| SGNet | ResNet-101 | 64.7M | 480×640 | 108.5G | 51.1 | 530×730 | 151.5G | 48.6 |
| SA-Gate | ResNet-101 | 110.9M | 480×640 | 192.6G | 52.4 | 530×730 | 248.7G | 49.4 |
| CEN | ResNet-101 | 118.2M | 480×640 | 617.9G | 51.7 | 530×730 | 789.2G | 50.2 |
| CEN | ResNet-152 | 133.9M | 480×640 | 663.3G | 52.5 | 530×730 | 848.3G | 51.1 |
| ShapeConv | ResNext-101 | 106.8M | 480×640 | 168.4G | 51.3 | 530×730 | 217.8G | 48.6 |
| FRNet | ResNet-34 | 87.8M | 480×640 | 109.5G | 53.6 | 530×730 | 142.7G | 51.8 |
| PGDENet | ResNet-34 | 107.4M | 480×640 | 162.9G | 53.7 | 530×730 | 212.2G | 51.0 |
| MultiMAE | ViT-B | 95.4M | 640×640 | 403.8G | 56.0 | 640×640 | – | – |
| Omnivore | Swin-B | 90.1M | 480×640 | 99.5G | 54.0 | 530×730 | – | – |
| CMX | MiT-B2 | 66.6M | 480×640 | 66.9G | 54.4 | 530×730 | 85.9G | 49.7 |
| CMX | MiT-B4 | 139.9M | 480×640 | 134.0G | 56.3 | 530×730 | 173.2G | 52.1 |
| CMX | MiT-B5 | 181.1M | 480×640 | 167.5G | 56.9 | 530×730 | 216.9G | 52.4 |
| CMNeXt | MiT-B4 | 116.6M | 480×640 | 131.4G | 56.9 | 530×730 | – | – |
| GeminiFusion | MiT-B3 | 75.8M | 480×640 | 138.2G | 56.8 | 530×730 | 179.0G | 52.7 |
| Sigma | VMamba-S | 69.8M | 480×640 | 138.9G | 57.0 | 480×640 | 138.9G | 52.4 |
| DFormerv2-L | DFormerv2-L | 95.5M | 480×640 | 124.1G | 58.4 | 530×730 | 160.5G | 53.3 |
| DFormer++-Base | DFormer++-B | 66.7M | 480×640 | 97.5G | 59.0 | 530×730 | 129.0G | 53.0 |
| **SEADP-Net-L** | RF-DETR ViT-S | 29.8M | 480×640 | 100.1G | **62.3** | 530×730 | 140.1G | **55.3** |

<sub>mIoU in %, MACs in G for a single forward pass; "–" indicates omitted or unavailable results, following the paper. Reference accuracies are the reported values from [DFormer++ Table 1](https://mftp.mmcheng.net/Papers/26PAMI-Deformer.pdf) and [DFormerv2 Table 1](https://openaccess.thecvf.com/content/CVPR2025/papers/Yin_DFormerv2_Geometry_Self-Attention_for_RGBD_Semantic_Segmentation_CVPR_2025_paper.pdf); external methods are not retrained here, and counting scopes are not guaranteed identical across papers. SEADP-Net rows use the released avg weights with five scales + horizontal flipping; our MACs count Conv/Linear/attention only (1 FMA = 1 MAC).</sub>

### Accuracy vs computation

<p align="center">
  <img src="docs/figures/performance_macs.png" width="65%" alt="NYUv2 mIoU vs single-forward MACs"><br>
  <em>NYUv2 mIoU (MS5+flip) vs single-forward MACs. SEADP-Net (stars): released
  weights; comparison values as reported by the original papers. Lines connect
  each family's S/M/L sizes.</em>
</p>

## Model zoo

| Variant | NYUv2 SS / MSF | SUN RGB-D SS / MSF | Weights |
|---|---|---|---|
| S | 58.27 / 60.16 | 52.76 / 53.69 | `weights/seadp-net-nyuv2-s.pth`, `weights/seadp-net-sunrgbd-s.pth` |
| M | 60.13 / 61.22 | 53.83 / 54.66 | `weights/seadp-net-nyuv2-m.pth`, `weights/seadp-net-sunrgbd-m.pth` |
| L | 60.94 / 62.27 | 54.76 / 55.28 | `weights/seadp-net-nyuv2-l.pth`, `weights/seadp-net-sunrgbd-l.pth` |

All six weights are attached to the
[v1.0.0 release](https://github.com/forever678-arch/SEADP-Net/releases/tag/v1.0.0);
each is an FP32 uniform parameter average of three listed mid-training
snapshots with sanitized metadata and dual SHA256 provenance in
[`model_zoo.json`](model_zoo.json).

## Getting started

### Requirements

Python ≥ 3.11, PyTorch ≥ 2.4 (verified on 2.8.0+cu128), plus the RF-DETR
package pinned to upstream commit `e96a71476bf09c738fa861c3d1ee2c8caeaa5ffe`.

```bash
pip install -r requirements.txt
git clone https://github.com/roboflow/rf-detr && cd rf-detr
git checkout e96a71476bf09c738fa861c3d1ee2c8caeaa5ffe && pip install -e .
cd .. && pip install -e .
```

Also fetch the fixed pretrained backbone `rf-detr-seg-small.pt`
(SHA256 `6de3da31b2572cac214a1c76cce4a92a13966d56390ac2b3a3de9a8dc2b2bca3`)
from the RF-DETR release; commands take its location via `--rf-checkpoint`.

### Data preparation

```bash
# NYUv2 — put the five pinned source files under data/nyuv2/raw/ first
python tools/prepare_nyuv2.py --root data/nyuv2
python tools/extract_nyuv2_depth.py --root data/nyuv2
python tools/prepare_nyuv2_validity.py --data-root data/nyuv2 --out data/nyuv2/nyuv2_raw_validity

# SUN RGB-D — self-contained resumable download of the official archive
python tools/prepare_sunrgbd.py data/sunrgbd_processed
```

Details and source checksums: [`docs/data_preparation.md`](docs/data_preparation.md).

### Training

```bash
python tools/train_nyuv2.py --config configs/nyuv2/seadp-net-l.yaml --run-dir runs/nyu_l
python tools/train_sunrgbd.py --config configs/sunrgbd/seadp-net-m.yaml --run-dir runs/sun_m
```

The entries preserve the training protocol exactly (batch-32 update with the
full-batch valid-pixel CE denominator, BF16 forward + FP32 CE, cosine LR with
500 warmup updates, the `backbone.*encoder*` LR grouping). **Model selection
monitored fixed subsets of the official test splits** (`test_sub100`,
`test_sub500`) — they are not independent validation sets.

### Evaluation

```bash
# single scale
python tools/evaluate_nyuv2.py --checkpoint weights/seadp-net-nyuv2-l.pth \
    --rf-checkpoint /path/rf-detr-seg-small.pt \
    --data-root data/nyuv2 --valid-root data/nyuv2/nyuv2_raw_validity \
    --split test --out nyu_l_ss.json

# 5-scale + flip
python tools/evaluate_sunrgbd.py --checkpoint weights/seadp-net-sunrgbd-m.pth \
    --rf-checkpoint /path/rf-detr-seg-small.pt \
    --data-root data/sunrgbd_processed \
    --scales 0.5 0.75 1.0 1.25 1.5 --flip --out sun_m_msf.json
```

The per-view order (FP32 logits → unflip → resize → crop → softmax → view
mean → 0.5/0.5 head fusion → argmax) is specified in
[`docs/evaluation_protocol.md`](docs/evaluation_protocol.md) — do not reorder.

### Inference

```bash
python tools/infer.py --checkpoint weights/seadp-net-sunrgbd-l.pth \
    --rf-checkpoint /path/rf-detr-seg-small.pt --dataset sunrgbd \
    --rgb sample/rgb.jpg --depth sample/depth.png --valid sample/valid.png \
    --out-seg seg.png --out-color seg_color.png
```

## Repository layout

```text
SEADP-Net/
  seadpnet/                  the package (models / data / losses / evaluation)
  configs/                   nyuv2/sunrgbd S/M/L training configs
  tools/                     train / evaluate / infer / average / export,
                             prepare_{nyuv2,sunrgbd} data pipelines
  tests/                     original-vs-package equivalence test
  docs/                      architecture figure, protocols, data prep, verification
  eval_records/              verification SS/MSF outputs for all six weights
  model_zoo.json             weights, dual SHA256, provenance, scores
```

## Verification

All six formal weights load bit-exact against the original training code
(identical parameters, zero per-head logit differences, equal confusion
matrices); the training entry reproduces the original first augmented batch
byte-for-byte; SS/MSF reruns match every recorded number. Details:
[`docs/equivalence_verification.md`](docs/equivalence_verification.md).
Checkpoint format strings such as `c11-csp-local-v1` are internal load keys,
not model names.

## License

Code is released under the [MIT License](LICENSE). Pretrained backbone and
third-party components remain under their own licenses (Apache-2.0; see
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)). Each release is archived
on Zenodo: [10.5281/zenodo.23294255](https://zenodo.org/doi/10.5281/zenodo.23294255).

## Citation

```bibtex
@software{seadpnet2026,
  title  = {SEADP-Net: A Shared-Encoder Asymmetric Dual-Pyramid Network for RGB-D Semantic Segmentation},
  author = {The SEADP-Net Authors},
  year   = {2026},
  doi    = {10.5281/zenodo.23294255},
  url    = {https://github.com/forever678-arch/SEADP-Net}
}
```

## Acknowledgements

The shared encoder builds on [RF-DETR](https://github.com/roboflow/rf-detr)
(Apache-2.0, including the SegSmall pretrained weights) and its DINOv2
backbone (Apache-2.0); see [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
The local CSP design is a self-implementation inspired by C3k2-style
split/concatenate blocks and contains no Ultralytics code.
