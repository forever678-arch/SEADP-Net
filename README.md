# SEADP-Net

**Shared-Encoder Asymmetric Dual-Pyramid Network** for RGB-D semantic segmentation.

English | [中文](README_zh-CN.md)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23294255.svg)](https://zenodo.org/doi/10.5281/zenodo.23294255)

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
a fixed 0.5/0.5 mean. There is no learnable gating, no point-cloud module and
no extra geometric attention. S / M / L share the same encoder and differ
only by the encoder input scale (0.5 / 0.75 / 1.0), local widths, repeat
depths and decoder width.

## Results

| Dataset | Variant | Params | SS mIoU | MS5+flip mIoU |
|---|---|---:|---:|---:|
| NYUv2 (40 cls) | S | 26.25 M | 58.27 | 60.16 |
| NYUv2 (40 cls) | M | 28.95 M | 60.13 | 61.22 |
| NYUv2 (40 cls) | L | 29.82 M | 60.94 | 62.27 |
| SUN RGB-D (37 cls) | S | 26.25 M | 52.76 | 53.69 |
| SUN RGB-D (37 cls) | M | 28.95 M | 53.83 | 54.66 |
| SUN RGB-D (37 cls) | L | 29.82 M | 54.76 | 55.28 |

<p align="center">
  <img src="docs/figures/performance_macs.png" width="65%" alt="NYUv2 mIoU vs single-forward MACs"><br>
  <em>NYUv2 mIoU (MS5+flip) vs single-forward MACs. SEADP-Net (stars): released
  weights; comparison values as reported by the original papers. Lines connect
  each family's S/M/L sizes.</em>
</p>

### Comparison with the state of the art — NYUv2 (40 classes, 654 test images)

|  | Method | Backbone | #Params (M) | Input H×W | Compute (G) | mIoU (%) ↑ |
| --- | --- | --- | ---: | --- | ---: | ---: |
| Ⅰ | Omnivore | Swin-T | 28.90 | 480×640 | 28.60 | 49.70 |
| Ⅰ | TokenFusion | MiT-B2 | 26.00 | 480×640 | 54.90 | 53.30 |
| Ⅰ | DFormer-T | DFormer-T | 6.00 | 480×640 | 14.70 | 51.80 |
| Ⅰ | DFormer-S | DFormer-S | 18.70 | 480×640 | 28.70 | 53.60 |
| Ⅰ | DFormer-B | DFormer-B | 29.50 | 480×640 | 45.30 | 55.60 |
| Ⅰ | DFormerv2-S | DFormerv2-S | 26.70 | 480×640 | 33.90 | 56.00 |
| Ⅰ | DFormer++-Tiny | DFormer++-T | 17.30 | 480×640 | 29.50 | 57.00 |
| Ⅰ | **SEADP-Net-S (Ours)** | RF-DETR ViT-S | 26.25 | 480×640 | 28.37 | **60.16** |
| Ⅱ | ESANet | ResNet-34 | 47.00 | 480×640 | 45.10 | 50.30 |
| Ⅱ | EMSANet | ResNet-34 | 46.90 | 480×640 | 45.10 | 51.00 |
| Ⅱ | TokenFusion | MiT-B3 | 45.90 | 480×640 | 94.00 | 54.20 |
| Ⅱ | Omnivore | Swin-S | 50.60 | 480×640 | 56.00 | 52.70 |
| Ⅱ | DFormer-L | DFormer-L | 39.00 | 480×640 | 69.30 | 57.20 |
| Ⅱ | AsymFormer | B0+Conv-T | 33.00 | 480×640 | 39.40 | 55.30 |
| Ⅱ | Sigma | VMamba-T | 48.30 | 480×640 | 89.50 | 53.90 |
| Ⅱ | DFormerv2-B | DFormerv2-B | 53.90 | 480×640 | 67.20 | 57.70 |
| Ⅱ | DFormer++-Small | DFormer++-S | 37.20 | 480×640 | 56.40 | 58.10 |
| Ⅱ | **SEADP-Net-M (Ours)** | RF-DETR ViT-S | 28.95 | 480×640 | 56.78 | **61.22** |
| Ⅲ | ACNet | ResNet-50 | 115.60 | 480×640 | 115.60 | 48.30 |
| Ⅲ | SGNet | ResNet-101 | 64.70 | 480×640 | 108.50 | 51.10 |
| Ⅲ | SA-Gate | ResNet-101 | 110.90 | 480×640 | 192.60 | 52.40 |
| Ⅲ | CEN | ResNet-101 | 118.20 | 480×640 | 617.90 | 51.70 |
| Ⅲ | CEN | ResNet-152 | 133.90 | 480×640 | 663.30 | 52.50 |
| Ⅲ | ShapeConv | ResNext-101 | 106.80 | 480×640 | 168.40 | 51.30 |
| Ⅲ | FRNet | ResNet-34 | 87.80 | 480×640 | 109.50 | 53.60 |
| Ⅲ | PGDENet | ResNet-34 | 107.40 | 480×640 | 162.90 | 53.70 |
| Ⅲ | MultiMAE | ViT-B | 95.40 | 640×640 | 403.80 | 56.00 |
| Ⅲ | Omnivore | Swin-B | 90.10 | 480×640 | 99.50 | 54.00 |
| Ⅲ | CMX | MiT-B2 | 66.60 | 480×640 | 66.90 | 54.40 |
| Ⅲ | CMX | MiT-B4 | 139.90 | 480×640 | 134.00 | 56.30 |
| Ⅲ | CMX | MiT-B5 | 181.10 | 480×640 | 167.50 | 56.90 |
| Ⅲ | CMNeXt | MiT-B4 | 116.60 | 480×640 | 131.40 | 56.90 |
| Ⅲ | GeminiFusion | MiT-B3 | 75.80 | 480×640 | 138.20 | 56.80 |
| Ⅲ | Sigma | VMamba-S | 69.80 | 480×640 | 138.90 | 57.00 |
| Ⅲ | DFormerv2-L | DFormerv2-L | 95.50 | 480×640 | 124.10 | 58.40 |
| Ⅲ | DFormer++-Base | DFormer++-B | 66.70 | 480×640 | 97.50 | 59.00 |
| Ⅲ | **SEADP-Net-L (Ours)** | RF-DETR ViT-S | 29.82 | 480×640 | 100.06 | **62.27** |

### Comparison with the state of the art — SUN RGB-D (37 classes, 5050 test images)

|  | Method | Backbone | #Params (M) | Input H×W | Compute (G) | mIoU (%) ↑ |
| --- | --- | --- | ---: | --- | ---: | ---: |
| Ⅰ | Omnivore | Swin-T | 28.90 | 530×730 | — | — |
| Ⅰ | TokenFusion | MiT-B2 | 26.00 | 530×730 | 70.70 | — |
| Ⅰ | DFormer-T | DFormer-T | 6.00 | 530×730 | 18.90 | 48.80 |
| Ⅰ | DFormer-S | DFormer-S | 18.70 | 530×730 | 37.00 | 50.00 |
| Ⅰ | DFormer-B | DFormer-B | 29.50 | 530×730 | 58.40 | 51.20 |
| Ⅰ | DFormerv2-S | DFormerv2-S | 26.70 | 530×730 | 43.70 | 51.50 |
| Ⅰ | DFormer++-Tiny | DFormer++-T | 17.30 | 530×730 | 39.30 | 51.50 |
| Ⅰ | **SEADP-Net-S (Ours)** | RF-DETR ViT-S | 26.25 | 530×730 | 38.82 | **53.69** |
| Ⅱ | ESANet | ResNet-34 | 47.00 | 480×640 | 34.90 | 45.00 |
| Ⅱ | EMSANet | ResNet-34 | 46.90 | 530×730 | 58.70 | 48.40 |
| Ⅱ | TokenFusion | MiT-B3 | 45.90 | 530×730 | 121.60 | — |
| Ⅱ | Omnivore | Swin-S | 50.60 | 530×730 | — | — |
| Ⅱ | DFormer-L | DFormer-L | 39.00 | 530×730 | 89.20 | 52.50 |
| Ⅱ | AsymFormer | B0+Conv-T | 33.00 | 530×730 | 52.60 | 49.10 |
| Ⅱ | Sigma | VMamba-T | 48.30 | 480×640 | 89.50 | 50.00 |
| Ⅱ | DFormerv2-B | DFormerv2-B | 53.90 | 530×730 | 86.90 | 52.80 |
| Ⅱ | DFormer++-Small | DFormer++-S | 37.20 | 530×730 | 74.60 | 52.70 |
| Ⅱ | **SEADP-Net-M (Ours)** | RF-DETR ViT-S | 28.95 | 530×730 | 76.72 | **54.66** |
| Ⅲ | ACNet | ResNet-50 | 115.60 | 530×730 | 150.60 | 48.10 |
| Ⅲ | SGNet | ResNet-101 | 64.70 | 530×730 | 151.50 | 48.60 |
| Ⅲ | SA-Gate | ResNet-101 | 110.90 | 530×730 | 248.70 | 49.40 |
| Ⅲ | CEN | ResNet-101 | 118.20 | 530×730 | 789.20 | 50.20 |
| Ⅲ | CEN | ResNet-152 | 133.90 | 530×730 | 848.30 | 51.10 |
| Ⅲ | ShapeConv | ResNext-101 | 106.80 | 530×730 | 217.80 | 48.60 |
| Ⅲ | FRNet | ResNet-34 | 87.80 | 530×730 | 142.70 | 51.80 |
| Ⅲ | PGDENet | ResNet-34 | 107.40 | 530×730 | 212.20 | 51.00 |
| Ⅲ | MultiMAE | ViT-B | 95.40 | 640×640 | 403.80 | — |
| Ⅲ | Omnivore | Swin-B | 90.10 | 530×730 | — | — |
| Ⅲ | CMX | MiT-B2 | 66.60 | 530×730 | 85.90 | 49.70 |
| Ⅲ | CMX | MiT-B4 | 139.90 | 530×730 | 173.20 | 52.10 |
| Ⅲ | CMX | MiT-B5 | 181.10 | 530×730 | 216.90 | 52.40 |
| Ⅲ | CMNeXt | MiT-B4 | 116.60 | 530×730 | 169.60 | — |
| Ⅲ | GeminiFusion | MiT-B3 | 75.80 | 530×730 | 179.00 | 52.70 |
| Ⅲ | Sigma | VMamba-S | 69.80 | 480×640 | 138.90 | 52.40 |
| Ⅲ | DFormerv2-L | DFormerv2-L | 95.50 | 530×730 | 160.50 | 53.30 |
| Ⅲ | DFormer++-Base | DFormer++-B | 66.70 | 530×730 | 129.00 | 53.00 |
| Ⅲ | **SEADP-Net-L (Ours)** | RF-DETR ViT-S | 29.82 | 530×730 | 140.10 | **55.28** |

<sub>
Comparison values (parameters, input size, computation, mIoU) are used as reported by the original papers — [DFormer++ Table 1](https://mftp.mmcheng.net/Papers/26PAMI-Deformer.pdf) and [DFormerv2 Table 1](https://openaccess.thecvf.com/content/CVPR2025/papers/Yin_DFormerv2_Geometry_Self-Attention_for_RGBD_Semantic_Segmentation_CVPR_2025_paper.pdf); external methods were **not** retrained in our environment, and counting scopes are not guaranteed identical across papers. “—” marks values not reported, or SUN-RGBD accuracies re-measured by the DFormer++ authors (TokenFusion MiT-B2/B3, MultiMAE, CMNeXt MiT-B4) that we omit. SEADP-Net computation is the single-forward major-operator MAC count of the full network (Conv/Linear/attention, 1 FMA = 1 MAC, padding and attention matmuls included, element-wise ops excluded); accuracies are MS5+flip. Bold marks our models — a system-level comparison without per-module attribution or cross-seed significance claims.
</sub>

Full provenance — weight SHA256, averaging windows, snapshot updates,
per-head scores and evaluation records — is in [`model_zoo.json`](model_zoo.json)
and [`eval_records/`](eval_records) (SS/MSF rerun outputs of this code base;
every number matches the frozen records, max |delta| 0.0005 except one
cross-GPU +0.005 case documented in
[`docs/equivalence_verification.md`](docs/equivalence_verification.md)).

## Installation

Python ≥ 3.11, PyTorch ≥ 2.4 (verified on 2.8.0+cu128), plus the RF-DETR
package pinned to upstream commit `e96a71476bf09c738fa861c3d1ee2c8caeaa5ffe`
(version string 1.11.0.dev0).

```bash
pip install -r requirements.txt   # frozen reference environment
git clone https://github.com/roboflow/rf-detr && cd rf-detr
git checkout e96a71476bf09c738fa861c3d1ee2c8caeaa5ffe && pip install -e .
cd .. && pip install -e .         # seadpnet
```

You also need the fixed pretrained backbone file `rf-detr-seg-small.pt`
(SHA256 `6de3da31b2572cac214a1c76cce4a92a13966d56390ac2b3a3de9a8dc2b2bca3`)
from the RF-DETR release; every command below takes its location via
`--rf-checkpoint`.

## Model zoo

| File | Dataset / variant |
|---|---|
| `weights/seadp-net-nyuv2-s.pth` | NYUv2 S |
| `weights/seadp-net-nyuv2-m.pth` | NYUv2 M |
| `weights/seadp-net-nyuv2-l.pth` | NYUv2 L |
| `weights/seadp-net-sunrgbd-s.pth` | SUN RGB-D S |
| `weights/seadp-net-sunrgbd-m.pth` | SUN RGB-D M |
| `weights/seadp-net-sunrgbd-l.pth` | SUN RGB-D L |

Each file is an FP32 uniform parameter average of three listed mid-training
snapshots (a single-model inference artifact, not an output ensemble) with
sanitized metadata; the tensor bytes are bit-identical to the corresponding
training average (dual SHA256 recorded in `model_zoo.json`). All six weights are attached to the
[v1.0.0 release](https://github.com/forever678-arch/SEADP-Net/releases/tag/v1.0.0);
verify each file against the SHA256 in `model_zoo.json`.

## Data preparation

Dataset bodies are not redistributed; the preparation tools download the
official sources (SHA256-pinned) and reproduce the exact processed layout.

```bash
# NYUv2 — put the five pinned source files under data/nyuv2/raw/ first
python tools/prepare_nyuv2.py data/nyuv2
python tools/extract_nyuv2_depth.py data/nyuv2
python tools/prepare_nyuv2_validity.py data/nyuv2 data/nyuv2/nyuv2_raw_validity

# SUN RGB-D — self-contained resumable download of the official archive
python tools/prepare_sunrgbd.py data/sunrgbd_processed
```

Details, source checksums and the directory contract:
[`docs/data_preparation.md`](docs/data_preparation.md).

## Evaluation

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

`load_experiment` rebuilds the model from the checkpoint's own config,
compares architecture metadata exactly and loads weights with `strict=True`;
`--rf-checkpoint` only overrides where the fixed pretrained file lives. The
per-view order (FP32 logits → unflip → resize → crop → softmax → view mean →
0.5/0.5 head fusion → argmax, confusion matrix over the whole split) is
specified in [`docs/evaluation_protocol.md`](docs/evaluation_protocol.md) —
do not reorder it.

## Training

```bash
python tools/train_nyuv2.py --config configs/nyuv2/seadp-net-l.yaml --run-dir runs/nyu_l
python tools/train_sunrgbd.py --config configs/sunrgbd/seadp-net-m.yaml --run-dir runs/sun_m
```

The entries preserve the original training math exactly: one effective
batch-32 update per AdamW step (microbatch CE **sums** divided by the full
batch's valid-pixel count), BF16 forward with FP32 CE, cosine LR with 500
warmup updates, gradient clip 0.1, and the LR grouping rule where only
`backbone.*encoder*` (without `depth`) parameters take the backbone LR.
Snapshot averaging uses `tools/average_checkpoints.py`.

**Honest reporting note:** training-time selection monitored fixed subsets of
the official *test* splits (NYUv2 `test_sub100`, SUN `test_sub500`); they are
not independent validation sets.

## Inference

```bash
python tools/infer.py --checkpoint weights/seadp-net-sunrgbd-l.pth \
    --rf-checkpoint /path/rf-detr-seg-small.pt --dataset sunrgbd \
    --rgb sample/rgb.jpg --depth sample/depth.png --valid sample/valid.png \
    --out-seg seg.png --out-color seg_color.png
```

`depth.png` is uint16 millimetres (or float32 metres `.npy`); `valid.png` is
nonzero at raw sensor returns.

## Repository layout

```text
SEADP-Net/
  seadpnet/                  the package
    blocks.py                geometry, ChannelNorm, decoder building blocks
    csp.py                   CSP-style local pyramids (self-implemented)
    dual_head_model.py       shared encoder + HR stem + dual-decoder graph
    scaled_encoder_model.py  S/M builder (encoder input scaling, 12 blocks kept)
    encoder_retention.py     full-encoder retention audit metadata
    model.py                 loaders for all three checkpoint formats
    losses.py                dual CE with the full-batch valid-pixel denominator
    rf/                      RF-DETR adapters + vendored semantic wrapper
    data/                    NYUv2 / SUN RGB-D loaders, class metadata
    evaluation/              SS / MSF protocols, metrics, native-GT SUN eval
  configs/                   nyuv2/{s,m,l}.yaml, sunrgbd/{s,m,l}.yaml
  tools/                     train / evaluate / infer / average / export,
                             prepare_{nyuv2,sunrgbd} data pipelines
  tests/                     original-vs-package equivalence test
  docs/                      architecture figure, protocols, data prep, verification
  eval_records/              verification SS/MSF outputs for all six weights
  model_zoo.json             weights, dual SHA256, provenance, scores
  weights/                   local release weights (gitignored; see model zoo)
```

## Verification

The package was verified against the original training code on the training
host: all six formal weights load bit-exact (identical parameters, zero
per-head logit differences, equal confusion matrices), the training entry
reproduces the original first augmented batch byte-for-byte, and the full
SS/MSF reruns match every recorded number. Details:
[`docs/equivalence_verification.md`](docs/equivalence_verification.md).

Checkpoint format strings such as `c11-csp-local-v1` inside the code and
weight metadata are internal load keys inherited from the training lineage,
not model names.

## License & citation

Code is released under the [MIT License](LICENSE). Pretrained backbone and
third-party components remain under their own licenses (Apache-2.0; see
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)). A archived DOI will be
provided via Zenodo for each release.

## Acknowledgements

The shared encoder builds on [RF-DETR](https://github.com/roboflow/rf-detr)
(Apache-2.0, including the SegSmall pretrained weights) and its DINOv2
backbone (Apache-2.0); see [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
The local CSP design is a self-implementation inspired by C3k2-style
split/concatenate blocks and contains no Ultralytics code.
