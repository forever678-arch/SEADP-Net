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
