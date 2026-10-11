# SEADP-Net

**共享编码器非对称双金字塔网络**（Shared-Encoder Asymmetric Dual-Pyramid Network），面向 RGB-D 语义分割。

[English](README.md) | 中文
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23294255.svg)](https://zenodo.org/doi/10.5281/zenodo.23294255)

## 简介

<p align="center">
  <img src="docs/figures/architecture_overview.png" width="88%" alt="SEADP-Net 架构总览"><br>
  <em>共享 RGB-D 上下文编码器（SCE）与高分辨率主干（HRS）同时供给两个非对称
  局部金字塔（ALP_A 输入 RGB，ALP_B 输入深度）与两个解码器；推理时两头概率以
  固定 0.5/0.5 均值融合。</em>
</p>

SEADP-Net 将一个完整的 RF-DETR SegSmall ViT-S 编码器（12 个 block、宽度 384、
patch 12）作为共享 RGB-D 上下文编码器，加入 stride-4 高分辨率主干，并用两个
**非对称**局部金字塔解码——RGB 分支（A）与更深的深度分支（B，输入为
`[填补深度, 原始传感器有效掩码, 深度梯度幅值]`）。训练以等权交叉熵分别监督两个
头；推理以固定 0.5/0.5 概率均值融合两头。S / M / L 共用同一编码器，仅通过编码
器输入尺度（0.5 / 0.75 / 1.0）、局部宽度、重复深度与解码器宽度区分。

## 主要结果

<p align="center">
  <img src="docs/figures/performance_macs.png" width="65%" alt="NYUv2 mIoU 与单次前向 MACs 对比"><br>
  <em>NYUv2 mIoU（MS5+flip）与单次前向 MACs。SEADP-Net（五角星）为发布权重；
  对比值取各论文报告数字。连线仅连接各家族 S/M/L 尺寸。</em>
</p>

## 模型库

| 版本 | NYUv2 SS / MSF | SUN RGB-D SS / MSF | 权重 |
|---|---|---|---|
| S | 58.27 / 60.16 | 52.76 / 53.69 | `weights/seadp-net-nyuv2-s.pth`、`weights/seadp-net-sunrgbd-s.pth` |
| M | 60.13 / 61.22 | 53.83 / 54.66 | `weights/seadp-net-nyuv2-m.pth`、`weights/seadp-net-sunrgbd-m.pth` |
| L | 60.94 / 62.27 | 54.76 / 55.28 | `weights/seadp-net-nyuv2-l.pth`、`weights/seadp-net-sunrgbd-l.pth` |

六个权重均已附于 [v1.0.0 发布](https://github.com/forever678-arch/SEADP-Net/releases/tag/v1.0.0)；
每个文件为三个列出中间快照的 FP32 等权参数平均（元数据已脱敏），双重 SHA256
溯源见 [`model_zoo.json`](model_zoo.json)。

## 快速上手

### 环境要求

Python ≥ 3.11、PyTorch ≥ 2.4（在 2.8.0+cu128 上验证），以及钉定在上游 commit
`e96a71476bf09c738fa861c3d1ee2c8caeaa5ffe` 的 RF-DETR 包。

```bash
pip install -r requirements.txt
git clone https://github.com/roboflow/rf-detr && cd rf-detr
git checkout e96a71476bf09c738fa861c3d1ee2c8caeaa5ffe && pip install -e .
cd .. && pip install -e .
```

还需固定预训练骨干文件 `rf-detr-seg-small.pt`
（SHA256 `6de3da31b2572cac214a1c76cce4a92a13966d56390ac2b3a3de9a8dc2b2bca3`），
命令通过 `--rf-checkpoint` 指定其位置。

### 数据准备

```bash
# NYUv2 —— 先把五个钉定来源文件放到 data/nyuv2/raw/
python tools/prepare_nyuv2.py --root data/nyuv2
python tools/extract_nyuv2_depth.py --root data/nyuv2
python tools/prepare_nyuv2_validity.py --data-root data/nyuv2 --out data/nyuv2/nyuv2_raw_validity

# SUN RGB-D —— 自包含，官方压缩包可断点续传下载
python tools/prepare_sunrgbd.py data/sunrgbd_processed
```

来源校验和与目录契约详见 [`docs/data_preparation.md`](docs/data_preparation.md)。

### 训练

```bash
python tools/train_nyuv2.py --config configs/nyuv2/seadp-net-l.yaml --run-dir runs/nyu_l
python tools/train_sunrgbd.py --config configs/sunrgbd/seadp-net-m.yaml --run-dir runs/sun_m
```

训练入口逐字保留原始训练协议（batch-32 更新、整批有效像素 CE 分母、BF16 前向 +
FP32 CE、500 步 warmup 余弦 LR、`backbone.*encoder*` LR 分组）。**模型选择监控的
是官方测试集的固定子集**（`test_sub100`、`test_sub500`）——它们不是独立验证集。

### 评测

```bash
# 单尺度
python tools/evaluate_nyuv2.py --checkpoint weights/seadp-net-nyuv2-l.pth \
    --rf-checkpoint /path/rf-detr-seg-small.pt \
    --data-root data/nyuv2 --valid-root data/nyuv2/nyuv2_raw_validity \
    --split test --out nyu_l_ss.json

# 5 尺度 + 翻转
python tools/evaluate_sunrgbd.py --checkpoint weights/seadp-net-sunrgbd-m.pth \
    --rf-checkpoint /path/rf-detr-seg-small.pt \
    --data-root data/sunrgbd_processed \
    --scales 0.5 0.75 1.0 1.25 1.5 --flip --out sun_m_msf.json
```

逐视图顺序（FP32 logits → 反翻转 → resize → 裁剪 → softmax → 视图平均 →
0.5/0.5 头融合 → argmax）规定于
[`docs/evaluation_protocol.md`](docs/evaluation_protocol.md)——不要调换顺序。

### 推理

```bash
python tools/infer.py --checkpoint weights/seadp-net-sunrgbd-l.pth \
    --rf-checkpoint /path/rf-detr-seg-small.pt --dataset sunrgbd \
    --rgb sample/rgb.jpg --depth sample/depth.png --valid sample/valid.png \
    --out-seg seg.png --out-color seg_color.png
```

## 仓库结构

```text
SEADP-Net/
  seadpnet/                  包本体（模型 / 数据 / 损失 / 评测）
  configs/                   nyuv2/sunrgbd 的 S/M/L 训练配置
  tools/                     训练 / 评测 / 推理 / 平均 / 导出、
                             prepare_{nyuv2,sunrgbd} 数据管线
  tests/                     原始代码 vs 本包等价性测试
  docs/                      架构图、协议、数据准备、验证记录
  eval_records/              六权重验证评测 SS/MSF 输出
  model_zoo.json             权重、双重 SHA256、溯源、成绩
```

## 验证

六个正式权重与原始训练代码逐位加载一致（参数相同、逐头 logits 零差异、混淆
矩阵相等）；训练入口逐字节复现原始首个增强批次；SS/MSF 全量重跑与所有记录
数字一致。详见
[`docs/equivalence_verification.md`](docs/equivalence_verification.md)。
代码中的 checkpoint 格式串（如 `c11-csp-local-v1`）是内部加载键，不是模型名。

## 许可

代码以 [MIT 许可证](LICENSE) 发布。预训练骨干与第三方组件仍遵循其自身许可
（Apache-2.0，见 [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)）。
每个发布版本均在 Zenodo 存档：[10.5281/zenodo.23294255](https://zenodo.org/doi/10.5281/zenodo.23294255)。

## 引用

```bibtex
@software{seadpnet2026,
  title  = {SEADP-Net: A Shared-Encoder Asymmetric Dual-Pyramid Network for RGB-D Semantic Segmentation},
  author = {The SEADP-Net Authors},
  year   = {2026},
  doi    = {10.5281/zenodo.23294255},
  url    = {https://github.com/forever678-arch/SEADP-Net}
}
```

## 致谢

共享编码器基于 [RF-DETR](https://github.com/roboflow/rf-detr)（Apache-2.0，
含 SegSmall 预训练权重）及其 DINOv2 骨干（Apache-2.0）；见
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)。局部 CSP 设计为受 C3k2
风格拆分/拼接块启发的自实现，不包含 Ultralytics 代码。
