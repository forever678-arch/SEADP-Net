"""Single-image RGB-D semantic segmentation inference.

Loads any SEADP-Net checkpoint (NYUv2 or SUN RGB-D lineage), runs the fused
0.5/0.5 probability prediction on one rgb/depth/valid sample and saves a
colour-mapped PNG plus the raw class-index PNG.

  python tools/infer.py --checkpoint model.pth --rgb in/rgb.png --depth in/depth.png \
      --valid in/valid.png --dataset sunrgbd --out-seg seg.png --out-color seg_color.png \
      --rf-checkpoint /path/rf-detr-seg-small.pt

depth.png: uint16 millimetres (SUN convention) or float32 .npy metres.
valid.png: nonzero pixels are valid RAW sensor returns (white = valid).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def load_depth(path, depth_max_meters):
    path = Path(path)
    if path.suffix == ".npy":
        depth = np.load(path, allow_pickle=False).astype(np.float32)
    else:
        raw = np.asarray(Image.open(path))
        if raw.dtype == np.uint8:
            raise ValueError("8-bit depth is not the stored convention; expected uint16 millimetres")
        depth = raw.astype(np.float32) / 1000.0
    depth = np.nan_to_num(depth, nan=0., posinf=0., neginf=0.).clip(0, depth_max_meters)
    return depth


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--rgb", required=True)
    parser.add_argument("--depth", required=True)
    parser.add_argument("--valid", required=True)
    parser.add_argument("--dataset", choices=("nyuv2", "sunrgbd"), required=True)
    parser.add_argument("--input-size", type=int, nargs=2, default=None,
                        help="Physical H W before padding (default 480 640 / 530 730)")
    parser.add_argument("--rf-checkpoint", required=True)
    parser.add_argument("--out-seg", required=True)
    parser.add_argument("--out-color")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    from torchvision.transforms import functional as TF
    from torch.nn import functional as F
    from seadpnet.model import load_experiment

    model, state = load_experiment(args.checkpoint, device=args.device,
                                   overrides={"rf_checkpoint": args.rf_checkpoint})
    model.eval()
    cfg = state["config"]
    num_classes = int(cfg["num_classes"])
    depth_max = float(cfg.get("depth_max_meters", 10.0))
    height, width = args.input_size or ((480, 640) if args.dataset == "nyuv2" else (530, 730))
    multiple = 24 if args.dataset == "nyuv2" else int(cfg.get("padding_multiple", 24))

    rgb = Image.open(args.rgb).convert("RGB").resize((width, height), Image.Resampling.BILINEAR)
    image = TF.normalize(TF.to_tensor(rgb), [.485, .456, .406], [.229, .224, .225])
    depth = load_depth(args.depth, depth_max)
    depth = torch.from_numpy(np.ascontiguousarray(depth))[None]
    depth = F.interpolate(depth[None], (height, width), mode="nearest")[0]
    valid = np.asarray(Image.open(args.valid)) > 0
    valid = torch.from_numpy(valid.astype(np.float32))[None]
    valid = F.interpolate(valid[None], (height, width), mode="nearest")[0]

    ph, pw = ((height + multiple - 1) // multiple) * multiple, ((width + multiple - 1) // multiple) * multiple
    pad = (0, pw - width, 0, ph - height)
    image, depth, valid = F.pad(image, pad), F.pad(depth, pad), F.pad(valid, pad)
    padding_mask = torch.ones((ph, pw), dtype=torch.bool)
    padding_mask[:height, :width] = False

    with torch.no_grad(), torch.autocast(args.device.split(":")[0], dtype=torch.bfloat16,
                                         enabled=args.device.startswith("cuda")):
        out = model(image[None].to(args.device), padding_mask[None].to(args.device),
                    depth[None].to(args.device), valid[None].to(args.device))
    prob = .5 * out["semantic_logits"].float().softmax(1) + .5 * out["semantic_logits_b"].float().softmax(1)
    logits = prob[0, :, :height, :width]
    seg = logits.argmax(0).byte().cpu().numpy()
    Image.fromarray(seg).save(args.out_seg)
    if args.out_color:
        palette = np.zeros((num_classes, 3), np.uint8)
        generator = np.random.RandomState(7)
        palette[1:] = generator.randint(32, 255, size=(num_classes - 1, 3))
        Image.fromarray(palette[seg]).save(args.out_color)
    print(f"saved {args.out_seg} ({height}x{width}, {num_classes} classes)")


if __name__ == "__main__":
    main()
