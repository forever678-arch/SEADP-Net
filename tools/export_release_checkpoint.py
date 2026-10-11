"""Export a training checkpoint as a clean release checkpoint.

Rewrites ONLY non-tensor metadata (config / architecture / provenance /
averaging strings): release naming for experiments, portable relative
references for file paths, and no absolute host paths. Every model tensor is
copied untouched and verified bit-exact before and after saving, so the
exported file computes identical outputs.

  python tools/export_release_checkpoint.py --checkpoint in.pth --out out.pth

The report prints source and output SHA256; keep both in the release records
(model zoo) so the lineage stays auditable.
"""
from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path

import torch

PATH_MAP = [
    ("/root/autodl-tmp/rfdetr-csp-branch-train/runs/y5_deeper_depth/u", "formal-run-nyuv2-l/u"),
    ("/root/autodl-tmp/rfdetr-hrd-v2/runs/hrd_s2/u", "formal-run-nyuv2-s/u"),
    ("/root/autodl-tmp/rfdetr-hrd-v2/runs/hrd_m2/u", "formal-run-nyuv2-m/u"),
    ("/root/autodl-tmp/seadp-sun-sm/runs/sun16000_lrb10_s2/formal16000/", "sunrgbd-formal-run-s/"),
    ("/root/autodl-tmp/seadp-sun-sm/runs/sun16000_lrb10_m2/formal16000/", "sunrgbd-formal-run-m/"),
    ("/root/autodl-tmp/seadp-net-sunrgbd/runs/sun16000_lrb10/formal16000/", "sunrgbd-formal-run-l/"),
    ("/root/autodl-tmp/seadp-net-sunrgbd/runs/", "sunrgbd-runs/"),
    ("/root/autodl-tmp/seadp-sun-sm/runs/", "sunrgbd-runs/"),
    ("/root/autodl-tmp/seadp-third-group-qualitative-20261008/weights/y5_avg234.pth", "weights/seadp-net-nyuv2-l.pth"),
    ("y5_avg234.pth", "weights/seadp-net-nyuv2-l.pth"),
    ("hrd_s2_average_primary.pth", "weights/seadp-net-nyuv2-s.pth"),
    ("hrd_m2_average_primary.pth", "weights/seadp-net-nyuv2-m.pth"),
    ("s2_avg_E_u6500_u7500_u8500.pth", "weights/seadp-net-sunrgbd-s.pth"),
    ("m2_avg_E_u6500_u7500_u8500.pth", "weights/seadp-net-sunrgbd-m.pth"),
    ("lrb10_avg_E_u6500_u7500_u8500.pth", "weights/seadp-net-sunrgbd-l.pth"),
    ("/root/autodl-tmp/seadp-net-sunrgbd/data/sunrgbd_processed", "sunrgbd_processed"),
    ("/root/autodl-tmp/rfdetr-y5-ablation/data/nyuv2", "nyuv2"),
    ("/root/autodl-tmp/rfdetr-y5-ablation/depth_validity", "nyuv2_raw_validity"),
    ("/root/autodl-tmp/rfdetr-y5-ablation/weights/rf-detr-seg-small.pt", "rf-detr-seg-small.pt"),
    ("/root/autodl-tmp/seadp-sun-sm/weights/rf-detr-seg-small.pt", "rf-detr-seg-small.pt"),
    ("/root/autodl-tmp/seadp-net-sunrgbd/weights/rf-detr-seg-small.pt", "rf-detr-seg-small.pt"),
    ("/root/autodl-tmp/seadp-avg/", ""),
    ("/root/autodl-tmp/seadp-third-group-qualitative-20261008/weights/", ""),
    ("SEADP-Net-SUNRGBD-S2-LRB10", "SEADP-Net-SUNRGBD-S"),
    ("SEADP-Net-SUNRGBD-M2-LRB10", "SEADP-Net-SUNRGBD-M"),
    ("SEADP-Net-SUNRGBD-L-LRB10", "SEADP-Net-SUNRGBD-L"),
    ("SEADP-Net-SUNRGBD-LRB10", "SEADP-Net-SUNRGBD-L"),
    ("sun16000_lrb10_m2", "sunrgbd-formal-run-m"),
    ("sun16000_lrb10_s2", "sunrgbd-formal-run-s"),
    ("sun16000_lrb10", "sunrgbd-formal-run-l"),
    ("/root/autodl-tmp/", ""),
    ("weights/rf-detr-seg-small.pt", "rf-detr-seg-small.pt"),
]
PATH_MAP.sort(key=lambda p: -len(p[0]))

CODE = re.compile(r"(?i)(lrb10|hrd_|y5_avg|m2_avg|s2_avg|y5_deeper|sun16000|autodl|/root/)")
DROPPED_KEYS = ("experiment_arm",)


def sanitize(obj):
    if isinstance(obj, str):
        for old, new in PATH_MAP:
            obj = obj.replace(old, new)
        return obj
    if isinstance(obj, list):
        return [sanitize(v) for v in obj]
    if isinstance(obj, dict):
        return {k: sanitize(v) for k, v in obj.items() if k not in DROPPED_KEYS}
    return obj


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def assert_clean_metadata(state):
    def walk(obj, where):
        if isinstance(obj, str):
            if CODE.search(obj):
                raise ValueError(f"unsanitized string at {where}: {obj[:100]!r}")
        elif isinstance(obj, dict):
            for k, v in obj.items():
                walk(v, f"{where}.{k}")
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                walk(v, f"{where}[{i}]")
    for key, value in state.items():
        if key != "model":
            walk(value, key)


def tensors_identical(a, b):
    if sorted(a) != sorted(b):
        return False, "key sets differ"
    for k in a:
        if not torch.equal(a[k], b[k]):
            return False, f"tensor differs: {k}"
    return True, ""


def export(source, target):
    source, target = Path(source), Path(target)
    state = torch.load(source, map_location="cpu", weights_only=False)
    exported = {k: (v if k == "model" else sanitize(v)) for k, v in state.items()}
    assert_clean_metadata(exported)
    ok, why = tensors_identical(state["model"], exported["model"])
    if not ok:
        raise RuntimeError(f"tensor set changed during export: {why}")
    with target.open("xb") as stream:
        torch.save(exported, stream)
    reread = torch.load(target, map_location="cpu", weights_only=False)
    assert_clean_metadata(reread)
    ok, why = tensors_identical(state["model"], reread["model"])
    if not ok:
        raise RuntimeError(f"saved file tensor mismatch: {why}")
    return {"source": str(source), "source_sha256": sha256(source),
            "output": str(target), "output_sha256": sha256(target),
            "tensors": len(reread["model"]), "tensor_bits_identical": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    import json
    print(json.dumps(export(args.checkpoint, args.out), indent=2))


if __name__ == "__main__":
    main()
