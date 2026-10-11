# Third-Party Notices

This project builds on the following third-party components. Per-file license
headers inside `seadpnet/rf/` are preserved verbatim.

## RF-DETR (roboflow/rf-detr)

- Used via the `rfdetr` Python package (pinned upstream commit
  `e96a71476bf09c738fa861c3d1ee2c8caeaa5ffe`, version string 1.11.0.dev0)
  as the backbone/runtime dependency, plus the vendored adapter package
  `seadpnet/rf/rfdetr_sem/` from this project's lineage.
- Upstream code: Apache-2.0. The `rf-detr-seg-small.pt` pretrained weights:
  Apache-2.0 per the official repository/licensing pages. Final release must
  verify the license text shipped with the pinned snapshot and reproduce it
  here.
- https://github.com/roboflow/rf-detr — https://roboflow.com/licensing

## DINOv2 (facebookresearch/dinov2)

- Indirect dependency through RF-DETR's backbone implementation. Apache-2.0.
- https://github.com/facebookresearch/dinov2/blob/main/LICENSE

## Ultralytics YOLO11

- NOT a code dependency. The local CSP pyramid (`seadpnet/csp.py`) is a
  self-implementation inspired by the C3k2 split/concatenate design; it does
  not import the Ultralytics runtime, and no Ultralytics source file is
  included. A file-level audit before release must confirm no copied code.

## PyTorch / torchvision / transformers / timm

- BSD-style / Apache-2.0 licenses by their respective distributors; pinned
  versions in `requirements.txt`.

---

**Pre-release checklist:** audit `seadpnet/` for any additional copied
third-party code; attach the RF-DETR NOTICE text of the pinned snapshot;
decide the project's own LICENSE (TBD by the authors).
