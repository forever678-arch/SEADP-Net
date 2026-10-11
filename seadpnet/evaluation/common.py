"""Shared evaluation geometry: BF16 autocast context and image views.

Logits are restored (unflip, resize to the padded input, crop) BEFORE the
softmax; multi-scale/flip views average per-head probabilities and the two
heads fuse with a fixed 0.5/0.5 mean.
"""
from __future__ import annotations

import contextlib

import torch
from torch.nn import functional as F

def amp_context(device="cuda"):
    if str(device).startswith("cuda"):
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def resize(x, size, mode="bilinear"):
    kwargs = {"align_corners": False} if mode == "bilinear" else {}
    return F.interpolate(x, size=size, mode=mode, **kwargs)


def restore_logits(logits, padded_size, mirrored=False, semantic_width=640):
    """Invert a view before computing probabilities; crop physical right padding."""
    logits = logits.float()
    if mirrored:
        logits = logits.flip(-1)
    logits = resize(logits, padded_size)
    if semantic_width > padded_size[1]:
        raise ValueError("Semantic width exceeds padded image width")
    return logits[..., :semantic_width]


def view_inputs(batch, device, scale, mirrored):
    if scale <= 0:
        raise ValueError("Positive scale required")
    image, depth, valid = [batch[key].to(device) for key in ("image", "depth", "valid")]
    keep = (~batch["padding_mask"]).to(device)[:, None].float()
    height, width = image.shape[-2:]
    size = (max(24, round(height * scale / 24) * 24),
            max(24, round(width * scale / 24) * 24))
    image = resize(image, size)
    depth, valid, keep = [resize(value, size, "nearest") for value in (depth, valid, keep)]
    if mirrored:
        image, depth, valid, keep = [value.flip(-1) for value in (image, depth, valid, keep)]
    return image, ~keep[:, 0].bool(), depth, valid


def view_head_logits(model, batch, device, scale, flip):
    """One shared forward; two FP32 logits restored and cropped to RGB geometry."""
    inputs = view_inputs(batch, device, scale, flip)
    with amp_context(device):
        output = model(*inputs)
    size = batch["image"].shape[-2:]
    return tuple(restore_logits(output[key], size, flip)
                 for key in ("semantic_logits", "semantic_logits_b"))


def view_probabilities(model, batch, device, scale, mirrored):
    """Restore logits before softmax, including the padding crop."""
    return tuple(logits.softmax(1) for logits in view_head_logits(model, batch, device, scale, mirrored))


def metrics_result(metric):
    result = metric.compute()
    result["miou_percent"] = 100 * result["miou"]
    if hasattr(metric, "matrix"):
        matrix = metric.matrix.detach().cpu()
        names = list(result["per_class_iou"])
        totals = matrix.sum(1)
        result["per_class_pixel_accuracy"] = {
            name: float(matrix[i, i].double() / totals[i]) if totals[i] > 0 else None
            for i, name in enumerate(names)
        }
        result["confusion_matrix"] = matrix.tolist()
    return result
