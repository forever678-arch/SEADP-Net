"""Microbatch splitting and the dual-head cross entropy.

Each microbatch CE is a SUM divided by the FULL effective batch's valid pixel
count, so gradient accumulation reproduces one batch-32 update exactly.
"""
import hashlib

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

def split_batch(batch, micro_batch):
    """Slice an already augmented batch without consuming any random numbers."""
    if micro_batch < 1:
        raise ValueError("micro_batch must be positive")
    size = int(batch["semantic"].shape[0])
    if size < 1:
        raise ValueError("Empty loader batch")
    pieces = []
    for start in range(0, size, micro_batch):
        stop = min(size, start + micro_batch)
        piece = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                if value.ndim < 1 or value.shape[0] != size:
                    raise ValueError(f"Inconsistent leading batch dimension: {key}")
                piece[key] = value[start:stop]
            elif isinstance(value, (list, tuple)) and len(value) == size:
                piece[key] = value[start:stop]
            else:
                raise ValueError(f"Unsupported batched value: {key}")
        pieces.append(piece)
    return pieces


def dual_ce(out, target, total_valid_pixels=None):
    """Microbatch contribution to (CE_A + CE_B)/2 over the FULL batch pixels."""
    pixels = int((target != 255).sum().item())
    denominator = pixels if total_valid_pixels is None else int(total_valid_pixels)
    if denominator <= 0 or pixels > denominator:
        raise ValueError("Invalid total valid-pixel count")
    def ce_sum(raw_logits, labels):
        # Casting must happen inside the recomputed region. Checkpoint inputs
        # then retain BF16 logits, rather than two full FP32 log-softmax graphs.
        return F.cross_entropy(raw_logits.float(), labels,
                               ignore_index=255, reduction="sum")

    def memory_efficient_ce(raw_logits):
        if torch.is_grad_enabled() and raw_logits.requires_grad:
            # CE has no stochastic operations: recomputation consumes no RNG.
            # Only this inexpensive loss is repeated, never the model forward.
            return checkpoint(ce_sum, raw_logits, target, use_reentrant=False,
                              preserve_rng_state=False)
        return ce_sum(raw_logits, target)

    a = memory_efficient_ce(out["semantic_logits"])
    b = memory_efficient_ce(out["semantic_logits_b"])
    loss = (a + b) / (2.0 * denominator)
    return loss, {"ce_a": a.detach().item() / denominator,
                  "ce_b": b.detach().item() / denominator}


def first_batch_fingerprint(batch):
    """Complete CPU loader-batch hash, independent of the microbatch size."""
    digest = hashlib.sha256()
    for key in ("image", "depth", "valid", "semantic", "padding_mask"):
        value = batch[key].detach().contiguous().cpu()
        digest.update(f"{key}|{value.dtype}|{tuple(value.shape)}".encode())
        digest.update(value.numpy().tobytes())
    ids = batch["image_id"]
    return {"first_update_image_ids": ids.tolist() if isinstance(ids, torch.Tensor) else list(ids),
            "first_update_batch_sha256": digest.hexdigest()}


def new_module_gradient_l2(model):
    result = {}
    parameters = list(model.named_parameters())
    for module in ("head_a_pyramid", "head_a_decoder", "head_b_pyramid", "head_b_decoder"):
        terms = [p.grad.detach().float().square().sum() for name, p in parameters
                 if name.startswith(module + ".") and p.grad is not None]
        result[module] = float(torch.stack(terms).sum().sqrt()) if terms else 0.0
    if not all(np.isfinite(value) for value in result.values()):
        raise RuntimeError("Nonfinite new-module gradient diagnostics")
    return result
