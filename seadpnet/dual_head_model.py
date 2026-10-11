"""Shared-backbone dual-decoder architecture.

Both arms share the SAME RGB-D RF-DETR backbone, semantic projection, and HR
stem. Two INDEPENDENT pyramid+decoder pipelines produce separate stride-4
logits, each supervised with its own CE (equal weight). Inference averages
the two heads' probabilities.

Head A is fed standardized RGB.
Head B takes RGB or [filled depth, raw validity, depth-gradient magnitude].
"""
from __future__ import annotations

from typing import Mapping

import torch
from torch import nn
from torch.nn import functional as F

from seadpnet.blocks import (
    resize, ChannelNorm, conv_norm, physical_keep, make_branch_input,
    mask_feature, InputPyramid, FusionStage, ProgressiveDecoder,
)

BRANCH_SOURCES = {"rgb", "depth"}


class DualHeadExperiment(nn.Module):
    """Shared backbone + two independent pyramid-decoder classification heads."""

    def __init__(self, checkpoint, head_a_source="rgb", head_b_source="rgb",
                 pyramid_widths=(32, 64, 96), decoder_width=128, num_classes=40):
        super().__init__()
        for src in (head_a_source, head_b_source):
            if src not in BRANCH_SOURCES:
                raise ValueError(f"Unknown branch source {src!r}")
        from seadpnet.rf.dense_head import DenseSemModel

        base = DenseSemModel(checkpoint, depth_mode="filled_validity", hr_mode="real",
                             trunk_depth="rgbd", hr_depth="rgbd")
        if base.hr is None:
            raise RuntimeError("H-series requires the RGB-D HR stem")
        self.backbone = base.backbone
        self.semantic = nn.Sequential(base.head.in_proj, base.head.refine)
        self.hr_stem = base.hr.stem
        self.head_a_source = head_a_source
        self.head_b_source = head_b_source
        self.depth_mode = "filled_validity"
        self.trunk_depth = self.hr_depth = "rgbd"
        self.pyramid_widths = tuple(int(w) for w in pyramid_widths)
        self.decoder_width = int(decoder_width)
        self.num_classes = int(num_classes)
        self.emit_legacy_fused = False
        self.checkpoint_heads = True
        for name, src in (("head_a", head_a_source), ("head_b", head_b_source)):
            setattr(self, f"{name}_source", src)
            setattr(self, f"{name}_pyramid", InputPyramid(self.pyramid_widths))
            setattr(self, f"{name}_decoder", ProgressiveDecoder(self.pyramid_widths, self.decoder_width,
                                                               num_classes=self.num_classes))

    def _head_logits(self, name, branch_input, semantic, hr, keep):
        def run_head(x, context, detail, support):
            pyramid = getattr(self, f"{name}_pyramid")(x, support)
            return getattr(self, f"{name}_decoder")(context, detail, pyramid, support)
        if self.checkpoint_heads and self.training and torch.is_grad_enabled():
            from torch.utils.checkpoint import checkpoint
            # These branches contain Conv/LayerNorm/GELU and no random layer or
            # running statistics. Recompute features rather than retaining two
            # full decoder graphs; the shared RF backbone executes only once.
            return checkpoint(run_head, branch_input, semantic, hr, keep,
                              use_reentrant=False, preserve_rng_state=False)
        return run_head(branch_input, semantic, hr, keep)

    def _shared_features(self, x_rgbd, padding_mask):
        from rfdetr.utilities.tensors import NestedTensor
        return self.backbone(NestedTensor(x_rgbd, padding_mask))

    def forward(self, images, padding_mask=None, depth=None, valid=None):
        if depth is None or valid is None:
            raise ValueError("H-series requires depth and RAW validity")
        if images.shape[-2] % 24 or images.shape[-1] % 24:
            raise ValueError("Input must be padded to multiples of 24")
        from rfdetr.utilities.tensors import NestedTensor

        keep = physical_keep(images, padding_mask)
        pmask = ~keep[:, 0]
        clean_depth = torch.nan_to_num(depth, nan=0., posinf=0., neginf=0.)
        clean_valid = torch.nan_to_num(valid, nan=0., posinf=0., neginf=0.)
        x_rgbd = torch.cat([images, clean_depth.to(images.dtype), clean_valid.to(images.dtype)], 1)
        feats, _, _ = self._shared_features(x_rgbd, pmask)
        semantic = self.semantic(feats[0].tensors)
        hr = mask_feature(self.hr_stem(x_rgbd), keep)

        logits_a = self._head_logits("head_a",
                                     make_branch_input(images, depth, valid, pmask, source=self.head_a_source),
                                     semantic, hr, keep)
        logits_b = self._head_logits("head_b",
                                     make_branch_input(images, depth, valid, pmask, source=self.head_b_source),
                                     semantic, hr, keep)

        full_a = resize(logits_a, images.shape[-2:])
        full_b = resize(logits_b, images.shape[-2:])
        result = {
            "semantic_logits": full_a,          # head A alone (for per-head eval)
            "semantic_logits_b": full_b,         # head B alone
            "stride4_a": logits_a,
            "stride4_b": logits_b,
        }
        # Only the isolated legacy-evaluation diagnostic requests this output.
        # Training and canonical evaluation operate on the two raw heads.
        if self.emit_legacy_fused:
            if self.training:raise RuntimeError('Legacy fused path is evaluation-only')
            fused = .5*(full_a.float().softmax(1)+full_b.float().softmax(1))
            result['semantic_logits_fused']=fused.clamp_min(1e-8).log().to(full_a.dtype)
        return result

