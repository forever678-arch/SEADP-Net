"""Dense semantic head on the shared RGB-D encoder, with a stride-4 detail branch.

DENSE decodes the projected [B,256,40,54] memory (stride 12) with a light
convolutional classifier.  A high-resolution residual branch reads the
raw 5-channel input at stride 4 and refines the upsampled semantic
feature; trunk_depth / hr_depth select real or zeroed depth at
the two entry points.
The residual output layer is zero-initialised: initial behaviour is exactly
the plain DENSE model.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from rfdetr.utilities.tensors import NestedTensor
from .small import SmallRFDETRSemantic


class DenseSemHead(nn.Module):
    def __init__(self, in_dim: int = 256, hidden: int = 128, num_classes: int = 40) -> None:
        super().__init__()
        # GroupNorm(1) = LayerNorm over the channel dim for [B, C, h, w] inputs.
        self.in_proj = nn.Sequential(nn.GroupNorm(1, in_dim), nn.Conv2d(in_dim, hidden, 1))
        self.refine = nn.Sequential(
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.GELU(),
        )
        self.cls = nn.Conv2d(hidden, num_classes, 1)

    def features(self, feat: torch.Tensor) -> torch.Tensor:
        return self.refine(self.in_proj(feat))

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.cls(self.features(feat))


class HRDetailBranch(nn.Module):
    """Stride-4 stem on the raw input plus a zero-init residual refinement."""

    def __init__(self, in_ch: int = 5, hidden: int = 64, sem_hidden: int = 128,
                 num_classes: int = 40) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(32, hidden, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(hidden + sem_hidden, hidden, 3, padding=1), nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.GELU(),
        )
        self.residual = nn.Conv2d(hidden, num_classes, 1)
        nn.init.zeros_(self.residual.weight)
        nn.init.zeros_(self.residual.bias)

    def forward(self, x_in: torch.Tensor, grid_feat: torch.Tensor,
                grid_logits: torch.Tensor) -> torch.Tensor:
        hr = self.stem(x_in)                                    # [B,hid,H/4,W/4]
        target = hr.shape[-2:]
        up_feat = F.interpolate(grid_feat, size=target, mode="bilinear", align_corners=False)
        res = self.residual(self.fuse(torch.cat([hr, up_feat], dim=1)))
        base = F.interpolate(grid_logits, size=target, mode="bilinear", align_corners=False)
        return base + res                                        # [B,40,H/4,W/4]


class DenseSemModel(nn.Module):
    def __init__(self, checkpoint: str, depth_mode: str = "filled_validity",
                 hr_mode: str = "none", trunk_depth: str | None = None,
                 hr_depth: str | None = None) -> None:
        super().__init__()
        if hr_mode not in ("none", "zero", "real"):
            raise ValueError(f"Unknown hr_mode {hr_mode!r}")
        # Legacy hr_mode semantics: 'zero' zeroed the shared input (RGB-only)
        # and 'real' kept real depth; the explicit trunk_depth/hr_depth keys
        # control the two entry points independently.
        #
        if hr_mode != "none" and trunk_depth is None and hr_depth is None:
            trunk_depth = hr_depth = ("rgbd" if hr_mode == "real" else "rgb")
        # Explicit trunk/hr depth keys imply the HR branch must exist: the
        # C10/C01 arms were silently run WITHOUT it when hr_mode was left at
        # its default in their configs.
        if hr_mode == "none" and (trunk_depth is not None or hr_depth is not None):
            hr_mode = "real"
        self.trunk_depth = trunk_depth or "rgbd"
        self.hr_depth = hr_depth or "rgbd"
        for v in (self.trunk_depth, self.hr_depth):
            if v not in ("rgb", "rgbd"):
                raise ValueError(f"depth control must be rgb|rgbd, got {v!r}")
        base = SmallRFDETRSemantic("s1", checkpoint, depth_mode=depth_mode)
        self.backbone = base.core.backbone
        self.depth_mode = depth_mode
        self.hr_mode = hr_mode
        self.head = DenseSemHead(num_classes=40)
        self.hr = HRDetailBranch() if hr_mode != "none" else None  # guarded above: depth keys force HR

    def forward(self, images, padding_mask=None, depth=None, valid=None):
        if depth is None or valid is None:
            raise ValueError("DenseSemModel requires depth and valid inputs")
        depth = torch.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
        x_full = torch.cat([images, depth.to(images.dtype), valid.to(images.dtype)], dim=1)

        def maybe_rgb_only(x: torch.Tensor) -> torch.Tensor:
            return torch.cat([x[:, :3], torch.zeros_like(x[:, 3:])], dim=1)

        x_trunk = maybe_rgb_only(x_full) if self.trunk_depth == "rgb" else x_full
        x_hr = maybe_rgb_only(x_full) if self.hr_depth == "rgb" else x_full
        feats, _, _ = self.backbone(NestedTensor(x_trunk, padding_mask if padding_mask is not None
                                                 else torch.zeros_like(images[:, 0, :, :], dtype=torch.bool)))
        grid_feat = self.head.features(feats[0].tensors)          # [B,128,40,54]
        grid = self.head.cls(grid_feat)                           # [B,40,40,54]
        if self.hr is not None:
            refined = self.hr(x_hr, grid_feat, grid)              # [B,40,H/4,W/4]
            logits = F.interpolate(refined, size=images.shape[-2:], mode="bilinear", align_corners=False)
        else:
            logits = F.interpolate(grid, size=images.shape[-2:], mode="bilinear", align_corners=False)
        return dict(semantic_logits=logits, grid_logits=grid)


__all__ = ["DenseSemModel", "DenseSemHead", "HRDetailBranch"]
