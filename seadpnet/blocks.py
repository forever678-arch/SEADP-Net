"""Shared building blocks of SEADP-Net.

Geometry helpers (physical padding, depth gradient input), ChannelNorm
convolutions, the original single-arm InputPyramid control and the
progressive decoder used by both heads. Kept verbatim from the training
implementation; import paths only were renamed.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

BRANCH_SOURCES = {"rgb", "depth"}

def resize(x: torch.Tensor, size, mode: str = "bilinear") -> torch.Tensor:
    if mode == "bilinear":
        return F.interpolate(x, size=size, mode=mode, align_corners=False)
    return F.interpolate(x, size=size, mode=mode)


class ChannelNorm(nn.Module):
    """Normalize channels independently at each spatial location."""

    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        return self.norm(x.movedim(1, -1)).movedim(-1, 1).contiguous()


def conv_norm(in_channels: int, out_channels: int, *, stride: int = 1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1),
        ChannelNorm(out_channels),
        nn.GELU(),
    )


def physical_keep(images: torch.Tensor, padding_mask=None) -> torch.Tensor:
    if padding_mask is None:
        return torch.ones_like(images[:, :1], dtype=torch.bool)
    if padding_mask.dtype != torch.bool:
        raise ValueError("padding_mask must be boolean, True means physical padding")
    if padding_mask.shape != (images.shape[0], *images.shape[-2:]):
        raise ValueError("padding_mask shape must be [B,H,W]")
    return ~padding_mask[:, None]


def make_branch_input(images, depth, valid, padding_mask=None, *, source="depth"):
    """Both alternatives have 3 channels and share the same physical support.

    Depth channels are filled depth, RAW validity and depth-gradient magnitude.
    Raw validity does not gate filled depth. Differences touching physical
    padding are excluded so the padded right/bottom edge creates no gradient.
    RGB is already standardized by the shared dataset implementation.
    """
    if source not in BRANCH_SOURCES:
        raise ValueError(f"branch source must be rgb|depth, got {source!r}")
    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError("images must be [B,3,H,W]")
    expected = (images.shape[0], 1, *images.shape[-2:])
    if depth.shape != expected or valid.shape != expected:
        raise ValueError("depth and validity must be [B,1,H,W]")
    keep = physical_keep(images, padding_mask)
    if source == "rgb":
        return images * keep.to(images.dtype)
    d = torch.nan_to_num(depth.float(), nan=0., posinf=0., neginf=0.).clamp(0, 1)
    v = torch.nan_to_num(valid.float(), nan=0., posinf=0., neginf=0.).clamp(0, 1)
    dx = (d[..., 1:] - d[..., :-1]) * (keep[..., 1:] & keep[..., :-1])
    dy = (d[..., 1:, :] - d[..., :-1, :]) * (keep[..., 1:, :] & keep[..., :-1, :])
    dx, dy = F.pad(dx, (0, 1)), F.pad(dy, (0, 0, 0, 1))
    magnitude = torch.linalg.vector_norm(torch.cat([dx, dy], 1), dim=1, keepdim=True)
    return (torch.cat([d, v, magnitude], 1) * keep.float()).to(images.dtype)


def mask_feature(feature: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
    return feature * resize(keep.float(), feature.shape[-2:], "nearest").to(feature.dtype)


class InputPyramid(nn.Module):
    """Actual learned stride 4/8/16 features, not resized copies of one grid."""

    def __init__(self, widths=(32, 64, 96)):
        super().__init__()
        if len(widths) != 3 or any(int(w) <= 0 for w in widths):
            raise ValueError("pyramid_widths must contain 3 positive widths")
        self.widths = tuple(int(w) for w in widths)
        w4, w8, w16 = self.widths
        self.stride2 = conv_norm(3, 16, stride=2)
        self.stride4 = nn.Sequential(conv_norm(16, w4, stride=2), conv_norm(w4, w4))
        self.stride8 = nn.Sequential(conv_norm(w4, w8, stride=2), conv_norm(w8, w8))
        self.stride16 = nn.Sequential(conv_norm(w8, w16, stride=2), conv_norm(w16, w16))

    def forward(self, branch_input, keep):
        x2 = mask_feature(self.stride2(branch_input), keep)
        x4 = mask_feature(self.stride4(x2), keep)
        x8 = mask_feature(self.stride8(x4), keep)
        x16 = mask_feature(self.stride16(x8), keep)
        return {4: x4, 8: x8, 16: x16}


class FusionStage(nn.Module):
    def __init__(self, in_channels, width):
        super().__init__()
        self.layers = nn.Sequential(conv_norm(in_channels, width), conv_norm(width, width))

    def forward(self, parts, keep):
        return mask_feature(self.layers(torch.cat(parts, 1)), keep)


class ProgressiveDecoder(nn.Module):
    """Semantics at stride 12 + pyramid at 16 -> 8 -> 4 + original HR stem."""

    def __init__(self, pyramid_widths=(32, 64, 96), width=128, semantic_channels=128,
                 hr_channels=64, num_classes=40):
        super().__init__()
        self.width = int(width)
        w4, w8, w16 = pyramid_widths
        self.fuse16 = FusionStage(semantic_channels + w16, self.width)
        self.fuse8 = FusionStage(self.width + w8, self.width)
        self.fuse4 = FusionStage(self.width + w4 + hr_channels, self.width)
        # Direct random-init classifier: every stage receives gradient at step 1.
        self.classifier = nn.Conv2d(self.width, num_classes, 1)

    def forward(self, semantic, hr, pyramid: Mapping[int, torch.Tensor], keep):
        p16, p8, p4 = pyramid[16], pyramid[8], pyramid[4]
        x16 = self.fuse16([resize(semantic, p16.shape[-2:]), p16], keep)
        x8 = self.fuse8([resize(x16, p8.shape[-2:]), p8], keep)
        x4 = self.fuse4([resize(x8, p4.shape[-2:]), p4, resize(hr, p4.shape[-2:])], keep)
        return self.classifier(x4)
