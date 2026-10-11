"""Independent, YOLO11-inspired CSP encoders for the two local pyramids.

The split/concatenate blocks are adapted from the C3k2 design; this is not an
official YOLO11 implementation and does not import the Ultralytics runtime.
ChannelNorm and GELU preserve the original branch's normalization convention.
There is no BatchNorm, running statistic, dropout, or attention module, so
non-reentrant activation recomputation is deterministic.
"""
from __future__ import annotations

from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F

from seadpnet.blocks import ChannelNorm, mask_feature, resize


def _mask(x: torch.Tensor, keep: Optional[torch.Tensor]) -> torch.Tensor:
    return x if keep is None else mask_feature(x, keep)


class ConvNorm(nn.Module):
    """A 1x1 or padded 3x3 convolution followed by ChannelNorm and GELU."""

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int = 3, stride: int = 1):
        super().__init__()
        if kernel_size not in (1, 3):
            raise ValueError("ConvNorm kernel_size must be 1 or 3")
        if in_channels <= 0 or out_channels <= 0 or stride <= 0:
            raise ValueError("ConvNorm channels and stride must be positive")
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=kernel_size // 2)
        self.norm = ChannelNorm(out_channels)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor, keep: Optional[torch.Tensor] = None):
        # Mask before and after every convolution: a padded value cannot leak
        # into a supported neighbour through a later spatial convolution.
        x = self.conv(_mask(x, keep))
        return _mask(self.activation(self.norm(x)), keep)


class ResidualBottleneck(nn.Module):
    """Two 3x3 transforms at the CSP branch width, with an optional shortcut."""

    def __init__(self, channels: int, *, shortcut: bool = True):
        super().__init__()
        self.channels = int(channels)
        self.shortcut = bool(shortcut)
        self.conv1 = ConvNorm(self.channels, self.channels, 3)
        self.conv2 = ConvNorm(self.channels, self.channels, 3)

    def forward(self, x: torch.Tensor, keep: Optional[torch.Tensor] = None):
        x = _mask(x, keep)
        transformed = self.conv2(self.conv1(x, keep), keep)
        return _mask(x + transformed if self.shortcut else transformed, keep)


class C3k(nn.Module):
    """Optional nested CSP transform containing two residual bottlenecks."""

    def __init__(self, in_channels: int, out_channels: int, *, repeats: int = 2,
                 expansion: float = 0.5):
        super().__init__()
        if repeats <= 0 or not 0 < expansion <= 1:
            raise ValueError("C3k repeats must be positive and expansion in (0,1]")
        self.hidden_channels = max(1, int(out_channels * expansion))
        hidden = self.hidden_channels
        self.branch = ConvNorm(in_channels, hidden, 1)
        self.bypass = ConvNorm(in_channels, hidden, 1)
        self.blocks = nn.ModuleList(ResidualBottleneck(hidden)
                                    for _ in range(repeats))
        self.project = ConvNorm(2 * hidden, out_channels, 1)

    def forward(self, x: torch.Tensor, keep: Optional[torch.Tensor] = None):
        branch = self.branch(x, keep)
        bypass = self.bypass(x, keep)
        for block in self.blocks:
            branch = block(branch, keep)
        return self.project(torch.cat((branch, bypass), dim=1), keep)


class C3k2(nn.Module):
    """C3k2-style split, sequential residual transforms, concatenate, project.

    The initial 1x1 convolution produces exactly two ``hidden_channels``
    chunks. Each transform consumes the latest chunk; all intermediate chunks
    are concatenated with the two initial chunks before the final projection.
    ``use_c3k=True`` substitutes a nested, two-bottleneck C3k for each transform.
    """

    def __init__(self, in_channels: int, out_channels: int, repeats: int = 1,
                 *, use_c3k: bool = False, expansion: float = 0.5):
        super().__init__()
        if repeats <= 0 or not 0 < expansion <= 1:
            raise ValueError("C3k2 repeats must be positive and expansion in (0,1]")
        self.repeats = int(repeats)
        self.use_c3k = bool(use_c3k)
        self.hidden_channels = max(1, int(out_channels * expansion))
        hidden = self.hidden_channels
        self.split = ConvNorm(in_channels, 2 * hidden, 1)
        self.blocks = nn.ModuleList(
            C3k(hidden, hidden, repeats=2) if self.use_c3k
            else ResidualBottleneck(hidden)
            for _ in range(self.repeats)
        )
        self.project = ConvNorm((2 + self.repeats) * hidden, out_channels, 1)

    def forward(self, x: torch.Tensor, keep: Optional[torch.Tensor] = None):
        chunks = list(self.split(x, keep).chunk(2, dim=1))
        for block in self.blocks:
            chunks.append(block(chunks[-1], keep))
        return self.project(torch.cat(chunks, dim=1), keep)


class PlainStage(nn.Module):
    """Width-matched control: two 3x3 transforms per requested repeat.

    This control matches stage widths, strides, normalization and repeat counts;
    its parameter count and computation are deliberately reported as actual
    values, rather than claimed to be matched to the CSP design.
    """

    def __init__(self, in_channels: int, out_channels: int, repeats: int = 1):
        super().__init__()
        if repeats <= 0:
            raise ValueError("PlainStage repeats must be positive")
        self.repeats = int(repeats)
        self.layers = nn.ModuleList()
        for index in range(2 * self.repeats):
            self.layers.append(ConvNorm(in_channels if index == 0 else out_channels,
                                        out_channels, 3))

    def forward(self, x: torch.Tensor, keep: Optional[torch.Tensor] = None):
        for layer in self.layers:
            x = layer(x, keep)
        return _mask(x, keep)


class SPPF(nn.Module):
    """Pure-torch, repeated 5x5 max pooling with a 1x1 fusion projection."""

    def __init__(self, channels: int):
        super().__init__()
        hidden = max(1, int(channels) // 2)
        self.reduce = ConvNorm(channels, hidden, 1)
        self.pool = nn.MaxPool2d(5, stride=1, padding=2)
        self.project = ConvNorm(4 * hidden, channels, 1)

    def _pool(self, x: torch.Tensor, keep: Optional[torch.Tensor]):
        if keep is not None:
            support = resize(keep.float(), x.shape[-2:], "nearest").bool()
            # Invalid zeros must not win a max against negative valid values.
            x = x.masked_fill(~support, torch.finfo(x.dtype).min)
        return _mask(self.pool(x), keep)

    def forward(self, x: torch.Tensor, keep: Optional[torch.Tensor] = None):
        x = self.reduce(x, keep)
        first = self._pool(x, keep)
        second = self._pool(first, keep)
        third = self._pool(second, keep)
        return self.project(torch.cat((x, first, second, third), dim=1), keep)


class StrongPyramid(nn.Module):
    """Learned P2/P3/P4 features with optional P5 context and top-down fusion.

    Default native widths are 64/128/256 with CSP repeat counts 1/2/2. The
    1x1 adapters emit 32/64/96 channels for the progressive decoder. Native
    spatial sizes are ``ceil(H / stride), ceil(W / stride)`` at strides 4/8/16;
    P2 remains at stride 4. ``keep`` is [B,1,H,W], with True for physical support.
    """

    def __init__(self, widths=(64, 128, 256), repeats=(1, 2, 2),
                 output_widths=(32, 64, 96), context_p5: bool = False,
                 topdown: bool = False, block_type: str = "csp"):
        super().__init__()
        self.widths = self._positive_triple(widths, "widths")
        self.repeats = self._positive_triple(repeats, "repeats")
        self.out_widths = self._positive_triple(output_widths, "output_widths")
        if block_type not in ("csp", "plain"):
            raise ValueError("block_type must be 'csp' or 'plain'")
        self.native_channels = self.native_widths = self.widths
        self.output_widths = self.out_widths
        self.mode = self.block_type = block_type
        self.context_p5 = bool(context_p5)
        self.topdown = bool(topdown)
        self.strides = (4, 8, 16)
        w4, w8, w16 = self.widths
        stage = C3k2 if block_type == "csp" else PlainStage
        stem_width = max(1, w4 // 2)
        self.stem = ConvNorm(3, stem_width, 3, stride=2)
        self.down4 = ConvNorm(stem_width, w4, 3, stride=2)
        self.stage4 = stage(w4, w4, self.repeats[0])
        self.down8 = ConvNorm(w4, w8, 3, stride=2)
        self.stage8 = stage(w8, w8, self.repeats[1])
        self.down16 = ConvNorm(w8, w16, 3, stride=2)
        self.stage16 = stage(w16, w16, self.repeats[2])
        self.down32 = self.stage32 = self.sppf = self.fuse16 = None
        if self.context_p5:
            w32 = 2 * w16
            self.down32 = ConvNorm(w16, w32, 3, stride=2)
            self.stage32 = stage(w32, w32, 2)
            self.sppf = SPPF(w32)
            self.fuse16 = stage(w16 + w32, w16, 1)
        self.fuse8 = self.fuse4 = None
        if self.topdown:
            self.fuse8 = stage(w8 + w16, w8, 1)
            self.fuse4 = stage(w4 + w8, w4, 1)
        self.adapters = nn.ModuleDict({
            str(stride): ConvNorm(native, output, 1)
            for stride, native, output in zip(self.strides, self.widths, self.out_widths)
        })

    @staticmethod
    def _positive_triple(values, name):
        values = tuple(values)
        if len(values) != 3 or any(isinstance(value, bool) or int(value) != value
                                   or int(value) <= 0 for value in values):
            raise ValueError(f"{name} must contain three positive integers")
        return tuple(int(value) for value in values)

    def forward(self, branch_input: torch.Tensor,
                keep: Optional[torch.Tensor] = None):
        if branch_input.ndim != 4 or branch_input.shape[1] != 3:
            raise ValueError("branch_input must be [B,3,H,W]")
        if keep is not None and (keep.dtype != torch.bool or
                                keep.shape != (branch_input.shape[0], 1,
                                               *branch_input.shape[-2:])):
            raise ValueError("keep must be boolean [B,1,H,W], True means support")
        x2 = self.stem(branch_input, keep)
        p4 = _mask(self.stage4(self.down4(x2, keep), keep), keep)
        p8 = _mask(self.stage8(self.down8(p4, keep), keep), keep)
        p16 = _mask(self.stage16(self.down16(p8, keep), keep), keep)
        if self.context_p5:
            p32 = self.down32(p16, keep)
            p32 = _mask(self.stage32(p32, keep), keep)
            p32 = _mask(self.sppf(p32, keep), keep)
            p16 = _mask(self.fuse16(torch.cat((p16, resize(p32, p16.shape[-2:])),
                                              dim=1), keep), keep)
        if self.topdown:
            p8 = _mask(self.fuse8(torch.cat((p8, resize(p16, p8.shape[-2:])),
                                           dim=1), keep), keep)
            p4 = _mask(self.fuse4(torch.cat((p4, resize(p8, p4.shape[-2:])),
                                           dim=1), keep), keep)
        return {stride: _mask(self.adapters[str(stride)](feature, keep), keep)
                for stride, feature in ((4, p4), (8, p8), (16, p16))}


__all__ = ["ConvNorm", "ResidualBottleneck", "C3k", "C3k2", "PlainStage",
           "SPPF", "StrongPyramid"]
