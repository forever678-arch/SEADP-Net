"""Parameter-matched RF-DETR-Seg Small semantic baselines.

S0 keeps the pretrained RGB encoder and the first two decoder/mask blocks.
S1 adds a zero-initialized depth input to the same backbone.  The depth
convolution is stored separately from the pretrained RGB convolution so it can
receive its own learning rate without changing the RGB kernel's optimizer
group.  Together they are exactly a four-channel patch projection.
"""

from __future__ import annotations

import torch
from torch import nn

from seadpnet.rf.rfdetr_sem.model import RFDETRSemantic


class SplitRGBDProjection(nn.Module):
    """A 4- or 5-channel patch convolution with independent extra parameters.

    Depth (and optionally validity) convolutions are stored separately from the
    pretrained RGB convolution so they can receive their own learning rate
    without changing the RGB kernel's optimizer group.  Together they are
    exactly a 3+N channel patch projection.
    """

    def __init__(self, rgb: nn.Conv2d, validity: bool = False) -> None:
        super().__init__()
        if not isinstance(rgb, nn.Conv2d) or rgb.in_channels != 3 or rgb.groups != 1:
            raise ValueError("Expected the pretrained three-channel patch Conv2d")
        self.rgb = rgb
        self.depth = nn.Conv2d(
            1,
            rgb.out_channels,
            kernel_size=rgb.kernel_size,
            stride=rgb.stride,
            padding=rgb.padding,
            dilation=rgb.dilation,
            padding_mode=rgb.padding_mode,
            bias=False,
        )
        nn.init.zeros_(self.depth.weight)
        self.validity = None
        if validity:
            self.validity = nn.Conv2d(
                1,
                rgb.out_channels,
                kernel_size=rgb.kernel_size,
                stride=rgb.stride,
                padding=rgb.padding,
                dilation=rgb.dilation,
                padding_mode=rgb.padding_mode,
                bias=False,
            )
            nn.init.zeros_(self.validity.weight)
        self.in_channels = 5 if validity else 4
        self.out_channels = rgb.out_channels
        self.kernel_size = rgb.kernel_size
        self.stride = rgb.stride

    def extra_parameters(self):
        params = list(self.depth.parameters())
        if self.validity is not None:
            params += list(self.validity.parameters())
        return params

    @property
    def weight(self) -> torch.Tensor:
        """DINOv2 reads projection.weight.dtype before calling forward."""
        return self.rgb.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != self.in_channels:
            raise ValueError(f"Expected N×{self.in_channels}×H×W RGB-D input, got {tuple(x.shape)}")
        out = self.rgb(x[:, :3]) + self.depth(x[:, 3:4])
        if self.validity is not None:
            out = out + self.validity(x[:, 4:5])
        return out


class SmallRFDETRSemantic(RFDETRSemantic):
    """Two-decoder-layer S0 (RGB) or S1 (RGB-D) NYU40 semantic model."""

    def __init__(self, variant: str, checkpoint: str, freeze_backbone: bool = False,
                 depth_mode: str = "zeroed", offset_mode: str = "none",
                 bias_mode: str = "none", decoder_layers: int = 2) -> None:
        variant = variant.lower()
        if variant not in {"s0", "s1"}:
            raise ValueError(f"Expected variant 's0' or 's1', got {variant!r}")
        if decoder_layers not in (2, 3, 4):
            raise ValueError(f"decoder_layers must be 2, 3 or 4, got {decoder_layers}")
        if depth_mode not in {"zeroed", "filled", "filled_validity", "zeroed_validity"}:
            raise ValueError(f"Unknown depth_mode {depth_mode!r}")
        if offset_mode not in {"none", "rgb", "depth"}:
            raise ValueError(f"Unknown offset_mode {offset_mode!r}")
        if bias_mode not in {"none", "three_state"}:
            raise ValueError(f"Unknown bias_mode {bias_mode!r}")
        if bias_mode != "none" and offset_mode != "depth":
            raise ValueError("three_state bias currently requires depth offset guidance")
        super().__init__(checkpoint, freeze_backbone=freeze_backbone, use_depth=False)
        self.variant = variant
        self.depth_mode = depth_mode if variant == "s1" else "zeroed"
        self.offset_mode = offset_mode if variant == "s1" else "none"
        self.bias_mode = bias_mode if variant == "s1" else "none"

        transformer = self.core.transformer
        decoder = transformer.decoder
        mask_head = self.core.segmentation_head
        if len(decoder.layers) != 4 or len(mask_head.blocks) != 4:
            raise RuntimeError("Expected four decoder layers and four mask blocks in RF-DETR-Seg Small")
        decoder.layers = nn.ModuleList(list(decoder.layers[:decoder_layers]))
        decoder.num_layers = decoder_layers
        transformer.dec_layers = decoder_layers
        mask_head.blocks = nn.ModuleList(list(mask_head.blocks[:decoder_layers]))

        if variant == "s1":
            needs_validity = self.depth_mode in {"filled_validity", "zeroed_validity"}
            patch = self.encoder.encoder.embeddings.patch_embeddings
            patch.projection = SplitRGBDProjection(patch.projection, validity=needs_validity)
            patch.num_channels = 5 if needs_validity else 4
            self.use_depth = True
            # The base adapter freezes the encoder *before* its own extra-channel
            # surgery.  Apply the requested Stage A policy to these new parameters.
            for param in patch.projection.extra_parameters():
                param.requires_grad_(not freeze_backbone)

        self._guide_stem = None
        self._guide_context = None
        self._guided_layers: list = []
        if self.offset_mode != "none":
            from .offset_guide import GuideStem, OffsetCorrectionHead, GuideContext, GuidedMSDeformAttn
            from .offset_guide import ThreeStateDepthBias
            self._guide_stem = GuideStem(3 if self.offset_mode == "rgb" else 2)
            self._guide_context = GuideContext()
            for layer in self.core.transformer.decoder.layers:
                head = OffsetCorrectionHead(layer.cross_attn.d_model)
                bias = ThreeStateDepthBias(layer.cross_attn.n_heads) if self.bias_mode != "none" else None
                layer.cross_attn = GuidedMSDeformAttn(layer.cross_attn, self._guide_context, head,
                                                      depth_bias=bias)
                self._guided_layers.append(layer.cross_attn)

    def forward(
        self,
        images: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
        depth: torch.Tensor | None = None,
        valid: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if self.variant in {"s0", "s2", "s3"}:
            return super().forward(images, padding_mask=padding_mask, depth=None)
        if depth is None:
            raise ValueError("S1 requires a depth tensor")
        if depth.ndim != 4 or depth.shape[:2] != (images.shape[0], 1) or depth.shape[-2:] != images.shape[-2:]:
            raise ValueError(f"Depth must have shape {(images.shape[0], 1, *images.shape[-2:])}")
        depth = torch.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
        if self.depth_mode in ("zeroed", "zeroed_validity"):
            if valid is not None:
                if valid.shape != depth.shape:
                    raise ValueError(f"Depth validity shape must be {tuple(depth.shape)}")
                depth = depth.masked_fill(~valid.bool(), 0)
        if self.depth_mode in ("filled_validity", "zeroed_validity"):
            if valid is None:
                raise ValueError(f"depth_mode {self.depth_mode} requires a validity tensor")
            if valid.shape != depth.shape:
                raise ValueError(f"Depth validity shape must be {tuple(depth.shape)}")
            depth_pair = torch.cat([depth, valid.to(depth.dtype)], dim=1)
        else:
            depth_pair = None
        if self._guide_context is not None:
            if self.offset_mode == "rgb":
                guide_input = images
            else:
                guide_input = depth_pair if depth_pair is not None else depth
            self._guide_context.set(self._guide_stem(guide_input), depth, valid)
            try:
                return super().forward(images, padding_mask=padding_mask, depth=depth_pair if depth_pair is not None else depth)
            finally:
                self._guide_context.clear()
        return super().forward(images, padding_mask=padding_mask, depth=depth_pair if depth_pair is not None else depth)

    def parameter_groups(
        self,
        lr: float,
        backbone_lr: float,
        depth_lr: float | None = None,
    ) -> list[dict]:
        if self.variant == "s0":
            return super().parameter_groups(lr, backbone_lr)
        depth_params = [p for p in self.encoder.encoder.embeddings.patch_embeddings.projection.extra_parameters()
                        if p.requires_grad]
        depth_ids = {id(p) for p in depth_params}
        encoder_ids = {id(p) for p in self.encoder.parameters()}
        groups = [
            dict(params=[p for p in self.parameters() if p.requires_grad and id(p) not in encoder_ids], lr=lr),
            dict(params=[p for p in self.encoder.parameters() if p.requires_grad and id(p) not in depth_ids],
                 lr=backbone_lr),
        ]
        if depth_params:
            groups.append(dict(params=depth_params, lr=lr if depth_lr is None else depth_lr))
        return groups
