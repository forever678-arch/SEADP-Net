"""NYUv2 S / M lineage (checkpoint format ``c11-hrd-v2-v1``).

Final S2/M2 keep the COMPLETE 12-block RF-DETR encoder
(``encoder_keep_layers`` 0..11) and scale the encoder input by 0.5 / 0.75;
layer-pruned variants were explorations and never entered the final model
zoo. ``prune_encoder`` verifies the retained pretrained state and records the
``encoder_scaling`` metadata that these checkpoints carry.
"""
from __future__ import annotations

from torch import nn

from seadpnet.encoder_retention import prune_encoder
from seadpnet.model import (
    EnhancedH, LATE_DECODER_INIT_SEEDS, NUM_CLASSES, PYRAMID_INIT_SEEDS,
    _isolated_cpu_initialization,
)

NYU_SM_CHECKPOINT_FORMAT = "c11-hrd-v2-v1"


class ScaledEncoderEnhancedH(EnhancedH):
    """EnhancedH exactly as the formal NYUv2 S/M training constructed it."""

    def __init__(self, checkpoint, head_a_source="rgb", head_b_source="depth",
                 pyramid_widths=(32, 64, 96), decoder_width=128,
                 branch_a=None, branch_b=None, shared_injection="standard",
                 encoder_keep_layers=None, encoder_input_scale=1.0):
        # Module construction order matches the original: every original
        # module first (inside EnhancedH), then the retention audit.
        super().__init__(checkpoint, head_a_source, head_b_source, pyramid_widths,
                         decoder_width, branch_a, branch_b, shared_injection, NUM_CLASSES,
                         encoder_input_scale)
        prune_encoder(self, encoder_keep_layers)


def build_scaled_encoder_model(cfg):
    """Cold construction wrapped in the recorded seed, as the loader did."""
    return _isolated_cpu_initialization(cfg.get("seed", 42), lambda: ScaledEncoderEnhancedH(
        cfg["pretrained_checkpoint"], cfg.get("head_a_source", "rgb"),
        cfg.get("head_b_source", "depth"), cfg.get("pyramid_widths", (32, 64, 96)),
        cfg.get("decoder_width", 128), cfg.get("branch_a"), cfg.get("branch_b"),
        cfg.get("shared_injection", "standard"), cfg.get("encoder_keep_layers"),
        cfg.get("encoder_input_scale", 1.0)))


def architecture_scaled_encoder(model):
    count = lambda module: sum(parameter.numel() for parameter in module.parameters())
    normalization = {
        "channel_norm": sum(type(module).__name__ == "ChannelNorm" for module in model.modules()),
        "layer_norm": sum(isinstance(module, nn.LayerNorm) for module in model.modules()),
        "batch_norm": sum(isinstance(module, nn.modules.batchnorm._BatchNorm) for module in model.modules()),
        "dropout": sum(isinstance(module, nn.modules.dropout._DropoutNd) for module in model.modules()),
    }
    metadata = {
        "format": NYU_SM_CHECKPOINT_FORMAT,
        "head_a_source": model.head_a_source, "head_b_source": model.head_b_source,
        "init": "rf_detr_pretrained", "trunk_depth": "rgbd", "hr_depth": "rgbd",
        "hr_enabled": True, "semantic_channels": 128, "hr_channels": 64,
        "num_classes": NUM_CLASSES, "pyramid_strides": [4, 8, 16],
        "pyramid_widths": list(model.pyramid_widths), "decoder_width": model.decoder_width,
        "shared_injection": model.shared_injection,
        "decoder_type": "ProgressiveDecoder" if model.shared_injection == "standard" else "LateSharedDecoder",
        "decoder_initialization": "original_h_cold_weights" if model.shared_injection == "standard"
                                  else "new_independent_late_decoder_cold_weights",
        "shared_first_injection_stride": 16 if model.shared_injection == "standard" else 4,
        "branch_a": dict(model.branch_specs["head_a"]),
        "branch_b": dict(model.branch_specs["head_b"]),
        "parameters_total": count(model),
        "parameters_backbone": count(model.backbone),
        "parameters_semantic": count(model.semantic),
        "parameters_hr_stem": count(model.hr_stem),
        "encoder_scaling": dict(model.encoder_scaling),
        "encoder_input_scale": model.encoder_input_scale,
        "local_input_scale": 1.0,
        "normalization": normalization,
        "local_normalization": "ChannelNorm(LayerNorm)/GELU; no BatchNorm or dropout",
    }
    for name in ("head_a", "head_b"):
        spec = model.branch_specs[name]
        pyramid, decoder = getattr(model, f"{name}_pyramid"), getattr(model, f"{name}_decoder")
        metadata[f"parameters_{name}_pyramid"] = count(pyramid)
        metadata[f"parameters_{name}_decoder"] = count(decoder)
        metadata[f"{name}_pyramid_init_seed"] = None if spec["kind"] == "original" else PYRAMID_INIT_SEEDS[name]
        metadata[f"{name}_decoder_init_seed"] = None if model.shared_injection == "standard" else LATE_DECODER_INIT_SEEDS[name]
        metadata[f"{name}_native_strides"] = [4, 8, 16, 32] if spec["context_p5"] else [4, 8, 16]
    return metadata
