"""SEADP-Net model: the shared H graph with CSP local arms (EnhancedH).

Handles both checkpoint lineages: NYUv2 (``c11-csp-local-v1``) and SUN RGB-D
(``seadp-sunrgbd-v1``). Construction, initialization order and architecture
metadata are the training implementations verbatim; only import paths were
renamed and the SUN loader was merged in.
"""
from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path

import torch
from torch import nn

from seadpnet.dual_head_model import DualHeadExperiment
from seadpnet.blocks import FusionStage, resize


CHECKPOINT_FORMAT = "c11-csp-local-v1"
BRANCH_KINDS = frozenset({"original", "csp", "plain"})
SHARED_INJECTIONS = frozenset({"standard", "late"})
PYRAMID_INIT_SEEDS = {"head_a": 42, "head_b": 43}
LATE_DECODER_INIT_SEEDS = {"head_a": 142, "head_b": 143}
NUM_CLASSES = 40


def _positive_triplet(values, field):
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{field} must contain 3 positive integers")
    try:
        result = tuple(int(value) for value in values)
        valid = len(result) == 3 and all(
            not isinstance(value, bool) and value == integer and integer > 0
            for value, integer in zip(values, result)
        )
    except (TypeError, ValueError, OverflowError):
        valid = False
    if not valid:
        raise ValueError(f"{field} must contain 3 positive integers")
    return result


def _positive_integer(value, field):
    try:
        integer = int(value)
        valid = not isinstance(value, bool) and value == integer and integer > 0
    except (TypeError, ValueError, OverflowError):
        valid = False
    if not valid:
        raise ValueError(f"{field} must be a positive integer")
    return integer


def validate_branch_spec(spec=None, *, output_widths=(32, 64, 96)):
    """Return the explicit, JSON-compatible configuration of one local arm.

    Missing specifications retain the original InputPyramid. Its widths are
    the decoder-facing widths, and stronger-encoder flags cannot silently
    become no-ops on that control arm.
    """
    output_widths = _positive_triplet(output_widths, "pyramid_widths")
    if spec is None:
        spec = {}
    if not isinstance(spec, Mapping):
        raise ValueError("branch specification must be a mapping")
    unknown = set(spec) - {"kind", "widths", "native_widths", "repeats", "context_p5", "topdown"}
    if unknown:
        raise ValueError(f"Unknown branch specification fields: {sorted(unknown)}")
    kind = spec.get("kind", "original")
    if kind not in BRANCH_KINDS:
        raise ValueError(f"branch kind must be original|csp|plain, got {kind!r}")
    default_widths = output_widths if kind == "original" else (64, 128, 256)
    widths = _positive_triplet(spec.get("widths", spec.get("native_widths", default_widths)),
                              "branch widths")
    if "widths" in spec and "native_widths" in spec:
        native = _positive_triplet(spec["native_widths"], "branch native_widths")
        if native != widths:
            raise ValueError("branch widths and native_widths disagree")
    repeats = _positive_triplet(spec.get("repeats", (1, 1, 1) if kind == "original" else (1, 2, 2)),
                               "branch repeats")
    flags = {}
    for name in ("context_p5", "topdown"):
        flags[name] = spec.get(name, False)
        if not isinstance(flags[name], bool):
            raise ValueError(f"branch {name} must be boolean")
    if kind == "original":
        if widths != output_widths:
            raise ValueError("original branch widths must match pyramid_widths")
        if repeats != (1, 1, 1) or any(flags.values()):
            raise ValueError("original branch requires repeats [1,1,1] and no context_p5/topdown")
    return {"kind": kind, "widths": list(widths), "repeats": list(repeats), **flags}


class LateSharedDecoder(nn.Module):
    """Decode local P16 -> P8, then add shared features at the final P4 fusion.

    No shared semantic or HR tensor enters fuse16 or fuse8. At stride 4 the
    four inputs are local decoder features, P4, shared semantics and HR.
    There is no output gate or zero-initialized residual path.
    """

    def __init__(self, pyramid_widths=(32, 64, 96), width=128,
                 semantic_channels=128, hr_channels=64, num_classes=NUM_CLASSES):
        super().__init__()
        self.pyramid_widths = _positive_triplet(pyramid_widths, "pyramid_widths")
        self.width = _positive_integer(width, "decoder_width")
        self.semantic_channels = _positive_integer(semantic_channels, "semantic_channels")
        self.hr_channels = _positive_integer(hr_channels, "hr_channels")
        self.num_classes = _positive_integer(num_classes, "num_classes")
        w4, w8, w16 = self.pyramid_widths
        self.fuse16 = FusionStage(w16, self.width)
        self.fuse8 = FusionStage(self.width + w8, self.width)
        self.fuse4 = FusionStage(self.width + w4 + self.semantic_channels + self.hr_channels,
                                 self.width)
        self.classifier = nn.Conv2d(self.width, self.num_classes, 1)

    def forward(self, semantic, hr, pyramid: Mapping[int, torch.Tensor], keep):
        p16, p8, p4 = pyramid[16], pyramid[8], pyramid[4]
        x16 = self.fuse16([p16], keep)
        x8 = self.fuse8([resize(x16, p8.shape[-2:]), p8], keep)
        x4 = self.fuse4([resize(x8, p4.shape[-2:]), p4,
                        resize(semantic, p4.shape[-2:]), resize(hr, p4.shape[-2:])], keep)
        return self.classifier(x4)


def _isolated_cpu_initialization(seed, constructor):
    # torch.manual_seed also seeds CUDA. Seed only the CPU default generator,
    # and restore that generator on exit, including constructor failures.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        return constructor()


class EnhancedH(DualHeadExperiment):
    """Original shared H graph with optional replacements of its local arms."""

    def __init__(self, checkpoint, head_a_source="rgb", head_b_source="depth",
                 pyramid_widths=(32, 64, 96), decoder_width=128,
                 branch_a=None, branch_b=None, shared_injection="standard", num_classes=40,
                 encoder_input_scale=1.0):
        if encoder_input_scale not in (0.5, 0.75, 1.0):
            raise ValueError("encoder_input_scale must be 0.5, 0.75 or 1.0")
        if head_a_source != "rgb":
            raise ValueError("head_a_source must be rgb")
        if head_b_source not in {"rgb", "depth"}:
            raise ValueError("head_b_source must be rgb|depth")
        widths = _positive_triplet(pyramid_widths, "pyramid_widths")
        decoder_width = _positive_integer(decoder_width, "decoder_width")
        if shared_injection not in SHARED_INJECTIONS:
            raise ValueError("shared_injection must be standard|late")
        specs = {
            "head_a": validate_branch_spec(branch_a, output_widths=widths),
            "head_b": validate_branch_spec(branch_b, output_widths=widths),
        }
        # Construct all original modules in the original order. The retained
        # modules therefore keep the same cold initialization as the H control.
        super().__init__(checkpoint, head_a_source, head_b_source, widths, decoder_width,
                         _positive_integer(num_classes, "num_classes"))
        self.shared_injection = shared_injection
        self.branch_specs = specs
        self.encoder_input_scale = float(encoder_input_scale)
        for head_name, spec in specs.items():
            if spec["kind"] != "original":
                from seadpnet.csp import StrongPyramid

                pyramid = _isolated_cpu_initialization(
                    PYRAMID_INIT_SEEDS[head_name],
                    lambda spec=spec: StrongPyramid(
                        widths=tuple(spec["widths"]), repeats=tuple(spec["repeats"]),
                        output_widths=widths, context_p5=spec["context_p5"],
                        topdown=spec["topdown"], block_type=spec["kind"]),
                )
                setattr(self, f"{head_name}_pyramid", pyramid)
            if shared_injection == "late":
                decoder = _isolated_cpu_initialization(
                    LATE_DECODER_INIT_SEEDS[head_name],
                    lambda: LateSharedDecoder(widths, decoder_width, num_classes=self.num_classes),
                )
                setattr(self, f"{head_name}_decoder", decoder)

    def _shared_features(self, x_rgbd, padding_mask):
        """Downscale only the RF-DETR encoder input, exactly as the NYUv2 S2/M2 arms did.

        The HR stem and both local pyramids keep running on the full-resolution
        input, so this knob changes the semantic pathway's cost without moving the
        dense supervision grid.
        """
        from torch.nn import functional as F
        from rfdetr.utilities.tensors import NestedTensor

        if self.encoder_input_scale != 1.0:
            size = tuple(max(24, round(dim * self.encoder_input_scale / 24) * 24)
                         for dim in x_rgbd.shape[-2:])
            rgb = F.interpolate(x_rgbd[:, :3], size=size, mode="bilinear", align_corners=False)
            rest = F.interpolate(x_rgbd[:, 3:], size=size, mode="nearest")
            x_rgbd = torch.cat([rgb, rest], 1)
            padding_mask = F.interpolate(padding_mask[:, None].float(), size=size,
                                         mode="nearest")[:, 0].bool()
        return self.backbone(NestedTensor(x_rgbd, padding_mask))


def build_model(cfg):
    """Build from a future-experiment configuration without starting training."""
    return EnhancedH(
        cfg["pretrained_checkpoint"], cfg.get("head_a_source", "rgb"),
        cfg.get("head_b_source", "depth"), cfg.get("pyramid_widths", (32, 64, 96)),
        cfg.get("decoder_width", 128), cfg.get("branch_a"), cfg.get("branch_b"),
        cfg.get("shared_injection", "standard"), cfg.get("num_classes", 40),
        cfg.get("encoder_input_scale", 1.0),
    )


def shared_state_digest(model):
    """Digest retained shared state using the original runtime byte contract."""
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        if name.startswith(("head_a_", "head_b_")):
            continue
        digest.update(f"{name}|{value.dtype}|{tuple(value.shape)}".encode())
        digest.update(value.detach().contiguous().cpu().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def architecture(model):
    count = lambda module: sum(parameter.numel() for parameter in module.parameters())
    normalization = {
        "channel_norm": sum(type(module).__name__ == "ChannelNorm" for module in model.modules()),
        "layer_norm": sum(isinstance(module, nn.LayerNorm) for module in model.modules()),
        "batch_norm": sum(isinstance(module, nn.modules.batchnorm._BatchNorm) for module in model.modules()),
        "dropout": sum(isinstance(module, nn.modules.dropout._DropoutNd) for module in model.modules()),
    }
    metadata = {
        "format": CHECKPOINT_FORMAT,
        "head_a_source": model.head_a_source, "head_b_source": model.head_b_source,
        "init": "rf_detr_pretrained", "trunk_depth": "rgbd", "hr_depth": "rgbd",
        "hr_enabled": True, "semantic_channels": 128, "hr_channels": 64,
        "num_classes": model.num_classes, "pyramid_strides": [4, 8, 16],
        "pyramid_widths": list(model.pyramid_widths), "decoder_width": model.decoder_width,
        "shared_injection": model.shared_injection,
        "decoder_type": "ProgressiveDecoder" if model.shared_injection == "standard" else "LateSharedDecoder",
        "decoder_initialization": "original_h_cold_weights" if model.shared_injection == "standard"
                                  else "new_independent_late_decoder_cold_weights",
        "shared_first_injection_stride": 16 if model.shared_injection == "standard" else 4,
        "branch_a": dict(model.branch_specs["head_a"]),
        "branch_b": dict(model.branch_specs["head_b"]),
        "encoder_input_scale": getattr(model, "encoder_input_scale", 1.0),
        "parameters_total": count(model),
        "parameters_backbone": count(model.backbone),
        "parameters_semantic": count(model.semantic),
        "parameters_hr_stem": count(model.hr_stem),
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



# --------------------------------------------------------------- SUN RGB-D
SUN_CHECKPOINT_FORMAT = "seadp-sunrgbd-v1"
NYU_SM_CHECKPOINT_FORMAT = "c11-hrd-v2-v1"
SUN_MODEL_NAME = "SEADP-Net-L"
SUN_EXPERIMENT_NAME = "SEADP-Net-SUNRGBD"

_PATH_OVERRIDES = {"rf_checkpoint": "pretrained_checkpoint",
                   "data_root": "data_root", "valid_root": "valid_root"}


def build_sun_model(cfg):
    """SUN transfer: identical graph, independent classifiers, cold RF start."""
    if cfg.get("warm_start_checkpoint") or cfg.get("c11_checkpoint") or cfg.get("nyu_checkpoint"):
        raise ValueError("SUN starts from original RF pretraining, never NYU semantic weights")
    model = build_model(cfg)
    # The size arms must identify themselves inside architecture metadata; the
    # module constants stay the L default for every pre-existing checkpoint.
    model.variant_identity = {"model_name": cfg["model_name"], "experiment_name": cfg["experiment_name"]}
    return model


def sun_architecture(model):
    identity = getattr(model, "variant_identity", {})
    result = architecture(model)
    result.update(format=SUN_CHECKPOINT_FORMAT,
                  model_name=identity.get("model_name", SUN_MODEL_NAME),
                  experiment_name=identity.get("experiment_name", SUN_EXPERIMENT_NAME),
                  method_name="SEADP-Net", dataset="SUNRGBD",
                  initialization="original_rf_detr_pretrained_cold_sun_heads", stride=12)
    return result


def sun_architecture_legacy_l(model):
    """The FIRST SUN-Net-L run's metadata schema, kept verbatim.

    That era's architecture() wrote code constants (experiment_name
    "SEADP-Net-SUNRGBD") and predated the encoder_input_scale field; S2/M2
    and every later checkpoint use the config-identity schema above.
    """
    result = architecture(model)
    result.pop("encoder_input_scale", None)
    result.update(format=SUN_CHECKPOINT_FORMAT,
                  model_name=SUN_MODEL_NAME,
                  experiment_name=SUN_EXPERIMENT_NAME,
                  method_name="SEADP-Net", dataset="SUNRGBD",
                  initialization="original_rf_detr_pretrained_cold_sun_heads", stride=12)
    return result


def load_experiment(path, device="cuda", overrides=None):
    """Strict format/architecture/state loading for both checkpoint lineages.

    ``overrides`` rewrites filesystem paths (RF checkpoint, data roots)
    recorded inside the checkpoint config; architecture metadata is rebuilt
    from the constructed model and compared exactly, and weights always load
    with strict=True.

    The only tolerated difference: ``encoder_input_scale`` may be MISSING from
    an older NYUv2 record (the field postdates those snapshots; the models
    are scale 1.0 by construction). A conflicting value is never accepted.
    """
    state = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(state.get("config"), Mapping):
        raise ValueError("Checkpoint config must be a mapping")
    cfg = dict(state["config"])
    for key, target in _PATH_OVERRIDES.items():
        if overrides and overrides.get(key):
            cfg[target] = overrides[key]
    fmt = state.get("format")
    if fmt == CHECKPOINT_FORMAT:
        model = build_model(cfg)
        expected = architecture(model)
    elif fmt == NYU_SM_CHECKPOINT_FORMAT:
        from seadpnet.scaled_encoder_model import architecture_scaled_encoder, build_scaled_encoder_model
        model = build_scaled_encoder_model(cfg)
        expected = architecture_scaled_encoder(model)
    elif fmt == SUN_CHECKPOINT_FORMAT:
        model = build_sun_model(cfg)
        expected = sun_architecture(model)
    else:
        raise ValueError(f"Unsupported checkpoint format: {fmt!r}")
    saved = state.get("architecture")
    candidates = [expected]
    if fmt == SUN_CHECKPOINT_FORMAT:
        candidates.append(sun_architecture_legacy_l(model))
    if saved not in candidates:
        differences = {key for key in set(saved) | set(expected)
                       if saved.get(key) != expected.get(key)}
        tolerated = differences == {"encoder_input_scale"} and "encoder_input_scale" not in saved
        if not (tolerated and model.encoder_input_scale == cfg.get("encoder_input_scale", 1.0)):
            raise ValueError(f"Saved and instantiated architectures differ: {sorted(differences)}")
    model.load_state_dict(state["model"], strict=True)
    return model.to(device), state
