"""Authorized cold CSP-arm training: one augmented loader batch per AdamW update.

Microbatches slice the original 32-image loader batch. Each CE contribution
uses that complete batch's valid-pixel denominator, including the 27-image
physical epoch tail. A diagnostic stop never changes the 10,000-update LR
horizon and its weights are never used to initialize a formal run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time
import traceback

import numpy as np
import torch
import yaml

from seadpnet.losses import dual_ce, first_batch_fingerprint, new_module_gradient_l2, split_batch
from seadpnet.schedule import lr_multiplier, run_updates

ROOT = Path(__file__).resolve().parents[1]
PRETRAINED_SHA256 = "6de3da31b2572cac214a1c76cce4a92a13966d56390ac2b3a3de9a8dc2b2bca3"
DATASET_SHA256 = "6078bea2d6753aead4708df6cb411b7df603a514cf53ecbc8df9118c476640cb"
FIRST_BATCH_SHA256 = "969c4f84f6e6283bf2bbba83133929875d71d05c8582cb598752365c2d92f678"
SHARED_DIGESTS = {
    "backbone": "6ba592d8971a910d7abc7fa9b34b0ee5ef149f44dd45ca62a1ee4661f1184aa4",
    "semantic": "fc5bcb9cf4d1d461fe40fd95d824c9a293b97b3eed10adcfe4b35559d8d1301a",
    "hr_stem": "574814fa8dca84bc73ccbbab224313d0c36ba9c5214323ae7ee6ab4958001ca3",
    "combined": "2ee3b697bb50012f585a468056d75e29a2d075ce3801b65f250921e4cf817551",
}
SNAPSHOT_UPDATES = [2000, 3000, 4000, 5000, 6000, 7000, 10000]
SCALES_MS5 = [0.5, 0.75, 1.0, 1.25, 1.5]
LOSS_DESCRIPTION = "0.5*CE_A+0.5*CE_B; global_valid_pixel_mean"
LOSS_WEIGHTS = {"ce_a": 0.5, "ce_b": 0.5}


def load_config(path):
    """Merge YAML bases without the legacy divisor-only microbatch check."""
    path = Path(path)
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise ValueError("Configuration must be a mapping")
    base = cfg.pop("base", None)
    if base:
        original = load_config(path.parent / base)
        original.update(cfg)
        cfg = original
    return cfg


def normalize_config(cfg, *, micro_batch=None, stop_after=None, skip_monitor=False, save_every=None):
    """Bind an authorized arm to the matched training and monitoring recipe."""
    cfg = dict(cfg)
    if cfg.get("debug_only") is not False or cfg.get("training_authorized") is not True:
        raise ValueError("Training requires debug_only:false and training_authorized:true")
    if any(key in cfg for key in ("c11_checkpoint", "warm_start_checkpoint", "pyramid_lr", "gate_lr")):
        raise ValueError("Cold CSP training cannot use warm starts or an additional module LR")
    expected = {
        "seed": 42, "train_updates": 10000, "warmup_updates": 500, "min_lr_ratio": 0.01,
        "head_lr": 0.000141, "backbone_lr": 0.0000141, "weight_decay": 0.0001,
        "gradient_clip": 0.1, "effective_batch": 32, "loader_batch": 32,
        "num_workers": 4, "monitor_split": "test_sub100", "monitor_scales": SCALES_MS5,
        "monitor_flip": True, "eval_protocol": "nyuv2_image_views", "eval_batch": 2,
        "val_interval": 500, "head_a_source": "rgb",
        "augmentation": {"hflip": 0.5, "color_jitter": [0.2, 0.2, 0.2, 0.05],
                         "scale_range": [0.75, 1.25]},
    }
    for key, value in expected.items():
        if cfg.get(key) != value:
            raise ValueError(f"Matched training configuration differs: {key}")
    if cfg.get("head_b_source") not in {"rgb", "depth"}:
        raise ValueError("Head B source must be rgb or depth")
    # Local widths, repeat depths and late/context injections belong to the
    # preregistered arm, including Y8's wider 64/128/256 and decoder width192.
    # branch_model validates these specifications while constructing the arm.
    if cfg.get("fusion", "fixed_equal_probability_mean") != "fixed_equal_probability_mean":
        raise ValueError("Fusion must remain the fixed equal probability mean")
    accepted_losses = {"0.5*CE_A+0.5*CE_B", "0.5*(CE_A+CE_B); global_valid_pixel_mean", LOSS_DESCRIPTION}
    if cfg.get("loss") not in accepted_losses:
        raise ValueError("The only training objective is equal-weight original dual CE")
    size = cfg.get("micro_batch") if micro_batch is None else micro_batch
    if isinstance(size, bool) or not isinstance(size, int) or not 1 <= size <= 32:
        raise ValueError("micro_batch must be an integer from 1 through 32")
    actual = run_updates(cfg["train_updates"], stop_after)
    diagnostic = actual != cfg["train_updates"]
    if skip_monitor and not diagnostic:
        raise ValueError("--skip-monitor is permitted only for a diagnostic early stop")
    interval = cfg.get("save_interval", 100) if save_every is None else save_every
    if isinstance(interval, bool) or not isinstance(interval, int) or interval < 1:
        raise ValueError("save_interval must be a positive integer")
    snapshots = cfg.get("snapshot_updates", SNAPSHOT_UPDATES)
    if snapshots != SNAPSHOT_UPDATES:
        raise ValueError("Snapshot updates must remain prespecified")
    if cfg.get("log_interval", 25) < 1:
        raise ValueError("log_interval must be positive")
    cfg.update(micro_batch=size, grad_accumulation=math.ceil(32 / size),
               max_grad_accumulation=math.ceil(32 / size), actual_updates=actual,
               diagnostic_stop=diagnostic, skip_monitor=bool(skip_monitor),
               save_interval=interval, snapshot_updates=list(snapshots),
               log_interval=cfg.get("log_interval", 25), initialization="original_rf_detr_pretrained",
               stage="direct", loss=LOSS_DESCRIPTION, loss_weights=dict(LOSS_WEIGHTS),
               loss_formula="mean_valid[0.5*CE_A+0.5*CE_B]", fusion="fixed_equal_probability_mean",
               ce_memory_strategy="nonreentrant_CE_recompute", train_limit=None, eval_limit=None,
               persistent_workers=False, drop_last=False,
               loader_batch_semantics="one full augmented loader batch split; no epoch-tail merging",
               scheduled_updates=cfg["train_updates"],
               resume_support="exact exhausted epoch-boundary optimizer/scheduler/RNG/loader continuation")
    return cfg


def validate_formal_config(cfg):
    if cfg.get("diagnostic_stop") or cfg.get("skip_monitor") or cfg.get("actual_updates", 10000) != 10000:
        raise ValueError("Formal artifacts require the complete monitored 10,000-update run")
    normalize_config(cfg)
    if cfg.get("average_updates") != [2000, 3000, 4000] or cfg.get("secondary_average_updates") != [3000, 4000, 5000]:
        raise ValueError("Averaging windows must remain prespecified")
    return True


def validate_architecture(arch):
    from seadpnet.model import CHECKPOINT_FORMAT
    if arch.get("format") != CHECKPOINT_FORMAT or arch.get("num_classes") != 40:
        raise ValueError("Unexpected CSP checkpoint format or semantic class count")
    if arch.get("normalization", {}).get("batch_norm") != 0:
        raise ValueError("CSP arms require LayerNorm/ChannelNorm without BatchNorm")
    if arch.get("local_normalization") != "ChannelNorm(LayerNorm)/GELU; no BatchNorm or dropout":
        raise ValueError("Unexpected local CSP normalization")
    if not isinstance(arch.get("parameters_total"), int) or arch["parameters_total"] < 1:
        raise ValueError("CSP architecture lacks a valid parameter count")
    return True


def source_code_hashes():
    from seadpnet.runtime import sha256
    paths = [Path(__file__)] + sorted((ROOT / "seadpnet").rglob("*.py"))
    return {path.relative_to(ROOT).as_posix(): sha256(path) for path in paths}


def data_provenance(cfg):
    """Verify the matched byte manifest and current split identities."""
    from seadpnet.runtime import sha256
    path = Path(cfg.get("data_fingerprint_path", ROOT / "checks" / "data_fingerprint.json"))
    manifest = json.loads(path.read_text(encoding="utf-8"))
    expected = {"train": 795, "test": 654, "test_sub100": 100}
    files = manifest.get("file_sha256")
    if manifest.get("split_sizes") != expected or not isinstance(files, dict) or len(files) != manifest.get("files"):
        raise ValueError("Dataset fingerprint lacks the matched complete NYUv2 manifest")
    digest = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    if manifest.get("dataset_sha256") != digest or digest != DATASET_SHA256:
        raise ValueError("Dataset fingerprint differs from the original matched data")
    splits = {}
    for name, count in expected.items():
        split_path = Path(cfg["data_root"]) / "splits" / f"{name}.txt"
        ids = [int(value) for value in split_path.read_text().split()]
        current = sha256(split_path)
        if len(ids) != count or len(set(ids)) != count or ids != manifest["split_ids"][name]:
            raise ValueError(f"Current {name} image IDs differ from preflight")
        if current != files[f"splits/{name}.txt"]:
            raise ValueError(f"Current {name} split bytes differ from preflight")
        splits[name] = current
    return {"manifest": str(path), "manifest_sha256": sha256(path), "dataset_sha256": digest,
            "files": len(files), "split_sizes": expected, "split_sha256": splits,
            "data_root": cfg["data_root"], "valid_root": cfg["valid_root"]}


def loss_identity_audit(value, ce_a, ce_b):
    if not all(math.isfinite(number) for number in (value, ce_a, ce_b)):
        raise RuntimeError("Nonfinite dual-head loss diagnostics")
    reconstructed = .5 * (ce_a + ce_b)
    error = abs(value - reconstructed)
    tolerance = max(1e-6, 2e-6 * abs(reconstructed))
    if error > tolerance:
        raise RuntimeError("Loss differs from the original 0.5*CE_A+0.5*CE_B")
    return {"formula": "loss=0.5*ce_a+0.5*ce_b", "absolute_error": error,
            "tolerance": tolerance, "passed": True}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    torch.cuda.set_rng_state_all(state["torch_cuda"])


def optimizer_update_audit(model, optimizer, update):
    """Check real AdamW moments/counters, allowing genuinely unused backbone tensors."""
    active, unused, mismatched = [], [], []
    active_by_module = {}
    for name, parameter in model.named_parameters():
        state = optimizer.state.get(parameter, {})
        if not state:
            unused.append(name)
            if not name.startswith("backbone."):
                mismatched.append({"name": name, "reason": "non-backbone parameter has no AdamW state"})
            continue
        step = int(state["step"].item()) if isinstance(state.get("step"), torch.Tensor) else state.get("step")
        if step != update or any(key not in state for key in ("exp_avg", "exp_avg_sq")):
            mismatched.append({"name": name, "adamw_step": step, "expected": update})
        if not all(bool(torch.isfinite(state[key]).all()) for key in ("exp_avg", "exp_avg_sq") if key in state):
            mismatched.append({"name": name, "reason": "nonfinite AdamW moments"})
        active.append(name)
        module = name.split(".", 1)[0]
        active_by_module[module] = active_by_module.get(module, 0) + 1
    if active_by_module.get("backbone", 0) == 0:
        mismatched.append({"reason": "backbone has no active AdamW state"})
    if mismatched or not active:
        raise RuntimeError(f"AdamW optimizer-update audit failed: {mismatched[:10]}")
    return {"optimizer": "AdamW", "expected_step": update, "active_parameter_tensors": len(active),
            "unused_backbone_parameter_tensors": len(unused), "unused_backbone_parameters": unused,
            "active_parameter_tensors_by_module": active_by_module,
            "all_active_steps_match": True, "moments_finite": True, "passed": True}


def module_gradient_diagnostics(model):
    """Witness finite, nonzero gradients through every participating model part."""
    groups = {name: [] for name in ("backbone", "semantic", "hr_stem", "head_a_pyramid",
                                   "head_a_decoder", "head_b_pyramid", "head_b_decoder")}
    for name, parameter in model.named_parameters():
        group = name.split(".", 1)[0]
        if group in groups:
            groups[group].append((name, parameter))
    result = {}
    for group, parameters in groups.items():
        used = [(name, parameter.grad) for name, parameter in parameters if parameter.grad is not None]
        unused = [name for name, parameter in parameters if parameter.grad is None]
        if not used or (group != "backbone" and unused):
            raise RuntimeError(f"Participating model module is disconnected: {group}; unused={unused[:5]}")
        terms = [gradient.detach().float().square().sum() for _, gradient in used]
        norm = float(torch.stack(terms).sum().sqrt())
        if not math.isfinite(norm) or norm <= 0:
            raise RuntimeError(f"Nonfinite or zero full-model gradient: {group}")
        result[group] = {"l2": norm, "finite": True, "with_gradient": len(used),
                         "without_gradient": len(unused), "unused_parameters": unused}
    return result


def validate_resume(state, cfg, arch, code_hashes):
    from seadpnet.model import CHECKPOINT_FORMAT
    if state.get("format") != CHECKPOINT_FORMAT or state.get("architecture") != arch:
        raise ValueError("Resume checkpoint architecture or format differs")
    if state.get("config") != cfg:
        raise ValueError("Resume configuration differs, including diagnostic/formal classification")
    if state.get("provenance", {}).get("code_sha256") != code_hashes:
        raise ValueError("Resume training source differs from the checkpoint source")
    loader_state = state.get("loader_state", {})
    if not state.get("exact_loader_continuation_supported") or not loader_state.get("epoch_boundary"):
        raise ValueError("Exact resume requires a fully exhausted epoch-boundary checkpoint")
    if not loader_state.get("sampler_exhausted") or loader_state.get("batches_in_current_epoch") != 0:
        raise ValueError("Resume checkpoint retains an active loader epoch")
    if any(key not in state for key in ("optimizer", "scheduler", "rng_state", "loader_generator_state")):
        raise ValueError("Resume requires the full last.pth training state")
    update = state.get("update")
    if not isinstance(update, int) or not 0 <= update <= cfg["actual_updates"]:
        raise ValueError("Resume update is outside the unchanged training horizon")


def run_training(args, cfg, root):
    from seadpnet.model import architecture, build_model, CHECKPOINT_FORMAT, shared_state_digest
    from seadpnet.runtime import seed_all, write_json, save_checkpoint, sha256, state_digest, optimizer_for
    from seadpnet.data.nyuv2 import loader, forward_batch
    from seadpnet.evaluation.common import amp_context
    from seadpnet.evaluation.nyuv2 import evaluate

    torch.set_num_threads(4)
    if not torch.cuda.is_available():
        raise RuntimeError("Training requires a CUDA device")
    seed_all(cfg["seed"])
    model = build_model(cfg)
    arch = architecture(model)
    validate_architecture(arch)
    if arch["parameters_total"] > cfg["max_parameters"]:
        raise ValueError("Configured CSP model exceeds its parameter budget")
    if not all(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("Full training must not freeze any parameters")
    model.checkpoint_heads = True
    pretrained_sha = sha256(cfg["pretrained_checkpoint"])
    if pretrained_sha != PRETRAINED_SHA256:
        raise RuntimeError("RF-DETR original pretrained checkpoint differs")
    shared = {name: state_digest(model, name + ".") for name in ("backbone", "semantic", "hr_stem")}
    shared["combined"] = shared_state_digest(model)
    if shared != SHARED_DIGESTS:
        raise RuntimeError(f"Shared cold state differs from original H: {shared}")
    initial_digest = state_digest(model)
    hashes = source_code_hashes()
    provenance = {
        "initialization": cfg["initialization"], "pretrained_sha256": pretrained_sha,
        "initial_state_digest": initial_digest, "initial_full_state_digest": initial_digest,
        "initial_backbone_digest": shared["backbone"], "shared_state_digest": shared,
        "shared_state_identical_to_plain_h": True, "architecture": arch,
        "all_parameters_trainable": True, "code_sha256": hashes, "data": data_provenance(cfg),
        "loss": LOSS_DESCRIPTION, "loss_weights": dict(LOSS_WEIGHTS), "ignore_index": 255,
        "fusion_weight_trainable": False, "head_weights": [.5, .5],
        "matched_first_augmented_batch_sha256": FIRST_BATCH_SHA256,
        "head_memory_strategy": "nonreentrant_deterministic_pyramid_decoder_recompute",
        "ce_memory_strategy": "nonreentrant_CE_recompute", "ce_recompute_consumes_random_numbers": False,
        "torch": str(torch.__version__), "gpu": torch.cuda.get_device_name(),
        "selection_split": cfg["monitor_split"], "selection_scales": cfg["monitor_scales"],
        "selection_flip": cfg["monitor_flip"],
        "selection_rule": "highest fused monitor mIoU; earliest update wins ties",
        "cuda_allocator": os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "default"),
        "resume_support": cfg["resume_support"], "diagnostic_stop": cfg["diagnostic_stop"],
        "fresh_cold_initialization": True, "trial_weights_used_for_formal_initialization": False,
    }
    model.cuda()
    opt = optimizer_for(model, cfg)
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lambda step: lr_multiplier(
        step, cfg["train_updates"], cfg["warmup_updates"], cfg["min_lr_ratio"]))
    dl = loader(cfg, "train", augment=True, limit=None, batch_size=cfg["loader_batch"])
    if dl.persistent_workers or dl.drop_last or dl.generator is None or len(dl.dataset) != 795 or len(dl) != 25:
        raise RuntimeError("Training requires the original 795-image, 25-batch unmerged-tail loader")
    iterator = None
    epoch_batches = len(dl)
    completed_epochs = batches_in_epoch = 0
    start_update = 0
    best = -1.0
    first_audit = None
    images_processed = 0
    elapsed_prior = 0.0
    final_optimizer_audit = None

    if args.resume:
        state = torch.load(args.resume, map_location="cpu", weights_only=False)
        validate_resume(state, cfg, arch, hashes)
        model.load_state_dict(state["model"], strict=True)
        opt.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_update = state["update"]
        if scheduler.last_epoch != start_update:
            raise ValueError("Resume scheduler counter differs from optimizer updates")
        loader_state = state["loader_state"]
        if (loader_state["epoch_batches"] != epoch_batches or loader_state["dataset_images"] != len(dl.dataset)
                or loader_state["loader_batch"] != dl.batch_size or loader_state["num_workers"] != dl.num_workers):
            raise ValueError("Resume loader geometry differs")
        completed_epochs = loader_state["completed_epochs"]
        dl.generator.set_state(state["loader_generator_state"])
        first_audit = state["first_batch_fingerprint"]
        if first_audit["first_update_batch_sha256"] != FIRST_BATCH_SHA256:
            raise ValueError("Resume first augmented batch differs")
        best = state["best_miou"]
        images_processed = state["images_processed"]
        elapsed_prior = state["elapsed_seconds"]
        provenance = state["provenance"]
        restore_rng(state["rng_state"])
        # Keep uncommitted diagnostics as evidence; restart from the last full
        # atomic checkpoint with an unambiguous history prefix.
        history_path = root / "history.jsonl"
        if history_path.exists():
            lines = history_path.read_text(encoding="utf-8").splitlines()
            kept = [line for line in lines if json.loads(line)["update"] <= start_update]
            discarded = [line for line in lines if json.loads(line)["update"] > start_update]
            if discarded:
                stamp = time.time_ns()
                (root / f"uncommitted_history_{stamp}.jsonl").write_text("\n".join(discarded) + "\n", encoding="utf-8")
                history_path.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")
    else:
        write_json(root / "config.json", cfg)
        write_json(root / "provenance.json", provenance)

    def next_batch():
        nonlocal iterator, completed_epochs, batches_in_epoch
        if iterator is None:
            iterator = iter(dl)
        batch = next(iterator)
        batches_in_epoch += 1
        expected_images = 27 if batches_in_epoch == epoch_batches else 32
        if int(batch["semantic"].shape[0]) != expected_images:
            raise RuntimeError("Loader changed the nominal batch or physical 27-image epoch tail")
        if batches_in_epoch == epoch_batches:
            try:
                next(iterator)
            except StopIteration:
                pass
            else:
                raise RuntimeError("Loader length differs from its emitted batches")
            iterator = None
            completed_epochs += 1
            batches_in_epoch = 0
        return batch

    def monitor():
        metrics = evaluate(model, cfg, split=cfg["monitor_split"], scales=cfg["monitor_scales"],
                           flip=cfg["monitor_flip"], limit=None, batch_size=cfg["eval_batch"])
        if metrics["images"] != 100:
            raise RuntimeError("Monitor must evaluate all 100 prespecified images")
        return metrics

    start = time.perf_counter()

    def elapsed():
        return elapsed_prior + time.perf_counter() - start

    def pack(update, training_state=False):
        payload = {"format": CHECKPOINT_FORMAT, "model": model.state_dict(), "config": cfg,
                   "architecture": arch, "update": update, "optimizer_steps": update,
                   "best_miou": best, "provenance": provenance}
        if training_state:
            payload.update(optimizer=opt.state_dict(), scheduler=scheduler.state_dict(),
                           rng_state={"python": random.getstate(), "numpy": np.random.get_state(),
                                      "torch_cpu": torch.get_rng_state(), "torch_cuda": torch.cuda.get_rng_state_all()},
                           loader_generator_state=dl.generator.get_state(),
                           exact_loader_continuation_supported=batches_in_epoch == 0,
                           loader_state={"epoch_batches": epoch_batches, "completed_epochs": completed_epochs,
                                         "batches_in_current_epoch": batches_in_epoch, "epoch_boundary": batches_in_epoch == 0,
                                         "sampler_exhausted": iterator is None, "num_workers": dl.num_workers,
                                         "persistent_workers": False, "drop_last": False,
                                         "dataset_images": len(dl.dataset), "loader_batch": dl.batch_size},
                           first_batch_fingerprint=first_audit, images_processed=images_processed,
                           elapsed_seconds=elapsed())
        return payload

    def select(update, metrics):
        nonlocal best
        if metrics["miou"] > best:
            best = metrics["miou"]
            save_checkpoint(root / "best.pth", pack(update))
            write_json(root / "selection.json", {"update": update, "miou_percent": 100 * best,
                       "split": cfg["monitor_split"], "images": 100, "scales": cfg["monitor_scales"],
                       "flip": cfg["monitor_flip"], "head": "fused", "rule": provenance["selection_rule"]})

    def status(update, state="running"):
        seconds = elapsed()
        return {"status": state, "updates": update, "optimizer_steps": update, "actual_updates": cfg["actual_updates"],
                "scheduled_updates": cfg["train_updates"], "diagnostic_stop": cfg["diagnostic_stop"],
                "skip_monitor": cfg["skip_monitor"], "architecture": arch, "initialization": cfg["initialization"],
                "loss": LOSS_DESCRIPTION, "loss_weights": dict(LOSS_WEIGHTS), "oom_count": 0,
                "micro_batch": cfg["micro_batch"], "max_grad_accumulation": cfg["max_grad_accumulation"],
                "images_processed": images_processed, "elapsed_seconds": seconds,
                "images_per_second": images_processed / seconds if seconds > 0 else 0.,
                "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
                "peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,
                "best_miou_percent": None if best < 0 else 100 * best,
                "loader_completed_epochs": completed_epochs, "loader_epoch_boundary": batches_in_epoch == 0,
                "optimizer_update_audit": final_optimizer_audit}

    torch.cuda.reset_peak_memory_stats()
    write_json(root / "status.json", status(start_update))
    if not cfg["skip_monitor"] and start_update == 0:
        initial = monitor()
        write_json(root / "initial.json", initial)
        select(0, initial)
    for update in range(start_update + 1, cfg["actual_updates"] + 1):
        step_start = time.perf_counter()
        full_batch = next_batch()
        if update == 1:
            first_audit = first_batch_fingerprint(full_batch)
            write_json(root / "first_batch.json", first_audit)
            if first_audit["first_update_batch_sha256"] != FIRST_BATCH_SHA256:
                raise RuntimeError("First augmented optimizer batch differs from the original matched control")
        pixels = int((full_batch["semantic"] != 255).sum().item())
        if pixels < 1:
            raise ValueError("Optimizer batch contains no supervised semantic pixels")
        batches = split_batch(full_batch, cfg["micro_batch"])
        sizes = [int(batch["semantic"].shape[0]) for batch in batches]
        counts = [int((batch["semantic"] != 255).sum().item()) for batch in batches]
        if sum(counts) != pixels:
            raise RuntimeError("Microbatch split lost supervised pixels")
        model.train()
        opt.zero_grad(set_to_none=True)
        lr_used = [group["lr"] for group in opt.param_groups]
        value = ce_a = ce_b = 0.0
        for batch in batches:
            with amp_context():
                output = forward_batch(model, batch, "cuda")
            target = batch["semantic"].cuda()
            loss, parts = dual_ce(output, target, total_valid_pixels=pixels)
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("Nonfinite original dual CE")
            loss.backward()
            value += float(loss.detach())
            ce_a += parts["ce_a"]
            ce_b += parts["ce_b"]
            del output, target, loss, parts
        identity = loss_identity_audit(value, ce_a, ce_b)
        gradients = new_module_gradient_l2(model)
        module_gradients = module_gradient_diagnostics(model) if update == 1 or update % cfg["log_interval"] == 0 or update == cfg["actual_updates"] else None
        if update == 1 and any(number <= 0 for number in gradients.values()):
            raise RuntimeError("A newly initialized head has no first-update gradient")
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["gradient_clip"], error_if_nonfinite=True)
        opt.step()
        scheduler.step()
        if update == cfg["actual_updates"]:
            final_optimizer_audit = optimizer_update_audit(model, opt, update)
        torch.cuda.synchronize()
        images = int(full_batch["semantic"].shape[0])
        images_processed += images
        seconds = time.perf_counter() - step_start
        event = {"update": update, "optimizer_steps": update, "loss": value, "ce_a": ce_a, "ce_b": ce_b,
                 "loss_weights": dict(LOSS_WEIGHTS), "loss_identity_audit": identity,
                 "gradient_norm": float(grad), "new_module_gradient_l2_preclip": gradients,
                 "module_gradient_diagnostics_preclip": module_gradients,
                 "all_observed_gradients_finite": True, "elapsed_seconds": elapsed(), "update_seconds": seconds,
                 "images_per_second": images / seconds, "effective_images": images, "supervised_pixels": pixels,
                 "microbatch_sizes": sizes, "microbatch_valid_pixels": counts, "actual_grad_accumulation": len(sizes),
                 "loader_completed_epochs": completed_epochs, "loader_epoch_boundary": batches_in_epoch == 0,
                 "lr_used_update": lr_used, "lr_next_update": [group["lr"] for group in opt.param_groups],
                 "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
                 "peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3, **first_audit}
        del full_batch, batches, batch
        if not cfg["skip_monitor"] and (update % cfg["val_interval"] == 0 or update == cfg["actual_updates"]):
            metrics = monitor()
            event.update(monitor_miou_percent=metrics["miou_percent"], monitor_images=metrics["images"],
                         monitor_heads=metrics["metrics_heads"])
            write_json(root / f"monitor_u{update}.json", metrics)
            select(update, metrics)
        with (root / "history.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, allow_nan=False) + "\n")
        if update % cfg["save_interval"] == 0 or update == cfg["actual_updates"]:
            save_checkpoint(root / "last.pth", pack(update, training_state=True))
        # Diagnostic trials retain only full last.pth. Formal immutable weight
        # snapshots are exactly the seven prespecified averaging/end points.
        if not cfg["diagnostic_stop"] and update in cfg["snapshot_updates"]:
            path = root / f"u{update}.pth"
            if path.exists():
                raise FileExistsError(f"Immutable snapshot exists: {path}")
            save_checkpoint(path, pack(update))
        if update == 1 or update % cfg["log_interval"] == 0 or update == cfg["actual_updates"]:
            write_json(root / "status.json", status(update))
            print(json.dumps(event, allow_nan=False), flush=True)
    if final_optimizer_audit is None:
        final_optimizer_audit = optimizer_update_audit(model, opt, cfg["actual_updates"])
    write_json(root / "status.json", status(cfg["actual_updates"], "complete"))


def main():
    from seadpnet.runtime import write_json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--stop-after", type=int)
    parser.add_argument("--micro-batch", type=int)
    parser.add_argument("--save-every", type=int)
    parser.add_argument("--skip-monitor", action="store_true")
    parser.add_argument("--resume", help="Full last.pth from the same config at an exhausted epoch boundary")
    args = parser.parse_args()
    cfg = normalize_config(load_config(args.config), micro_batch=args.micro_batch, stop_after=args.stop_after,
                           skip_monitor=args.skip_monitor, save_every=args.save_every)
    root = Path(args.run_dir)
    if args.resume:
        if not root.is_dir() or not (root / "config.json").is_file():
            raise ValueError("Resume must use the existing original run directory")
        if json.loads((root / "config.json").read_text(encoding="utf-8")) != cfg:
            raise ValueError("Existing run configuration differs")
        if Path(args.resume).resolve().parent != root.resolve():
            raise ValueError("Resume checkpoint must belong to the selected run directory")
    elif root.exists() and any(root.iterdir()):
        raise FileExistsError(f"Fresh runs never overwrite existing evidence: {root}")
    root.mkdir(parents=True, exist_ok=True)
    try:
        run_training(args, cfg, root)
    except BaseException as exc:
        previous = json.loads((root / "status.json").read_text(encoding="utf-8")) if (root / "status.json").exists() else {}
        previous.update(status="failed", error=traceback.format_exc(), diagnostic_stop=cfg["diagnostic_stop"],
                        actual_updates=cfg["actual_updates"], scheduled_updates=cfg["train_updates"],
                        oom_count=int(isinstance(exc, torch.cuda.OutOfMemoryError)), loss=LOSS_DESCRIPTION)
        write_json(root / "status.json", previous)
        raise


if __name__ == "__main__":
    main()
