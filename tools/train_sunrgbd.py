"""Cold SUN transfer; user-declared test-subset monitoring and exact loader resume."""
from __future__ import annotations
import argparse
import json
import math
import os
from pathlib import Path
import time
import traceback
import torch
from seadpnet.evaluation.common import amp_context
from seadpnet.losses import dual_ce, first_batch_fingerprint, split_batch
from seadpnet.schedule import lr_multiplier
from seadpnet.data.sunrgbd import loader
from seadpnet.evaluation.sunrgbd import evaluate
from seadpnet.model import build_sun_model as build_model, sun_architecture as architecture, SUN_CHECKPOINT_FORMAT as CHECKPOINT_FORMAT
from seadpnet.sun_runtime import (load_config, seed_all, rng_state, restore_rng, source_code_hashes,
                                 data_provenance, sha256, write_json, save_checkpoint, optimizer_for, state_digest)

MODULES = ("backbone", "semantic", "hr_stem", "head_a_pyramid", "head_a_decoder", "head_b_pyramid", "head_b_decoder")


def gradient_audit(model):
    result = {}
    for prefix in MODULES:
        terms = [parameter.grad.detach().float().square().sum() for name, parameter in model.named_parameters()
                 if name.startswith(prefix+".") and parameter.grad is not None]
        value = float(torch.stack(terms).sum().sqrt()) if terms else 0.
        if not math.isfinite(value) or value <= 0:
            raise RuntimeError(f"Missing or nonfinite participating-module gradient: {prefix}")
        result[prefix] = value
    return result


def optimizer_audit(model, optimizer, update):
    active = {}; unused = []
    for name, parameter in model.named_parameters():
        state = optimizer.state.get(parameter)
        if not state:
            unused.append(name)
            if not name.startswith("backbone."):
                raise RuntimeError(f"No optimizer state for participating head: {name}")
            continue
        if int(state["step"].item()) != update:
            raise RuntimeError(f"Optimizer step mismatch: {name}")
        if not all(bool(torch.isfinite(state[key]).all()) for key in ("exp_avg", "exp_avg_sq")):
            raise RuntimeError(f"Nonfinite AdamW moments: {name}")
        prefix = name.split(".", 1)[0]; active[prefix] = active.get(prefix, 0)+1
    if not all(active.get(prefix, 0) for prefix in MODULES):
        raise RuntimeError("An active module has no optimizer state")
    return {"passed": True, "optimizer_steps": update, "active_tensors_by_module": active,
            "unused_backbone_tensors": unused}


def smoke_config(cfg):
    cfg = dict(cfg)
    cfg.update(run_type="sunrgbd_smoke", smoke=True, max_updates=8, monitor_limit=2)
    return cfg


def validate_resume(state, cfg, arch, hashes, provenance):
    if state.get("format") != CHECKPOINT_FORMAT or state.get("config") != cfg or state.get("architecture") != arch:
        raise ValueError("Resume format/config/architecture differs")
    if state.get("provenance", {}).get("source_code_sha256") != hashes:
        raise ValueError("Runtime source changed; exact continuation requires frozen code")
    if state["provenance"]["data"] != provenance:
        raise ValueError("Resume dataset fingerprint differs")
    required = ("optimizer", "scheduler", "rng_state", "loader_state", "loader_generator_state")
    if not all(key in state for key in required) or not state.get("exact_loader_continuation_supported"):
        raise ValueError("Resume requires complete training last.pth")


def run_training(cfg, root, resume=None, device="cuda", model_factory=build_model):
    seed_all(cfg["seed"])
    pretrained_hash = sha256(cfg["pretrained_checkpoint"])
    if cfg.get("pretrained_sha256") and pretrained_hash != cfg["pretrained_sha256"]:
        raise ValueError("Original RF checkpoint hash mismatch")
    verified_manifest = os.environ.get("SUN_VERIFIED_FINGERPRINT_SHA256")
    trusted = verified_manifest == sha256(cfg["data_fingerprint_path"])
    data = data_provenance(cfg, verify_files=not bool(resume) and not trusted)
    hashes = source_code_hashes()
    model = model_factory(cfg)
    arch = architecture(model)
    if arch["parameters_total"] > cfg["max_parameters"]:
        raise RuntimeError("Model exceeds declared parameter budget")
    if cfg.get("expected_parameters") and arch["parameters_total"] != cfg["expected_parameters"]:
        raise RuntimeError("SUN model differs from the expected parameter count")
    provenance = {"pretrained_checkpoint": cfg["pretrained_checkpoint"], "pretrained_sha256": pretrained_hash,
                  "data": data, "source_code_sha256": hashes, "cold_model_state_sha256": state_digest(model),
                  "torch": str(torch.__version__), "device": str(device), "method_name": cfg["method_name"],
                  "model_name": cfg["model_name"], "experiment_name": cfg["experiment_name"], "dataset": cfg["dataset"],
                  "selection_split": cfg["monitor_split"], "selection_rule": cfg["selection_rule"],
                  "trial_weights_used_for_formal_initialization": False,
                  "data_verification": "supervisor-verified byte manifest" if trusted else "direct byte verification",
                  "average_updates": cfg["average_updates"], "test_used_for_selection": cfg["test_used_for_selection"],
                  "monitor_origin": cfg["monitor_origin"], "monitor_description": cfg["monitor_description"],
                  "optimizer_group_rule": "backbone. prefix AND encoder substring AND no depth -> backbone_lr; else head_lr"}
    model.to(device)
    opt = optimizer_for(model, cfg)
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lambda step: lr_multiplier(
        step, cfg["train_updates"], cfg["warmup_updates"], cfg["min_lr_ratio"]))
    update = images_processed = 0; best = -1.; elapsed_prior = 0.; first_audit = None
    position = {"epoch": 0, "batches_in_current_epoch": 0, "completed_epochs": 0}
    if resume:
        state = torch.load(resume, map_location="cpu", weights_only=False)
        validate_resume(state, cfg, arch, hashes, data)
        model.load_state_dict(state["model"], strict=True); opt.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        update, best, position = state["update"], state["best_miou"], state["loader_state"]
        if scheduler.last_epoch != update:
            raise ValueError("Resume scheduler step mismatch")
        if position["epoch"]*cfg["epoch_updates"]+position["batches_in_current_epoch"] != update:
            raise ValueError("Resume loader position mismatch")
        images_processed, elapsed_prior = state["images_processed"], state["elapsed_seconds"]
        first_audit, provenance = state["first_batch_fingerprint"], state["provenance"]
        restore_rng(state["rng_state"])
        history = root/"history.jsonl"
        if history.exists():
            lines = history.read_text(encoding="utf-8").splitlines()
            committed = [line for line in lines if json.loads(line)["update"] <= update]
            uncommitted = [line for line in lines if json.loads(line)["update"] > update]
            if uncommitted:
                (root/f"uncommitted_history_{time.time_ns()}.jsonl").write_text("\n".join(uncommitted)+"\n", encoding="utf-8")
                history.write_text("\n".join(committed)+("\n" if committed else ""), encoding="utf-8")
    else:
        write_json(root/"config.json", cfg); write_json(root/"provenance.json", provenance)
        write_json(root/"architecture.json", arch)
    start = time.perf_counter(); dl = None

    def elapsed():
        return elapsed_prior+time.perf_counter()-start

    def pack(training_state=False):
        payload = {"format": CHECKPOINT_FORMAT, "config": cfg, "architecture": arch,
                   "model": model.state_dict(), "update": update, "optimizer_steps": update,
                   "epoch": update/cfg["epoch_updates"], "best_miou": best, "provenance": provenance,
                   "method_name": cfg["method_name"], "model_name": cfg["model_name"],
                   "experiment_name": cfg["experiment_name"], "dataset": cfg["dataset"]}
        if training_state:
            payload.update(optimizer=opt.state_dict(), scheduler=scheduler.state_dict(), rng_state=rng_state(),
                           loader_state=dict(position), loader_generator_state=dl.generator.get_state() if dl else None,
                           exact_loader_continuation_supported=True, first_batch_fingerprint=first_audit,
                           images_processed=images_processed, elapsed_seconds=elapsed())
        return payload

    def status(state="running"):
        value = {"status": state, "update": update, "updates": update, "optimizer_steps": update,
                 "scheduled_updates": cfg["train_updates"], "actual_updates": cfg["max_updates"],
                 "epoch": update/cfg["epoch_updates"], "epochs": cfg["epochs"], "loader_state": dict(position),
                 "images_processed": images_processed, "elapsed_seconds": elapsed(),
                 "images_per_second": images_processed/max(elapsed(), 1e-9), "best_monitor_miou_percent": 100*best if best >= 0 else None,
                 "run_type": cfg["run_type"], "method_name": cfg["method_name"], "model_name": cfg["model_name"],
                 "experiment_name": cfg["experiment_name"], "dataset": cfg["dataset"], "micro_batch": cfg["micro_batch"],
                 "effective_batch": cfg["effective_batch"], "architecture": arch, "oom_count": 0,
                 "monitor_split": cfg["monitor_split"], "monitor_origin": cfg["monitor_origin"],
                 "test_used_for_selection": cfg["test_used_for_selection"]}
        if str(device).startswith("cuda"):
            value.update(peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,
                         peak_reserved_gib=torch.cuda.max_memory_reserved()/1024**3)
        write_json(root/"status.json", value)

    status()
    start_epoch, consumed = position["epoch"], position["batches_in_current_epoch"]
    for epoch in range(start_epoch, cfg["epochs"]):
        if update >= cfg["max_updates"]:
            break
        dl = loader(cfg, cfg["train_split"], augment=True, batch_size=cfg["loader_batch"], epoch=epoch)
        if len(dl) != cfg["epoch_updates"] or len(dl.dataset) != cfg["split_sizes"][cfg["train_split"]]:
            raise RuntimeError("Actual training loader differs from declared split")
        for batch_index, full_batch in enumerate(dl):
            if epoch == start_epoch and batch_index < consumed:
                continue
            if update >= cfg["max_updates"]:
                break
            model.train(); opt.zero_grad(set_to_none=True)
            batch_images = int(full_batch["semantic"].shape[0])
            expected = cfg["epoch_tail"] if batch_index+1 == len(dl) else cfg["loader_batch"]
            if batch_images != expected:
                raise RuntimeError("Physical epoch tail or nominal batch changed")
            if update == 0:
                first_audit = first_batch_fingerprint(full_batch)
                write_json(root/"first_batch_fingerprint.json", first_audit)
            total_valid = int((full_batch["semantic"] != cfg["ignore_index"]).sum())
            if total_valid < 1:
                raise ValueError("Entire training batch has ignored GT")
            loss_value = ce_a = ce_b = 0.; pieces = split_batch(full_batch, cfg["micro_batch"])
            for batch in pieces:
                inputs = [batch[key].to(device) for key in ("image", "padding_mask", "depth", "valid")]
                with amp_context(device):
                    output = model(*inputs)
                    loss, losses = dual_ce(output, batch["semantic"].to(device), total_valid)
                if not bool(torch.isfinite(loss)):
                    raise RuntimeError("Nonfinite training loss")
                loss.backward(); loss_value += float(loss.detach()); ce_a += losses["ce_a"]; ce_b += losses["ce_b"]
                del output, loss, inputs
            if abs(loss_value-.5*(ce_a+ce_b)) > max(1e-6, 2e-6*abs(loss_value)):
                raise RuntimeError("Dual CE identity failed")
            gradient = gradient_audit(model) if update == 0 or update+1 == cfg["max_updates"] else None
            total_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["gradient_clip"], error_if_nonfinite=True))
            rates = [group["lr"] for group in opt.param_groups]
            opt.step(); scheduler.step(); update += 1; images_processed += batch_images
            boundary = batch_index+1 == len(dl)
            position = {"epoch": epoch+1 if boundary else epoch,
                        "completed_epochs": epoch+1 if boundary else epoch,
                        "batches_in_current_epoch": 0 if boundary else batch_index+1,
                        "epoch_batches": len(dl), "dataset_images": len(dl.dataset), "loader_batch": cfg["loader_batch"],
                        "epoch_tail": cfg["epoch_tail"], "epoch_boundary": boundary, "num_workers": cfg["num_workers"],
                        "drop_last": False, "persistent_workers": False,
                        "augmentation_rng": "stateless_sha256(seed|epoch|sample_id); isolated CPU jitter",
                        "sampler_seed": cfg["seed"]+epoch,
                        "resume_regeneration": "recreate epoch sampler from seed; skip committed batches"}
            event = {"update": update, "epoch": update/cfg["epoch_updates"], "batch_images": batch_images,
                     "microbatch_sizes": [int(batch["semantic"].shape[0]) for batch in pieces], "valid_pixels": total_valid,
                     "loss": loss_value, "ce_a": ce_a, "ce_b": ce_b, "gradient_norm_before_clip": total_norm,
                     "lr": rates, "elapsed_seconds": elapsed(), "image_ids": list(full_batch["image_id"])}
            if gradient is not None:
                audit = optimizer_audit(model, opt, update)
                event.update(gradient_l2_by_module=gradient, optimizer_audit=audit)
                write_json(root/("first_runtime_audit.json" if update == 1 else "final_runtime_audit.json"), event)
            monitor_now = update%cfg["monitor_every_updates"] == 0 or update == cfg["max_updates"]
            if monitor_now:
                values = evaluate(model, cfg, cfg["monitor_split"], cfg["monitor_scales"], cfg["monitor_flip"],
                                  limit=cfg.get("monitor_limit"), device=device)
                event.update(val_miou_percent=values["miou_percent"], val_images=values["images"])
                write_json(root/f"val_u{update}.json", values)
                if values["miou"] > best:
                    best = values["miou"]
                    save_checkpoint(root/"best_monitor.pth", pack())
                    write_json(root/"selection.json", {"update": update, "epoch": update/cfg["epoch_updates"],
                               "miou_percent": 100*best, "images": values["images"], "split": cfg["monitor_split"],
                               "scales": cfg["monitor_scales"], "flip": cfg["monitor_flip"], "rule": cfg["selection_rule"]})
            with (root/"history.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(event, allow_nan=False)+"\n")
            snapshot = update in cfg["average_updates"] and not cfg.get("smoke")
            if snapshot:
                path = root/f"u{update}.pth"
                if path.exists():
                    raise FileExistsError(f"Immutable SUN snapshot exists: {path}")
                save_checkpoint(path, pack())
            if update%cfg["save_interval"] == 0 or boundary or snapshot or update == cfg["max_updates"]:
                save_checkpoint(root/"last.pth", pack(training_state=True))
            if update == 1 or update%cfg["log_interval"] == 0 or update == cfg["max_updates"]:
                status(); print(json.dumps(event, allow_nan=False), flush=True)
            del full_batch, pieces
    if update != cfg["max_updates"]:
        raise RuntimeError("Runner stopped before the declared update limit")
    status("complete")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True); parser.add_argument("--run-dir", required=True)
    parser.add_argument("--resume"); parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.smoke:
        cfg = smoke_config(cfg)
    root = Path(args.run_dir)
    if args.resume:
        if not (root/"config.json").is_file() or Path(args.resume).resolve().parent != root.resolve():
            raise ValueError("Resume requires last.pth inside the original run directory")
    elif root.exists() and any(root.iterdir()):
        raise FileExistsError("Cold runs must use a new empty output directory")
    root.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    try:
        run_training(cfg, root, args.resume, args.device)
    except BaseException as exc:
        prior = json.loads((root/"status.json").read_text()) if (root/"status.json").exists() else {}
        prior.update(status="failed", error=traceback.format_exc(),
                     oom_count=int(isinstance(exc, torch.cuda.OutOfMemoryError)))
        write_json(root/"status.json", prior)
        raise


if __name__ == "__main__":
    main()
