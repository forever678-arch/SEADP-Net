"""Independent SUN configuration, byte provenance, RNG and literal LR grouping."""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path
import random
import numpy as np
from PIL import Image
import torch
import yaml
from seadpnet.runtime import sha256, write_json, save_checkpoint, optimizer_for, state_digest
from seadpnet.data.sunrgbd import read_ids, sample_paths

ROOT = Path(__file__).resolve().parent
DATA_FORMAT = "seadp-sunrgbd-data-v1"


def load_config(path):
    path = Path(path)
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise ValueError("Configuration must be a mapping")
    base = cfg.pop("base", None)
    if base:
        merged = load_config(path.parent/base); merged.update(cfg); cfg = merged
    return normalize_config(cfg)


def normalize_config(cfg):
    cfg = dict(cfg)
    if cfg.get("training_authorized") is not True or cfg.get("debug_only") is not False:
        raise ValueError("Authorized configuration required")
    if any(cfg.get(key) for key in ("c11_checkpoint", "nyu_checkpoint", "warm_start_checkpoint")):
        raise ValueError("Cold original RF initialization required")
    if cfg.get("monitor_split") != "test_sub500" or cfg.get("monitor_origin") != cfg.get("test_split"):
        raise ValueError("User-selected test_sub500 monitor must originate from test5050")
    if cfg["train_split"] == cfg["test_split"]:
        raise ValueError("Test data cannot train")
    for key in ("num_classes", "loader_batch", "effective_batch", "micro_batch", "epochs", "padding_multiple", "save_interval", "log_interval", "monitor_every_epochs"):
        if isinstance(cfg.get(key), bool) or not isinstance(cfg.get(key), int) or cfg[key] < 1:
            raise ValueError(f"Positive integer required: {key}")
    if cfg["num_classes"] != len(cfg["class_names"]):
        raise ValueError("num_classes and class_names differ")
    if cfg["loader_batch"] != cfg["effective_batch"] or cfg["micro_batch"] > cfg["loader_batch"]:
        raise ValueError("Split one effective loader batch into positive microbatches")
    if cfg["ignore_index"] != 255 or cfg["eval_batch"] != 1 or cfg["padding_multiple"] != 24:
        raise ValueError("SUN ignore255/native batch1/RF padding24 required")
    if cfg["depth_format"] not in {"png", "npy"}:
        raise ValueError("depth_format must be png or npy")
    if cfg["eval_protocol"] != "sun_native_gt_logits_restore_then_softmax":
        raise ValueError("Native GT evaluation required")
    if cfg["loss"] != "0.5*CE_A+0.5*CE_B; global_valid_pixel_mean" or cfg["fusion"] != "fixed_equal_probability_mean":
        raise ValueError("Original equal dual CE and probability fusion required")
    if cfg["head_a_source"] != "rgb" or cfg["head_b_source"] != "depth":
        raise ValueError("SUN transfer uses RGB A and depth B")
    if cfg["shared_injection"] != "standard":
        raise ValueError("SUN transfer uses the original standard shared injection")
    for key in ("head_lr", "backbone_lr", "weight_decay", "gradient_clip", "depth_max_meters"):
        if not math.isfinite(float(cfg[key])) or cfg[key] <= 0:
            raise ValueError(f"Positive finite value required: {key}")
    train_images = cfg["split_sizes"][cfg["train_split"]]
    epoch_updates = math.ceil(train_images/cfg["loader_batch"])
    total = cfg.get("planned_updates", epoch_updates*cfg["epochs"])
    if isinstance(total, bool) or not isinstance(total, int) or total < 1:
        raise ValueError("Positive integer planned_updates required")
    average = cfg["average_updates"]
    if not average or average != sorted(set(average)) or any(step < 1 or step > total for step in average):
        raise ValueError("Prespecified averaging updates must be ordered and within horizon")
    interval = cfg.get("monitor_every_updates", 500)
    if isinstance(interval, bool) or not isinstance(interval, int) or interval < 1:
        raise ValueError("Positive monitor_every_updates required")
    cfg.update(train_updates=total, epochs=math.ceil(total/epoch_updates),
               epoch_updates=epoch_updates,
               epoch_tail=train_images%cfg["loader_batch"] or cfg["loader_batch"],
               grad_accumulation=math.ceil(cfg["loader_batch"]/cfg["micro_batch"]),
               initialization="original_rf_detr_pretrained_cold_sun_heads",
               resume_support="exact_mid_epoch_stateless_augmentation_optimizer_scheduler_rng_loader",
               drop_last=False, persistent_workers=False,
               selection_rule="highest fused test_sub500 single-scale mIoU; earliest update wins ties",
               test_used_for_selection=True,
               monitor_description="fixed random 500-image official-test subset; not independent validation",
               loader_batch_semantics="one augmented loader batch split; unmerged epoch tail",
               max_updates=cfg.get("max_updates", total))
    return cfg


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state["python"]); np.random.set_state(state["numpy"]); torch.set_rng_state(state["torch_cpu"])
    if state["torch_cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def source_code_hashes():
    # Hash all runtime Python plus the actual resolved config stored separately.
    return {path.relative_to(ROOT).as_posix(): sha256(path) for path in sorted(ROOT.rglob("*.py"))
            if not any(part in {"runs", "checks", "tests", "__pycache__"} for part in path.relative_to(ROOT).parts)}


def ensure_monitor_subset(cfg):
    """Create the user-authorized random test monitor after official splits exist."""
    root = Path(cfg["data_root"])
    source = read_ids(root, cfg["monitor_origin"])
    count = cfg["split_sizes"][cfg["monitor_split"]]
    indices = set(np.random.RandomState(cfg["split_seed"]).choice(len(source), size=count, replace=False).tolist())
    chosen = [ident for index, ident in enumerate(source) if index in indices]
    target = root/"splits"/f"{cfg['monitor_split']}.txt"
    text = "\n".join(chosen)+"\n"
    if target.exists() and target.read_text(encoding="utf-8") != text:
        raise ValueError("Existing test monitor subset differs; preserve evidence")
    if not target.exists():
        temporary = target.with_suffix(".tmp")
        temporary.write_text(text, encoding="utf-8")
        import os
        os.replace(temporary, target)
    return chosen


def fingerprint_data(cfg):
    root = Path(cfg["data_root"])
    ids = {split: read_ids(root, split) for split in cfg["split_sizes"]}
    actual = {split: len(values) for split, values in ids.items()}
    if actual != cfg["split_sizes"]:
        raise ValueError(f"Split sizes differ: {actual}")
    train, val, test = [set(ids[key]) for key in ("train5285", cfg["monitor_split"], "test5050")]
    if train & test or not val.issubset(test):
        raise ValueError("SUN train/test disjointness or test-subset membership is invalid")
    ordered = list(ids[cfg["monitor_origin"]])
    chosen_indices = set(np.random.RandomState(cfg["split_seed"]).choice(len(ordered), size=len(val), replace=False).tolist())
    chosen = [ident for index, ident in enumerate(ordered) if index in chosen_indices]
    if chosen != ids[cfg["monitor_split"]]:
        raise ValueError("test_sub500 differs from RandomState(split_seed) selection in official test order")
    paths = {root/"splits"/f"{split}.txt" for split in ids}
    for ident in train | test:
        paths.update(sample_paths(root, ident, cfg["depth_format"]).values())
    file_hashes = {path.relative_to(root).as_posix(): sha256(path) for path in sorted(paths)}
    counts = {}; sizes = {}
    for ident in sorted(train | test):
        pair = sample_paths(root, ident, cfg["depth_format"])
        semantic_path = pair["semantic"]
        with Image.open(semantic_path) as source:
            gt = np.asarray(source, dtype=np.int64)
        keep = gt != cfg["ignore_index"]
        if not np.all((gt[keep] >= 0) & (gt[keep] < cfg["num_classes"])):
            raise ValueError(f"Invalid processed labels while fingerprinting: {ident}")
        counts[ident] = np.bincount(gt[keep], minlength=cfg["num_classes"]).tolist()
        sizes[ident] = list(gt.shape)
        for modality in ("rgb", "valid"):
            with Image.open(pair[modality]) as source:
                if (source.height, source.width) != gt.shape:
                    raise ValueError(f"Unaligned native {modality} while fingerprinting: {ident}")
        if cfg["depth_format"] == "png":
            with Image.open(pair["depth"]) as source:
                depth_shape = (source.height, source.width)
        else:
            depth_shape = np.load(pair["depth"], allow_pickle=False, mmap_mode="r").shape
        if depth_shape != gt.shape:
            raise ValueError(f"Unaligned native depth while fingerprinting: {ident}")
    class_support = {split: np.asarray([counts[ident] for ident in values], dtype=np.int64).sum(0).tolist()
                     for split, values in ids.items()}
    digest = hashlib.sha256(json.dumps(file_hashes, sort_keys=True).encode()).hexdigest()
    return {"format": DATA_FORMAT, "depth_format": cfg["depth_format"], "num_classes": cfg["num_classes"],
            "ignore_index": cfg["ignore_index"], "split_seed": cfg["split_seed"],
            "split_algorithm": "numpy.RandomState(seed).choice(len(test_ids),monitor_count,replace=False); original order",
            "monitor_origin": cfg["monitor_origin"], "test_used_for_selection": True,
            "split_ids": ids, "split_sizes": actual, "files": len(file_hashes),
            "file_sha256": file_hashes, "dataset_sha256": digest,
            "class_pixels_by_split": class_support, "native_semantic_sizes": sizes}


def data_provenance(cfg, verify_files=True):
    path = Path(cfg["data_fingerprint_path"])
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("format") != DATA_FORMAT or manifest.get("split_sizes") != cfg["split_sizes"]:
        raise ValueError("Missing or mismatched SUN fingerprint")
    if manifest.get("depth_format") != cfg["depth_format"] or manifest.get("num_classes") != cfg["num_classes"]:
        raise ValueError("Fingerprint modality or classes differ")
    files = manifest["file_sha256"]
    digest = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    if digest != manifest["dataset_sha256"] or len(files) != manifest["files"]:
        raise ValueError("Corrupt fingerprint manifest")
    expected_files = 4*(cfg["split_sizes"]["train5285"]+cfg["split_sizes"]["test5050"])+len(cfg["split_sizes"])
    if len(files) != expected_files or any(not count for count in manifest["class_pixels_by_split"][cfg["test_split"]]):
        raise ValueError("Fingerprint is missing paired files or official test GT classes")
    for split in cfg["split_sizes"]:
        split_path = Path(cfg["data_root"])/"splits"/f"{split}.txt"
        if read_ids(cfg["data_root"], split) != manifest["split_ids"][split] or sha256(split_path) != files[f"splits/{split}.txt"]:
            raise ValueError(f"Split changed after fingerprint: {split}")
    if verify_files:
        actual = fingerprint_data(cfg)
        if actual != manifest:
            raise ValueError("Processed data or split bytes changed after fingerprint")
    return {"manifest": str(path), "manifest_sha256": sha256(path), "dataset_sha256": digest,
            "files": len(files), "split_sizes": cfg["split_sizes"], "data_root": cfg["data_root"],
            "depth_format": cfg["depth_format"], "native_gt_evaluation": True}
