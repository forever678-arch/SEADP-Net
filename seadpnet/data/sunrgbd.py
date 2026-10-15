"""Processed SUN RGB-D pairs; stateless paired augmentation and native GT eval."""
from __future__ import annotations
import hashlib
from pathlib import Path
import random
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageOps
from torchvision.transforms import ColorJitter, functional as TF


def read_ids(root, split):
    ids = (Path(root) / "splits" / f"{split}.txt").read_text(encoding="utf-8").split()
    if not ids or len(ids) != len(set(ids)):
        raise ValueError(f"Empty or duplicate split: {split}")
    if any(Path(ident).name != ident or ident in {".", ".."} for ident in ids):
        raise ValueError("Split IDs must be plain stems without paths")
    return ids


def sample_paths(root, ident, depth_format):
    root = Path(root)
    candidates = [root / "rgb" / f"{ident}{suffix}" for suffix in (".jpg", ".png", ".jpeg")]
    rgb = next((path for path in candidates if path.is_file()), candidates[0])
    return {"rgb": rgb, "depth": root / "depth" / f"{ident}.{depth_format}",
            "valid": root / "valid" / f"{ident}.png", "semantic": root / "semantic" / f"{ident}.png"}


def _nearest(array, size):
    value = torch.from_numpy(np.ascontiguousarray(array, dtype=np.float32))[None, None]
    return F.interpolate(value, size=size, mode="nearest")[0, 0].numpy()


class SUNRGBD(Dataset):
    def __init__(self, cfg, split, augment=False, limit=None, epoch=0):
        self.cfg, self.root, self.split = cfg, Path(cfg["data_root"]), split
        self.ids = read_ids(self.root, split)
        if limit is not None:
            self.ids = self.ids[:limit]
        if not self.ids:
            raise ValueError("Empty selected dataset")
        self.augment, self.epoch = augment, int(epoch)
        self.depth_format = cfg["depth_format"]
        if self.depth_format not in {"png", "npy"}:
            raise ValueError("depth_format must be png(uint16 mm) or npy(float meters)")
        self.height, self.width = map(int, cfg["input_size"])
        self.num_classes, self.ignore = int(cfg["num_classes"]), int(cfg["ignore_index"])
        self.aug = cfg["augmentation"]
        self.jitter = ColorJitter(*self.aug["color_jitter"]) if self.aug.get("color_jitter") else None

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        ident = self.ids[index]
        paths = sample_paths(self.root, ident, self.depth_format)
        with Image.open(paths["rgb"]) as source:
            rgb = source.convert("RGB")
        with Image.open(paths["semantic"]) as source:
            semantic = np.array(source, dtype=np.int64)
        with Image.open(paths["valid"]) as source:
            valid = np.array(source) > 0
        if self.depth_format == "npy":
            depth = np.load(paths["depth"], allow_pickle=False).astype(np.float32)
        else:
            with Image.open(paths["depth"]) as source:
                raw = np.asarray(source)
                # Pillow may decode a 16-bit grayscale PNG to int32 (mode I).
                # Check the file bit depth rather than rejecting that valid backend.
                with paths["depth"].open("rb") as stream:
                    header = stream.read(26)
                png16 = len(header) == 26 and header[:8] == b"\x89PNG\r\n\x1a\n" and header[24:26] == bytes([16, 0])
                if not png16 or raw.ndim != 2 or raw.min() < 0 or raw.max() > 65535:
                    raise ValueError(f"PNG depth must contain uint16 millimeters: {ident}")
                depth = raw.astype(np.float32) / 1000.0
        native_size = (rgb.height, rgb.width)
        if semantic.shape != native_size or depth.shape != native_size or valid.shape != native_size:
            raise ValueError(f"Unaligned native pair: {ident}")
        if not np.all((semantic == self.ignore) | ((semantic >= 0) & (semantic < self.num_classes))):
            raise ValueError(f"Processed labels must be 0..{self.num_classes-1} or {self.ignore}: {ident}")
        native_gt = torch.from_numpy(semantic.copy())
        depth = np.nan_to_num(depth, nan=0., posinf=0., neginf=0.).clip(0, self.cfg["depth_max_meters"])
        # Filled depth and RAW validity are separate inputs, as in the final model.
        # Fixed network geometry never replaces native GT used by evaluation.
        h, w = self.height, self.width
        rgb = rgb.resize((w, h), Image.Resampling.BILINEAR)
        gt = Image.fromarray(semantic.astype(np.uint8)).resize((w, h), Image.Resampling.NEAREST)
        depth, valid = _nearest(depth, (h, w)), _nearest(valid, (h, w)) > .5
        seed_bytes = hashlib.sha256(f"{self.cfg['seed']}|{self.epoch}|{ident}".encode()).digest()
        sample_seed = int.from_bytes(seed_bytes[:8], "little") % (2**63-1)
        rng = random.Random(sample_seed)
        flipped = self.augment and rng.random() < self.aug["hflip"]
        if flipped:
            rgb, gt = ImageOps.mirror(rgb), ImageOps.mirror(gt)
            depth, valid = depth[:, ::-1], valid[:, ::-1]
        ch, cw = h, w
        if self.augment and self.aug.get("scale_range"):
            scale = rng.uniform(*self.aug["scale_range"])
            nh, nw = max(24, round(h * scale)), max(24, round(w * scale))
            rgb, gt = rgb.resize((nw, nh), Image.Resampling.BILINEAR), gt.resize((nw, nh), Image.Resampling.NEAREST)
            depth, valid = _nearest(depth, (nh, nw)), _nearest(valid, (nh, nw)) > .5
            top, left = max(0, (nh-h)//2), max(0, (nw-w)//2)
            ch, cw = min(h, nh), min(w, nw)
            crop = (left, top, left+cw, top+ch)
            rgb, gt = rgb.crop(crop), gt.crop(crop)
            depth, valid = depth[top:top+ch, left:left+cw], valid[top:top+ch, left:left+cw]
            canvas = Image.new("RGB", (w, h)); canvas.paste(rgb, (0, 0)); rgb = canvas
            canvas = Image.new("L", (w, h), self.ignore); canvas.paste(gt, (0, 0)); gt = canvas
            d, v = np.zeros((h, w), np.float32), np.zeros((h, w), bool)
            d[:ch, :cw], v[:ch, :cw] = depth, valid
            depth, valid = d, v
        if self.augment and self.jitter:
            # Each sample owns its augmentation RNG, including workers/prefetch.
            # Resume can regenerate an epoch and skip committed batches exactly.
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(sample_seed)
                rgb = self.jitter(rgb)
        image = TF.normalize(TF.to_tensor(rgb), [.485, .456, .406], [.229, .224, .225])
        target = torch.from_numpy(np.array(gt, dtype=np.int64))
        multiple = int(self.cfg["padding_multiple"])
        ph, pw = ((h+multiple-1)//multiple)*multiple, ((w+multiple-1)//multiple)*multiple
        padding = torch.ones((ph, pw), dtype=torch.bool)
        padding[:ch, :cw] = False
        output = {"image": F.pad(image, (0, pw-w, 0, ph-h)),
                  "semantic": F.pad(target, (0, pw-w, 0, ph-h), value=self.ignore),
                  "depth": F.pad(torch.from_numpy(np.ascontiguousarray(depth/self.cfg["depth_max_meters"]))[None], (0, pw-w, 0, ph-h)),
                  "valid": F.pad(torch.from_numpy(np.ascontiguousarray(valid, dtype=np.float32))[None], (0, pw-w, 0, ph-h)),
                  "padding_mask": padding, "image_id": ident, "flipped": bool(flipped),
                  "resized_size": torch.tensor([h, w]), "native_size": torch.tensor(native_size)}
        if not self.augment:
            output["native_semantic"] = native_gt
        return output


def collate(samples):
    from torch.utils.data import default_collate
    native = [sample.get("native_semantic") for sample in samples]
    output = default_collate([{key: value for key, value in sample.items() if key != "native_semantic"}
                              for sample in samples])
    if native[0] is not None:
        output["native_semantic"] = native
    return output


def loader(cfg, split, augment=False, limit=None, batch_size=None, num_workers=None, epoch=0):
    ds = SUNRGBD(cfg, split, augment, limit, epoch)
    generator = torch.Generator().manual_seed(int(cfg["seed"]) + int(epoch))
    return DataLoader(ds, batch_size=batch_size or cfg["loader_batch"], shuffle=augment,
                      num_workers=cfg["num_workers"] if num_workers is None else num_workers,
                      pin_memory=True, generator=generator, collate_fn=collate,
                      persistent_workers=False, drop_last=False)
