"""Paired RGB/depth/validity augmentation with explicit physical padding metadata."""
from pathlib import Path
import random
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageOps
from torchvision.transforms import ColorJitter, functional as TF


class NYU40(Dataset):
    def __init__(self, cfg, split, augment=False, limit=None):
        self.root = Path(cfg["data_root"])
        self.valid_root = Path(cfg["valid_root"])
        self.ids = [int(x) for x in (self.root / "splits" / f"{split}.txt").read_text().split()]
        if limit is not None:
            self.ids = self.ids[:limit]
        if not self.ids:
            raise ValueError("Empty dataset")
        self.augment = augment
        self.aug = cfg["augmentation"]
        self.jitter = ColorJitter(*self.aug["color_jitter"])

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        ident = self.ids[index]
        rgb = Image.open(self.root / "rgb" / f"{ident:04d}.png").convert("RGB")
        gt = Image.open(self.root / "semantic" / f"{ident:04d}.png")
        depth = np.load(self.root / "depth" / f"{ident:04d}.npy").astype(np.float32)
        valid = np.asarray(Image.open(self.valid_root / f"{ident:04d}.png")) > 0
        if rgb.size != (640, 480) or gt.size != rgb.size or depth.shape != (480, 640) or valid.shape != depth.shape:
            raise ValueError(f"Unaligned sample {ident}")
        flip = self.augment and random.random() < self.aug["hflip"]
        if flip:
            rgb, gt = ImageOps.mirror(rgb), ImageOps.mirror(gt)
            depth, valid = depth[:, ::-1], valid[:, ::-1]
        ch, cw = 480, 640
        if self.augment and self.aug.get("scale_range"):
            scale = random.uniform(*self.aug["scale_range"])
            nh, nw = max(48, round(480 * scale)), max(48, round(640 * scale))
            rgb = rgb.resize((nw, nh), Image.Resampling.BILINEAR)
            gt = gt.resize((nw, nh), Image.Resampling.NEAREST)
            depth = F.interpolate(torch.from_numpy(np.ascontiguousarray(depth))[None, None],
                                  (nh, nw), mode="nearest")[0, 0].numpy()
            valid = F.interpolate(torch.from_numpy(valid.astype(np.float32))[None, None],
                                  (nh, nw), mode="nearest")[0, 0].numpy() > .5
            top, left = max(0, (nh-480)//2), max(0, (nw-640)//2)
            ch, cw = min(480, nh), min(640, nw)
            crop = (left, top, left+cw, top+ch)
            rgb, gt = rgb.crop(crop), gt.crop(crop)
            depth, valid = depth[top:top+ch, left:left+cw], valid[top:top+ch, left:left+cw]
            canvas = Image.new("RGB", (640, 480)); canvas.paste(rgb, (0, 0)); rgb = canvas
            canvas = Image.new(gt.mode, (640, 480), 255); canvas.paste(gt, (0, 0)); gt = canvas
            d = np.zeros((480, 640), np.float32); d[:ch, :cw] = depth; depth = d
            v = np.zeros((480, 640), bool); v[:ch, :cw] = valid; valid = v
        if self.augment:
            rgb = self.jitter(rgb)
        image = TF.normalize(TF.to_tensor(rgb), [.485, .456, .406], [.229, .224, .225])
        semantic = torch.from_numpy(np.array(gt, dtype=np.int64))
        if not ((semantic == 255) | ((semantic >= 0) & (semantic < 40))).all():
            raise ValueError(f"Invalid labels: {ident}")
        # Padding comes only from geometry, never from semantic labels.
        padding = torch.ones((480, 648), dtype=torch.bool)
        padding[:ch, :cw] = False
        depth = np.nan_to_num(np.ascontiguousarray(depth), nan=0., posinf=0., neginf=0.)
        return {"image": F.pad(image, (0, 8)), "semantic": F.pad(semantic, (0, 8), value=255),
                "depth": F.pad(torch.from_numpy(depth.clip(0, 10) / 10)[None], (0, 8)),
                "valid": F.pad(torch.from_numpy(np.ascontiguousarray(valid.astype(np.float32)))[None], (0, 8)),
                "padding_mask": padding, "image_id": ident, "flipped": bool(flip)}


def worker_seed(worker_id):
    seed = torch.initial_seed() % 2**32
    random.seed(seed); np.random.seed(seed)


def loader(cfg, split, augment=False, limit=None, batch_size=None, num_workers=None):
    ds = NYU40(cfg, split, augment, limit)
    generator = torch.Generator().manual_seed(cfg["seed"])
    # A dedicated generator prevents addon initialisation from changing image order.
    return DataLoader(ds, batch_size=batch_size or cfg["micro_batch"], shuffle=augment,
                      num_workers=cfg["num_workers"] if num_workers is None else num_workers,
                      pin_memory=True, generator=generator, worker_init_fn=worker_seed)


def forward_batch(model, batch, device):
    return model(batch["image"].to(device), batch["padding_mask"].to(device),
                 batch["depth"].to(device), batch["valid"].to(device))
