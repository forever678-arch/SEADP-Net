"""Deterministic seeding, hashing, atomic IO and the original LR grouping.

The optimizer grouping rule is part of the published recipe: only parameters
whose name starts with ``backbone.``, contains ``encoder`` and does not
contain ``depth`` use the backbone LR; everything else (decoders, pyramids,
depth/validity projections) uses the head LR.
"""
import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np
import torch

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""): h.update(block)
    return h.hexdigest()


def state_digest(model, prefix=None):
    h = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        if prefix and not name.startswith(prefix): continue
        h.update(f"{name}|{value.dtype}|{tuple(value.shape)}".encode())
        h.update(value.detach().contiguous().cpu().reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def write_json(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+".tmp")
    tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False),encoding="utf-8")
    os.replace(tmp,path)


def save_checkpoint(path, value):
    path=Path(path);tmp=path.with_name(path.name+".tmp")
    torch.save(value,tmp);os.replace(tmp,path)


def optimizer_for(model,cfg):
    groups={}
    for name,param in model.named_parameters():
        if not param.requires_grad:continue
        # Preserve the original encoder/projector LR grouping. New decoder and
        # pyramid always receive head LR, rather than matching an 'encoder' substring.
        lr=cfg["backbone_lr"] if name.startswith("backbone.") and "encoder" in name and "depth" not in name else cfg["head_lr"]
        groups.setdefault(lr,[]).append(param)
    return torch.optim.AdamW([{"params":ps,"lr":lr} for lr,ps in groups.items()],weight_decay=cfg["weight_decay"])
