"""NYUv2 evaluation over canonical image views (multi-scale/flip protocol).

Identical math to the formal evaluation: per-head logits restore/crop/softmax,
view-mean probabilities, fixed 0.5/0.5 head fusion, 40x40 confusion matrix
accumulated over the whole split. The right 8 pixels are input padding.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from seadpnet.data.nyuv2 import loader
from seadpnet.evaluation.common import (
    amp_context, metrics_result, view_probabilities,
)
from seadpnet.evaluation.metrics import SemanticMetrics
from seadpnet.data.nyu40_classes import CLASSES

HEADS = ("head_a", "head_b", "fused")
PROTOCOL = "nyuv2_image_views"
AGGREGATION = "canonical_per_head_logits_restore_crop_softmax"


def new_metrics():
    return SemanticMetrics(40, CLASSES, 255)


def architecture(model):
    from seadpnet.model import architecture as describe
    return describe(model)


@torch.no_grad()
def evaluate(model, cfg, split="test_sub100", scales=(1.,), flip=False, *,
             limit=None, batch_size=2, device="cuda"):
    """Evaluate all three predictions using the same samples and image views.

    Limit/batch/device are deliberately keyword-only: the legacy positional call
    silently evaluated two images rather than batches of size two.
    """
    scales = tuple(float(scale) for scale in scales)
    if not scales or any(scale <= 0 for scale in scales):
        raise ValueError("At least one positive scale required")
    was_training = model.training
    model.eval()
    metrics = {head: new_metrics() for head in HEADS}
    seen, image_ids = 0, []
    views = len(scales) * (2 if flip else 1)
    try:
        for batch in loader(cfg, split, augment=False, limit=limit, batch_size=batch_size):
            scores_a = scores_b = None
            for scale in scales:
                for mirrored in ((False, True) if flip else (False,)):
                    probability_a, probability_b = view_probabilities(model, batch, device, scale, mirrored)
                    if scores_a is None:
                        scores_a, scores_b = probability_a, probability_b
                    else:
                        scores_a.add_(probability_a)
                        scores_b.add_(probability_b)
                    del probability_a, probability_b
            scores_a.div_(views)
            scores_b.div_(views)
            predictions = {"head_a": scores_a.argmax(1).cpu(),
                           "head_b": scores_b.argmax(1).cpu(),
                           "fused": (scores_a + scores_b).argmax(1).cpu()}
            target = batch["semantic"][..., :640]
            if predictions["fused"].shape != target.shape:
                raise ValueError(f"Prediction/GT mismatch: {predictions['fused'].shape} != {target.shape}")
            ids = batch.get("image_id")
            if ids is not None:
                image_ids.extend(int(ident) for ident in ids)
            for index in range(len(target)):
                for head in HEADS:
                    metrics[head].update(predictions[head][index], target[index])
                seen += 1
            del predictions, scores_a, scores_b
        heads = {head: metrics_result(metric) for head, metric in metrics.items()}
        result = dict(heads["fused"])
        result.pop("confusion_matrix", None)
        result.update(metrics_heads=heads, images=seen, image_ids=image_ids, split=split,
                      scales=list(scales), flip=bool(flip), protocol=PROTOCOL,
                      aggregation=AGGREGATION, head="fused", head_weights=[.5, .5],
                      limit=limit, batch_size=batch_size,
                      **architecture(model))
        return result
    finally:
        model.train(was_training)


def checkpoint_metadata(state, checkpoint):
    from seadpnet.runtime import sha256
    metadata = {"checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
                "checkpoint_update": state.get("update", -1),
                "pretrained_sha256": state.get("provenance", {}).get("pretrained_sha256")}
    for key in ("averaging", "checkpoint_averaging", "average_sources", "weight_average"):
        if key in state:
            metadata[key] = state[key]
    return metadata
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test", choices=("test", "test_sub100", "val", "train"))
    parser.add_argument("--scales", type=float, nargs="+", default=[1.])
    parser.add_argument("--flip", action="store_true")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", required=True)
    parser.add_argument("--rf-checkpoint", help="Override config pretrained_checkpoint path")
    parser.add_argument("--data-root", help="Override config data_root")
    parser.add_argument("--valid-root", help="Override config valid_root")
    args = parser.parse_args()
    if Path(args.out).exists():
        raise FileExistsError(args.out)
    from seadpnet.model import architecture, load_experiment
    from seadpnet.runtime import write_json
    torch.set_num_threads(4)
    model, state = load_experiment(args.checkpoint, device=args.device,
                                   overrides=dict(rf_checkpoint=args.rf_checkpoint,
                                                  data_root=args.data_root,
                                                  valid_root=args.valid_root))
    cfg = dict(state["config"])
    for key, value in (("data_root", args.data_root), ("valid_root", args.valid_root)):
        if value:
            cfg[key] = value
    result = evaluate(model, cfg, args.split, args.scales, args.flip,
                      limit=args.limit, batch_size=args.batch_size, device=args.device)
    result.update(architecture(model))
    result.update(fusion="fixed_equal_probability_mean", head_weights=[.5, .5],
                  fusion_weight_trainable=False)
    result.update(checkpoint_metadata(state, args.checkpoint))
    write_json(args.out, result)
    print(json.dumps({"checkpoint_update": result["checkpoint_update"], "images": result["images"],
                      "miou_percent": result["miou_percent"], "scales": result["scales"], "flip": result["flip"],
                      "metrics_heads": {name: values["miou_percent"] for name, values in result["metrics_heads"].items()}},
                     allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
