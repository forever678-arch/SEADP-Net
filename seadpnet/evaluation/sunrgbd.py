"""Native-label SUN evaluation: unflip, restore padded logits, crop, native resize, softmax."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import torch
from torch.nn import functional as F
from seadpnet.evaluation.common import amp_context, view_inputs
from seadpnet.data.sunrgbd import loader
from seadpnet.evaluation.metrics import SemanticMetrics

HEADS = ("head_a", "head_b", "fused")
PROTOCOL = "sun_native_gt_logits_restore_then_softmax"


def restore_native_logits(logits, padded_size, resized_size, native_size, mirrored=False):
    logits = logits.float()
    if mirrored:
        logits = logits.flip(-1)
    logits = F.interpolate(logits, size=tuple(padded_size), mode="bilinear", align_corners=False)
    h, w = map(int, resized_size)
    logits = logits[..., :h, :w]
    return F.interpolate(logits, size=tuple(map(int, native_size)), mode="bilinear", align_corners=False)


@torch.no_grad()
def evaluate(model, cfg, split=None, scales=(1.,), flip=False, *, limit=None, batch_size=1, device="cuda"):
    if batch_size != 1:
        raise ValueError("Native SUN GT has variable sizes; evaluation requires batch=1")
    scales = tuple(float(scale) for scale in scales)
    if not scales or any(scale <= 0 for scale in scales):
        raise ValueError("Positive scales required")
    split = split or cfg["test_split"]
    was_training = model.training
    model.eval()
    metrics = {head: SemanticMetrics(cfg["num_classes"], cfg.get("class_names"), cfg["ignore_index"]) for head in HEADS}
    views = len(scales)*(2 if flip else 1)
    ids, sizes = [], []
    try:
        for batch in loader(cfg, split, augment=False, limit=limit, batch_size=1):
            scores = None
            for scale in scales:
                for mirrored in ((False, True) if flip else (False,)):
                    with amp_context(device):
                        output = model(*view_inputs(batch, device, scale, mirrored))
                    logits = [restore_native_logits(output[key], batch["image"].shape[-2:],
                                                     batch["resized_size"][0], batch["native_size"][0], mirrored)
                              for key in ("semantic_logits", "semantic_logits_b")]
                    probabilities = [value.softmax(1) for value in logits]
                    if scores is None:
                        scores = probabilities
                    else:
                        for index, value in enumerate(probabilities):
                            scores[index].add_(value)
                    del output, logits, probabilities
            scores = [value/views for value in scores]
            predictions = (scores[0].argmax(1)[0].cpu(), scores[1].argmax(1)[0].cpu(),
                           (scores[0]+scores[1]).argmax(1)[0].cpu())
            target = batch["native_semantic"][0]
            for head, prediction in zip(HEADS, predictions):
                metrics[head].update(prediction, target)
            ids.append(batch["image_id"][0]); sizes.append(list(target.shape))
        heads = {head: metric.compute() for head, metric in metrics.items()}
        if split == cfg["test_split"] and limit is None:
            expected = cfg["split_sizes"][split]
            if len(ids) != expected:
                raise RuntimeError(f"Full official test needs {expected} images")
            if any(result["classes_in_union"] != cfg["num_classes"] for result in heads.values()):
                raise RuntimeError("Full test must include all semantic classes")
        result = {key: value for key, value in heads["fused"].items() if key != "confusion_matrix"}
        result.update(metrics_heads=heads, images=len(ids), image_ids=ids, native_sizes=sizes,
                      split=split, scales=list(scales), flip=bool(flip), batch_size=1, limit=limit,
                      head="fused", head_weights=[.5, .5], protocol=PROTOCOL,
                      aggregation="per_head_native_logits_then_softmax_view_mean_equal_probability_fusion",
                      input_size=cfg["input_size"], evaluation_gt="unchanged_native_semantic",
                      model_name=cfg["model_name"], experiment_name=cfg["experiment_name"],
                      test_used_for_selection=cfg["test_used_for_selection"],
                      selection_split=cfg["monitor_split"], monitor_origin=cfg["monitor_origin"])
        return result
    finally:
        model.train(was_training)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True); parser.add_argument("--out", required=True)
    parser.add_argument("--split"); parser.add_argument("--scales", nargs="+", type=float, default=[1.])
    parser.add_argument("--flip", action="store_true"); parser.add_argument("--limit", type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--rf-checkpoint", help="Override config pretrained_checkpoint path")
    parser.add_argument("--data-root", help="Override config data_root")
    args = parser.parse_args()
    if Path(args.out).exists():
        raise FileExistsError(args.out)
    from seadpnet.model import load_experiment
    from seadpnet.runtime import sha256, write_json
    model, state = load_experiment(args.checkpoint, device=args.device,
                                   overrides=dict(rf_checkpoint=args.rf_checkpoint,
                                                  data_root=args.data_root))
    cfg = dict(state["config"])
    if args.data_root:
        cfg["data_root"] = args.data_root
    result = evaluate(model, cfg, args.split, args.scales, args.flip, limit=args.limit, device=args.device)
    result.update(checkpoint=str(args.checkpoint), checkpoint_sha256=sha256(args.checkpoint),
                  checkpoint_update=state["update"], checkpoint_epoch=state["epoch"],
                  training_updates=state.get("training_updates", state["update"]),
                  averaging=state.get("averaging"), run_type=state["config"]["run_type"])
    write_json(args.out, result)
    print(json.dumps({"images": result["images"], "miou_percent": result["miou_percent"], "split": result["split"]}), flush=True)


if __name__ == "__main__":
    main()
