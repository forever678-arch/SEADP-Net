"""Stage-C equivalence: original training code vs this package.

Both implementations are built from the SAME checkpoint (strict load) and run
on the SAME real dataset batch; per-head logits must match to floating point
noise, and a small single-scale evaluation must produce identical confusion
matrices.

Run on the training server with the original project tree intact:

  cd /path/to/original/project      # cwd: the original training tree (its relative paths)
  PYTHONPATH=/path/to/SEADP-Net python /path/to/SEADP-Net/tests/test_equivalence.py \
      --dataset sun --checkpoint /path/to/seadp-net-sunrgbd-m.pth \
      --rf-checkpoint /path/to/original/project/weights/rf-detr-seg-small.pt \
      --data-root /path/to/sunrgbd_processed
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


def max_diff(a, b):
    return float((a.float() - b.float()).abs().max().item())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("sun", "nyu"), required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--rf-checkpoint", required=True)
    parser.add_argument("--data-root")
    parser.add_argument("--valid-root")
    parser.add_argument("--limit", type=int, default=4, help="images for the logits comparison")
    parser.add_argument("--eval-limit", type=int, default=20, help="images for the mIoU comparison")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    original_root = Path.cwd()
    package_root = Path(__file__).resolve().parents[1]

    # --- original implementation (top-level modules of the original project) ---
    sys.path.insert(0, str(original_root))
    torch.set_num_threads(4)
    if args.dataset == "sun":
        import sun_model as original_model_mod
        import sun_data as original_data_mod
        from sun_evaluate import evaluate as original_evaluate

        model_o, state = original_model_mod.load_experiment(args.checkpoint, device=args.device)
        cfg_o = state["config"]
        if args.data_root:
            cfg_o = dict(cfg_o, data_root=args.data_root)
        loader_o = original_data_mod.loader(cfg_o, cfg_o["test_split"], augment=False,
                                            limit=args.limit, batch_size=1)
    else:
        import branch_model as original_model_mod
        import pyramid_experiments.data as original_data_mod
        from evaluate import evaluate as original_evaluate

        model_o, state = original_model_mod.load_experiment(args.checkpoint, device=args.device)
        cfg_o = state["config"]
        for key, value in (("data_root", args.data_root), ("valid_root", args.valid_root)):
            if value:
                cfg_o = dict(cfg_o, **{key: value})
        loader_o = original_data_mod.loader(cfg_o, "test", augment=False,
                                            limit=args.limit, batch_size=1)

    # --- package implementation ---
    sys.path.insert(0, str(package_root))
    import seadpnet.model as package_model_mod
    overrides = {"rf_checkpoint": args.rf_checkpoint, "data_root": args.data_root,
                 "valid_root": args.valid_root}
    model_p, state_p = package_model_mod.load_experiment(args.checkpoint, device=args.device,
                                                         overrides=overrides)
    assert state_p["format"] == state["format"]
    assert state_p["architecture"] == state["architecture"], "architecture metadata diverged"
    cfg_p = dict(state_p["config"])
    for key, value in (("data_root", args.data_root), ("valid_root", args.valid_root)):
        if value:
            cfg_p[key] = value

    # --- identical weights -> identical parameter tensors ---
    params_o = dict(model_o.named_parameters())
    params_p = dict(model_p.named_parameters())
    assert set(params_o) == set(params_p), "parameter names diverged"
    worst_param = max(max_diff(params_o[name], params_p[name]) for name in params_o)
    buffers_o = dict(model_o.named_buffers())
    buffers_p = dict(model_p.named_buffers())
    worst_buffer = max((max_diff(buffers_o[name], buffers_p[name]) for name in buffers_o),
                       default=0.0)

    # --- same input, same forward ---
    from seadpnet.evaluation.common import amp_context
    model_o.eval(); model_p.eval()
    report = {"checkpoint": args.checkpoint, "format": state["format"],
              "max_param_diff": worst_param, "max_buffer_diff": worst_buffer,
              "logits_max_abs_diff": {}, "fused_prediction_mismatches": None}

    with torch.no_grad():
        for batch in loader_o:
            inputs = [batch[key].to(args.device) for key in ("image", "padding_mask", "depth", "valid")]
            with amp_context(args.device):
                out_o = model_o(*inputs)
                out_p = model_p(*inputs)
            for key_o, key_p in (("semantic_logits", "semantic_logits"),
                                 ("semantic_logits_b", "semantic_logits_b")):
                diff = max_diff(out_o[key_o], out_p[key_p])
                report["logits_max_abs_diff"][key_o] = max(
                    diff, report["logits_max_abs_diff"].get(key_o, 0.0))
            prob_o = .5 * out_o["semantic_logits"].float().softmax(1) + .5 * out_o["semantic_logits_b"].float().softmax(1)
            prob_p = .5 * out_p["semantic_logits"].float().softmax(1) + .5 * out_p["semantic_logits_b"].float().softmax(1)
            mismatches = int((prob_o.argmax(1) != prob_p.argmax(1)).sum().item())
            report["fused_prediction_mismatches"] = mismatches + (report["fused_prediction_mismatches"] or 0)
            break  # one batch of real images is enough for logits equality

    # --- identical small evaluation (confusion matrix equality) ---
    if args.dataset == "sun":
        from seadpnet.evaluation.sunrgbd import evaluate as package_evaluate
        from seadpnet.data.sunrgbd import loader as package_loader
        cfg_p = dict(state_p["config"])
        if args.data_root:
            cfg_p["data_root"] = args.data_root
        result_o = original_evaluate(model_o, cfg_o, limit=args.eval_limit, device=args.device)
        result_p = package_evaluate(model_p, cfg_p, limit=args.eval_limit, device=args.device)
    else:
        from seadpnet.evaluation.nyuv2 import evaluate as package_evaluate
        result_o = original_evaluate(model_o, cfg_o, split="test", limit=args.eval_limit, batch_size=1, device=args.device)
        result_p = package_evaluate(model_p, cfg_p, split="test", limit=args.eval_limit, batch_size=1, device=args.device)
    report["eval_images"] = result_o["images"]
    report["miou_percent_original"] = result_o["miou_percent"]
    report["miou_percent_package"] = result_p["miou_percent"]
    matrix_o = result_o["metrics_heads"]["fused"]["confusion_matrix"]
    matrix_p = result_p["metrics_heads"]["fused"]["confusion_matrix"]
    report["confusion_matrix_equal"] = matrix_o == matrix_p
    report["per_head_miou_original"] = {k: v["miou_percent"] for k, v in result_o["metrics_heads"].items()}
    report["per_head_miou_package"] = {k: v["miou_percent"] for k, v in result_p["metrics_heads"].items()}

    passed = (worst_param == 0.0 and worst_buffer == 0.0
              and report["fused_prediction_mismatches"] == 0
              and max(report["logits_max_abs_diff"].values()) <= 1e-4
              and report["confusion_matrix_equal"])
    report["passed"] = bool(passed)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
