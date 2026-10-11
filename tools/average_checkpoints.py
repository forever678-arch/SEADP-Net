"""Uniform FP32 parameter averaging of mid-training snapshots.

This reproduces the formal averaging artifacts: an equal-weight FP32 mean of
the MODEL STATE of explicitly listed checkpoints (e.g. NYU u2000/3000/4000
"avg234" or SUN E windows u6500/7500/8500). It is an inference artifact, not
SWA or EMA. All sources must share format/config/architecture/provenance and
come from one run.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def sha256(path):
    import hashlib
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def average_state_dicts(states):
    keys = list(states[0])
    if not keys:
        raise ValueError("Empty model state")
    if any(set(state) != set(keys) for state in states[1:]):
        raise ValueError("Model state keys differ")
    bn_names = {"running_mean", "running_var", "num_batches_tracked"}
    found = [key for key in keys if key.rsplit(".", 1)[-1] in bn_names]
    if found:
        raise ValueError("BatchNorm buffers require recalibration; refusing: " + ", ".join(found))
    result = {}
    for key in keys:
        values = [state[key] for state in states]
        reference = values[0]
        if any(value.shape != reference.shape or value.dtype != reference.dtype for value in values[1:]):
            raise ValueError(f"Model tensor shape/dtype differ: {key}")
        cpu_values = [value.detach().cpu() for value in values]
        if reference.is_floating_point():
            if not all(torch.isfinite(value).all().item() for value in cpu_values):
                raise ValueError(f"Nonfinite source tensor: {key}")
            accumulated = torch.zeros(reference.shape, dtype=torch.float32)
            for value in cpu_values:
                accumulated.add_(value.float(), alpha=1.0 / len(cpu_values))
            averaged = accumulated.to(reference.dtype)
            if not torch.isfinite(averaged).all().item():
                raise ValueError(f"Nonfinite averaged tensor: {key}")
            result[key] = averaged
        else:
            if any(not torch.equal(cpu_values[0], value) for value in cpu_values[1:]):
                raise ValueError(f"Nonfloating model tensors differ: {key}")
            result[key] = cpu_values[0].clone()
    return result


def average_checkpoints(paths, out, metadata_out):
    paths = [Path(path) for path in paths]
    out, metadata_out = Path(out), Path(metadata_out)
    if len(paths) < 2 or len({path.resolve() for path in paths}) != len(paths):
        raise ValueError("Distinct source checkpoints required")
    if len({path.resolve().parent for path in paths}) != 1:
        raise ValueError("Snapshot averaging sources must come from one run directory")
    if any(target.exists() for target in (out, metadata_out)):
        raise FileExistsError("Refusing to overwrite existing outputs")
    source_hashes = [sha256(path) for path in paths]
    checkpoints = [torch.load(path, map_location="cpu", weights_only=False) for path in paths]
    for expected, checkpoint in zip(checkpoints[1:], checkpoints):
        for field in ("format", "config", "architecture", "provenance"):
            if checkpoint.get(field) != checkpoints[0].get(field):
                raise ValueError(f"Sources do not share the same {field}")
    updates = [checkpoint.get("update") for checkpoint in checkpoints]
    if any(not isinstance(update, int) or update < 0 for update in updates):
        raise ValueError("Sources lack integer update counters")
    model = average_state_dicts([checkpoint["model"] for checkpoint in checkpoints])
    averaging = {
        "rule": "uniform model-state mean of explicitly listed mid-training snapshots",
        "accumulation_dtype": "float32",
        "updates": updates,
        "sources": [{"checkpoint": str(path), "sha256": digest, "update": update}
                    for path, digest, update in zip(paths, source_hashes, updates)],
    }
    first = checkpoints[0]
    payload = {key: first[key] for key in ("format", "config", "architecture", "provenance")
               if key in first}
    payload.update(model=model, update=-1, optimizer_steps=-1, averaging=averaging)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("xb") as stream:
        torch.save(payload, stream)
    report = {"checkpoint": str(out), "checkpoint_sha256": sha256(out),
              "parameters_total": first["architecture"].get("parameters_total"),
              "model_state_keys": len(model), "sources": averaging["sources"]}
    metadata_out.parent.mkdir(parents=True, exist_ok=True)
    with metadata_out.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--metadata", required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    print(json.dumps(average_checkpoints(args.checkpoints, args.out, args.metadata), indent=2))


if __name__ == "__main__":
    main()
