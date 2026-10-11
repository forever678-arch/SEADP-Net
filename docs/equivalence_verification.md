# Equivalence Verification Record

The package was verified against the ORIGINAL training/evaluation code on
the training server (RTX 4080 SUPER, Python 3.12.3, torch 2.8.0+cu128):
same checkpoint, same real dataset batch, both implementations side by side
(`tests/test_equivalence.py`). Acceptance: identical parameter tensors
(strict load on both sides), zero per-head logit differences, zero fused
prediction mismatches, and byte-equal confusion matrices on a 20-image
single-scale run.

| Checkpoint | Format | params/buffers | logits A/B | fused preds | confusion matrix |
|---|---|---|---|---|---|
| NYUv2 L, avg of u2000/3000/4000 | c11-csp-local-v1 | 0.0 / 0.0 | 0.0 / 0.0 | 0 mismatch | equal |
| NYUv2 S, avg of u2000/3000/4000 | c11-hrd-v2-v1 | 0.0 / 0.0 | 0.0 / 0.0 | 0 mismatch | equal |
| NYUv2 M, avg of u2000/3000/4000 | c11-hrd-v2-v1 | 0.0 / 0.0 | 0.0 / 0.0 | 0 mismatch | equal |
| SUN RGB-D M, avg of u6500/7500/8500 | seadp-sunrgbd-v1 | 0.0 / 0.0 | 0.0 / 0.0 | 0 mismatch | equal |
| SUN RGB-D S, avg of u6500/7500/8500 | seadp-sunrgbd-v1 | 0.0 / 0.0 | 0.0 / 0.0 | 0 mismatch | equal |
| SUN RGB-D L, avg of u6500/7500/8500 | seadp-sunrgbd-v1 | 0.0 / 0.0 | 0.0 / 0.0 | 0 mismatch | equal |

All six formal weights (NYUv2 S/M/L, SUN RGB-D S/M/L) are bit-exact between
the original code and this package.

Identifier note: strings like `c11-csp-local-v1` are internal CHECKPOINT
FORMAT identifiers stored inside the weight files; they are load keys, not
model names, and cannot change without rewriting every checkpoint.

Schema note: older NYUv2 L snapshots predate the `encoder_input_scale`
architecture field, and the first SUN-Net-L run additionally recorded code
constants instead of config identity (`experiment_name` "SEADP-Net-SUNRGBD")
with no `encoder_input_scale` field. `load_experiment` accepts exactly those
two recorded schemas (the legacy-L dict is reproduced verbatim as an
alternative candidate) plus a missing-`encoder_input_scale`-only tolerance
whose value must match the config; a conflicting value is never accepted.

Full-split single-scale reruns with the package:

- SUN RGB-D M, 5050 images: **53.833370910003644** — bit-identical to the
  recorded official SS value (same GPU class produced both).
- NYUv2 L, 654 images: **60.947887** vs recorded 60.9428 (+0.0051) — within
  the documented same-weight cross-machine noise (~0.005); same-machine
  logits were verified bit-identical above.
- SUN RGB-D M MS5+flip rerun queued on the training host (expected 54.660).

Verification rerun outputs (SS and MS5+flip for all six weights, paper-named
and path-sanitized, no numeric field touched) are in [`eval_records/`](../eval_records/).
