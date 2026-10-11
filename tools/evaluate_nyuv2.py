"""Evaluate a SEADP-Net NYUv2 checkpoint (SS or multi-scale + flip).

Examples:
  python tools/evaluate_nyuv2.py --checkpoint avg234.pth --split test --out ss.json \
      --rf-checkpoint /path/rf-detr-seg-small.pt --data-root /path/nyudv2 --valid-root /path/valid
  python tools/evaluate_nyuv2.py ... --scales 0.5 0.75 1.0 1.25 1.5 --flip --out msf.json
"""
from seadpnet.evaluation.nyuv2 import main

if __name__ == "__main__":
    main()
