"""Evaluate a SEADP-Net SUN RGB-D checkpoint on native-size ground truth.

Single-scale:
  python tools/evaluate_sunrgbd.py --checkpoint m2_avg_E.pth --out ss.json \
      --rf-checkpoint /path/rf-detr-seg-small.pt --data-root /path/sunrgbd_processed
MS5+flip:
  python tools/evaluate_sunrgbd.py ... --scales 0.5 0.75 1.0 1.25 1.5 --flip --out msf.json
"""
from seadpnet.evaluation.sunrgbd import main

if __name__ == "__main__":
    main()
