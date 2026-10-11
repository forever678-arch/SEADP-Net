"""Extract original sensor validity, separate from filled NYUv2 depth."""
import argparse
import hashlib
import json
from pathlib import Path
import h5py
import numpy as np
from PIL import Image


def main(data_root: Path, out: Path):
    mat = data_root / "raw" / "nyu_depth_v2_labeled.mat"
    out.mkdir(parents=True, exist_ok=True)
    hasher = hashlib.sha256()
    with mat.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    counts = []
    with h5py.File(mat, "r") as handle:
        raw = handle["rawDepths"]
        if raw.shape != (1449, 640, 480):
            raise ValueError(f"Unexpected rawDepths storage shape {raw.shape}")
        for image_id in range(1, 1450):
            sensor = raw[image_id - 1].T  # MATLAB HDF5 is W,H inside each image.
            valid = np.isfinite(sensor) & (sensor > 0)
            counts.append(int(valid.sum()))
            Image.fromarray(valid.astype(np.uint8) * 255).save(out / f"{image_id:04d}.png")
    info = {
        "source_file_sha256": hasher.hexdigest(),
        "source_field": "rawDepths (>0 and finite)",
        "filled_depth_source": "depths field, stored separately in data/nyuv2/depth",
        "count": len(counts),
        "image_shape_hw": [480, 640],
        "valid_fraction_min": min(counts) / (480 * 640),
        "valid_fraction_mean": float(np.mean(counts) / (480 * 640)),
        "valid_fraction_max": max(counts) / (480 * 640),
    }
    (out / "manifest.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/nyuv2"))
    parser.add_argument("--out", type=Path, default=Path("depth_validity"))
    args = parser.parse_args()
    main(args.data_root, args.out)
