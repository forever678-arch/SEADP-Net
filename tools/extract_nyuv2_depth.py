"""Extract per-image filled depth from the official labeled mat.

Reads nyu_depth_v2_labeled.mat (MATLAB 7.3 / HDF5), takes the in-painted
`depths` volume, writes one float16 metres .npy per official image ID, and
proves alignment with the existing rgb/semantic files by comparing the mat's
own `images` against our rgb PNGs for a few IDs.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import h5py
from PIL import Image


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024**2), b''):
            h.update(block)
    return h.hexdigest()


def main(root):
    root = Path(root); raw = root/'raw'; out = root/'depth'
    out.mkdir(exist_ok=True)
    mat_path = raw/'nyu_depth_v2_labeled.mat'
    with h5py.File(mat_path, 'r') as f:
        keys = list(f.keys())
        assert {'depths', 'images', 'rawDepths'} <= set(keys), keys
        depths = f['depths']
        # MATLAB 480x640x1449 is stored C-ordered as (1449, 640, 480); each
        # slice is (W, H) and must be transposed to (H, W).
        assert tuple(depths.shape) == (1449, 640, 480), depths.shape
        probe_ids = [1, 400, 1200]
        stored = f['images'][[i - 1 for i in probe_ids]].transpose(0, 3, 2, 1)
        diffs = {}
        for k, image_id in enumerate(probe_ids):
            ours = np.array(Image.open(root/'rgb'/f'{image_id:04d}.png').convert('RGB'))
            diffs[image_id] = float(np.abs(stored[k].astype(int) - ours.astype(int)).mean())
        volume = np.stack([depths[i].T for i in range(1449)])  # (1449, 480, 640) metres
    assert max(diffs.values()) < 1.0, f'RGB alignment failed: {diffs}'
    assert np.isfinite(volume).all(), 'Filled depth contains non-finite values'
    for image_id in range(1, 1450):
        np.save(out/f'{image_id:04d}.npy', volume[image_id - 1].astype(np.float16))
    stats = dict(count=1449,
                 min_m=float(volume.min()), max_m=float(volume.max()),
                 mean_m=float(volume.mean()), std_m=float(volume.std()))
    manifest = dict(source='https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html',
                    file='raw/nyu_depth_v2_labeled.mat', file_sha256=sha256(mat_path),
                    field='depths (in-painted/filled depth, metres)', dtype='float16',
                    alignment_rgb_mean_abs_diff=diffs, value_stats=stats,
                    convention='depth/{image_id:04d}.npy aligned with rgb/semantic IDs',
                    status='validated')
    (root/'depth_manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--root', default='data/nyuv2')
    main(p.parse_args().root)
