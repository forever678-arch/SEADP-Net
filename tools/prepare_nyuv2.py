"""Prepare RGB/NYU40 from Ankur Handa's NYUv2 metadata and RGB archives."""
import argparse
import hashlib
import json
import re
import tarfile
from pathlib import Path
import numpy as np
from scipy.io import loadmat
from PIL import Image
from seadpnet.data.nyu40_classes import CLASSES


def map_labels(raw):
    """Official NYU40 IDs 0..40 -> 0..39 with 255 for the invalid 0 class."""
    raw = np.asarray(raw)
    if not np.isin(raw, np.arange(41)).all():
        raise ValueError('Expected raw NYU40 IDs 0..40')
    output = np.full(raw.shape, 255, dtype=np.uint8)
    valid = raw > 0
    output[valid] = raw[valid] - 1
    return output


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024**2), b''):
            h.update(block)
    return h.hexdigest()


def main(root):
    root = Path(root); raw = root/'raw'
    for directory in ('rgb', 'semantic', 'splits'):
        (root/directory).mkdir(exist_ok=True)
    split_mat = loadmat(raw/'splits.mat')
    splits = {s: sorted(split_mat[k].ravel().astype(int).tolist()) for s,k in [('train','trainNdxs'),('test','testNdxs')]}
    assert len(splits['train']) == 795 and len(splits['test']) == 654
    assert not set(splits['train']) & set(splits['test'])
    assert set(splits['train']+splits['test']) == set(range(1,1450))
    classes = loadmat(raw/'classMapping40.mat')
    names = [str(x.item()) for x in classes['className'].ravel()]
    # Source metadata has the historical typo "refridgerator" at ID 24.
    normalize = lambda x: re.sub(r'[^a-z]', '', x.lower()).replace('refridgerator', 'refrigerator')
    assert list(map(normalize, names)) == list(map(normalize, CLASSES)), names
    labels = loadmat(raw/'labels40.mat')['labels40']
    assert labels.shape == (480, 640, 1449)
    assert labels.min() == 0 and labels.max() == 40
    histogram = np.zeros(256, dtype=np.int64)
    for split in ('train','test'):
        seen = set()
        with tarfile.open(raw/f'nyu_{split}_rgb.tgz', 'r:gz') as archive:
            for member in archive:
                if not member.isfile(): continue
                match = re.fullmatch(r'nyu_rgb_(\d+)\.png', Path(member.name).name)
                if not match: continue
                image_id = int(match[1])
                assert image_id in splits[split] and image_id not in seen
                seen.add(image_id)
                image = Image.open(archive.extractfile(member)).convert('RGB')
                assert image.size == (640,480)
                image.save(root/'rgb'/f'{image_id:04d}.png')
                mapped = map_labels(labels[:,:,image_id-1])
                histogram += np.bincount(mapped.ravel(), minlength=256)
                Image.fromarray(mapped).save(root/'semantic'/f'{image_id:04d}.png')
        assert seen == set(splits[split])
    # Internal validation, untouched official test. All IDs remain original 1-based IDs.
    ids = np.asarray(splits['train']); rng = np.random.default_rng(42)
    val = sorted(rng.choice(ids, size=80, replace=False).tolist())
    splits['val'] = val; splits['train_dev'] = sorted(set(ids.tolist()) - set(val))
    splits['overfit'] = splits['train_dev'][:8]
    for key, ids in splits.items():
        (root/'splits'/f'{key}.txt').write_text('\n'.join(map(str,ids))+'\n')
    files = ['nyu_train_rgb.tgz','nyu_test_rgb.tgz','labels40.mat','splits.mat','classMapping40.mat']
    manifest = dict(source='https://github.com/ankurhanda/nyuv2-meta-data',
                    rgb_source='https://www.doc.ic.ac.uk/~ahanda/nyu_{train,test}_rgb.tgz',
                    label_source='labels40.mat: precomputed NYU40; classMapping40.mat records original 894-to-40 map',
                    sha256={f:sha256(raw/f) for f in files}, splits={k:len(v) for k,v in splits.items()},
                    classes=CLASSES, raw_class_names=names, seed=42,
                    label_convention='0..39 valid; 255 ignore; official image IDs 1..1449',
                    histogram=histogram.tolist(), status='validated')
    (root/'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest['splits']), flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser(); p.add_argument('--root', default='data/nyuv2'); main(p.parse_args().root)
