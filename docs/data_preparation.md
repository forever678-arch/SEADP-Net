# Data Preparation

Dataset bodies are NOT redistributed. Both preparation pipelines download the
official sources (SHA256-pinned), decode them into a fixed layout, and record
manifests; every later integrity check (training preflight, SUN fingerprint)
compares against those pinned hashes.

## NYUv2 (40 classes, 795 train / 654 test)

Sources (all SHA256-pinned in the generated `manifest.json`):

| File | Source |
|---|---|
| `nyu_train_rgb.tgz`, `nyu_test_rgb.tgz` | https://www.doc.ic.ac.uk/~ahanda/nyu_{train,test}_rgb.tgz |
| `labels40.mat` (precomputed NYU40 GT) | https://github.com/ankurhanda/nyuv2-meta-data |
| `splits.mat` (official 795/654) | same |
| `classMapping40.mat` (894→40 official map) | same |
| `nyu_depth_v2_labeled.mat` (HDF5) | https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html |

Put the five files under `data/nyuv2/raw/`, then:

```bash
python tools/prepare_nyuv2.py data/nyuv2           # rgb/ semantic/ splits/ + manifest
python tools/extract_nyuv2_depth.py --root data/nyuv2   # depth/*.npy  (filled 'depths', float16 metres)
python tools/prepare_nyuv2_validity.py data/nyuv2 nyuv2_raw_validity   # original-sensor valid PNGs
```

Layout produced: `rgb/{id:04d}.png`, `semantic/{id:04d}.png` (0–39 + 255),
`depth/{id:04d}.npy` (in-painted metric depth), `nyuv2_raw_validity/{id:04d}.png`
(`rawDepths > 0` mask — the model's RAW validity input V, kept separate from
the filled depth D), `splits/{train,test,val,...}.txt`, `manifest.json` +
`depth_manifest.json` (sources, hashes, alignment proof).

## SUN RGB-D (37 classes, 5285 train / 5050 test)

Single self-contained pipeline (resumable ranged download, never extracts the
full 6.9 GB archive):

```bash
python tools/prepare_sunrgbd.py data/sunrgbd_processed
```

Sources: official `SUNRGBD.zip` (rgbd.cs.princeton.edu, SHA256-pinned) plus
the public Handa meta-data (37-class label tarball and official train/test
image lists). Per sample it writes:

- `rgb/` — official RGB (native geometry retained everywhere),
- `depth/` — `depth_bfx` member, bit-rotation decoded to uint16 **millimetres**,
- `valid/` — decoded original depth `> 0` mask (RAW sensor validity V),
- `semantic/` — native-resolution 37-class GT,
- `splits/` — official `train5285` / `test5050`, plus a fixed
  RandomState(42) `test_sub500` monitoring subset drawn from the official
  test split (**selection used the test set** — recorded in every artifact),
- `fingerprint.json` — per-file SHA256 + class-pixel statistics.

## Loader contract (both datasets)

Filled depth D and RAW validity V are ALWAYS two separate inputs; depth is
clipped to 0–10 m and divided by 10; RGB uses ImageNet normalization. See
`seadpnet/data/{nyuv2,sunrgbd}.py` for the exact paired augmentation recipe.
