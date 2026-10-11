"""Prepare metric SUN-RGBD data without extracting the entire official archive.

RGB/native 37-class labels are paired by the public Handa train/test path lists.
Official encoded raw and depth_bfx PNGs supply metric depth and raw validity.
Downloads are resumable, ranged into one file, and never replace old experiments.
"""
from __future__ import annotations
import argparse
import concurrent.futures as cf
import hashlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import tarfile
import threading
import time
import zipfile

import numpy as np
from PIL import Image
import requests

OFFICIAL_ZIP = "https://rgbd.cs.princeton.edu/data/SUNRGBD.zip"
ZIP_BYTES = 6885481608
EXPECTED_ARCHIVE_SHA256 = "1a6dbf2a1c9044c4805a35ee648d616ea39a231fd5bd6f77e84cd2b8287fe41c"
DOWNLOAD_ZIP = os.environ.get("SEADP_SUN_ARCHIVE_URL", OFFICIAL_ZIP)
META_BASE = "https://raw.githubusercontent.com/ankurhanda/sunrgbd-meta-data/master/"
LABEL_URL = META_BASE + "sunrgbd_train_test_labels.tar.gz"
TRAIN_URL = META_BASE + "sunrgbd_training_images.txt"
TEST_URL = META_BASE + "sunrgbd_testing_images.txt"


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(tmp, path)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def small_download(url, dest):
    if dest.exists():
        return
    response = requests.get(url, timeout=(30, 180))
    response.raise_for_status()
    tmp = dest.with_suffix(dest.suffix + ".part")
    tmp.write_bytes(response.content)
    os.replace(tmp, dest)


def ranged_download(url, dest, total, status_path, workers=4):
    if dest.exists() and dest.stat().st_size == total:
        with zipfile.ZipFile(dest):
            return
    part = dest.with_suffix(dest.suffix + ".part")
    ledger = dest.with_suffix(dest.suffix + ".ranges.json")
    size = 8 << 20
    count = math.ceil(total / size)
    saved = json.loads(ledger.read_text()) if ledger.exists() else {}
    allowed_urls = {OFFICIAL_ZIP,
                   "https://huggingface.co/datasets/ZHAO11564/SUNRGBD.zip/resolve/main/SUNRGBD.zip",
                   "https://hf-mirror.com/datasets/ZHAO11564/SUNRGBD.zip/resolve/main/SUNRGBD.zip"}
    if saved and (saved.get("url") not in allowed_urls or url not in allowed_urls
                  or saved.get("bytes") != total or saved.get("chunk") not in (8 << 20, 32 << 20)):
        raise RuntimeError("Download source changed; preserve existing partial evidence")
    old_size = saved.get("chunk", size)
    done = {offset // size for index in saved.get("complete", [])
            for offset in range(index * old_size, min(total, (index + 1) * old_size), size)}
    if any(index < 0 or index >= count for index in done):
        raise RuntimeError("Invalid completed range index")
    write_json(ledger, {"url": url, "bytes": total, "chunk": size, "complete": sorted(done),
                        "migration": f"preserve completed {old_size}-byte blocks as {size}-byte blocks"})
    fd = os.open(part, os.O_CREAT | os.O_RDWR, 0o644)
    os.ftruncate(fd, total)
    started = time.time()
    started_bytes = sum(min(size, total - i * size) for i in done)
    thread_local = threading.local()
    resolve_lock = threading.Lock()
    resolved = {"url": None, "time": 0.0}

    def resolve_url(force=False):
        if resolved["url"] and not force and time.time() - resolved["time"] < 1200:
            return resolved["url"]
        with resolve_lock:
            if resolved["url"] and not force and time.time() - resolved["time"] < 1200:
                return resolved["url"]
            with requests.get(url, headers={"Range": "bytes=0-0", "Accept-Encoding": "identity"},
                              stream=True, timeout=(12, 20)) as response:
                if response.status_code != 206 or response.headers.get("Content-Range") != f"bytes 0-0/{total}":
                    raise RuntimeError(f"Cannot resolve archive storage: HTTP {response.status_code}")
                _ = response.content
                resolved.update(url=response.url, time=time.time())
            return resolved["url"]

    def one(index):
        if not hasattr(thread_local, "client"):
            thread_local.client = requests.Session()
        client = thread_local.client
        left, right = index * size, min(total, (index + 1) * size) - 1
        last = None
        for attempt in range(5):
            try:
                attempt_started = time.time()
                with client.get(resolve_url(), headers={"Range": f"bytes={left}-{right}", "Accept-Encoding": "identity"},
                                stream=True, timeout=(12, 20)) as r:
                    if r.status_code != 206:
                        if r.status_code in (401, 403):
                            resolve_url(force=True)
                        raise RuntimeError(f"Range request returned {r.status_code}")
                    expected = f"bytes {left}-{right}/{total}"
                    if r.headers.get("Content-Range") != expected:
                        raise RuntimeError(f"Unexpected Content-Range {r.headers.get('Content-Range')}")
                    position = left
                    for block in r.iter_content(128 << 10):
                        if block:
                            written = os.pwrite(fd, block, position)
                            if written != len(block):
                                raise OSError("Short archive write")
                            position += written
                        if time.time() - attempt_started > 45:
                            raise TimeoutError("Range exceeded 45-second wall-clock budget")
                    if position != right + 1:
                        raise RuntimeError("Truncated range")
                return index
            except Exception as exc:
                last = exc
                time.sleep(min(2 ** attempt, 5))
        raise RuntimeError(f"Range {index} failed") from last
    try:
        # Continue recording successful chunks if a different request fails.
        # Raising inside the pool would otherwise wait silently for all queued
        # work, losing the successful range ledger on a transient HTTP error.
        for retry_round in range(3):
            failed = []
            with cf.ThreadPoolExecutor(max_workers=max(2, workers // (retry_round + 1))) as pool:
                futures = {pool.submit(one, i): i for i in range(count) if i not in done}
                for future in cf.as_completed(futures):
                    try:
                        done.add(future.result())
                    except Exception as exc:
                        failed.append(futures[future])
                        print(json.dumps({"stage": "range_retry", "index": futures[future],
                                          "round": retry_round, "error": repr(exc),
                                          "cause_type": type(exc.__cause__).__name__}), flush=True)
                        continue
                    write_json(ledger, {"url": url, "bytes": total, "chunk": size, "complete": sorted(done)})
                    progress = {"status": "running", "stage": "download", "source": url,
                                "downloaded_bytes": sum(min(size, total - i * size) for i in done),
                                "total_bytes": total, "chunks": len(done), "total_chunks": count,
                                "retry_round": retry_round, "failed_ranges_this_round": failed,
                                "new_bytes_since_resume": sum(min(size, total - i * size) for i in done) - started_bytes,
                                "average_MBps_since_resume": (sum(min(size, total - i * size) for i in done) - started_bytes) / max(time.time() - started, 1e-6) / 1e6,
                                "transport": "direct_storage_with_per_worker_persistent_TLS_session",
                                "elapsed_seconds": time.time() - started}
                    write_json(status_path, progress)
                    print(json.dumps(progress), flush=True)
            if len(done) == count:
                break
            print(json.dumps({"stage": "retry_remaining", "ranges": failed, "round": retry_round}), flush=True)
        os.fsync(fd)
    finally:
        os.close(fd)
    if len(done) != count:
        raise RuntimeError("Incomplete range ledger")
    with zipfile.ZipFile(part):
        pass
    os.replace(part, dest)


def decode_depth(encoded):
    # Official SUN toolbox depth encoding, then millimetres -> metres in loader.
    value = np.asarray(encoded, dtype=np.uint16)
    return ((value >> 3) | (value << 13)).astype(np.uint16)


def prepare(root, workers=4):
    root = Path(root).resolve()
    raw, data = root / "raw", root / "processed"
    raw.mkdir(parents=True, exist_ok=True)
    data.mkdir(parents=True, exist_ok=True)
    status_path = root / "prepare_status.json"
    write_json(status_path, {"status": "running", "stage": "metadata"})
    for url, name in ((TRAIN_URL, "train_paths.txt"), (TEST_URL, "test_paths.txt"),
                      (LABEL_URL, "labels37.tar.gz")):
        small_download(url, raw / name)
    train_paths = [str(PurePosixPath(x.strip().replace("\\", "/")))
                   for x in (raw / "train_paths.txt").read_text().splitlines() if x.strip()]
    test_paths = [str(PurePosixPath(x.strip().replace("\\", "/")))
                  for x in (raw / "test_paths.txt").read_text().splitlines() if x.strip()]
    if len(train_paths) != 5285 or len(test_paths) != 5050 or len(set(train_paths + test_paths)) != 10335:
        raise RuntimeError("SUN official split sizes or uniqueness mismatch")
    archive = raw / "SUNRGBD.zip"
    ranged_download(DOWNLOAD_ZIP, archive, ZIP_BYTES, status_path, workers)
    archive_sha = sha256(archive)
    if archive_sha != EXPECTED_ARCHIVE_SHA256:
        raise RuntimeError(f"Archive checksum mismatch: {archive_sha}")
    for directory in ("rgb", "depth", "valid", "semantic", "splits"):
        (data / directory).mkdir(exist_ok=True)
    paths = test_paths + train_paths
    labels = {}
    with tarfile.open(raw / "labels37.tar.gz", "r:gz") as tf:
        for member in tf.getmembers():
            if member.isfile() and PurePosixPath(member.name).suffix.lower() == ".png":
                stem = PurePosixPath(member.name).stem
                if stem.startswith("img-"):
                    identifier = int(stem.split("-")[-1])
                    labels[identifier] = tf.extractfile(member).read()
    if set(labels) != set(range(1, 10336)):
        raise RuntimeError(f"Label index mismatch, examples {sorted(labels)[:3]} / {sorted(labels)[-3:]}")
    histogram = np.zeros(37, dtype=np.int64)
    shape_histogram = {}
    samples = []
    with zipfile.ZipFile(archive) as z:
        names = set(z.namelist())
        depth_members = {}
        for name in names:
            p = PurePosixPath(name)
            if p.suffix.lower() == ".png" and p.parent.name in ("depth", "depth_bfx"):
                depth_members.setdefault((str(p.parent.parent), p.parent.name), []).append(name)
        for index, rgb_name in enumerate(paths, start=1):
            identifier = f"{index:06d}"
            if rgb_name not in names:
                raise RuntimeError(f"RGB member missing {rgb_name}")
            scene = str(PurePosixPath(rgb_name).parent.parent)
            original = depth_members.get((scene, "depth"), [])
            filled = depth_members.get((scene, "depth_bfx"), [])
            if len(original) != 1 or len(filled) != 1:
                raise RuntimeError(f"Nonunique depth pair {scene}: {original} / {filled}")
            rgb_bytes = z.read(rgb_name)
            rgb = Image.open(io.BytesIO(rgb_bytes)).convert("RGB")
            raw_depth = decode_depth(Image.open(io.BytesIO(z.read(original[0]))))
            filled_depth = decode_depth(Image.open(io.BytesIO(z.read(filled[0]))))
            label = np.array(Image.open(io.BytesIO(labels[index])), dtype=np.uint8)
            expected = (rgb.height, rgb.width)
            if label.shape != expected or raw_depth.shape != expected or filled_depth.shape != expected:
                raise RuntimeError(f"Alignment mismatch sample {index}: RGB {expected}, label {label.shape}, D {filled_depth.shape}")
            if np.any(label > 37):
                raise RuntimeError(f"Non37 labels in sample {index}: {np.unique(label)}")
            target = np.full(label.shape, 255, dtype=np.uint8)
            keep = (label >= 1) & (label <= 37)
            target[keep] = label[keep] - 1
            histogram += np.bincount(target[keep], minlength=37)
            valid = (raw_depth > 0).astype(np.uint8) * 255
            # uint16 PNG retains the decoded native millimetre precision exactly.
            rgb_extension = ".png" if PurePosixPath(rgb_name).suffix.lower() == ".png" else ".jpg"
            rgb_dest = data / "rgb" / f"{identifier}{rgb_extension}"
            depth_dest = data / "depth" / f"{identifier}.png"
            valid_dest = data / "valid" / f"{identifier}.png"
            semantic_dest = data / "semantic" / f"{identifier}.png"
            rgb_dest.write_bytes(rgb_bytes)
            Image.fromarray(filled_depth).save(depth_dest)
            Image.fromarray(valid).save(valid_dest)
            Image.fromarray(target).save(semantic_dest)
            key = f"{expected[0]}x{expected[1]}"
            shape_histogram[key] = shape_histogram.get(key, 0) + 1
            samples.append({"id": identifier, "rgb_member": rgb_name, "raw_depth_member": original[0],
                            "filled_depth_member": filled[0], "shape": list(expected)})
            if index % 100 == 0 or index == 10335:
                progress = {"status": "running", "stage": "prepare", "images": index, "total_images": 10335}
                write_json(status_path, progress)
                print(json.dumps(progress), flush=True)
    train_ids = [f"{i:06d}" for i in range(5051, 10336)]
    test_ids = [f"{i:06d}" for i in range(1, 5051)]
    generator = np.random.RandomState(42)
    selected = set(generator.choice(len(train_ids), size=200, replace=False).tolist())
    val_ids = [x for i, x in enumerate(train_ids) if i in selected]
    fit_ids = [x for i, x in enumerate(train_ids) if i not in selected]
    for name, values in (("train5285", train_ids), ("test5050", test_ids),
                         ("val200", val_ids), ("train_fit5085", fit_ids)):
        (data / "splits" / f"{name}.txt").write_text("\n".join(values) + "\n", encoding="utf-8")
    if set(val_ids) & set(fit_ids) or set(train_ids) & set(test_ids):
        raise RuntimeError("Split leakage")
    manifest = {"dataset": "SUNRGBD37", "classes": 37, "ignore": 255,
                "source_label_range": "0 background,1..37 classes", "target_label_range": "0..36 classes,255 ignore",
                "depth": "official depth_bfx bit-rotation decoded uint16 millimetres", "depth_format": "png",
                "raw_validity": "decoded original depth>0", "native_geometry_retained": True,
                "official_train": 5285, "official_test": 5050, "train_fit": 5085, "validation": 200,
                "validation_policy": "fixed RandomState42 selection from official train, no test use",
                "sources": {"rgb_depth": OFFICIAL_ZIP, "archive_transport": DOWNLOAD_ZIP,
                            "archive_expected_sha256": EXPECTED_ARCHIVE_SHA256,
                            "labels": LABEL_URL, "train_paths": TRAIN_URL, "test_paths": TEST_URL},
                "source_sha256": {name: sha256(raw / name) for name in ("SUNRGBD.zip", "labels37.tar.gz", "train_paths.txt", "test_paths.txt")},
                "shape_histogram": shape_histogram, "class_support_pixels_all_data": histogram.tolist(), "samples": samples}
    write_json(data / "manifest.json", manifest)
    write_json(status_path, {"status": "complete", "stage": "ready", "root": str(data),
                             "images": 10335, "splits": {"train_fit5085": 5085, "val200": 200, "test5050": 5050},
                             "manifest_sha256": sha256(data / "manifest.json")})
    print(json.dumps({"status": "complete", "data_root": str(data)}), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True)
    p.add_argument("--workers", type=int, default=4)
    a = p.parse_args()
    try:
        prepare(a.root, a.workers)
    except BaseException as exc:
        write_json(Path(a.root) / "prepare_status.json", {"status": "failed", "error": repr(exc)})
        raise
