#!/usr/bin/env python3
"""Prepare shared normalized FP32 vectors, eligibility masks and exact filtered top-k."""

import argparse
import hashlib
import json
import logging
from pathlib import Path

import h5py
import numpy as np


def require(condition, message):
    """Fail before publishing an incomplete prepared-data manifest."""
    if not condition:
        raise ValueError(message)


def fingerprint(path):
    """Bind reuse to source path, size and modification timestamp."""
    path = Path(path).resolve()
    stat = path.stat()
    return {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def normalize(block):
    """Ensure nonzero finite unit vectors so L2 ranking equals cosine ranking."""
    block = np.array(block, dtype="<f4", order="C", copy=True)
    norms = np.linalg.norm(block, axis=1)
    require(np.isfinite(block).all() and np.all(norms > 0), "nonfinite or zero vector in dataset")
    block /= norms[:, None]
    return block


def export_matrix(source, path, rows):
    """Stream HDF5 rows into a matrix with a uint64 [rows, dimension] header."""
    require(source.ndim == 2 and source.shape[1] > 0, "expected a two-dimensional nonempty matrix")
    with path.open("wb") as output:
        np.asarray([rows, source.shape[1]], dtype="<u8").tofile(output)
        for start in range(0, rows, 8192):
            normalize(source[start:min(rows, start + 8192)]).tofile(output)


def mapped_matrix(path):
    """Read a prepared matrix without loading a second copy into memory."""
    shape = tuple(int(value) for value in np.fromfile(path, dtype="<u8", count=2))
    return np.memmap(path, dtype="<f4", mode="r", offset=16, shape=shape)


def synthetic(path):
    """Create a deterministic 2048-dimensional smoke fixture; it is not Kling performance evidence."""
    rng = np.random.default_rng(20260928)
    centers = rng.normal(size=(40, 2048)).astype("float32")
    with h5py.File(path, "w") as output:
        for name, count in (("train", 4000), ("test", 120)):
            labels = rng.integers(0, len(centers), count)
            vectors = centers[labels] + rng.normal(scale=0.3, size=(count, 2048)).astype("float32")
            output.create_dataset(name, data=normalize(vectors))
        output.attrs["synthetic_smoke_only"] = True


def mask_paths(arguments):
    """Resolve named packed exclusion bitsets without accepting path traversal in scene names."""
    result = {}
    for item in arguments.bitset:
        name, path = item.split("=", 1)
        require(name.replace("_", "").isalnum(), "bitset scene name must be alphanumeric or underscore")
        require(name not in result, "duplicate bitset scene name")
        result[name] = Path(path).resolve()
    return result


def create_masks(rows, arguments):
    """Match Knowhere id%100 scenes, or seeded random masks, plus arbitrary packed exclusion bitsets."""
    result = {}
    order = np.random.default_rng(arguments.seed).permutation(rows)
    for percentage in arguments.percentages:
        require(0 < percentage <= 100, "accepted percentage must be in (0,100]")
        mask = np.arange(rows) % 100 < percentage
        if arguments.mask_mode == "random":
            mask = np.zeros(rows, dtype=bool)
            mask[order[:int(rows * percentage / 100)]] = True
        result[f"{arguments.mask_mode}_p{percentage}"] = mask
    for name, path in mask_paths(arguments).items():
        packed = np.fromfile(path, dtype=np.uint8)
        require(len(packed) == (rows + 7) // 8, f"packed bitset length mismatch: {path}")
        result[f"custom_{name}"] = np.unpackbits(packed, bitorder="little", count=rows) == 0
    return result


def merge_topk(scores, ids, state, topk):
    """Keep the best exact dot products seen so far; temporary memory is block-bounded."""
    combined_scores = np.concatenate((state[0], scores), axis=1)
    combined_ids = np.concatenate((state[1], np.broadcast_to(ids, scores.shape)), axis=1)
    positions = np.argpartition(combined_scores, -topk, axis=1)[:, -topk:]
    return (np.take_along_axis(combined_scores, positions, axis=1),
            np.take_along_axis(combined_ids, positions, axis=1))


def exact_truth(directory, masks, arguments):
    """Scan FP32 data once; share exact scores across masks, never derive truth from ANN results."""
    train = mapped_matrix(directory / "train.f32")
    queries = mapped_matrix(directory / "test.f32")[:arguments.recall_nq]
    shape = (arguments.recall_nq, arguments.topk)
    states = {name: (np.full(shape, -np.inf, dtype="float32"), np.full(shape, -1, dtype="int64"))
              for name in masks}
    for start in range(0, len(train), 8192):
        end = min(len(train), start + 8192)
        scores = queries @ train[start:end].T
        ids = np.arange(start, end, dtype="int64")
        for name, mask in masks.items():
            passing = mask[start:end]
            states[name] = merge_topk(scores[:, passing], ids[passing], states[name], arguments.topk)
        print(f"exact truth scanned={end}/{len(train)}", flush=True)
    for name, (scores, ids) in states.items():
        require(np.isfinite(scores).all() and (ids >= 0).all(), f"insufficient exact truth for {name}")
        order = np.argsort(-scores, axis=1)
        np.take_along_axis(ids, order, axis=1).astype("<i8").tofile(directory / f"{name}.gt")


def prepare_new(source, directory, arguments):
    """Materialize a cache before its completion manifest is published."""
    with h5py.File(source, "r") as data:
        rows = data["train"].shape[0]
        query_rows = min(data["test"].shape[0], max(arguments.recall_nq, arguments.nq * arguments.concurrency))
        require(query_rows >= max(arguments.recall_nq, arguments.nq), "insufficient test rows")
        require(data["train"].shape[1] == data["test"].shape[1], "train/test dimension mismatch")
        masks = create_masks(rows, arguments)
        require(bool(masks), "at least one filter scene is required")
        for name, mask in masks.items():
            require(int(mask.sum()) >= arguments.topk, f"fewer than topk eligible vectors: {name}")
            mask.astype("uint8").tofile(directory / f"{name}.mask")
        export_matrix(data["train"], directory / "train.f32", rows)
        export_matrix(data["test"], directory / "test.f32", query_rows)
    exact_truth(directory, masks, arguments)
    (directory / "scenes.txt").write_text("\n".join(masks) + "\n")


def prepare(source, root, arguments):
    """Reuse only a completed cache with identical inputs and preparation semantics."""
    description = {"version": 1, "source": fingerprint(source), "topk": arguments.topk,
                   "recall_nq": arguments.recall_nq, "nq": arguments.nq, "concurrency": arguments.concurrency,
                   "percentages": arguments.percentages, "mask_mode": arguments.mask_mode, "seed": arguments.seed,
                   "bitsets": {name: fingerprint(path) for name, path in mask_paths(arguments).items()},
                   "preparer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    key = hashlib.sha256(json.dumps(description, sort_keys=True).encode()).hexdigest()[:16]
    directory = root / key
    directory.mkdir(parents=True, exist_ok=True)
    manifest = directory / "manifest.json"
    if not manifest.exists():
        prepare_new(source, directory, arguments)
        manifest.write_text(json.dumps(description, indent=2) + "\n")
    return directory


def parse_arguments():
    """Expose the same recall and concurrency controls as the benchmark runner."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--topk", type=int, default=100)
    parser.add_argument("--recall-nq", type=int, default=100)
    parser.add_argument("--nq", type=int, default=10)
    parser.add_argument("--concurrency", type=int, default=12)
    parser.add_argument("--percentages", default="5,10,15,20,30")
    parser.add_argument("--mask-mode", choices=("modulo", "random"), default="modulo")
    parser.add_argument("--seed", type=int, default=20260928)
    parser.add_argument("--bitset", action="append", default=[])
    parser.add_argument("--synthetic", action="store_true")
    args = parser.parse_args()
    args.percentages = [int(value) for value in args.percentages.split(",") if value]
    require(min(args.topk, args.recall_nq, args.nq, args.concurrency) > 0, "counts must be positive")
    return args


def main():
    """Print the final prepared directory after all files and exact truth are ready."""
    args = parse_arguments()
    args.cache.mkdir(parents=True, exist_ok=True)
    if args.synthetic and not args.dataset.exists():
        synthetic(args.dataset)
    print(prepare(args.dataset, args.cache, args), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logging.exception("dataset preparation failed; action=abort, no new completion manifest published")
        raise SystemExit(1)
