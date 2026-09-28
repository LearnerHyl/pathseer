#!/usr/bin/env python3
"""Materialize ordered Kling Parquet shards as one normalized HDF5 matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def source_files(directory: Path, shard_count: int) -> tuple[list[Path], Path]:
    train_paths = [directory / f"train-{shard:02d}-of-1000.parquet" for shard in range(shard_count)]
    test_path = directory / "test.parquet"

    if any(not path.is_file() for path in train_paths):
        raise ValueError(f"one or more of the first {shard_count} training shards are missing")
    if not test_path.is_file():
        raise ValueError(f"test.parquet does not exist in {directory}")
    return train_paths, test_path


def fingerprint(paths: list[Path]) -> list[dict[str, int | str]]:
    result: list[dict[str, int | str]] = []

    for path in paths:
        stat = path.stat()
        result.append({"path": str(path.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns})
    return result


def manifest(directory: Path, column: str, shard_count: int) -> dict[str, object]:
    train_paths, test_path = source_files(directory, shard_count)
    result: dict[str, object] = {
        "version": 1,
        "vector_column": column,
        "normalize_l2": True,
        "sources": fingerprint(train_paths + [test_path]),
    }
    return result


def cache_key(description: dict[str, object]) -> str:
    encoded = json.dumps(description, sort_keys=True, separators=(",", ":")).encode()
    result = hashlib.sha256(encoded).hexdigest()[:12]
    return result


def vector_matrix(values: pa.Array, path: Path, column: str) -> np.ndarray:
    result = np.empty((0, 0), dtype=np.float32)
    dimension = 0

    if values.null_count:
        raise ValueError(f"null vector found in {path}:{column}")
    if pa.types.is_fixed_size_list(values.type):
        dimension = values.type.list_size
    elif pa.types.is_list(values.type) or pa.types.is_large_list(values.type):
        offsets = values.offsets.to_numpy(zero_copy_only=False)
        lengths = offsets[1:] - offsets[:-1]
        if not len(lengths) or not np.all(lengths == lengths[0]) or int(lengths[0]) <= 0:
            raise ValueError(f"vectors must have one positive dimension in {path}:{column}")
        dimension = int(lengths[0])
    else:
        raise ValueError(f"expected list<float> in {path}:{column}, got {values.type}")

    flat = values.flatten().to_numpy(zero_copy_only=False)
    result = np.asarray(flat, dtype=np.float32).reshape(len(values), dimension)
    if not np.isfinite(result).all():
        raise ValueError(f"NaN or infinity found in {path}:{column}")
    result = np.array(result, dtype=np.float32, order="C", copy=True)
    return result


def append_parquet(output: h5py.File, name: str, paths: list[Path], column: str,
                   batch_size: int, expected_dimension: int | None) -> tuple[int, int]:
    dataset = None
    rows = 0
    dimension = expected_dimension
    expected_rows = sum(pq.ParquetFile(path).metadata.num_rows for path in paths)

    for path in paths:
        parquet = pq.ParquetFile(path)
        if column not in parquet.schema_arrow.names:
            raise ValueError(f"vector column '{column}' not found in {path}")
        for batch in parquet.iter_batches(batch_size=batch_size, columns=[column]):
            matrix = vector_matrix(batch.column(0), path, column)
            norms = np.linalg.norm(matrix, axis=1)
            nonzero = norms > 0
            matrix[nonzero] /= norms[nonzero, np.newaxis]
            if dimension is None:
                dimension = matrix.shape[1]
            elif matrix.shape[1] != dimension:
                raise ValueError(f"dimension mismatch in {path}: expected {dimension}, got {matrix.shape[1]}")
            if dataset is None:
                dataset = output.create_dataset(name, shape=(expected_rows, dimension), dtype="float32")
            dataset[rows:rows + len(matrix)] = matrix
            rows += len(matrix)
        print(f"prepared {name}: {path.name}, total_rows={rows}", flush=True)

    if dataset is None or dimension is None or rows != expected_rows:
        raise ValueError(f"no vectors were read for {name}")
    return rows, dimension


def prepare(directory: Path, output_path: Path, column: str, batch_size: int,
            shard_count: int) -> dict[str, object]:
    train_paths, test_path = source_files(directory, shard_count)
    expected = manifest(directory, column, shard_count)
    manifest_path = output_path.with_suffix(output_path.suffix + ".manifest.json")
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    result: dict[str, object] = {}

    if output_path.is_file() and manifest_path.is_file() and json.loads(manifest_path.read_text()) == expected:
        with h5py.File(output_path, "r") as cached:
            result = {
                "cached": True,
                "path": str(output_path),
                "train_rows": int(cached["train"].shape[0]),
                "test_rows": int(cached["test"].shape[0]),
                "dimension": int(cached["train"].shape[1]),
                "train_shards": len(train_paths),
            }
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path.unlink(missing_ok=True)
        with h5py.File(temporary_path, "w") as output:
            train_rows, dimension = append_parquet(output, "train", train_paths, column, batch_size, None)
            test_rows, _ = append_parquet(output, "test", [test_path], column, batch_size, dimension)
            output.attrs["train_shards"] = len(train_paths)
            output.attrs["normalized_l2"] = True
        temporary_path.replace(output_path)
        manifest_path.write_text(json.dumps(expected, indent=2, sort_keys=True) + "\n")
        result = {
            "cached": False,
            "path": str(output_path),
            "train_rows": train_rows,
            "test_rows": test_rows,
            "dimension": dimension,
            "train_shards": len(train_paths),
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare Kling shards for the Curator benchmark")
    parser.add_argument("--parquet-dir", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--cache-key", action="store_true")
    parser.add_argument("--vector-column", default="emb")
    parser.add_argument("--batch-size", type=int, default=16384)
    parser.add_argument("--train-shard-count", type=int, default=10)
    arguments = parser.parse_args()
    directory = arguments.parquet_dir.resolve()
    description = manifest(directory, arguments.vector_column, arguments.train_shard_count)

    if arguments.cache_key:
        print(cache_key(description))
    elif arguments.output is not None:
        result = prepare(directory, arguments.output.resolve(), arguments.vector_column,
                         arguments.batch_size, arguments.train_shard_count)
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        parser.error("--output is required unless --cache-key is used")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
