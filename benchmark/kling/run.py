#!/usr/bin/env python3
"""Build and benchmark this repository directly, using the Kling workload contract."""

import argparse
import csv
import datetime
import hashlib
import json
import logging
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parent.parent
ALGORITHM = "acorn" if (REPOSITORY / "faiss/IndexACORN.h").exists() else "navix"


def run(command, log, environment=None):
    """Append exact commands and output; preserve nonzero exits instead of emitting false results."""
    command = [str(value) for value in command]
    print(shlex.join(command), flush=True)
    with log.open("a") as output:
        output.write("\n$ " + shlex.join(command) + "\n")
        output.flush()
        subprocess.run(command, stdout=output, stderr=subprocess.STDOUT, env=environment, check=True)


def output(command):
    """Collect a short read-only command result."""
    return subprocess.check_output([str(value) for value in command], text=True).strip()


def require(condition, message):
    """Reject invalid configuration before expensive work."""
    if not condition:
        raise ValueError(message)


def parse_arguments():
    """Keep independent overrides explicit while inheriting current Knowhere measurement defaults."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, help="existing Knowhere HDF5 with train/test matrices")
    parser.add_argument("--parquet-dir", type=Path, help="Kling directory; prepares first six shards")
    parser.add_argument("--work-dir", type=Path, default=REPOSITORY / "build/kling")
    parser.add_argument("--result-dir", type=Path)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="synthetic 2048D functional test only")
    parser.add_argument("--percentages", default="5,10,15,20,30")
    parser.add_argument("--mask-mode", choices=("modulo", "random"), default="modulo")
    parser.add_argument("--bitset", action="append", default=[], help="NAME=packed exclusion bitset path")
    parser.add_argument("--gammas", default="10,1", help="ACORN gamma sweep; 1 is the control")
    args = parser.parse_args()
    args.algorithm = ALGORITHM
    require(args.jobs > 0, "--jobs must be positive")
    has_input = any((args.build_only, args.smoke, args.dataset, args.parquet_dir))
    require(has_input, "provide --dataset, --parquet-dir, --smoke or --build-only")
    require(args.dataset is None or args.dataset.is_file(), "HDF5 dataset does not exist")
    require(args.parquet_dir is None or args.parquet_dir.is_dir(), "Parquet directory does not exist")
    require(not (args.dataset and args.parquet_dir), "choose --dataset or --parquet-dir")
    require(not (args.smoke and (args.dataset or args.parquet_dir)), "--smoke cannot use real data inputs")
    args.work_dir = args.work_dir.resolve()
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    args.result_dir = args.result_dir or REPOSITORY / f"benchmark-results/independent-{stamp}-{os.getpid()}"
    args.result_dir.mkdir(parents=True, exist_ok=False)
    return args


def compiler():
    """Prefer an explicit CXX, then this host's available GCC 11, then system c++."""
    gcc11 = Path("/opt/rh/devtoolset-11/root/usr/bin/g++")
    return os.environ.get("CXX", str(gcc11) if gcc11.exists() else "c++")


def source_directory(algorithm, arguments):
    """Record the current revision and local changes, allowing experiments in this checkout."""
    source = REPOSITORY
    revision = output(["git", "-C", source, "rev-parse", "HEAD"])
    status = output(["git", "-C", source, "diff", "HEAD", "--stat"])
    (arguments.result_dir / f"{algorithm}-status.txt").write_text(status + "\n")
    run(["git", "-C", source, "diff", "HEAD"], arguments.result_dir / f"{algorithm}-changes.patch")
    (arguments.result_dir / f"{algorithm}-revision.txt").write_text(revision + "\n")
    return source


def build(algorithm, arguments):
    """Compile only the needed upstream library and benchmark; no install or Knowhere rebuild."""
    source = source_directory(algorithm, arguments)
    directory = arguments.work_dir / algorithm
    log = arguments.result_dir / f"{algorithm}-build.log"
    command = ["cmake", "-S", HERE, "-B", directory, f"-DFANN_ALGORITHM={algorithm}",
               f"-DFANN_SOURCE={source}", f"-DCMAKE_CXX_COMPILER={compiler()}"]
    if algorithm == "acorn":
        dependency = arguments.work_dir / "json"
        ensure_json(dependency, log)
        command.append(f"-DFETCHCONTENT_SOURCE_DIR_NLOHMANN_JSON={dependency}")
    run(command, log)
    run(["cmake", "--build", directory, "--target", "filtered_benchmark", "--parallel", arguments.jobs], log)
    return directory / "filtered_benchmark"


def ensure_json(directory, log):
    """Fetch ACORN's pinned configure-time dependency only when missing."""
    if not directory.exists():
        run(["git", "clone", "--depth", "1", "--branch", "v3.10.4",
             "https://github.com/nlohmann/json.git", directory], log)
    require(output(["git", "-C", directory, "rev-parse", "HEAD"]) ==
            "fec56a1a16c6e1c1b1f4e116a20e79398282626c", "unexpected nlohmann/json revision")


def environment(arguments):
    """Bound inner thread pools so 12 clients do not each launch an OpenMP team."""
    result = os.environ.copy()
    defaults = {"TOPK": "100", "RECALL_NQ": "100", "NQ": "10", "CONCURRENCY": "12",
                "SECONDS": "60", "BUILD_THREADS": "16", "M": "32", "EFC": "200",
                "EFS": "100,200,400,800,1600,3200", "GT_THREADS": "16"}
    for key, value in defaults.items():
        result.setdefault(f"FANN_{key}", value)
    result.update(OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", OMP_NUM_THREADS="1")
    if arguments.smoke:
        result.update(FANN_RECALL_NQ="10", FANN_NQ="2", FANN_CONCURRENCY="2", FANN_SECONDS="1",
                      FANN_BUILD_THREADS="4", FANN_EFS="100,400")
    return result


def resolve_dataset(arguments, log):
    """Use existing HDF5 directly, or reuse Knowhere's unmodified six-shard Parquet preparer."""
    dataset = arguments.dataset
    if arguments.smoke:
        dataset = arguments.work_dir / "synthetic-2048.hdf5"
    elif arguments.parquet_dir:
        script = HERE / "prepare_kling_parquet_dataset.py"
        key = output([sys.executable, script, "--parquet-dir", arguments.parquet_dir,
                      "--train-shard-count", "6", "--cache-key"])
        dataset = arguments.work_dir / f"kling-6shards-{key}.hdf5"
        run([sys.executable, script, "--parquet-dir", arguments.parquet_dir,
             "--train-shard-count", "6", "--output", dataset], log)
    require(dataset is not None, "provide --dataset, --parquet-dir, or --smoke")
    return dataset.resolve()


def prepare(arguments, env):
    """Prepare shared vectors/masks/truth once for all selected algorithms."""
    log = arguments.result_dir / "prepare.log"
    dataset = resolve_dataset(arguments, log)
    command = [sys.executable, HERE / "prepare.py", "--dataset", dataset,
               "--cache", arguments.work_dir / "data", "--percentages", arguments.percentages,
               "--mask-mode", arguments.mask_mode]
    for option, key, fallback in (("topk", "FANN_TOPK", "100"), ("recall-nq", "FANN_RECALL_NQ", "100"),
                                  ("nq", "FANN_NQ", "10"), ("concurrency", "FANN_CONCURRENCY", "12")):
        command.extend([f"--{option}", env.get(key, fallback)])
    for item in arguments.bitset:
        command.extend(["--bitset", item])
    command.extend(["--synthetic"] if arguments.smoke else [])
    prep_env = env.copy()
    prep_env["OPENBLAS_NUM_THREADS"] = env.get("FANN_GT_THREADS", "16")
    run(command, log, prep_env)
    directory = Path(log.read_text().splitlines()[-1])
    shutil.copyfile(directory / "manifest.json", arguments.result_dir / "data-manifest.json")
    return directory


def measure(algorithm, binary, context):
    """Build each graph once per gamma, then sweep all scenes and ef values in one process."""
    arguments, directory, env = context
    gammas = arguments.gammas.split(",") if algorithm == "acorn" else ["1"]
    for gamma in gammas:
        require(gamma.isdigit() and int(gamma) > 0, "gamma must be a positive integer")
        label = algorithm if algorithm == "navix" else f"acorn-gamma{gamma}"
        current = dict(env, FANN_GAMMA=gamma)
        (arguments.result_dir / f"{label}-settings.json").write_text(json.dumps(
            {key: value for key, value in current.items() if key.startswith("FANN_")}, indent=2) + "\n")
        command = [binary, directory, arguments.result_dir / f"{label}.csv"]
        run(command, arguments.result_dir / f"{label}-run.log", current)


def summarize(arguments):
    """Report only measured points meeting each recall threshold; do not interpolate claimed QPS."""
    lines = ["# Independent filtered ANN benchmark", "", f"Synthetic smoke only: {arguments.smoke}", "",
             "| Algorithm | Scene | Recall target | Measured recall | QPS | ef |",
             "|---|---|---:|---:|---:|---:|"]
    for path in sorted(arguments.result_dir.glob("*.csv")):
        with path.open() as source:
            rows = list(csv.DictReader(source))
        for scene in sorted({row["scene"] for row in rows}):
            for target in (0.90, 0.95, 0.98, 0.99):
                points = [row for row in rows if row["scene"] == scene and float(row["recall"]) >= target]
                best = max(points, key=lambda row: float(row["qps"]), default=None)
                values = f"{best['recall']} | {best['qps']} | {best['ef']}" if best else "not reached | — | —"
                lines.append(f"| {path.stem} | {scene} | {target} | {values} |")
    (arguments.result_dir / "summary.md").write_text("\n".join(lines) + "\n")


def main():
    """Respect priority order and leave build/data failures visible in per-stage logs."""
    args = parse_arguments()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    print(f"Results and logs: {args.result_dir}", flush=True)
    env = environment(args)
    (args.result_dir / "arguments.json").write_text(json.dumps(vars(args), default=str, indent=2) + "\n")
    files = sorted(HERE.glob("*")) + [REPOSITORY / "run_kling_benchmark.sh"]
    hashes = {str(path.relative_to(REPOSITORY)): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in files if path.is_file()}
    (args.result_dir / "benchmark-sha256.json").write_text(json.dumps(hashes, indent=2) + "\n")
    run(["lscpu"], args.result_dir / "cpu.txt")
    run(["git", "-C", REPOSITORY, "rev-parse", "HEAD"], args.result_dir / "repository-revision.txt")
    algorithms = [args.algorithm]
    directory = None
    for algorithm in algorithms:
        binary = build(algorithm, args)
        directory = run_measurements(algorithm, binary, (args, directory, env))
    print(f"Completed: {args.result_dir}", flush=True)


def run_measurements(algorithm, binary, context):
    """Allow a build-only invocation without importing data dependencies or creating fake results."""
    args, directory, env = context
    if not args.build_only:
        directory = directory or prepare(args, env)
        measure(algorithm, binary, (args, directory, env))
        summarize(args)
    return directory


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logging.exception("independent benchmark failed; action=abort, inspect the printed result directory logs")
        raise SystemExit(1)
