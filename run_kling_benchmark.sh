#!/usr/bin/env bash
# Build and benchmark this checkout; keep dependencies and outputs inside it.
set -eEuo pipefail
readonly REPOSITORY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly ENV_DIR="${REPOSITORY_DIR}/build/kling/venv"
trap 'status=$?; echo "ERROR: line=${LINENO} status=${status}; benchmark aborted" >&2; exit "${status}"' ERR

# Bootstrap an isolated environment once; avoid changing the user's system Python.
prepare_python() {
    if [[ ! -x "${ENV_DIR}/bin/python" ]]; then
        "${FANN_BOOTSTRAP_PYTHON:-python3}" -m venv "${ENV_DIR}"
    fi
    "${ENV_DIR}/bin/python" -m pip install --only-binary=:all: --retries 0 \
        --index-url "${FANN_PYPI_INDEX_URL:-https://pypi.org/simple}" \
        -r "${REPOSITORY_DIR}/benchmark/kling/requirements.txt"
}

# No arguments means the fixed online Kling run, using the recorded Knowhere data location.
if [[ $# -eq 0 ]]; then
    export FANN_TOPK=100 FANN_RECALL_NQ=100 FANN_NQ=10 FANN_CONCURRENCY=12
    export FANN_SECONDS=60 FANN_BUILD_THREADS=16 FANN_GT_THREADS=16
    export FANN_M=32 FANN_EFC=200 FANN_EFS=100,200,400,800,1600,3200
    set -- --parquet-dir /media/nvme1n1/huayelin/knowhere/datasets/kling_1b \
        --percentages 5,10,15,20,30 --mask-mode modulo --jobs 4
fi

prepare_python
exec "${ENV_DIR}/bin/python" "${REPOSITORY_DIR}/benchmark/kling/run.py" "$@"
