#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

export PYTHONHASHSEED=42
export CUBLAS_WORKSPACE_CONFIG=:4096:8

for run_id in assamble1; do
    output_dir="outputs/raw${run_id}"

    echo "[$(date '+%F %T')] 开始训练：${output_dir}"
    python -u train.py \
        --seed 42 \
        --full-deterministic \
        --output "${output_dir}"

    echo "[$(date '+%F %T')] 训练完成：${output_dir}"
done
