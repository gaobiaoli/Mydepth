#!/usr/bin/env bash

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

export PYTHONHASHSEED=42
export CUBLAS_WORKSPACE_CONFIG=:4096:8

extra_dataset_root="/mnt/priorbimda-data/s23_syncbim_area2_5_504"
output_dir="outputs/area1_syncbim_stride3"

echo "[$(date '+%F %T')] 将在 4 小时后开始训练：${output_dir}"

sleep 4h

echo "[$(date '+%F %T')] 开始训练：${output_dir}"

python -u train.py \
    --seed 42 \
    --full-deterministic \
    --extra-dataset-root "${extra_dataset_root}" \
    --extra-dataset-stride 3 \
    --output "${output_dir}"

echo "[$(date '+%F %T')] 训练完成：${output_dir}"