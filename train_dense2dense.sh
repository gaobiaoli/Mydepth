#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

export PYTHONHASHSEED=42
export CUBLAS_WORKSPACE_CONFIG=:4096:8

extra_dataset_root="/mnt/priorbimda-data/s23_syncbim_area2_5_504"
output_dir="outputs/dense2dense_unbounded_syncbim_stride1"

echo "[$(date '+%F %T')] 开始训练：${output_dir}"
python -u train_dense2dense.py \
    --seed 42 \
    --full-deterministic \
    --local-files-only \
    --epochs 6 \
    --batch-size 4 \
    --accumulation 4 \
    --extra-dataset-root "${extra_dataset_root}" \
    --extra-dataset-stride 1 \
    --output "${output_dir}" \
    "$@"

echo "[$(date '+%F %T')] 训练完成：${output_dir}"
