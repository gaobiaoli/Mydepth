#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

export PYTHONHASHSEED=42
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONPATH="/home/bgao491/Depth-Anything-3/src:/home/bgao491/S3-SAM3D-ToolKit/src${PYTHONPATH:+:${PYTHONPATH}}"
# extra_dataset_root="/mnt/priorbimda-data/s23_syncbim_area2_5_fill_slab_504"
extra_dataset_root="/mnt/priorbimda-data/s23_syncbim_area2_5_504"
# extra_dataset_root2="/mnt/priorbimda-data/s23_syncbim_area6_504"
output_dir="outputs/area1_syncbim_stride1-25-s41"

echo "[$(date '+%F %T')] 开始训练：${output_dir}"
python -u train.py \
    --seed 41 \
    --full-deterministic \
    --extra-dataset-root "${extra_dataset_root}" \
    --extra-dataset-stride 1 \
    --output "${output_dir}"

echo "[$(date '+%F %T')] 训练完成：${output_dir}"
