#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

export PYTHONHASHSEED=42
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONPATH="/home/bgao491/Depth-Anything-3/src:/home/bgao491/S3-SAM3D-ToolKit/src${PYTHONPATH:+:${PYTHONPATH}}"

area2_5_root="/mnt/priorbimda-data/s23_syncbim_area2_5_504"
area6_root="/mnt/priorbimda-data/s23_syncbim_area6_504"
scannet_root="/mnt/priorbimda-data/scannet_syncbim_504"
output_dir="outputs/area1_syncbim_full_stride1"

# ScanNet 标签由 ScanNetDatasetAdapter 自动转换，与原 S23 loss 兼容。
echo "[$(date '+%F %T')] 开始训练：${output_dir}"
python -u train.py \
    --seed 42 \
    --full-deterministic \
    --local-files-only \
    --epochs 6 \
    --batch-size 4 \
    --accumulation 4 \
    --extra-dataset-root "${area2_5_root}" \
    --extra-dataset-root "${area6_root}" \
    --extra-dataset-root "${scannet_root}" \
    --extra-dataset-stride 1 \
    --output "${output_dir}" \
    "$@"

echo "[$(date '+%F %T')] 训练完成：${output_dir}"
