#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

export PYTHONHASHSEED=42
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# --------------------------------------------------
# Extra datasets
# --------------------------------------------------

extra_dataset_base="/root/autodl-tmp/2d3d-mde"
areas=(2 3 4 5 6)

extra_dataset_args=()

for area in "${areas[@]}"; do
    dataset_root="${extra_dataset_base}/area_${area}"

    if [[ ! -d "${dataset_root}" ]]; then
        echo "ERROR: dataset not found: ${dataset_root}"
        exit 1
    fi

    extra_dataset_args+=(
        --extra-dataset-root "${dataset_root}"
    )
done

# --------------------------------------------------
# Training
# --------------------------------------------------

for run_id in train_full1; do
    output_dir="outputs/${run_id}"

    echo "[$(date '+%F %T')] 开始训练：${output_dir}"
    echo "Extra datasets:"
    for area in "${areas[@]}"; do
        echo "  ${extra_dataset_base}/area_${area}"
    done

    python -u train.py \
        --dataset-root /root/autodl-tmp/area1_priorbimda_504 \
        --s23-root /root/autodl-tmp \
        --num-workers 12 \
        --seed 42 \
        --full-deterministic \
        "${extra_dataset_args[@]}" \
        --extra-dataset-stride 1 \
        --output "${output_dir}" \
        --no-zero-shot \
        --batch-size 16 \
        --accumulation 1
    echo "[$(date '+%F %T')] 训练完成：${output_dir}"
done