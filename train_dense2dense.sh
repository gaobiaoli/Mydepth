#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

export PYTHONHASHSEED=42
export CUBLAS_WORKSPACE_CONFIG=:4096:8

area2_5_root="/mnt/priorbimda-data/s23_syncbim_area2_5_504"
area6_root="/mnt/priorbimda-data/s23_syncbim_area6_504"

sleep 5h
run_experiment() {
    local name="$1"
    local trainer="$2"
    local output_dir="$3"
    shift 3

    echo "[$(date '+%F %T')] 开始训练 ${name}：${output_dir}"
    python -u "${trainer}" \
        --seed 42 \
        --full-deterministic \
        --local-files-only \
        --epochs 6 \
        --batch-size 4 \
        --accumulation 4 \
        --extra-dataset-root "${area2_5_root}" \
        --extra-dataset-root "${area6_root}" \
        --extra-dataset-stride 1 \
        --zero-shot \
        "$@" \
        --output "${output_dir}"

    echo "[$(date '+%F %T')] 评测 ${name} latest.pt：全部 OBJ 场景"
    python -u zero_shot_eval.py \
        --checkpoint "${output_dir}/latest.pt" \
        --output "${output_dir}/zero_shot_all_metrics_latest_obj.json" \
        --scenes all \
        --mesh-source obj

    echo "[$(date '+%F %T')] 评测 ${name} best.pt：全部 OBJ 场景"
    python -u zero_shot_eval.py \
        --checkpoint "${output_dir}/best.pt" \
        --output "${output_dir}/zero_shot_all_metrics_best_obj.json" \
        --scenes all \
        --mesh-source obj

    echo "[$(date '+%F %T')] ${name} 全部评测完成"
}

run_experiment \
    dense2dense \
    train_dense2dense.py \
    outputs/dense2dense_unbounded_syncbim_area26_stride1 \
    "$@"

run_experiment \
    dense2dense_noweight \
    train_dense2dense_noweight.py \
    outputs/dense2dense_noweight_unbounded_syncbim_area26_stride1 \
    "$@"
