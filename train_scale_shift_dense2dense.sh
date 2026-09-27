#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

python_bin="/home/bgao491/miniconda3/envs/priorbimda/bin/python"
export PYTHONHASHSEED=42
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONPATH="/home/bgao491/Depth-Anything-3/src:/home/bgao491/S3-SAM3D-ToolKit/src${PYTHONPATH:+:${PYTHONPATH}}"

calibration_dir="outputs/train_onlyscale_shift_area2-6-s42"
calibration_checkpoint="${calibration_dir}/latest.pt"
area2_5_root="/mnt/priorbimda-data/s23_syncbim_area2_5_504"
area6_root="/mnt/priorbimda-data/s23_syncbim_area6_504"
da3_cache="/mnt/priorbimda-data/zero_shot_raw/da3_cache"
output_dir="outputs/train_scale_shift_dense2dense_noweight_area2-6-s42"
weighted_output_dir="outputs/train_scale_shift_dense2dense_area2-6-s42"

calibration_metrics="${calibration_dir}/zero_shot_all_metrics_latest_obj.json"
if [[ -s "${calibration_metrics}" ]]; then
    echo "[$(date '+%F %T')] 已有 scale-shift 全场景 OBJ 结果，跳过重复评测"
else
    echo "[$(date '+%F %T')] 评测 scale-shift latest.pt：全部 OBJ 场景"
    "${python_bin}" -u zero_shot_eval.py \
        --checkpoint "${calibration_checkpoint}" \
        --output "${calibration_metrics}" \
        --da3-cache "${da3_cache}" \
        --scenes all \
        --mesh-source obj
fi

echo "[$(date '+%F %T')] 训练冻结 scale-shift + dense2dense_noweight"
"${python_bin}" -u train_scale_shift_dense2dense_noweight.py \
    --seed 42 \
    --full-deterministic \
    --local-files-only \
    --epochs 6 \
    --batch-size 4 \
    --accumulation 4 \
    --num-workers 8 \
    --extra-dataset-root "${area2_5_root}" \
    --extra-dataset-root "${area6_root}" \
    --extra-dataset-stride 1 \
    --da3-cache "${da3_cache}" \
    --zero-shot \
    "$@" \
    --output "${output_dir}"

echo "[$(date '+%F %T')] 评测组合模型 latest.pt：全部 OBJ 场景"
"${python_bin}" -u zero_shot_eval.py \
    --checkpoint "${output_dir}/latest.pt" \
    --output "${output_dir}/zero_shot_all_metrics_latest_obj.json" \
    --da3-cache "${da3_cache}" \
    --scenes all \
    --mesh-source obj

echo "[$(date '+%F %T')] 评测组合模型 best.pt：全部 OBJ 场景"
"${python_bin}" -u zero_shot_eval.py \
    --checkpoint "${output_dir}/best.pt" \
    --output "${output_dir}/zero_shot_all_metrics_best_obj.json" \
    --da3-cache "${da3_cache}" \
    --scenes all \
    --mesh-source obj

echo "[$(date '+%F %T')] 训练冻结 scale-shift + dense2dense（加权 loss）"
"${python_bin}" -u train_scale_shift_dense2dense.py \
    --seed 42 \
    --full-deterministic \
    --local-files-only \
    --epochs 6 \
    --batch-size 4 \
    --accumulation 4 \
    --num-workers 8 \
    --extra-dataset-root "${area2_5_root}" \
    --extra-dataset-root "${area6_root}" \
    --extra-dataset-stride 1 \
    --da3-cache "${da3_cache}" \
    --zero-shot \
    "$@" \
    --output "${weighted_output_dir}"

echo "[$(date '+%F %T')] 评测加权组合模型 latest.pt：全部 OBJ 场景"
"${python_bin}" -u zero_shot_eval.py \
    --checkpoint "${weighted_output_dir}/latest.pt" \
    --output "${weighted_output_dir}/zero_shot_all_metrics_latest_obj.json" \
    --da3-cache "${da3_cache}" \
    --scenes all \
    --mesh-source obj

echo "[$(date '+%F %T')] 评测加权组合模型 best.pt：全部 OBJ 场景"
"${python_bin}" -u zero_shot_eval.py \
    --checkpoint "${weighted_output_dir}/best.pt" \
    --output "${weighted_output_dir}/zero_shot_all_metrics_best_obj.json" \
    --da3-cache "${da3_cache}" \
    --scenes all \
    --mesh-source obj

echo "[$(date '+%F %T')] 全部任务完成"
