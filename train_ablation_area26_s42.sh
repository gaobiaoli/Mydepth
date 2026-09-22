#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

python_bin="/home/bgao491/miniconda3/envs/priorbimda/bin/python"
export PYTHONHASHSEED=42
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONPATH="/home/bgao491/Depth-Anything-3/src:/home/bgao491/S3-SAM3D-ToolKit/src${PYTHONPATH:+:${PYTHONPATH}}"

area1_root="/mnt/priorbimda-data/area1_priorbimda_504"
area2_5_root="/mnt/priorbimda-data/s23_syncbim_area2_5_504"
area6_root="/mnt/priorbimda-data/s23_syncbim_area6_504"
s23_root="/home/bgao491/Stanford2D3DS/no_xyz"

# train_noweight_pixelloss.py runs Area1 test and the default three-scene
# zero-shot evaluation with best.pt. The full-scene OBJ evaluation below uses
# the final epoch, matching the Area2-6 seed-42 baseline protocol.
output_dir="outputs/ablation_area26_s42_noweight_pixelloss"

echo "[$(date '+%F %T')] Training noweight_pixelloss: ${output_dir}"
"${python_bin}" -u train_noweight_pixelloss.py \
    --dataset-root "${area1_root}" \
    --s23-root "${s23_root}" \
    --extra-dataset-root "${area2_5_root}" \
    --extra-dataset-root "${area6_root}" \
    --extra-dataset-stride 1 \
    --seed 42 \
    --full-deterministic \
    --epochs 6 \
    --batch-size 4 \
    --accumulation 4 \
    --num-workers 8 \
    --device cuda \
    --zero-shot \
    --output "${output_dir}"

echo "[$(date '+%F %T')] Evaluating 25 OBJ scenes: noweight_pixelloss"
"${python_bin}" -u zero_shot_eval.py \
    --checkpoint "${output_dir}/latest.pt" \
    --output "${output_dir}/zero_shot_all_metrics_latest_obj.json" \
    --scenes all \
    --mesh-source obj

echo "[$(date '+%F %T')] noweight_pixelloss ablation completed"

# This run differs from the completed noweight experiment only by disabling
# the equivariance loss and its perturbed-depth scale prediction.
output_dir="outputs/ablation_area26_s42_noequal_noweight"

echo "[$(date '+%F %T')] Training noequal_noweight: ${output_dir}"
"${python_bin}" -u train_noequal_noweight.py \
    --dataset-root "${area1_root}" \
    --s23-root "${s23_root}" \
    --extra-dataset-root "${area2_5_root}" \
    --extra-dataset-root "${area6_root}" \
    --extra-dataset-stride 1 \
    --seed 42 \
    --full-deterministic \
    --epochs 6 \
    --batch-size 4 \
    --accumulation 4 \
    --num-workers 8 \
    --device cuda \
    --zero-shot \
    --output "${output_dir}"

echo "[$(date '+%F %T')] Evaluating 25 OBJ scenes: noequal_noweight"
"${python_bin}" -u zero_shot_eval.py \
    --checkpoint "${output_dir}/latest.pt" \
    --output "${output_dir}/zero_shot_all_metrics_latest_obj.json" \
    --scenes all \
    --mesh-source obj

echo "[$(date '+%F %T')] All queued ablations completed"
