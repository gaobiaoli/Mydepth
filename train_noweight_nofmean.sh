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

# Each trainer evaluates Area1 test and the default three zero-shot scenes with
# best.pt. The two explicit evaluations below use the same 25-scene OBJ protocol.
run_experiment() {
    local name="$1"
    local trainer="$2"
    local output_dir="$3"

    echo "[$(date '+%F %T')] Training ${name}: ${output_dir}"
    "${python_bin}" -u "${trainer}" \
        --dataset-root "${area1_root}" \
        --s23-root "${s23_root}" \
        --extra-dataset-root "${area2_5_root}" \
        --extra-dataset-root "${area6_root}" \
        --extra-dataset-stride 1 \
        --seed 42 \
        --full-deterministic \
        --local-files-only \
        --epochs 6 \
        --batch-size 4 \
        --accumulation 4 \
        --num-workers 8 \
        --device cuda \
        --zero-shot \
        --output "${output_dir}"

    echo "[$(date '+%F %T')] Evaluating 25 OBJ scenes with latest.pt: ${name}"
    "${python_bin}" -u zero_shot_eval.py \
        --checkpoint "${output_dir}/latest.pt" \
        --output "${output_dir}/zero_shot_all_metrics_latest_obj.json" \
        --scenes all \
        --mesh-source obj

    echo "[$(date '+%F %T')] Evaluating 25 OBJ scenes with best.pt: ${name}"
    "${python_bin}" -u zero_shot_eval.py \
        --checkpoint "${output_dir}/best.pt" \
        --output "${output_dir}/zero_shot_all_metrics_best_obj.json" \
        --scenes all \
        --mesh-source obj

    echo "[$(date '+%F %T')] ${name} completed"
}

run_experiment \
    noweight_nofmean \
    train_noweight_nofmean_noequal.py \
    outputs/ablation_area26_s42_noequal_noweight_nofmean

run_experiment \
    noweight_nofmean_meanscale \
    train_noweight_nofmean_meanscale_noequal.py \
    outputs/ablation_area26_s42_noequal_noweight_nofmean_meanscale
