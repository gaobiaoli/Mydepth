#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

# Compare best.pt only: Area2-5 seed 42 has no latest.pt checkpoint.
python_bin="/home/bgao491/miniconda3/envs/priorbimda/bin/python"
export PYTHONPATH="/home/bgao491/Depth-Anything-3/src:/home/bgao491/S3-SAM3D-ToolKit/src${PYTHONPATH:+:${PYTHONPATH}}"

dry_run=false
if [[ ${1:-} == "--dry-run" && $# -eq 1 ]]; then
    dry_run=true
elif [[ $# -ne 0 ]]; then
    echo "Usage: bash eval_area25_area26.sh [--dry-run]" >&2
    exit 2
fi

# The 17 floor scenes with a wall-filled mesh in the existing benchmark.
wall_scenes=(
    train/1px train/759 train/7y3_1 train/ac2 train/b6b
    train/e9z train/hxp train/i5n train/i5n_1 train/px4_1
    test/7y3 test/d7n test/e9z_1 test/px4 test/px4_2
    test/q9v test/zsn
)

valid_result() {
    local path=$1 run=$2 mesh=$3 scene_count=$4
    jq -e --arg run "$run" --arg mesh "$mesh" --argjson scene_count "$scene_count" '
        (.checkpoint | split("/")[-2]) as $source
        | ($source == $run or
           ($run == "area1_syncbim_stride1-25-s42" and
            $source == "area1_syncbim_stride1"))
          and (.checkpoint | endswith("/best.pt"))
          and (.protocol.scenes | length == $scene_count)
          and (if $mesh == "obj"
               then .protocol.mesh == "registered BIMNet obj"
               else (.protocol.mesh == "registered BIMNet obj_wall_filled" or
                     .protocol.mesh == "registered wall-filled BIMNet OBJ")
               end)
          and (.overall.final.pixel_micro.count > 0)
    ' "$path" >/dev/null
}

evaluate() {
    local run=$1 mesh=$2
    local run_dir="outputs/${run}"
    local checkpoint="${run_dir}/best.pt"
    local output scene_count

    if [[ "$mesh" == "obj" ]]; then
        scene_count=25
        output="${run_dir}/zero_shot_all_metrics_best_obj.json"
        # This completed evaluation predates the common output filename.
        if [[ "$run" == "area1_syncbim_stride1-25-s42" ]]; then
            output="${run_dir}/zero_shot_full_metrics_best_obj.json"
        fi
    else
        scene_count=17
        output="${run_dir}/zero_shot_all_metrics_best_wall_filled_17.json"
        if [[ "$run" == "area1_syncbim_stride1-25-s42" ]]; then
            output="${run_dir}/zero_shot_all_metrics_best.json"
        fi
    fi

    if [[ ! -f "$checkpoint" ]]; then
        echo "Missing checkpoint: $checkpoint" >&2
        exit 1
    fi
    if [[ -e "$output" ]]; then
        if valid_result "$output" "$run" "$mesh" "$scene_count"; then
            echo "[$(date '+%F %T')] Reusing $output"
            return
        fi
        echo "Existing result has the wrong protocol or is incomplete: $output" >&2
        exit 1
    fi
    if [[ "$dry_run" == true ]]; then
        echo "Would evaluate: $checkpoint -> $output ($mesh, $scene_count scenes)"
        return
    fi

    echo "[$(date '+%F %T')] Evaluating $checkpoint ($mesh, $scene_count scenes)"
    if [[ "$mesh" == "obj" ]]; then
        "$python_bin" -u zero_shot_eval.py \
            --checkpoint "$checkpoint" \
            --output "$output" \
            --scenes all \
            --mesh-source obj
    else
        "$python_bin" -u zero_shot_eval.py \
            --checkpoint "$checkpoint" \
            --output "$output" \
            --scenes "${wall_scenes[@]}" \
            --mesh-source wall-filled
    fi
    echo "[$(date '+%F %T')] Saved $output"
}

# Complete the 25-scene OBJ comparison first; then the 17-scene wall-filled one.
for mesh in obj wall-filled; do
    for seed in 40 41 42; do
        for area in 25 26; do
            evaluate "area1_syncbim_stride1-${area}-s${seed}" "$mesh"
        done
    done
done

if [[ "$dry_run" == true ]]; then
    echo "Dry run complete; no evaluations were started"
else
    echo "[$(date '+%F %T')] All requested zero-shot evaluations are complete"
fi
