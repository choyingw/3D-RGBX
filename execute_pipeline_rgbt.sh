#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/scripts/pipeline_common.sh"

scene_name="Building"

SCENE_ROOT="./RGBT-Scenes/$scene_name"
RGB_DIR="$SCENE_ROOT/rgb/train"
TARGET_DIR="$SCENE_ROOT/thermal_aug_1chan/train"

MATCHER="loftr"
FILTER_MATCHER="xoftr"
MATCH_THRESHOLDS=(0.10 0.13 0.15)
SAMPLE_RATES=(0.85 0.75 0.65)

PIPELINE_ROOT="$RGBX_ROOT/demo/pipelines/$scene_name"
MATCHING_ROOT="$PIPELINE_ROOT/matching"
DENS_ROOT="$PIPELINE_ROOT/dens"
MEAN_DIR="$PIPELINE_ROOT/mean"
FILTERED_DIR="$PIPELINE_ROOT/filtered"
REFINED_ROOT="$PIPELINE_ROOT/refined"

ckpt="${RGBT_DENSIFICATION_CKPT:-$CHECKPOINT_ROOT/densification_rgbt.pt}"

dens_folders=()
for thr in "${MATCH_THRESHOLDS[@]}"; do
    matching_dir="$MATCHING_ROOT/thr_$thr"
    dens_dir="$DENS_ROOT/thr_$thr"

    run_rgbx_matching semidense_matching.py "$MATCHER" --match_thr "$thr" \
        "$RGB_DIR" "$TARGET_DIR" "$matching_dir"

    run_densification_first "$matching_dir" "$RGB_DIR" "$dens_dir"
    dens_folders+=("dens/thr_$thr")
done

average_folders "$PIPELINE_ROOT" mean "${dens_folders[@]}"

filter_folder "$RGB_DIR" "$MEAN_DIR" "$FILTERED_DIR" "$FILTER_MATCHER"

base_matching_dir="$MATCHING_ROOT/thr_${MATCH_THRESHOLDS[0]}"
refined_folders=()
for sample_rate in "${SAMPLE_RATES[@]}"; do
    refined_dir="$REFINED_ROOT/sample_$sample_rate"
    run_densification_second "$base_matching_dir" "$FILTERED_DIR" "$RGB_DIR" "$refined_dir" "$sample_rate"
    refined_folders+=("refined/sample_$sample_rate")
done

average_folders "$PIPELINE_ROOT" refined_mean "${refined_folders[@]}"
