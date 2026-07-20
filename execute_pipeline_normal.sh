#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/scripts/pipeline_common.sh"

first_existing_dir() {
    local candidate
    for candidate in "$@"; do
        if [[ -d "$candidate" ]]; then
            printf '%s\n' "$candidate"
            return 0
        fi
    done
    printf '%s\n' "$1"
}

require_dir() {
    local label="$1"
    local path="$2"
    if [[ ! -d "$path" ]]; then
        echo "Missing $label directory: $path" >&2
        return 1
    fi
}

require_file() {
    local label="$1"
    local path="$2"
    if [[ ! -f "$path" ]]; then
        echo "Missing $label file: $path" >&2
        return 1
    fi
}

scene_name="${scene_name:-scene_1}"

if [[ -z "${SCENE_ROOT:-}" ]]; then
    if [[ -n "${METU_CONDITION:-}" ]]; then
        SCENE_ROOT="$RGBX_ROOT/METU_VisTIR/$METU_CONDITION/$scene_name"
    else
        SCENE_ROOT="$RGBX_ROOT/METU_VisTIR/$scene_name"
    fi
fi

if [[ -z "${RGB_DIR:-}" ]]; then
    RGB_DIR="$(first_existing_dir \
        "$SCENE_ROOT/visible/images_518_518" \
        "$SCENE_ROOT/visible/images_1920_1080" \
        "$SCENE_ROOT/visible/images")"
fi

if [[ -z "${TARGET_DIR:-}" ]]; then
    TARGET_DIR="$(first_existing_dir \
        "$SCENE_ROOT/normal_generation" \
        "$SCENE_ROOT/normal/images" \
        "$SCENE_ROOT/normal")"
fi

MATCHER="${MATCHER:-loftr}"
MATCHING_SCRIPT="${NORMAL_MATCHING_SCRIPT:-semidense_matching.py}"
MATCH_THRESHOLDS=(0.15 0.3 0.5)
DENSIFICATION_EXTRA_ARGS=(--train_data_name NORMAL --pred_confidence_input 0)

if [[ -z "${PIPELINE_ROOT:-}" ]]; then
    if [[ -n "${METU_CONDITION:-}" ]]; then
        PIPELINE_ROOT="$RGBX_ROOT/demo/pipelines/METU_VisTIR/$METU_CONDITION/$scene_name/normal"
    else
        PIPELINE_ROOT="$RGBX_ROOT/demo/pipelines/METU_VisTIR/$scene_name/normal"
    fi
fi

MATCHING_ROOT="$PIPELINE_ROOT/matching"
DENS_ROOT="$PIPELINE_ROOT/dens"

ckpt="${NORMAL_DENSIFICATION_CKPT:-$CHECKPOINT_ROOT/densification_normal.pt}"

require_dir "normal-map scene root" "$SCENE_ROOT"
require_dir "RGB image" "$RGB_DIR"
require_dir "normal target image" "$TARGET_DIR"
require_file "normal densification checkpoint" "$ckpt"
require_file "$MATCHER matcher checkpoint" "$(matcher_checkpoint "$MATCHER")"
require_file "normal matching script" "$RGBX_ROOT/$MATCHING_SCRIPT"

echo "Running normal-map pipeline"
if [[ -n "${METU_CONDITION:-}" ]]; then
    echo "Condition:   $METU_CONDITION"
fi
echo "Scene:       $scene_name"
echo "Scene root:  $SCENE_ROOT"
echo "RGB images:  $RGB_DIR"
echo "Target X:    $TARGET_DIR"
echo "Output root: $PIPELINE_ROOT"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "DRY_RUN=1: path validation passed; skipping processing."
    exit 0
fi

dens_folders=()
for thr in "${MATCH_THRESHOLDS[@]}"; do
    matching_dir="$MATCHING_ROOT/thr_$thr"
    dens_dir="$DENS_ROOT/thr_$thr"

    run_rgbx_matching "$MATCHING_SCRIPT" "$MATCHER" --match_thr "$thr" \
        "$RGB_DIR" "$TARGET_DIR" "$matching_dir" \
        --processing_mode normal \
        --context_frames 3 \
        --region_sample_rate 0.10 \
        --pad 0

    run_densification_first "$matching_dir" "$RGB_DIR" "$dens_dir"
    dens_folders+=("dens/thr_$thr")
done

average_folders "$PIPELINE_ROOT" mean "${dens_folders[@]}"
