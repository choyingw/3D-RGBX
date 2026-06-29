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
        "$SCENE_ROOT/visible/images_1920_1080" \
        "$SCENE_ROOT/visible/images" \
        "$SCENE_ROOT/visible/train" \
        "$SCENE_ROOT/rgb/images" \
        "$SCENE_ROOT/rgb/train")"
fi

if [[ -z "${TARGET_DIR:-}" ]]; then
    TARGET_DIR="$(first_existing_dir \
        "$SCENE_ROOT/infrared/images_1920_1080" \
        "$SCENE_ROOT/infrared/images" \
        "$SCENE_ROOT/infrared/train" \
        "$SCENE_ROOT/infared/images_1920_1080" \
        "$SCENE_ROOT/infared/images" \
        "$SCENE_ROOT/thermal/images_1920_1080" \
        "$SCENE_ROOT/thermal/images" \
        "$SCENE_ROOT/thermal/train")"
fi

MATCHER="${MATCHER:-xoftr}"
FILTER_MATCHER="${FILTER_MATCHER:-xoftr}"
MATCH_THRESHOLDS=(0.15 0.3 0.5)
SAMPLE_RATES=(0.85 0.75 0.65)

if [[ -z "${PIPELINE_ROOT:-}" ]]; then
    if [[ -n "${METU_CONDITION:-}" ]]; then
        PIPELINE_ROOT="$RGBX_ROOT/demo/pipelines/METU_VisTIR/$METU_CONDITION/$scene_name"
    else
        PIPELINE_ROOT="$RGBX_ROOT/demo/pipelines/METU_VisTIR/$scene_name"
    fi
fi
MATCHING_ROOT="$PIPELINE_ROOT/matching"
DENS_ROOT="$PIPELINE_ROOT/dens"
MEAN_DIR="$PIPELINE_ROOT/mean"
FILTERED_DIR="$PIPELINE_ROOT/filtered"
REFINED_ROOT="$PIPELINE_ROOT/refined"

ckpt="${RGBT_DENSIFICATION_CKPT:-$CHECKPOINT_ROOT/densification_rgbt.pt}"

require_dir "METU scene root" "$SCENE_ROOT"
require_dir "METU RGB image" "$RGB_DIR"
require_dir "METU thermal/target image" "$TARGET_DIR"
require_dir "COLMAP sparse model" "$SCENE_ROOT/sparse/0"
require_file "densification checkpoint" "$ckpt"
require_file "$MATCHER matcher checkpoint" "$(matcher_checkpoint "$MATCHER")"
require_file "$FILTER_MATCHER filter matcher checkpoint" "$(matcher_checkpoint "$FILTER_MATCHER")"

echo "Running METU-VisTIR RGBT pipeline"
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

GS_ROOT="${GS_ROOT:-$RGBX_ROOT/gaussian-splatting-archive}"
GS_THERMAL_DIR="$PIPELINE_ROOT/refined_mean"
GS_MODEL_DIR="${GS_MODEL_DIR:-$PIPELINE_ROOT/gs_model}"
GS_RENDER_DIR="${GS_RENDER_DIR:-$PIPELINE_ROOT/gs_rendered}"

require_file "3DGS training script" "$GS_ROOT/train.py"
require_file "3DGS rendering script" "$GS_ROOT/render.py"
require_dir "refined thermal" "$GS_THERMAL_DIR"

echo "Running 3DGS RGBT training and rendering for METU-VisTIR $scene_name"
echo "Thermal:     $GS_THERMAL_DIR"
echo "Model dir:   $GS_MODEL_DIR"
echo "Render dir:  $GS_RENDER_DIR"

(
    cd "$GS_ROOT"
    "$PYTHON_BIN" train.py \
        -s "$SCENE_ROOT" \
        --thermal "$GS_THERMAL_DIR" \
        --image "$RGB_DIR" \
        -m "$GS_MODEL_DIR"

    "$PYTHON_BIN" render.py -m "$GS_MODEL_DIR" --render_output "$GS_RENDER_DIR"
)
