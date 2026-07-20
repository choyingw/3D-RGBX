#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/scripts/pipeline_common.sh"

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

scene_name="${scene_name:-09-28-16-48-17-1}"

NIR_ROOT="${NIR_ROOT:-$RGBX_ROOT/NIR_colmap}"
SCENE_ROOT="${SCENE_ROOT:-$NIR_ROOT/$scene_name}"
RGB_DIR="${RGB_DIR:-$SCENE_ROOT/rgb_fil}"
TARGET_DIR="${TARGET_DIR:-$SCENE_ROOT/nir_fil_aug}"

MATCHER="${MATCHER:-xoftr}"
FILTER_MATCHER="${FILTER_MATCHER:-$MATCHER}"
MATCH_THRESHOLDS=(0.15 0.3 0.5)
SAMPLE_RATES=(0.85 0.75 0.65)

PIPELINE_ROOT="${PIPELINE_ROOT:-$RGBX_ROOT/demo/pipelines/NIR_colmap/$scene_name}"
MATCHING_ROOT="$PIPELINE_ROOT/matching"
DENS_ROOT="$PIPELINE_ROOT/dens"
MEAN_DIR="$PIPELINE_ROOT/mean"
FILTERED_DIR="$PIPELINE_ROOT/filtered"
REFINED_ROOT="$PIPELINE_ROOT/refined"

ckpt="${NIR_DENSIFICATION_CKPT:-$CHECKPOINT_ROOT/densification_nir.pt}"

require_dir "NIR scene root" "$SCENE_ROOT"
require_dir "RGB image" "$RGB_DIR"
require_dir "NIR target image" "$TARGET_DIR"
require_dir "COLMAP sparse model" "$SCENE_ROOT/sparse/0"
require_file "NIR densification checkpoint" "$ckpt"
require_file "$MATCHER matcher checkpoint" "$(matcher_checkpoint "$MATCHER")"
require_file "$FILTER_MATCHER filter matcher checkpoint" "$(matcher_checkpoint "$FILTER_MATCHER")"
require_file "semi-dense matching script" "$RGBX_ROOT/semidense_matching.py"

echo "Running NIR pipeline"
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
        "$RGB_DIR" "$TARGET_DIR" "$matching_dir" \
        --processing_mode nir

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
GS_NIR_DIR="$PIPELINE_ROOT/refined_mean"
GS_MODEL_DIR="${GS_MODEL_DIR:-$PIPELINE_ROOT/gs_model}"
GS_RENDER_DIR="${GS_RENDER_DIR:-$PIPELINE_ROOT/gs_rendered}"

if [[ ! -f "$GS_ROOT/train.py" ]]; then
    echo "Missing 3DGS training script: $GS_ROOT/train.py" >&2
    exit 1
fi
if [[ ! -f "$GS_ROOT/render.py" ]]; then
    echo "Missing 3DGS rendering script: $GS_ROOT/render.py" >&2
    exit 1
fi
if [[ ! -d "$GS_NIR_DIR" ]]; then
    echo "Missing refined NIR directory: $GS_NIR_DIR" >&2
    exit 1
fi

echo "Running 3DGS NIR training and rendering for $scene_name"
echo "Scene root:  $SCENE_ROOT"
echo "RGB images:  $RGB_DIR"
echo "NIR:         $GS_NIR_DIR"
echo "Model dir:   $GS_MODEL_DIR"
echo "Render dir:  $GS_RENDER_DIR"

(
    cd "$GS_ROOT"
    "$PYTHON_BIN" train.py \
        -s "$SCENE_ROOT" \
        --thermal "$GS_NIR_DIR" \
        --image "$RGB_DIR" \
        -m "$GS_MODEL_DIR"

    "$PYTHON_BIN" render.py -m "$GS_MODEL_DIR" --render_output "$GS_RENDER_DIR"
)
