#!/usr/bin/env bash

RGBX_ROOT="${RGBX_ROOT:-/path/to/3D-RGBX}"
RGBX_ROOT="${RGBX_ROOT%/}"
DENSIFICATION_ROOT="${DENSIFICATION_ROOT:-$RGBX_ROOT/densification}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-$RGBX_ROOT/checkpoints}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

load_dav2="${load_dav2:-0}"
resolution="${resolution:-3}"
pred_confidence_input="${pred_confidence_input:-1}"
multi_resolution_learnable_gradients_weights="${multi_resolution_learnable_gradients_weights:-uniform}"
optim_layer_input_clamp="${optim_layer_input_clamp:-1.0}"
depth_activation_format="${depth_activation_format:-exp}"
max_depth="${max_depth:-300.0}"
whiten_sparse_depths="${whiten_sparse_depths:-1}"
backbone="${backbone:-rgbd}"
gpus="${gpus:-0}"

matcher_checkpoint() {
    local method="$1"
    case "$method" in
        xoftr)
            printf '%s\n' "$CHECKPOINT_ROOT/weights_xoftr_640.ckpt"
            ;;
        loftr)
            printf '%s\n' "$CHECKPOINT_ROOT/minima_loftr.ckpt"
            ;;
        sp_lg)
            printf '%s\n' "$CHECKPOINT_ROOT/minima_lightglue.pth"
            ;;
        roma)
            printf '%s\n' "$CHECKPOINT_ROOT/minima_roma.pth"
            ;;
        *)
            echo "Unknown matcher: $method" >&2
            return 1
            ;;
    esac
}

run_rgbx_matching() {
    local script_name="$1"
    local method="$2"
    local threshold_flag="$3"
    local threshold="$4"
    local fold1="$5"
    local fold2="$6"
    local save_dir="$7"

    mkdir -p "$(dirname "$save_dir")"
    (
        cd "$RGBX_ROOT"
        "$PYTHON_BIN" "$script_name" \
            --fold1 "$fold1" \
            --fold2 "$fold2" \
            --save_dir "$save_dir" \
            --method "$method" \
            --ckpt "$(matcher_checkpoint "$method")" \
            "$threshold_flag" "$threshold"
    )
}

run_densification_first() {
    local sparse_dir="$1"
    local rgb_dir="$2"
    local save_path="$3"

    (
        cd "$RGBX_ROOT"
        "$PYTHON_BIN" densification/src/first_densi.py \
            --max_depth "$max_depth" --data_normalize_median 1 \
            --num_resolution "$resolution" \
            --multi_resolution_learnable_gradients_weights "$multi_resolution_learnable_gradients_weights" \
            --load_dav2 "$load_dav2" \
            --gpus "$gpus" \
            --GRU_iters 1 \
            --optim_layer_input_clamp "$optim_layer_input_clamp" \
            --depth_activation_format "$depth_activation_format" \
            --whiten_sparse_depths "$whiten_sparse_depths" \
            --gru_internal_whiten_method median \
            --backbone_mode "$backbone" \
            --pred_confidence_input "$pred_confidence_input" \
            --pretrain "$ckpt" \
            --ori "$sparse_dir" \
            --rgb "$rgb_dir" \
            --save_path "$save_path"
    )
}

run_densification_second() {
    local sparse_dir="$1"
    local filtered_dir="$2"
    local rgb_dir="$3"
    local save_path="$4"
    local sample_rate="$5"

    (
        cd "$RGBX_ROOT"
        "$PYTHON_BIN" densification/src/second_densi.py \
            --max_depth "$max_depth" --data_normalize_median 1 \
            --num_resolution "$resolution" \
            --multi_resolution_learnable_gradients_weights "$multi_resolution_learnable_gradients_weights" \
            --load_dav2 "$load_dav2" \
            --gpus "$gpus" \
            --GRU_iters 1 \
            --optim_layer_input_clamp "$optim_layer_input_clamp" \
            --depth_activation_format "$depth_activation_format" \
            --whiten_sparse_depths "$whiten_sparse_depths" \
            --gru_internal_whiten_method median \
            --backbone_mode "$backbone" \
            --pred_confidence_input "$pred_confidence_input" \
            --pretrain "$ckpt" \
            --ori "$sparse_dir" \
            --filtered "$filtered_dir" \
            --rgb "$rgb_dir" \
            --save_path "$save_path" \
            --sample_rate "$sample_rate"
    )
}

average_folders() {
    local parent="$1"
    local save_folder="$2"
    shift 2

    if [[ "$#" -ne 3 ]]; then
        echo "level_mean.py expects exactly 3 input folders, got $#." >&2
        return 1
    fi

    local args=(--parent "$parent" --save_folder "$save_folder")
    local idx=1
    local folder
    for folder in "$@"; do
        args+=("--f${idx}" "$folder")
        idx=$((idx + 1))
    done

    (
        cd "$RGBX_ROOT"
        "$PYTHON_BIN" level_mean.py "${args[@]}"
    )
}

filter_folder() {
    local rgb_dir="$1"
    local sparse_dir="$2"
    local save_dir="$3"
    local method="$4"

    (
        cd "$RGBX_ROOT"
        "$PYTHON_BIN" filtering.py \
            --fold1 "$rgb_dir" \
            --fold2 "$sparse_dir" \
            --save_dir "$save_dir" \
            --method "$method" \
            --ckpt "$(matcher_checkpoint "$method")"
    )
}
