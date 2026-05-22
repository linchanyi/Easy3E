#!/bin/bash
# =============================================================
# Batch preprocessing script (preprocess_batch.sh)
# Function: Iterate over each model subfolder under a root directory and preprocess its model/
# Input structure (must conform to):
#   <BASE_DIR>/
#       <model_name_1>/
#           model/                  # Contains one of .glb/.obj/.ply/... 
#       <model_name_2>/
#           model/
#       ...
#
# Preprocessing artifacts (auto-generated, used by downstream edit_batch.sh):
#   <BASE_DIR>/<model_name>/render/          (mesh.ply, voxels.ply, feature.npz, transforms.json, multi-view images)
#   <BASE_DIR>/<model_name>/edit_views/      (orthographic reference views for selecting edit viewpoint)
#   <BASE_DIR>/<model_name>/ori/012.png      (reference image used by edit stage)
#
# Usage:
#   bash preprocess_batch.sh <root_folder_path>
# Example:
#   bash preprocess_batch.sh /path/to/your/models
# =============================================================

set -u

if [ $# -lt 1 ]; then
    echo "Usage: $0 <root_folder_path>"
    echo "Example: $0 /path/to/your/models"
    exit 1
fi

BASE_DIR="$1"

if [ ! -d "$BASE_DIR" ]; then
    echo "[ERROR] Root folder does not exist: $BASE_DIR"
    exit 1
fi

# Statistics
total=0
success=0
skipped=0
failed=0

echo "=============================================="
echo " Batch preprocessing started"
echo " Root directory: $BASE_DIR"
echo "=============================================="

for MODEL_DIR in "$BASE_DIR"/*/; do
    [ -d "$MODEL_DIR" ] || continue
    MODEL_DIR="${MODEL_DIR%/}"
    MODEL_NAME="$(basename "$MODEL_DIR")"
    total=$((total + 1))

    echo ""
    echo "---------- [$total] $MODEL_NAME ----------"

    OBJ_PATH="$MODEL_DIR/model"
    if [ ! -d "$OBJ_PATH" ]; then
        echo "[SKIP] Missing model/ subfolder: $OBJ_PATH"
        skipped=$((skipped + 1))
        continue
    fi

    # Check if model/ contains a supported model file
    if ! ls "$OBJ_PATH"/*.{glb,obj,ply,gltf,fbx,stl,usd,usdz,dae,vrm,blend} 2>/dev/null | head -n 1 | grep -q .; then
        echo "[SKIP] No supported model file found in model/ (.glb/.obj/.ply/...)"
        skipped=$((skipped + 1))
        continue
    fi

    # Preprocessing does not use input_dir / output_dir / original_image (all optional in argparse)
    ROOT_DIR="$MODEL_DIR"

    echo "[RUN] root_dir = $ROOT_DIR"
    echo "[RUN] obj_path = $OBJ_PATH"

    python inference.py \
        --preprocess \
        --root_dir "$ROOT_DIR" \
        --obj_path "$OBJ_PATH"

    rc=$?
    if [ $rc -eq 0 ] && [ -f "$ROOT_DIR/render/mesh.ply" ] && [ -f "$ROOT_DIR/render/feature.npz" ]; then
        echo "[OK]   $MODEL_NAME preprocessing complete"
        success=$((success + 1))
    else
        echo "[FAIL] $MODEL_NAME preprocessing failed (exit=$rc)"
        failed=$((failed + 1))
    fi
done

echo ""
echo "=============================================="
echo " Preprocessing finished: total=$total success=$success skipped=$skipped failed=$failed"
echo "=============================================="
