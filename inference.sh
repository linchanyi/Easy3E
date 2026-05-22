#!/bin/bash
# =============================================================
# Single-case edit inference script
#
# Directory layout:
#   $BASE/$ID/
#     ├── model/                       # Original 3D model (.glb / .obj)
#     ├── render/                      # Preprocess artifacts: mesh.ply / feature.npz / transforms.json
#     ├── edit_views/                  # Preprocess artifacts: orthographic reference views (16 images)
#     ├── ori/012.png                  # Reference orthographic image (used by edit stage)
#     ├── cond/<edit_name>.png         # User edit intent image (RGBA)
#     ├── mask_model/                  # Flat mask directory
#     │   ├── <edit_name>.glb          # Required: same name as cond/<edit_name>.png
#     │   └── <edit_name>_texture.glb  # Optional: textured mask (for blender_mask rendering)
#     └── output/                      # Inference results
# =============================================================

# === Modify these variables ===
ID="tiger"                                                                  # Case ID (subdirectory name)
BASE="./examples"                                                           # Root directory containing cases

# --- FlowEdit / guidance parameters (adjust as needed) ---
CFG_SRC=5.0                        # Source condition CFG strength in FlowEdit (default 5.0)
CFG_TAR=5.0                        # Target condition CFG strength in FlowEdit (default 5.0)

# === Auto-construct paths and run ===
INPUT_DIR="$BASE/$ID/cond"
MASK_DIR="$BASE/$ID/mask"
OUTPUT_DIR="$BASE/$ID/output"
ROOT_DIR="$BASE/$ID"
OBJ_PATH="$BASE/$ID/model"
ORI_IMAGE="$BASE/$ID/ori/012.png"

python inference.py \
  --input_dir "$INPUT_DIR" \
  --mask_dir "$MASK_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --root_dir "$ROOT_DIR" \
  --obj_path "$OBJ_PATH" \
  --mode "edit" \
  --original_image "$ORI_IMAGE" \
  --cfg_src "$CFG_SRC" \
  --cfg_tar "$CFG_TAR"
