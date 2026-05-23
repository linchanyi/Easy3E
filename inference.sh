#!/bin/bash
# =============================================================
# Single-case edit inference script
#
# Directory layout:
#   $BASE/$ID/
#     ├── model/                       # Original 3D model (.glb / .obj)
#     ├── render/                      # Preprocess artifacts: mesh.ply / feature.npz / transforms.json
#     ├── edit_views/                  # Preprocess artifacts: orthographic reference views (16 images)
#     │   └── camera/                  # Camera params (.npy) for each ortho view
#     ├── ori/<VIEW_IDX>.png           # Reference orthographic image (determines editing view angle)
#     ├── cond/<edit_name>.png         # User edit intent image (RGBA)
#     ├── mask_model/                  # Flat mask directory
#     │   ├── <edit_name>.glb          # Required: same name as cond/<edit_name>.png
#     │   └── <edit_name>_texture.glb  # Optional: textured mask (for blender_mask rendering)
#     └── output/                      # Inference results
# =============================================================

# === Modify these variables ===
ID="tiger"                                                                  # Case ID (subdirectory name)
BASE="./examples"                                                           # Root directory containing cases

# --- View selection ---
# View index (000-015): determines the editing view angle.
# 012 = front view (azimuth 270°), 000 = azimuth 0°, 004 = azimuth 90°, etc.
# The ori/ folder filename must match this index.
VIEW_IDX="012"

# --- FlowEdit / guidance parameters (adjust as needed) ---
CFG_SRC=5.0                        # Source condition CFG strength in FlowEdit (default 5.0)
CFG_TAR=5.0                        # Target condition CFG strength in FlowEdit (default 5.0)

# === Auto-construct paths and run ===
INPUT_DIR="$BASE/$ID/cond"
MASK_DIR="$BASE/$ID/mask"
OUTPUT_DIR="$BASE/$ID/output"
ROOT_DIR="$BASE/$ID"
OBJ_PATH="$BASE/$ID/model"
ORI_IMAGE="$BASE/$ID/ori/${VIEW_IDX}.png"

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
  # --debug_guidance_viz          # Uncomment to save guidance debug visualizations
