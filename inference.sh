ID="girl"
BASE="../example"
VIEW_IDX="012"

CFG_SRC=5.0
CFG_TAR=5.0

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
    --cfg_tar "$CFG_TAR" \
