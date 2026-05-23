import os
os.environ['SPCONV_ALGO'] = 'native'

import argparse
import time

import torch
from tqdm import tqdm

from pipeline_edit.edit import process_single_sample
from pipeline_edit.preprocess import preprocess_single_sample
from pipeline_edit.utils import load_pipeline


if torch.cuda.is_available():
    torch.cuda.synchronize()
t0 = time.time()
last_t = t0


def parse_args():
    """Parse command-line arguments for the pipeline."""
    parser = argparse.ArgumentParser(description="TRELLIS 3D Reconstruction Pipeline")

    parser.add_argument("--input_dir", default="",
                      help="Input image directory path")
    parser.add_argument("--output_dir", default="",
                      help="Output result root directory")
    parser.add_argument("--mask_dir", default=None,
                      help="Input mask directory path (flat layout: <mask_dir>/<edit_name>.glb)")

    parser.add_argument("--root_dir", default="", help="Processing root directory")
    parser.add_argument("--mode", default="baseline",
                      help="Run mode: baseline | edit")
    parser.add_argument("--preprocess", action="store_true", help="Run preprocessing only")

    parser.add_argument("--checkpoint", default="checkpoint/TRELLIS-image-large",
                      help="Pretrained model path")
    parser.add_argument("--obj_path", default="assets/example_mesh/T.ply",
                      help="Source 3D model directory")
    parser.add_argument("--original_image", default="",
                      help="Original reference image path")
    parser.add_argument("--encoder_name", default="ss_enc_conv3d_16l8_fp16",
                      help="SS encoder weight subdirectory name")
    parser.add_argument('--enc_pretrained', type=str,
                      default='checkpoint/TRELLIS-image-large/ckpts/slat_enc_swin8_B_64l8_fp16',
                      help='SLAT encoder weight path')

    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    parser.add_argument("--no_gpu", action="store_true", help="Disable GPU acceleration")

    parser.add_argument("--cfg_src", type=float, default=5.0,
                      help="Source condition CFG strength in FlowEdit (default 5.0)")
    parser.add_argument("--cfg_tar", type=float, default=5.0,
                      help="Target condition CFG strength in FlowEdit (default 5.0)")
    parser.add_argument("--enable_step_viz", action="store_true",
                      help="Export per-step FlowEdit visualization")
    parser.add_argument("--no_guidance", action="store_true",
                      help="Disable orthographic silhouette guidance")
    parser.add_argument("--debug_guidance_viz", action="store_true",
                      help="Save debug visualization of guidance alignment (silhouette vs edit mask)")

    return parser.parse_args()


def _list_input_images(input_dir):
    """List all supported images (png/jpg/jpeg) under input_dir."""
    return sorted([
        os.path.join(input_dir, fname)
        for fname in os.listdir(input_dir)
        if fname.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])


def main():
    """CLI entry point. Routes based on --preprocess / --mode."""
    args = parse_args()

    if args.preprocess:
        preprocess_single_sample(args)
        return

    image_paths = _list_input_images(args.input_dir)
    pipeline = load_pipeline(args)

    if args.mode == "edit":
        print("[INFO] Running in Edit mode...")
    else:
        print(f"[INFO] Running in {args.mode.upper()} mode...")

    for image_path in tqdm(image_paths, desc="Processing samples"):
        process_single_sample(args, pipeline, encoder=None, image_path=image_path)

    del pipeline
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
