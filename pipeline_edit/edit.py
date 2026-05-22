"""Edit / baseline stage for the Edit pipeline.

Tasks performed in this stage (once per input image):
1. Load cached mesh / latent from the preprocess stage
2. Extract edit_mask from the input RGBA image (alpha channel)
3. Optionally: load mask.glb, construct 3D latent_mask / raw_mask
4. Call `pipeline.flowedit` (edit mode) or `pipeline.run` (baseline mode) to produce output
5. Export the edited result as `<output_dir>/<edit_name>.glb` (the only user-facing output)
6. Write intermediate artifacts under root_dir for Blender post-processing and subsequent edits

Entry point: `process_single_sample(args, pipeline, encoder, image_path)`
"""

import os
import traceback

import numpy as np
import open3d as o3d
import torch
from PIL import Image

import trellis.modules.sparse as sp
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import postprocessing_utils

from .utils import (
    _build_masks_from_mask_glb,
    _normalize_mask,
    encode_slat_latent_from_feature_tensor,
    get_voxels,
    load_encoder,
    voxelize,
)


def load_edit_mask_from_rgba(
    image_path: str,
    target_size=None,
    return_type="torch",
    threshold=0,
    soft=False,
    invert=False,
    save_path=None,
    device=None
):
    """Extract an edit mask from the alpha channel of an RGBA image.
    White(1) = object, Black(0) = background.
    soft=True uses normalized alpha directly; soft=False applies threshold binarization.
    """
    img = Image.open(image_path).convert("RGBA")

    if target_size is not None:
        resample = Image.BILINEAR if soft else Image.NEAREST
        img = img.resize((target_size[1], target_size[0]), resample=resample)

    alpha = np.array(img, dtype=np.uint8)[..., 3].astype(np.float32) / 255.0

    if soft:
        mask = alpha
    else:
        thr = float(threshold) / 255.0
        mask = (alpha > thr).astype(np.float32)

    if invert:
        mask = 1.0 - mask

    if save_path is not None:
        m8 = (np.clip(mask, 0, 1) * 255).astype(np.uint8)
        Image.fromarray(m8, mode="L").save(save_path)

    if return_type == "numpy":
        return mask
    elif return_type == "pil":
        m8 = (np.clip(mask, 0, 1) * 255).astype(np.uint8)
        return Image.fromarray(m8, mode="L")
    else:
        t = torch.from_numpy(mask)[None, None, ...]
        t = t.to(dtype=torch.float32)
        if device is not None:
            t = t.to(device)
        return t


def process_single_sample(args, pipeline=None, encoder=None, image_path=None):
    """Run the edit / baseline pipeline for a single input image.

    Args:
        args: Namespace from `parse_args()`.
        pipeline: A loaded TrellisImageTo3DPipeline instance; required.
        encoder: Reserved parameter; loaded internally as needed.
        image_path: Target condition image for this edit (RGBA or regular image).

    Returns:
        bool: Always returns True (exceptions are caught and printed).
    """
    filename = os.path.basename(image_path)
    output_root = args.output_dir
    render_path = os.path.join(args.root_dir, "render")
    feature_path = os.path.join(args.root_dir, "feature")
    mesh_path = os.path.join(render_path, "mesh.ply")
    voxels_path = os.path.join(render_path, "voxels.ply")
    slat_feature_path = os.path.join(render_path, "feature.npz")

    try:

        if args.mode == "edit":
            image = Image.open(image_path)
            ori_image_path = args.original_image
            edit_mask = load_edit_mask_from_rgba(image_path, target_size=(512, 512))
            mesh = o3d.io.read_triangle_mesh(mesh_path) if os.path.exists(mesh_path) else None
            ori_image = Image.open(ori_image_path)
            voxels, positions = voxelize(mesh, voxels_path=voxels_path)

            ss_latent_path = os.path.join(render_path, "ss_latent.pt")
            slat_latent_path = os.path.join(render_path, "slat_latent.pt")

            if os.path.exists(ss_latent_path):
                latent = torch.load(ss_latent_path, map_location="cuda")
            else:
                encoder = load_encoder(args)
                ss = get_voxels(voxels)[None].float().to("cuda")
                latent = encoder(ss, sample_posterior=False)
                del encoder
                torch.cuda.empty_cache()

            if os.path.exists(slat_latent_path):
                slat_cache = torch.load(slat_latent_path, map_location="cuda")
                slat_latent_tensor = sp.SparseTensor(
                    feats=slat_cache["feats"].cuda(),
                    coords=slat_cache["coords"].cuda(),
                )
            else:
                slat_latent_tensor = encode_slat_latent_from_feature_tensor(
                    feature_npz_path=slat_feature_path,
                    enc_pretrained_path=args.enc_pretrained,
                )

            edit_name = os.path.splitext(filename)[0]
            mask_path = os.path.join(args.mask_dir, f"{edit_name}.glb")
            mask_cache_dir = os.path.join(args.root_dir, ".mask_cache", edit_name)
            os.makedirs(mask_cache_dir, exist_ok=True)
            _normalize_mask(args, mask_path, mask_cache_dir)
            normalized_ply = os.path.join(mask_cache_dir, f"{edit_name}.ply")
            raw_mask, latent_mask = _build_masks_from_mask_glb(args, normalized_ply)

            outputs = pipeline.flowedit(
                ori_image,
                image,
                seed=args.seed,
                latent=latent,
                latent_slat=slat_latent_tensor,
                coords=voxels,
                latent_mask=latent_mask,
                raw_mask=raw_mask,
                feature_path=feature_path,
                edit_mask=edit_mask,
                cfg_src_strength=args.cfg_src,
                cfg_tar_strength=args.cfg_tar,
                enable_step_viz=args.enable_step_viz,
                use_guidance=(not args.no_guidance),
            )

        else:
            image = Image.open(image_path)
            outputs = pipeline.run(
                image,
                seed=args.seed,
            )

        name_without_ext = os.path.splitext(filename)[0]
        os.makedirs(output_root, exist_ok=True)
        glb_path = os.path.join(output_root, f"{name_without_ext}.glb")

        glb = postprocessing_utils.to_glb(
            outputs['gaussian'][0],
            outputs['mesh'][0],
            simplify=0,
            texture_size=1024,
        )
        glb.export(glb_path)

    except Exception as e:
        print(f"Error processing {filename}: {str(e)}")
        traceback.print_exc()

    return True
