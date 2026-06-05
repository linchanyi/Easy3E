import os
import traceback

import numpy as np
import open3d as o3d
import torch
from PIL import Image

import trellis.modules.sparse as sp
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import postprocessing_utils
from trellis.utils.general_utils import get_view_azimuth_from_ori

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


def _get_azimuth_from_ori(args) -> float:
    try:
        ori_dir = os.path.join(args.root_dir, "ori")
        _, azimuth_deg = get_view_azimuth_from_ori(ori_dir)
        return azimuth_deg
    except Exception as e:
        print(f"[WARN] Could not extract azimuth from ori folder: {e}. Using default 270°.")
        return 270.0


def process_single_sample(args, pipeline=None, encoder=None, image_path=None):
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
            if not os.path.exists(mask_path):
                glb_files = [f for f in os.listdir(args.mask_dir) if f.endswith('.glb')]
                if not glb_files:
                    raise FileNotFoundError(f"[mask] No .glb file found in {args.mask_dir}")
                mask_path = os.path.join(args.mask_dir, glb_files[0])
                print(f"[INFO] Mask '{edit_name}.glb' not found, using fallback: {glb_files[0]}")
            mask_cache_dir = os.path.join(args.root_dir, ".mask_cache", edit_name)
            os.makedirs(mask_cache_dir, exist_ok=True)
            _normalize_mask(args, mask_path, mask_cache_dir)
            mask_stem = os.path.splitext(os.path.basename(mask_path))[0]
            normalized_ply = os.path.join(mask_cache_dir, f"{mask_stem}.ply")
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
                azimuth_deg=_get_azimuth_from_ori(args),
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
