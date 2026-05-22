"""Preprocess stage for the Edit pipeline.

Tasks performed in this stage (run once to prepare for subsequent edit stages):
1. Render multi-view images + orthographic reference views via Blender
2. Voxelize the mesh to 64^3
3. Extract per-voxel patch-level features using DINOv2
4. Encode ss_latent / slat_latent and save to disk for the edit stage

Entry point: `preprocess_single_sample(args)`
"""

import json
import os
import shutil
from pathlib import Path

import numpy as np
import open3d as o3d
import torch
import torch.nn.functional as F
import utils3d
from PIL import Image
from torchvision import transforms

from trellis.utils.random_utils import sphere_hammersley_sequence

from .utils import (
    _run_blender_script,
    encode_slat_latent_from_feature_tensor,
    find_model_path,
    get_voxels,
    load_encoder,
    voxelize,
)


def _render(args, obj_path, output_dir, num_views=150, original_image=None,
            edit_views_path=None):
    """Render multi-view images and orthographic reference views via Blender."""
    os.makedirs(output_dir, exist_ok=True)

    model_path = find_model_path(obj_path)

    yaws, pitchs = [], []
    offset = (np.random.rand(), np.random.rand())
    for i in range(num_views):
        y, p = sphere_hammersley_sequence(i, num_views, offset)
        yaws.append(y)
        pitchs.append(p)
    views = [
        {'yaw': y, 'pitch': p, 'radius': 2, 'fov': 40 / 180 * np.pi}
        for y, p in zip(yaws, pitchs)
    ]

    _run_blender_script(
        'render.py',
        [
            '--task', 'views',
            '--object', os.path.expanduser(model_path),
            '--output_folder', output_dir,
            '--resolution', '512',
            '--engine', 'CYCLES',
            '--save_mesh',
            '--export_normalized_glb',
            '--views', json.dumps(views),
        ],
    )

    if edit_views_path is not None:
        os.makedirs(edit_views_path, exist_ok=True)
        _run_blender_script(
            'render.py',
            [
                '--task', 'ortho_ref',
                '--object', os.path.expanduser(model_path),
                '--output_folder', edit_views_path,
                '--resolution', '512',
                '--engine', 'CYCLES',
            ],
        )


def copy_camera_png(args, fname="012.png"):
    """Copy {root}/edit_views/image_ori/{fname} to {root}/ori/{fname}."""
    root = Path(args.root_dir)
    src = root / "edit_views" / "image_ori" / fname
    dst_dir = root / "ori"
    dst = dst_dir / fname

    if not src.exists():
        raise FileNotFoundError(f"[copy] Source file not found: {src}")

    dst_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    print(f"[copy] {src} -> {dst}")


def extract_feature_single_object(
    render_dir,
    positions,
    indices,
    save_path,
    model_name='dinov2_vitl14_reg',
    image_size=518,
    batch_size=8,
):
    """Extract patch-level voxel features for a single object using DINOv2."""
    dinov2_model = torch.hub.load('facebookresearch/dinov2', model_name)
    dinov2_model.eval().cuda()

    transform = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    n_patch = image_size // 14

    transforms_path = os.path.join(render_dir, 'transforms.json')
    with open(transforms_path, 'r') as f:
        meta = json.load(f)
    frames = meta['frames']

    views = []
    for view in frames:
        image_path = os.path.join(render_dir, view['file_path'])
        image = Image.open(image_path).resize((image_size, image_size), Image.Resampling.LANCZOS)
        image = np.array(image).astype(np.float32) / 255.0
        image = image[:, :, :3] * image[:, :, 3:]
        image = torch.from_numpy(image).permute(2, 0, 1).float()

        c2w = torch.tensor(view['transform_matrix'], dtype=torch.float32)
        c2w[:3, 1:3] *= -1
        extrinsics = torch.inverse(c2w)

        fov = view['camera_angle_x']
        intrinsics = utils3d.torch.intrinsics_from_fov_xy(torch.tensor(fov), torch.tensor(fov))

        views.append({
            'image': transform(image),
            'extrinsics': extrinsics,
            'intrinsics': intrinsics
        })

    assert torch.all(indices >= 0) and torch.all(indices < 64), "Some voxel indices are out of bounds"

    patchtokens_lst = []
    uv_lst = []

    for i in range(0, len(views), batch_size):
        batch = views[i:i + batch_size]
        images = torch.stack([v['image'] for v in batch]).cuda()
        extrinsics = torch.stack([v['extrinsics'] for v in batch]).cuda()
        intrinsics = torch.stack([v['intrinsics'] for v in batch]).cuda()

        with torch.no_grad():
            features = dinov2_model(images, is_training=True)
        uv = utils3d.torch.project_cv(positions, extrinsics, intrinsics)[0] * 2 - 1

        patchtokens = features['x_prenorm'][:, dinov2_model.num_register_tokens + 1:].permute(0, 2, 1)
        patchtokens = patchtokens.reshape(images.size(0), 1024, n_patch, n_patch)

        patchtokens_lst.append(patchtokens)
        uv_lst.append(uv)

    patchtokens = torch.cat(patchtokens_lst, dim=0)
    uv = torch.cat(uv_lst, dim=0)

    sampled = F.grid_sample(patchtokens, uv.unsqueeze(1),
                            mode='bilinear', align_corners=False)
    sampled = sampled.squeeze(2).permute(0, 2, 1).cpu().numpy()
    final_tokens = np.mean(sampled, axis=0).astype(np.float16)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.savez_compressed(save_path,
                        patchtokens=final_tokens,
                        indices=indices.cpu().numpy().astype(np.uint8))

    print(f"[Done] Saved feature to: {save_path}")

    del dinov2_model
    torch.cuda.empty_cache()


def preprocess_single_sample(args):
    """Preprocessing entry point: convert a raw 3D model into all caches needed by the edit stage.

    All artifacts are saved under `<args.root_dir>/`:
        render/mesh.ply, render/feature.npz, render/transforms.json,
        render/ss_latent.pt, render/slat_latent.pt,
        edit_views/, ori/012.png
    """
    image_path = args.original_image
    filename = os.path.basename(image_path)
    feature_path = os.path.join(args.root_dir, "feature")
    render_path = os.path.join(args.root_dir, "render")
    edit_views_path = os.path.join(args.root_dir, "edit_views")
    mesh_path = os.path.join(render_path, "mesh.ply")
    voxels_path = os.path.join(render_path, "voxels.ply")
    slat_feature_path = os.path.join(render_path, "feature.npz")

    _render(args,
            args.obj_path,
            render_path,
            num_views=150,
            original_image=args.original_image,
            edit_views_path=edit_views_path,
            )
    copy_camera_png(args, fname="012.png")

    mesh = o3d.io.read_triangle_mesh(mesh_path) if os.path.exists(mesh_path) else None
    voxels, positions = voxelize(mesh, voxels_path=voxels_path)

    extract_feature_single_object(
        render_path,
        positions,
        voxels,
        slat_feature_path,
    )

    encoder = load_encoder(args)
    ss = get_voxels(voxels)[None].float().to("cuda")
    latent = encoder(ss, sample_posterior=False)
    del encoder
    torch.cuda.empty_cache()

    slat_latent_tensor = encode_slat_latent_from_feature_tensor(
        feature_npz_path=slat_feature_path,
        enc_pretrained_path=args.enc_pretrained
    )

    ss_latent_path = os.path.join(render_path, "ss_latent.pt")
    slat_latent_path = os.path.join(render_path, "slat_latent.pt")
    torch.save(latent.detach().cpu(), ss_latent_path)
    torch.save({
        "feats": slat_latent_tensor.feats.detach().cpu(),
        "coords": slat_latent_tensor.coords.detach().cpu(),
    }, slat_latent_path)

    print("[preprocess] Done: generated mesh.ply / voxels.ply / feature.npz / "
          "ss_latent.pt / slat_latent.pt.")
