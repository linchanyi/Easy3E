"""Shared utilities for the Edit pipeline.

This module provides low-level helpers used by both the preprocess and edit stages:
Blender invocation, geometry/mask construction, voxelization, SLAT latent encoding,
model loading, and visualization.
"""

import os
import json
import subprocess
import time
from collections import deque
from pathlib import Path

import numpy as np
import open3d as o3d
import torch
import torch.nn.functional as F
from einops import repeat
from PIL import Image

import trellis.models as models
import trellis.modules.sparse as sp
from trellis.pipelines import TrellisImageTo3DPipeline


_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_PKG_DIR)
BLENDER_INSTALLATION_PATH = os.path.join(_REPO_ROOT, 'tmp')
BLENDER_PATH = os.environ.get(
    'BLENDER_PATH',
    f'{BLENDER_INSTALLATION_PATH}/blender-4.0.0-linux-x64/blender'
)
BLENDER_SCRIPTS_DIR = os.path.join(_REPO_ROOT, 'blender')


def _run_blender_script(script_name, script_args, *, background=True, check=False):
    """Execute a Blender script in headless mode.

    Args:
        script_name: Script filename under the blender/ directory.
        script_args: Arguments passed to the script (placed after `--`).
        background: Whether to run in -b headless mode (default True).
        check: Whether to raise on failure (default False).
    """
    script_path = os.path.join(BLENDER_SCRIPTS_DIR, script_name)
    cmd = [BLENDER_PATH]
    if background:
        cmd.append('-b')
    cmd += ['-P', script_path, '--'] + [str(a) for a in script_args]

    env = os.environ.copy()
    env.pop('WAYLAND_DISPLAY', None)
    env.pop('DISPLAY', None)
    xdg_runtime = env.get('XDG_RUNTIME_DIR') or f"/tmp/runtime-{os.geteuid()}"
    try:
        os.makedirs(xdg_runtime, exist_ok=True)
        os.chmod(xdg_runtime, 0o700)
    except OSError:
        pass
    env['XDG_RUNTIME_DIR'] = xdg_runtime
    env['MESA_GL_VERSION_OVERRIDE'] = '4.3'
    env['LIBGL_ALWAYS_SOFTWARE'] = '1'
    env['QT_QPA_PLATFORM'] = 'offscreen'

    try:
        subprocess.run(cmd, env=env, check=check)
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Blender script {script_name} failed: {e}")


def print_time_cost(start_time, stage_name):
    """Print elapsed time since start_time for the given stage."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    end_time = time.time()
    cost = end_time - start_time
    print(f"[{stage_name}] Time cost: {cost:.4f}s")
    return end_time


def build_latent_mask_from_mesh_regions(mask_boxes, latent_shape=(16, 16, 16), device="cuda") -> torch.Tensor:
    """Union multiple boxes on a 64^3 mask, downsample to latent_shape, and expand to 8 channels.

    Args:
        mask_boxes: List of boxes, each box = [[x0,x1],[y0,y1],[z0,z1]]

    Returns:
        latent_mask: [1, 8, D', H', W'] (float 0/1)
        raw_mask64:  [1, 1, 64, 64, 64] (float 0/1)
    """
    raw_mask64 = torch.zeros((1, 1, 64, 64, 64), dtype=torch.float32, device=device)
    for box in mask_boxes:
        (x0, x1), (y0, y1), (z0, z1) = box
        raw_mask64[0, 0, z0:z1, y0:y1, x0:x1] = 1.0

    latent_mask = F.interpolate(raw_mask64, size=latent_shape, mode='trilinear', align_corners=False)
    latent_mask = (latent_mask > 0).float()
    latent_mask = repeat(latent_mask, 'b c d h w -> b (rep c) d h w', rep=8)
    return latent_mask, raw_mask64


def downsample_voxel_mask(mask: torch.Tensor, target_shape=(16, 16, 16)) -> torch.Tensor:
    """Downsample a 3D voxel mask [B,1,D,H,W] to target_shape via trilinear interpolation."""
    if mask.dtype != torch.float32:
        mask = mask.float()
    downsampled_mask = F.interpolate(mask, size=target_shape, mode='trilinear', align_corners=False)
    return downsampled_mask


def find_model_path(folder: str) -> str:
    """Find the first supported 3D model file in the given folder (non-recursive)."""
    exts = ('.obj', '.glb', '.gltf', '.fbx', '.ply', '.stl', '.usd', '.usdz', '.dae', '.vrm', '.blend')
    p = Path(folder)
    for ext in exts:
        hit = next(p.glob(f'*{ext}'), None)
        if hit:
            return str(hit.resolve())
    raise FileNotFoundError(f'No model file found in {folder}')


def _normalize_mask(args, mask_path, output_dir):
    """Normalize mask.glb to the same coordinate system as the main mesh."""
    _run_blender_script(
        'render.py',
        [
            '--task', 'views',
            '--object', os.path.expanduser(mask_path),
            '--output_folder', output_dir,
            '--resolution', '512',
            '--engine', 'CYCLES',
            '--save_mesh',
            '--export_normalized_glb',
            '--transform_path', os.path.join(args.root_dir, 'render', 'transforms.json'),
        ],
    )


def _solid_mask_via_sdf(mesh_legacy: o3d.geometry.TriangleMesh, res: int = 64) -> torch.Tensor:
    """Compute solid mask [1,1,res,res,res] via SDF sampling on a res^3 grid.
    World coordinates assumed in [-0.5, 0.5].
    """
    tmesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh_legacy)
    scene = o3d.t.geometry.RaycastingScene()
    _ = scene.add_triangles(tmesh)

    lin = np.linspace(-0.5 + 0.5 / res, 0.5 - 0.5 / res, res, dtype=np.float32)
    X, Y, Z = np.meshgrid(lin, lin, lin, indexing="ij")
    pts = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)

    pts_t = o3d.core.Tensor(pts, dtype=o3d.core.Dtype.Float32)
    sdf = scene.compute_signed_distance(pts_t).numpy().reshape(res, res, res)
    inside = (sdf <= 0).astype(np.float32)

    raw = torch.from_numpy(inside).unsqueeze(0).unsqueeze(0)
    return raw


def _fill_solid_from_surface(surface_bool: np.ndarray) -> np.ndarray:
    """Infer solid voxels from surface voxels via 6-connected exterior flood-fill.

    Args:
        surface_bool: [res,res,res], True = surface occupied

    Returns:
        solid_bool: [res,res,res], True = solid (including surface)
    """
    res = surface_bool.shape[0]
    occ = surface_bool.astype(bool)
    visited = np.zeros_like(occ, dtype=bool)
    q = deque()

    def push(i, j, k):
        if 0 <= i < res and 0 <= j < res and 0 <= k < res:
            if (not occ[i, j, k]) and (not visited[i, j, k]):
                visited[i, j, k] = True
                q.append((i, j, k))

    for i in range(res):
        for j in range(res):
            push(i, j, 0);            push(i, j, res - 1)
            push(i, 0, j);            push(i, res - 1, j)
            push(0, i, j);            push(res - 1, i, j)

    nbrs = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
    while q:
        i, j, k = q.popleft()
        for di, dj, dk in nbrs:
            push(i + di, j + dj, k + dk)

    outside = visited
    solid = ~outside
    return solid


def _solid_mask_via_floodfill(mesh_legacy: o3d.geometry.TriangleMesh, res: int = 64) -> torch.Tensor:
    """Fallback: surface voxelization + exterior flood-fill to get solid mask [1,1,res,res,res]."""
    surf_idx, _ = voxelize(mesh_legacy)
    surf_idx_np = surf_idx.detach().cpu().numpy().astype(np.int64)

    surface = np.zeros((res, res, res), dtype=bool)
    surface[surf_idx_np[:, 0], surf_idx_np[:, 1], surf_idx_np[:, 2]] = True

    solid = _fill_solid_from_surface(surface).astype(np.float32)
    raw = torch.from_numpy(solid).unsqueeze(0).unsqueeze(0)
    return raw


def _build_masks_from_mask_glb(args, mask_glb_path, res: int = 64):
    """Generate raw_mask64 [1,1,64,64,64] and latent_mask [1,8,16,16,16] from a mask mesh file."""
    if not os.path.exists(mask_glb_path):
        raise FileNotFoundError(f"[mask] Not found mask glb: {mask_glb_path}")

    mask_mesh = o3d.io.read_triangle_mesh(mask_glb_path)
    if mask_mesh is None or len(mask_mesh.vertices) == 0 or len(mask_mesh.triangles) == 0:
        raise RuntimeError(f"[mask] Invalid mesh: {mask_glb_path}")

    try:
        raw_cpu = _solid_mask_via_sdf(mask_mesh, res=res)
        if torch.isnan(raw_cpu).any():
            raise RuntimeError("SDF produced NaN")
    except Exception as e:
        print(f"[mask] SDF failed ({e}), fallback to flood-fill from surface...")
        raw_cpu = _solid_mask_via_floodfill(mask_mesh, res=res)

    raw_mask64 = raw_cpu.to(device="cuda", dtype=torch.float32)

    latent16 = F.interpolate(raw_mask64, size=(16, 16, 16), mode="trilinear", align_corners=False)
    latent16 = (latent16 > 0.5).float()
    latent_mask = repeat(latent16, "b c d h w -> b (rep c) d h w", rep=8)

    return raw_mask64, latent_mask


def _orthographic_projections_64(grid_64):
    """Compute orthographic max-projections of a 64^3 voxel grid, returning front/back/left/right views."""
    assert grid_64.ndim == 5 and grid_64.shape[2:] == (64, 64, 64)

    front = grid_64.max(dim=4).values[0, 0].detach().float().cpu().numpy()
    back = np.flip(front, axis=1)
    left = grid_64.max(dim=2).values[0, 0].detach().float().cpu().numpy()
    right = np.flip(left, axis=1)

    front = front.T
    back = back.T

    return {"front": front, "back": back, "left": left, "right": right}


def visualize_voxel_and_mask(args, raw_mask64=None, save_dir=None):
    """Generate four-view overlay visualization (orthographic projection): object voxels vs mask voxels."""
    render_path = os.path.join(args.root_dir, "render")
    mesh_path = os.path.join(render_path, "mesh.ply")
    save_dir = save_dir or os.path.join(render_path, "viz_mask")
    os.makedirs(save_dir, exist_ok=True)

    if raw_mask64 is None:
        latent_mask64_path = os.path.join(render_path, "latent_mask64.pt")
        if not os.path.exists(latent_mask64_path):
            raise FileNotFoundError(f"[viz] missing {latent_mask64_path}")
        raw_mask64 = torch.load(latent_mask64_path).to("cuda")

    mesh = o3d.io.read_triangle_mesh(mesh_path)
    vox_idx, _ = voxelize(mesh)
    obj64 = torch.zeros((1, 1, 64, 64, 64), dtype=torch.float32, device="cuda")
    obj64[0, 0, vox_idx[:, 0], vox_idx[:, 1], vox_idx[:, 2]] = 1.0

    obj_bin = obj64 > 0
    mask_bin = raw_mask64 > 0
    inter = (obj_bin & mask_bin).sum().item()
    union = (obj_bin | mask_bin).sum().item()
    iou = inter / union if union else 1.0
    cover = inter / obj_bin.sum().item() if obj_bin.sum().item() else 1.0
    print(f"[viz] 3D IoU={iou:.4f},  Mask coverage={cover:.4f},  "
          f"obj_voxels={int(obj_bin.sum())}, mask_voxels={int(mask_bin.sum())}, inter={int(inter)}")

    obj_views = _orthographic_projections_64(obj64)
    mask_views = _orthographic_projections_64(raw_mask64)

    def _save_single(name, arr):
        img = np.uint8(np.clip(arr, 0, 1) * 255)
        Image.fromarray(img, mode="L").save(os.path.join(save_dir, name))

    for k in ["front", "back", "left", "right"]:
        obj = (obj_views[k] > 0).astype(np.float32)
        msk = (mask_views[k] > 0).astype(np.float32)
        overlay = np.clip(0.4 * obj + 0.9 * msk - 0.3 * (obj * msk), 0.0, 1.0)

        _save_single(f"obj_{k}.png", obj)
        _save_single(f"mask_{k}.png", msk)
        _save_single(f"overlay_{k}.png", overlay)

    print(f"[viz] saved to: {save_dir}")


def voxelize(mesh: o3d.geometry.TriangleMesh, voxels_path=None) -> torch.Tensor:
    """Voxelize a mesh onto a 64^3 regular grid (world box [-0.5, 0.5]).

    Returns:
        indices:  [M, 3] int CUDA, occupied voxel coordinates, range [0, 63].
        positions:[M, 3] float CUDA, voxel center positions, range [-0.5, 0.5].
    """
    vertices = np.asarray(mesh.vertices)
    vertices = np.clip(vertices, -0.5 + 1e-6, 0.5 - 1e-6)
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh_within_bounds(
        mesh, voxel_size=1 / 64, min_bound=(-0.5, -0.5, -0.5), max_bound=(0.5, 0.5, 0.5))
    vertices = np.array([voxel.grid_index for voxel in voxel_grid.get_voxels()])
    positions = (vertices + 0.5) / 64 - 0.5
    positions = torch.tensor(positions).float().cuda()

    return torch.tensor(vertices).int().cuda(), positions


def get_voxels(coords):
    """Convert integer voxel coordinates [M, 3] to occupancy tensor [1, 64, 64, 64]."""
    coords = coords.int().contiguous()
    ss = torch.zeros(1, 64, 64, 64, dtype=torch.long, device=coords.device)
    ss[:, coords[:, 0], coords[:, 1], coords[:, 2]] = 1
    return ss


def encode_slat_latent_from_feature_tensor(
    feature_npz_path: str,
    enc_pretrained_path: str
) -> torch.Tensor:
    """Encode SLAT latent from feature.npz using the given encoder weights.

    Args:
        feature_npz_path: Path to npz with patchtokens and indices.
        enc_pretrained_path: Path to encoder model weights.

    Returns:
        latent: Encoded SparseTensor.
    """
    encoder1 = models.from_pretrained(enc_pretrained_path).eval().cuda()

    data = np.load(feature_npz_path)
    patchtokens = torch.from_numpy(data['patchtokens']).float()
    indices = torch.from_numpy(data['indices']).int()

    coords = torch.cat([
        torch.zeros(indices.shape[0], 1).int(),
        indices
    ], dim=1)

    feats = sp.SparseTensor(
        feats=patchtokens,
        coords=coords,
    ).cuda()

    with torch.no_grad():
        latent = encoder1(feats, sample_posterior=False)
        assert torch.isfinite(latent.feats).all(), "Non-finite latent detected"

    del encoder1
    torch.cuda.empty_cache()

    return latent


def load_encoder(args):
    """Load the SS encoder in inference mode."""
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_gpu else "cpu")
    encoder_path = f"{args.checkpoint}/ckpts/{args.encoder_name}"
    encoder = models.from_pretrained(encoder_path).eval().to(device)
    return encoder


def load_pipeline(args):
    """Load the TrellisImageTo3DPipeline onto GPU."""
    pipeline = TrellisImageTo3DPipeline.from_pretrained(args.checkpoint)
    pipeline.cuda()
    return pipeline


def initialize_models(args):
    """Initialize both encoder and pipeline."""
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_gpu else "cpu")

    encoder_path = f"{args.checkpoint}/ckpts/{args.encoder_name}"
    encoder = models.from_pretrained(encoder_path).eval().to(device)

    pipeline = TrellisImageTo3DPipeline.from_pretrained(args.checkpoint)
    pipeline.cuda()

    return pipeline, encoder
