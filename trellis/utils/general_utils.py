import re
import numpy as np
import cv2
import torch
import contextlib
import pickle
import os
import trimesh
import trellis.modules.sparse as sp
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
# flowedit_utils.py
# -*- coding: utf-8 -*-

from typing import Optional, Tuple, Union
import os
import numpy as np
import torch
import torch.nn.functional as F

# PIL optional dependency: if not available, grid only returns None
try:
    from PIL import Image
    HAS_PIL = True
except Exception:
    HAS_PIL = False


ArrayLike = Union[np.ndarray, torch.Tensor]

import torch

def _smoothstep(x: torch.Tensor):
    # x ∈ [0,1] → [0,1], C1 continuous
    x = x.clamp(0, 1)
    return x * x * (3 - 2 * x)

def soft_box_weight_zyx(coords: torch.Tensor,
                        boxes_zyx,
                        band_in: float = 2.0,     # Inward smoothing bandwidth inside box (in voxels)
                        band_out: float = 1.0,    # Outward smoothing bandwidth outside box
                        alpha_at_boundary: float = 0.5  # New feature weight at boundary (0.5 = half-half)
                        ) -> torch.Tensor:
    """
    coords: [N,4] where 1/2/3 -> x/y/z
    boxes_zyx: list of [[z0,z1],[y0,y1],[x0,x1], ...] (closed intervals)
    Returns: w_new ∈ [0,1], shape [N], representing "new feature weight"
    Rules:
      - Deep inside box (>=band_in): w=1
      - Far outside box (>=band_out): w=0
      - At boundary (distance=0): w=alpha_at_boundary
      - Transition zone: smooth interpolation
      - Multiple boxes: take max (any box influence can increase w)
    """
    if not boxes_zyx:
        return torch.zeros(coords.size(0), device=coords.device)

    x, y, z = coords[:, 1].float(), coords[:, 2].float(), coords[:, 3].float()
    w_total = torch.zeros_like(x)

    for (zr, yr, xr) in boxes_zyx:
        z0, z1 = float(zr[0]), float(zr[1])
        y0, y1 = float(yr[0]), float(yr[1])
        x0, x1 = float(xr[0]), float(xr[1])

        # --- Inside logic: L∞ distance to nearest face (closer to "grid thickness" on voxel grid)
        dx_in = torch.minimum(x - x0, x1 - x)
        dy_in = torch.minimum(y - y0, y1 - y)
        dz_in = torch.minimum(z - z0, z1 - z)
        d_in = torch.minimum(torch.minimum(dx_in, dy_in), dz_in)  # <0 means not inside box

        inside = d_in >= 0
        w_inside = torch.zeros_like(w_total)
        if band_in > 0:
            t = (d_in / band_in).clamp(0, 1)  # 0=boundary, 1=inner edge of bandwidth
            w_inside = alpha_at_boundary + (1 - alpha_at_boundary) * _smoothstep(t)
        else:
            w_inside = torch.ones_like(w_total)  # No inner bandwidth: all 1 inside box
        w_inside = torch.where(inside, w_inside, torch.zeros_like(w_inside))

        # --- Outside logic: L∞ external distance to box (0=at boundary/projected inside)
        # L∞ external distance: max of per-axis distances to interval
        dx_out = torch.maximum(x0 - x, x - x1)
        dy_out = torch.maximum(y0 - y, y - y1)
        dz_out = torch.maximum(z0 - z, z - z1)
        d_out_inf = torch.maximum(torch.maximum(dx_out, dy_out), dz_out)  # >0 outside box, =0 aligned with boundary/projected inside

        outside = d_out_inf > 0
        w_outside = torch.zeros_like(w_total)
        if band_out > 0:
            t = (d_out_inf / band_out).clamp(0, 1)  # 0=boundary, 1=outer edge of bandwidth
            # Boundary α → far outside 0, smooth decay
            w_outside = alpha_at_boundary * (1 - _smoothstep(t))
        else:
            w_outside = torch.zeros_like(w_total)
        w_outside = torch.where(outside, w_outside, torch.zeros_like(w_outside))

        # Merge single-box contribution: both inside and outside give a w, take max
        w_box = torch.maximum(w_inside, w_outside)
        w_total = torch.maximum(w_total, w_box)  # Multiple boxes: take max

    return w_total  # [N]


import torch

import torch

def _hash4(c: torch.Tensor) -> torch.Tensor:
    c = c.long()
    return (c[:,0] << 26) | (c[:,1] << 20) | (c[:,2] << 14) | (c[:,3] << 8)

def weight_ring_distance1_outside(coords: torch.Tensor, boxes_zyx, alpha: float = 0.5) -> torch.Tensor:
    """
    Only blend voxels that are outside the box with L∞ distance==1: w=alpha.
    Inside box at any position: w=1; outside with distance>=2: w=0. Multiple boxes: take max.
    coords: [N,4] (1/2/3 -> x/y/z; boxes use closed intervals)
    Returns: w ∈ [0,1], [N]
    """
    if not boxes_zyx:
        return torch.zeros(coords.size(0), device=coords.device)

    # Use integer grid to avoid floating-point ==1 comparison issues
    x = coords[:,1].long()
    y = coords[:,2].long()
    z = coords[:,3].long()
    device = coords.device

    # Use float for w, convenient for subsequent linear blending
    w_total = torch.zeros_like(x, dtype=torch.float32, device=device)

    for (zr, yr, xr) in boxes_zyx:
        z0, z1 = int(zr[0]), int(zr[1])
        y0, y1 = int(yr[0]), int(yr[1])
        x0, x1 = int(xr[0]), int(xr[1])

        # inside (closed interval)
        inside = (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1) & (z >= z0) & (z <= z1)

        # L∞ external distance (strict integer "layer count")
        dx_out = torch.maximum(x0 - x, x - x1).clamp_min(0)
        dy_out = torch.maximum(y0 - y, y - y1).clamp_min(0)
        dz_out = torch.maximum(z0 - z, z - z1).clamp_min(0)
        d_out = torch.maximum(torch.maximum(dx_out, dy_out), dz_out)  # long

        # Inside: w=1; outside with d_out==1: w=alpha; outside with d_out>=2: w=0
        w_box = torch.zeros_like(w_total)
        w_box = torch.where(inside, torch.ones_like(w_box), w_box)
        w_box = torch.where((~inside) & (d_out == 1), torch.full_like(w_box, float(alpha)), w_box)

        # Multiple boxes: take max (whichever favors new wins)
        w_total = torch.maximum(w_total, w_box)

    return w_total

def blend_feats_ring1_outside(sample, ori_sample, boxes_zyx, alpha=0.5):
    """
    Only blend at voxels with distance=1 outside box; inside=new, far outside=original.
    For new coordinates not present in ori: force new (w=1).
    """
    device = sample.feats.device
    coords      = sample.coords.to(device)
    coords_ori  = ori_sample.coords.to(device)

    # Weight
    w = weight_ring_distance1_outside(coords, boxes_zyx, alpha=alpha).to(device)  # [N], float32

    # New coordinates (not in ori): force new
    is_new = ~torch.isin(_hash4(coords), _hash4(coords_ori))
    w = torch.where(is_new, torch.ones_like(w), w)

    # Coordinate alignment (iterate over tensor rows, use c.tolist())
    ori_map = {tuple(c.tolist()): i for i, c in enumerate(coords_ori)}  # Key fix: don't .tolist() then .tolist() again
    idx_s, idx_o = [], []
    for i, c in enumerate(coords):
        j = ori_map.get(tuple(c.tolist()))
        if j is not None:
            idx_s.append(i); idx_o.append(j)

    if not idx_s:
        return sample

    idx_s = torch.tensor(idx_s, dtype=torch.long, device=device)
    idx_o = torch.tensor(idx_o, dtype=torch.long, device=device)

    new_feats = sample.feats.clone()
    # Ensure weight dtype matches feats
    w_exp = w[idx_s].unsqueeze(1).to(new_feats.dtype)  # [K,1]
    new_feats[idx_s] = w_exp * new_feats[idx_s] + (1 - w_exp) * ori_sample.feats[idx_o].to(new_feats.dtype)
    return sample.replace(feats=new_feats)


def _stat(x, name):
    x = x.detach()
    return {
        "name": name,
        "shape": tuple(x.shape),
        "min": float(x.min()),
        "max": float(x.max()),
        "mean": float(x.mean()),
        "abs_mean": float(x.abs().mean()),
        "norm": float(x.norm()),
        "max_abs": float(x.abs().max()),
        "frac_tiny(<1e-12)": float((x.abs() < 1e-12).float().mean()),
    }

def make_loss_closure(decode_voxel_fn, edit_mask, ortho_scale, tx, ty, flip_y):
    # You can also pass tau/rot90k etc. as parameters
    def closure(z):
        with torch.enable_grad():
            sigma = decode_voxel_fn(z)
            # Consistent with main pipeline (soft version is better for gradients)
            tau = 0.55
            sil = _silhouette_from_sigma(sigma, tau=tau)    # Or your OR-soft version
            sil_img = project_ortho_no_center(
                sil, out_hw=edit_mask.shape[-2:],
                ortho_scale=ortho_scale, tx=tx, ty=ty, flip_y=flip_y
            )
            sil_img = apply_orient_2d(sil_img, rot90k=3)
            # Stable BCE without clamp, to avoid clamp cutting off gradients
            eps = 1e-6
            pred = sil_img.clamp(eps, 1-eps)  # Only for numerical stability (if you suspect clamp cuts gradients, try removing temporarily)
            L = -(edit_mask*torch.log(pred) + (1-edit_mask)*torch.log(1-pred)).mean()
            return L
    return closure


def fd_directional_check(loss_closure, x, num_dirs=3, eps_scale=1e-3):
    """
    Verify that autograd's <grad, v> ≈ finite difference (L(x+eps v)-L(x-eps v))/(2eps)
    - loss_closure: a function f(z) -> scalar L (will re-run forward under no_grad)
    - x: current latent (leaf tensor)
    """
    x = x.detach()
    x.requires_grad_(True)
    L = loss_closure(x)                # scalar
    g = torch.autograd.grad(L, x)[0]   # autograd gradient

    gdot_vs, fd_vs, rel_errs = [], [], []
    for k in range(num_dirs):
        v = torch.randn_like(x)
        v = v / (v.norm() + 1e-8)
        # Directional derivative (autograd)
        gdot = float((g * v).sum())

        # Finite difference (no_grad)
        with torch.no_grad():
            eps = eps_scale * (x.std() + 1e-8)
            Lp = loss_closure(x + eps * v)
            Lm = loss_closure(x - eps * v)
            fd = float((Lp - Lm) / (2 * eps))

        rel = abs(gdot - fd) / (abs(fd) + 1e-8)
        gdot_vs.append(gdot); fd_vs.append(fd); rel_errs.append(rel)

    return {"g_v": gdot_vs, "fd": fd_vs, "rel_err": rel_errs, "grad_norm": float(g.norm())}
@torch.no_grad()
def print_stats(d):
    print("— gradient stats —")
    for k,v in d.items():
        print(f"{k}: {v}")

def grad_chain_diagnostics(L, x_t, *, sigma, sil, sil_img):
    """
    Print gradient strength of L w.r.t. each layer: dL/d(sil_img), dL/d(sil), dL/d(sigma), dL/d(x_t)
    Note: must be called after L is computed and the computation graph is still alive.
    """
    stats = {}

    # dL/d(sil_img)
    g_silimg = torch.autograd.grad(L, sil_img, retain_graph=True, create_graph=False, allow_unused=True)[0]
    if g_silimg is not None:
        stats["dL/d(sil_img)"] = _stat(g_silimg, "g_silimg")
    else:
        stats["dL/d(sil_img)"] = "None (gradient did not reach sil_img, usually due to clamp/detach/hard-threshold truncation)"

    # dL/d(sil)
    g_sil = torch.autograd.grad(L, sil, retain_graph=True, create_graph=False, allow_unused=True)[0]
    if g_sil is not None:
        stats["dL/d(sil)"] = _stat(g_sil, "g_sil")
    else:
        stats["dL/d(sil)"] = "None (projection/orientation processing may have truncated gradient)"

    # dL/d(sigma)
    g_sigma = torch.autograd.grad(L, sigma, retain_graph=True, create_graph=False, allow_unused=True)[0]
    if g_sigma is not None:
        stats["dL/d(sigma)"] = _stat(g_sigma, "g_sigma")
    else:
        stats["dL/d(sigma)"] = "None (decoder may have internal detach/non-differentiable ops)"

    # dL/d(x_t)
    g_x = torch.autograd.grad(L, x_t, retain_graph=False, create_graph=False, allow_unused=True)[0]
    if g_x is not None:
        stats["dL/d(x_t)"] = _stat(g_x, "g_x")
    else:
        stats["dL/d(x_t)"] = "None (x_t did not participate in differentiable path / was replaced)"

    print_stats(stats)
    return g_x, stats
def save_overlay_png(pred01, gt01, path):
    """pred01/gt01: [1,1,H,W] or [H,W]. Red=GT(mask), Green=Pred(projection)."""
    import numpy as np, os
    from PIL import Image
    p = pred01.detach().squeeze().clamp(0,1).cpu().numpy()
    g = gt01.detach().squeeze().clamp(0,1).cpu().numpy()
    H, W = p.shape
    rgb = np.zeros((H, W, 3), dtype=np.uint8)
    rgb[..., 0] = (g * 255).astype(np.uint8)  # R: GT
    rgb[..., 1] = (p * 255).astype(np.uint8)  # G: Pred
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(rgb).save(path)
def _to_mask_tensor(x, like_tensor):
        if isinstance(x, Image.Image):
            x = np.array(x)
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        if x.ndim == 2:
            x = x[None, None, ...]  # [1,1,H,W]
        elif x.ndim == 3:           # [B,H,W] or [1,H,W]
            x = x[:, None, ...]
        x = x.to(like_tensor.device, dtype=like_tensor.dtype)
        return (x > 0.5).float()

def _save_gray_png(tensor_01, path):
    img = tensor_01.detach().squeeze().clamp(0,1).mul(255).byte().cpu().numpy()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(img).save(path)

def _silhouette_from_sigma(
    sigma,                       # [B,1,*,*,*]
    depth_axis='y',              # 'z'|'y'|'x'
    invert_depth=False,
    prob_mode='logit',           # 'logit': use sigmoid(σ/τ); 'density': treat σ as non-negative density then compress to [0,1]
    tau=0.6,
    hard=False,                  # True: forward 0/1 + STE; False: pure soft
    thr_voxel=0.5,
    edge_blur_ks=1.2,
    # Additional:
    agg='poisson',               # 'poisson' (recommended) | 'noisy_or'
    kappa=10.0,                   # Poisson-OR strength (can anneal 4→12)
    topk=4                    # Only use top K depths; None = use all, recommend 8 or 16
):
    """
    Goal: any hit along the ray makes it white.
    - Poisson-OR:  S = 1 - exp(-kappa * sum_z p_z)      (recommended, stable gradients)
    - Noisy-OR:    S = 1 - Π_z (1 - p_z)                 (gradients diminish with many depths)
    - hard=True: forward hard threshold + straight-through estimator (backprop through soft S)
    Returns: [B,1,H,W]
    """
    import torch
    import torch.nn.functional as F

    # --- 1) Move "depth axis" to dim=2 -> [B,1,D,H,W]
    axis_to_dim = {'z': 2, 'y': 3, 'x': 4}
    da = axis_to_dim[depth_axis]
    if da != 2:
        perm = [0,1,2,3,4]
        perm[2], perm[da] = perm[da], perm[2]
        sigma = sigma.permute(*perm).contiguous()
    if invert_depth:
        sigma = sigma.flip(2)

    # --- 2) Convert to probability p_z ∈ [0,1]
    if prob_mode == 'logit':
        p = torch.sigmoid(sigma / tau)       # If σ is logit, σ=0 <-> p=0.5
        thr = float(thr_voxel)               # Usually 0.5 (equivalent to σ>0)
    elif prob_mode == 'density':
        # If σ is non-negative density, 1-exp(-c*σ) is more physically meaningful; here a conservative version:
        p = torch.clamp(sigma, min=0)
        p = p / (p.max().detach() + 1e-8)
        thr = float(thr_voxel)
    else:
        raise ValueError("prob_mode must be 'logit' or 'density'.")

    # --- 3) Only take Top-K depths (optional; surface voxels usually respond on only a few layers)
    if topk is not None and topk > 0 and topk < p.shape[2]:
        p = torch.topk(p, k=topk, dim=2).values  # [B,1,K,H,W]

    # --- 4) Composite along depth
    if agg == 'poisson':
        # S = 1 - exp(-kappa * sum_z p_z)
        S_soft = 1.0 - torch.exp(-kappa * p.sum(dim=2))
    elif agg == 'noisy_or':
        # S = 1 - Π_z (1 - p_z)
        S_soft = 1.0 - torch.prod(1.0 - p + 1e-6, dim=2)
    else:
        raise ValueError("agg must be 'poisson' or 'noisy_or'.")

    # --- 5) Optional 2D smoothing (soften aliasing)
    if edge_blur_ks and edge_blur_ks > 1:
        k = int(edge_blur_ks); pad = k//2
        S_soft = F.avg_pool2d(F.pad(S_soft, (pad,pad,pad,pad), mode='reflect'), k, stride=1)

    if not hard:
        return S_soft  # [B,1,H,W]

    # --- 6) Hard silhouette + STE (forward hard, backprop through soft)
    P_hard = (p.max(dim=2).values > thr).float()   # [B,1,H,W]
    return P_hard + (S_soft - S_soft.detach())

def apply_orient_2d(img, flip_x=False, flip_y=False, rot90k=0, swap_xy=False):
    """img: [B,1,H,W]. Apply 2D orientation correction as needed; differentiable. rot90k ∈ {0,1,2,3}"""
    # Rotate 90*k
    
    k = int(rot90k) % 4
    k = 1
    if k == 1:  # +90°
        img = img.transpose(-2, -1).flip(-2)
    elif k == 2:  # 180°
        img = img.flip(-2).flip(-1)
    elif k == 3:  # -90°
        img = img.transpose(-2, -1).flip(-1)
    # Swap XY (if needed)
    if swap_xy:
        img = img.transpose(-2, -1)
    # Horizontal/vertical flip
    if flip_x:
        img = img.flip(-1)
    if flip_y:
        img = img.flip(-2)
    return img
def project_ortho_no_center(sil_world, out_hw, ortho_scale: float,
                                tx: float = 0.0, ty: float = 0.0, flip_y: bool = False):
    """
    Orthographic projection sampling without "auto-centering":
        - sil_world: [B,1,Hs,Ws], corresponds to front-view projection of world plane (x,y)∈[-0.5,0.5]^2
        - out_hw: (H,W) target resolution (= edit mask resolution)
        - ortho_scale: == Blender's cam.data.ortho_scale (world width covered by image width)
        - tx, ty: world-unit translation (same semantics as Blender), default 0
        - flip_y: vertical flip switch (set True if upside-down)
    """
    B = sil_world.shape[0]
    H, W = int(out_hw[0]), int(out_hw[1])
    theta = sil_world.new_zeros(B, 2, 3)
    theta[:, 0, 0] = ortho_scale
    theta[:, 0, 2] = 2.0 * tx
    theta[:, 1, 1] = (-ortho_scale if flip_y else ortho_scale)
    theta[:, 1, 2] = (-2.0 * ty if flip_y else 2.0 * ty)
    grid = F.affine_grid(theta, size=(B, 1, H, W), align_corners=True)
    img = F.grid_sample(sil_world, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
    return img
# ---------- Shape/Spatial Alignment & Pooling ----------
def _resolve_viz_base(feature_path, fallback="."):
    if feature_path is None:
        return fallback
    # If exists and is a directory, use directly
    if os.path.exists(feature_path) and os.path.isdir(feature_path):
        return feature_path
    # If exists and is a file -> use its parent directory
    if os.path.exists(feature_path) and os.path.isfile(feature_path):
        return os.path.dirname(feature_path)
    # If doesn't exist but looks like a "file path" (has extension) -> use parent directory
    root, ext = os.path.splitext(feature_path)
    if ext:  # .pkl/.pt/.npy/.png etc.
        return os.path.dirname(feature_path) or fallback
    # Otherwise treat as directory
    return feature_path
def _align_spatial(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """
    Align x's dimensions to ref: (B,C,[D,]H,W), and expand along channel dim.
    """
    while x.dim() < ref.dim():
        x = x.unsqueeze(1)  # Add C dim
    if x.size(1) != ref.size(1):
        x = x.expand(-1, ref.size(1), *([-1] * (ref.dim() - 2)))
    return x


def _avg_pool_nd(x: torch.Tensor, k: int) -> torch.Tensor:
    """
    2D/3D adaptive average pooling (stride=1, padding=kernel//2).
    """
    if x.dim() == 4:
        return F.avg_pool2d(x, k, 1, k // 2)
    elif x.dim() == 5:
        k3 = 2 * max(1, k // 2) + 1 if isinstance(k, int) else k
        return F.avg_pool3d(x, k3, 1, k3 // 2)
    return x


def _max_pool_nd(x: torch.Tensor, k: int) -> torch.Tensor:
    """
    2D/3D adaptive max pooling (stride=1, padding=kernel//2).
    """
    if x.dim() == 4:
        return F.max_pool2d(x, k, 1, k // 2)
    elif x.dim() == 5:
        k3 = 2 * max(1, k // 2) + 1 if isinstance(k, int) else k
        return F.max_pool3d(x, k3, 1, k3 // 2)
    return x


# ---------- Mask Construction/Combination/Metrics ----------

def apply_mask_blend(new: torch.Tensor, old: torch.Tensor, soft_mask: Optional[torch.Tensor]) -> torch.Tensor:
    """
    Convex combination using soft mask in spatial domain: soft*new + (1-soft)*old.
    If soft_mask is None, returns new directly.
    """
    if soft_mask is None:
        return new
    return soft_mask * new + (1.0 - soft_mask) * old


def combine_masks(
    manual_soft: Optional[torch.Tensor],
    auto_soft: Optional[torch.Tensor],
    mode: str = "blend",
    w: float = 0.5
) -> Optional[torch.Tensor]:
    """
    Combine manual and auto masks:
      - "union":   A ∪ B  ≈ A + B - A*B
      - "intersect": A ∩ B = A*B
      - "blend":   (1-w)*manual + w*auto
    If either is None, returns the other; if both None, returns None.
    """
    if manual_soft is None and auto_soft is None:
        return None
    if manual_soft is None:
        return auto_soft
    if auto_soft is None:
        return manual_soft
    if mode == "union":
        return torch.clamp(manual_soft + auto_soft - manual_soft * auto_soft, 0, 1)
    if mode == "intersect":
        return manual_soft * auto_soft
    # default blend
    return torch.clamp((1.0 - w) * manual_soft + w * auto_soft, 0, 1)


def build_soft_masks(
    mask: Optional[torch.Tensor],
    ref: torch.Tensor,
    feather: int = 2,
    guard: int = 1
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    From binary/soft mask, generate:
      - soft: feathered soft mask ([0,1])
      - guard_band: guard band (dilation - original mask), [0,1]
    """
    if mask is None:
        return None, None
    m = _align_spatial(mask.float(), ref)
    # Feathering
    soft = _avg_pool_nd(m, 2 * feather + 1) if feather > 0 else m
    soft = soft.clamp(0, 1)
    # Guard band
    if guard > 0:
        dil = _max_pool_nd(m, 2 * guard + 1)
        guard_band = (dil - m).clamp(0, 1)
    else:
        guard_band = torch.zeros_like(soft)
    return soft, guard_band


def make_soft_mask(m: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    """
    Simple soft mask compatible with legacy logic (light average filtering).
    """
    if m is None:
        return None
    m = m.float()
    if m.dim() == 4:
        m = F.avg_pool2d(m, kernel_size=5, stride=1, padding=2)
    elif m.dim() == 5:
        m = F.avg_pool3d(m, kernel_size=3, stride=1, padding=1)
    return m.clamp(0, 1)


def binarize(x: torch.Tensor, thr: float = 0.5) -> torch.Tensor:
    """Binarize soft mask to {0,1}."""
    return (x >= thr).float()


def mask_metrics(
    auto_soft: Optional[torch.Tensor],
    manual_soft: Optional[torch.Tensor],
    thr: float = 0.5,
    eps: float = 1e-6
) -> Optional[dict]:
    """
    Compute IoU/Precision/Recall of auto vs manual (binarized at thr).
    Input can be 2D/3D/Batched tensors, automatically summed.
    """
    if auto_soft is None or manual_soft is None:
        return None
    a = binarize(auto_soft, thr)
    m = binarize(manual_soft, thr)
    inter = (a * m).sum()
    union = (a + m - a * m).sum() + eps
    iou = (inter + eps) / union
    prec = (inter + eps) / (a.sum() + eps)
    rec = (inter + eps) / (m.sum() + eps)
    return {"iou": iou.item(), "prec": prec.item(), "rec": rec.item()}


# ---------- Auto-mask Construction & Scoring ----------

def reduce_vector_norm(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    (B,C,[D,]H,W) -> (B,1,[D,]H,W), L2 norm.
    """
    return (x.pow(2).sum(dim=1, keepdim=True) + eps).sqrt()


def build_auto_mask_from_map(
    score_map: torch.Tensor,
    ref: Optional[torch.Tensor] = None,  # Kept for compatibility, not used
    quantile: float = 0.7,
    tau: float = 0.25,
    feather: int = 2
) -> torch.Tensor:
    """
    Generate soft mask from score map (higher = more likely to edit):
      - per-sample quantile threshold
      - temperature sigmoid
      - light feathering
    Returns values in range [0,1].
    """
    s = score_map
    B = s.size(0)
    s_flat = s.view(B, -1)
    th = torch.quantile(s_flat, torch.tensor([quantile], device=s.device), dim=1) \
         .view(B, 1, *([1] * (s.dim() - 2)))
    soft = torch.sigmoid((s - th) / max(1e-6, tau))
    if feather > 0:
        k = 2 * feather + 1
        soft = _avg_pool_nd(soft, k)
    return soft.clamp(0, 1)


# ---------- Visualization (np normalization + 2x2 grid) ----------

def _to_np_2d(x: ArrayLike) -> np.ndarray:
    """
    Convert torch / numpy data to (H,W) numpy.float32, normalized to [0,1].
    Rules:
      - If batched: take first sample
      - If (C,D,H,W): take ch0 + middle slice of D
      - If (C,H,W) or (1,H,W): take ch0
    """
    if isinstance(x, torch.Tensor):
        a = x.detach().float().cpu().numpy()
    else:
        a = np.asarray(x, dtype=np.float32)

    if a.ndim >= 4:
        a = a[0]
    if a.ndim == 4:
        a = a[0][a.shape[1] // 2]  # (C,D,H,W) -> ch0, center slice
    if a.ndim == 3:
        a = a[0]  # (1,H,W) or (C,H,W) -> ch0
    if a.ndim != 2:
        a = a.reshape(a.shape[-2], a.shape[-1])

    mn, mx = a.min(), a.max()
    if mx - mn < 1e-8:
        return np.zeros_like(a, dtype=np.float32)
    return ((a - mn) / (mx - mn + 1e-8)).astype(np.float32)


def _save_panel_png(arr2d, path_png):
        os.makedirs(os.path.dirname(path_png), exist_ok=True)
        if HAS_PIL:
            Image.fromarray((arr2d*255).astype(np.uint8), mode="L").convert("RGB").save(path_png)
        else:
            # Fallback to .npy when PIL is not available
            np.save(path_png.replace(".png", ".npy"), arr2d)


def _save_grid_2x2(
    auto: Optional[np.ndarray],
    manual: Optional[np.ndarray],
    edited_mask: Optional[np.ndarray],
    delta: Optional[np.ndarray],
    out_png: str
) -> Optional[str]:
    """
    Combine four grayscale panels (normalized to [0,1] 2D numpy) into a 2x2 grid image and save.
    Returns None if PIL is not available.
    Panel order:
      [auto, manual]
      [edited_mask, delta]
    """
    if not HAS_PIL:
        return None

    def to_rgb(x: Optional[np.ndarray], ref=(256, 256)) -> Image.Image:
        if x is None:
            return Image.new("RGB", ref, (0, 0, 0))
        im = Image.fromarray((x * 255).astype(np.uint8), mode="L").convert("RGB")
        return im

    ref_img = to_rgb(auto) if auto is not None else to_rgb(manual)
    w, h = ref_img.size
    A = to_rgb(auto, (w, h))
    M = to_rgb(manual, (w, h))
    E = to_rgb(edited_mask, (w, h))
    D = to_rgb(delta, (w, h))

    grid = Image.new("RGB", (2 * w, 2 * h))
    grid.paste(A, (0, 0))
    grid.paste(M, (w, 0))
    grid.paste(E, (0, h))
    grid.paste(D, (w, h))
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    grid.save(out_png)
    return out_png


__all__ = [
    # Shape/pooling
    "_align_spatial", "_avg_pool_nd", "_max_pool_nd",
    # Mask and blending
    "apply_mask_blend", "combine_masks", "build_soft_masks", "make_soft_mask",
    "binarize", "mask_metrics",
    # auto-mask
    "reduce_vector_norm", "build_auto_mask_from_map",
    # Visualization
    "_to_np_2d", "_save_grid_2x2",
    # Flags
    "HAS_PIL",
]

def replace_feats(sample, ori_sample, sample_indices, ori_indices):
    sample_feats = sample.feats.clone()
    sample_feats[sample_indices] = ori_sample.feats[ori_indices]
    return sample_feats

def build_coord_index_map(sample_coords, ori_coords, mask):
    """
    sample_coords: [N, 3] int tensor
    ori_coords: [M, 3] int tensor
    mask: [N] uint8 or bool tensor, 0 means to be replaced
    Returns:
        sample_indices: indices in sample that need replacement
        ori_indices: corresponding indices in ori_sample
    """
    # Find coords in sample that need replacement
    target_indices = torch.nonzero(mask == 0, as_tuple=False).squeeze(1)
    target_coords = sample_coords[target_indices]  # [K, 3]

    # Use hash table for fast matching
    ori_map = {tuple(c.tolist()): i for i, c in enumerate(ori_coords)}
    matched_sample_idx = []
    matched_ori_idx = []

    for i, coord in zip(target_indices.tolist(), target_coords.tolist()):
        key = tuple(coord)
        if key in ori_map:
            matched_sample_idx.append(i)
            matched_ori_idx.append(ori_map[key])

    return (
        torch.tensor(matched_sample_idx, dtype=torch.long),
        torch.tensor(matched_ori_idx, dtype=torch.long)
    )
def combine(source,sample):
    return sample

import json, ast

import torch
import torch.nn.functional as F

def weight_ring1_from_rawmask(coords: torch.Tensor, raw_mask: torch.Tensor, alpha: float = 0.5) -> torch.Tensor:
    """
    Compute per-coordinate blending weight from raw_mask ([B,1,64,64,64]):
      - Inside mask: w=1
      - Outside mask with L∞ distance==1 (one-ring outside): w=alpha
      - Otherwise: w=0
    Coordinate format (b,x,y,z), aligned with grid[b,0,x,y,z].
    """
    grid = raw_mask.to(coords.device)
    if grid.dtype != torch.bool:  # Convert 0/1 float to bool
        grid = grid > 0

    # Morphological dilation for L∞-1 neighborhood: max_pool3d kernel=3, padding=1
    # Note: pooling must be done on float, then threshold back to bool
    dilated = F.max_pool3d(grid.float(), kernel_size=3, stride=1, padding=1) > 0.5
    ring = (~grid) & dilated  # One-ring outside

    b, x, y, z = coords.long().unbind(dim=1)  # (b,x,y,z)
    inside = grid[b, 0, x, y, z]
    in_ring = ring[b, 0, x, y, z]

    w = torch.zeros(coords.size(0), device=coords.device, dtype=torch.float32)
    w = torch.where(inside, torch.ones_like(w), w)
    w = torch.where((~inside) & in_ring, torch.full_like(w, float(alpha)), w)
    return w  # [N]


def blend_feats_ring1_outside_rawmask(sample, ori_sample, raw_mask: torch.Tensor, alpha=0.5):
    """
    raw_mask version blending:
      - Inside mask: use sample's new features
      - One-ring outside mask (L∞=1): blend new/old by alpha
      - Far outside mask: keep ori
      - For "new coordinates" (not in ori): force new (w=1)
    """
    device = sample.feats.device
    coords     = sample.coords.to(device)      # [Ns,4] (b,x,y,z)
    coords_ori = ori_sample.coords.to(device)  # [No,4]

    # Compute weights
    w = weight_ring1_from_rawmask(coords, raw_mask.to(device), alpha=alpha)  # [Ns]

    # New coordinates: force w=1
    # Assumes _hash4(c) -> int64 hash function exists (consistent with existing code)
    is_new = ~torch.isin(_hash4(coords), _hash4(coords_ori))
    w = torch.where(is_new, torch.ones_like(w), w)

    # Find overlapping rows between new and old coords for blending
    # Keep the dict lookup approach, simple and reliable
    ori_map = {tuple(c.tolist()): i for i, c in enumerate(coords_ori)}
    idx_s, idx_o = [], []
    for i, c in enumerate(coords):
        j = ori_map.get(tuple(c.tolist()))
        if j is not None:
            idx_s.append(i); idx_o.append(j)

    if not idx_s:
        return sample  # No overlap, return directly

    idx_s = torch.tensor(idx_s, dtype=torch.long, device=device)
    idx_o = torch.tensor(idx_o, dtype=torch.long, device=device)

    # Blend: new = w*new + (1-w)*old
    new_feats = sample.feats.clone()
    w_exp = w[idx_s].unsqueeze(1).to(new_feats.dtype)  # [K,1]
    new_feats[idx_s] = w_exp * new_feats[idx_s] + (1 - w_exp) * ori_sample.feats[idx_o].to(new_feats.dtype)

    return sample.replace(feats=new_feats)

def _normalize_boxes_zyx(mask_list):
    """
    Normalize mask_list to a list of "multiple boxes", with fixed order [z,y,x].
      - Single box: [[z0,z1],[y0,y1],[x0,x1]] -> [ [[z0,z1],[y0,y1],[x0,x1]] ]
      - Multiple boxes: [[[z0,z1],[y0,y1],[x0,x1]], ...] -> as-is
    Performs basic clamp to [0, 64], but does not change "closed interval" semantics (still <=).
    """
    if mask_list is None:
        return []

    if isinstance(mask_list, str):
        try:
            mask_list = json.loads(mask_list)
        except Exception:
            mask_list = ast.literal_eval(mask_list)

    # Single box -> multiple boxes
    if len(mask_list) == 3 and all(isinstance(v, (list, tuple)) for v in mask_list):
        boxes = [mask_list]
    else:
        boxes = mask_list

    out = []
    for box in boxes:
        assert len(box) == 3, f"mask box must have 3 axes [z,y,x], got: {box}"
        (z0, z1), (y0, y1), (x0, x1) = box

        def clamp_pair(a, b):
            a = int(max(0, min(64, a)))
            b = int(max(0, min(64, b)))
            if b < a: a, b = b, a
            return a, b

        z0, z1 = clamp_pair(z0, z1)
        y0, y1 = clamp_pair(y0, y1)
        x0, x1 = clamp_pair(x0, x1)

        # Allow z1==64 as upper bound; since <= is used for comparison, coords max 63 won't overflow
        if (z1 >= z0) and (y1 >= y0) and (x1 >= x0):
            out.append([[z0, z1], [y0, y1], [x0, x1]])
    return out

def inside_any_zyx_mask(t: torch.Tensor, boxes_zyx):
    """Return boolean vector: whether each point falls inside any [z,y,x] box (closed interval)."""
    if not boxes_zyx:
        return torch.zeros(t.size(0), dtype=torch.bool, device=t.device)
    x, y, z = t[:,1], t[:,2], t[:,3]
    mask = torch.zeros_like(x, dtype=torch.bool)
    for (zr, yr, xr) in boxes_zyx:
        z0,z1 = zr; y0,y1 = yr; x0,x1 = xr
        m = (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1) & (z >= z0) & (z <= z1)
        mask = mask | m
    return mask

def filter_coords_by_boxes_zyx(t: torch.Tensor, boxes_zyx, keep: str):
    """keep='inside' or 'outside'. Returns filtered coords (preserving dtype/shape)."""
    m = inside_any_zyx_mask(t, boxes_zyx)
    return t[m] if keep == 'inside' else t[~m]

import torch

def merge_coords(
    coords1: torch.Tensor,   # [N1, 4] (b,x,y,z)
    coords:  torch.Tensor,   # [N2, 4] (b,x,y,z)
    raw_mask: torch.Tensor,  # [B,1,64,64,64], bool or values>0
    device=None,             # Kept for legacy signature (ignored)
    hole_fill_radius: int = 0  # Kept for legacy signature (ignored)
) -> torch.Tensor:
    # Ensure dtype/device consistency (stay on GPU)
    grid   = raw_mask.to(coords.device)
    if grid.dtype != torch.bool:
        grid = grid > 0

    coords1 = coords1.to(dtype=torch.long, device=coords.device)
    coords  = coords.to(dtype=torch.long, device=coords.device)

    # Inside check: grid[b,0,x,y,z]
    b1, x1, y1, z1 = coords1.unbind(dim=1)
    b0, x0, y0, z0 = coords.unbind(dim=1)

    inside1 = grid[b1, 0, x1, y1, z1]
    inside0 = grid[b0, 0, x0, y0, z0]

    # Use new coords1 inside mask, keep old coords outside mask
    out = torch.cat([coords1[inside1], coords[~inside0]], dim=0)
    out = torch.unique(out, dim=0).to(torch.int32).contiguous()
    return out


def read_feature(feature_path, img_name):
    """
    Read the tensor corresponding to img_name from feature_path (e.g., features.pkl).
    Returns None if img_name is not found.
    """
    with open(feature_path, 'rb') as f:
        while True:
            try:
                item = pickle.load(f)
                if img_name in item:
                    data = item[img_name]
                    if isinstance(data, dict) and data.get('type') == 'SparseTensor':
                        feats = torch.from_numpy(data['feats']).float()
                        coords = torch.from_numpy(data['coords']).int()
                        return sp.SparseTensor(feats=feats, coords=coords)
                    else:
                        return data
            except EOFError:
                break
    return None  # Not found


def load_img_features(feature_path):
    img_feature_dict = {}

    # Open file in rb mode
    with open(feature_path, 'rb') as f:
        os.makedirs(os.path.dirname(feature_path), exist_ok=True)
        while True:
            try:
                # Load one dictionary
                feature_item = pickle.load(f)
                for key, value in feature_item.items():
                    if key.endswith('_img'):
                        img_feature_dict[key] = value
            except EOFError:
                break  # Finished reading, exit loop

    return img_feature_dict

def save_feature(feature_path, img_name, sample):
    # Ensure directory exists
    os.makedirs(os.path.dirname(feature_path), exist_ok=True)

    # Ensure sample is on CPU
    if isinstance(sample, torch.Tensor):
        sample_cpu = sample.detach().cpu()
        feature_item = {img_name: sample_cpu}
    elif isinstance(sample, sp.SparseTensor):
        feature_item = {
            img_name: {
                'type': 'SparseTensor',
                'feats': sample.feats.cpu().numpy(),
                'coords': sample.coords.cpu().numpy(),
            }
        }
    else:
        raise ValueError(f"[ERROR] Unsupported sample type: {type(sample)}")

    # Write to pickle file in append mode
    with open(feature_path, 'ab') as f:
        pickle.dump(feature_item, f)

# Dictionary utils
def _dict_merge(dicta, dictb, prefix=''):
    """
    Merge two dictionaries.
    """
    assert isinstance(dicta, dict), 'input must be a dictionary'
    assert isinstance(dictb, dict), 'input must be a dictionary'
    dict_ = {}
    all_keys = set(dicta.keys()).union(set(dictb.keys()))
    for key in all_keys:
        if key in dicta.keys() and key in dictb.keys():
            if isinstance(dicta[key], dict) and isinstance(dictb[key], dict):
                dict_[key] = _dict_merge(dicta[key], dictb[key], prefix=f'{prefix}.{key}')
            else:
                raise ValueError(f'Duplicate key {prefix}.{key} found in both dictionaries. Types: {type(dicta[key])}, {type(dictb[key])}')
        elif key in dicta.keys():
            dict_[key] = dicta[key]
        else:
            dict_[key] = dictb[key]
    return dict_


def dict_merge(dicta, dictb):
    """
    Merge two dictionaries.
    """
    return _dict_merge(dicta, dictb, prefix='')


def dict_foreach(dic, func, special_func={}):
    """
    Recursively apply a function to all non-dictionary leaf values in a dictionary.
    """
    assert isinstance(dic, dict), 'input must be a dictionary'
    for key in dic.keys():
        if isinstance(dic[key], dict):
            dic[key] = dict_foreach(dic[key], func)
        else:
            if key in special_func.keys():
                dic[key] = special_func[key](dic[key])
            else:
                dic[key] = func(dic[key])
    return dic


def dict_reduce(dicts, func, special_func={}):
    """
    Reduce a list of dictionaries. Leaf values must be scalars.
    """
    assert isinstance(dicts, list), 'input must be a list of dictionaries'
    assert all([isinstance(d, dict) for d in dicts]), 'input must be a list of dictionaries'
    assert len(dicts) > 0, 'input must be a non-empty list of dictionaries'
    all_keys = set([key for dict_ in dicts for key in dict_.keys()])
    reduced_dict = {}
    for key in all_keys:
        vlist = [dict_[key] for dict_ in dicts if key in dict_.keys()]
        if isinstance(vlist[0], dict):
            reduced_dict[key] = dict_reduce(vlist, func, special_func)
        else:
            if key in special_func.keys():
                reduced_dict[key] = special_func[key](vlist)
            else:
                reduced_dict[key] = func(vlist)
    return reduced_dict


def dict_any(dic, func):
    """
    Recursively apply a function to all non-dictionary leaf values in a dictionary.
    """
    assert isinstance(dic, dict), 'input must be a dictionary'
    for key in dic.keys():
        if isinstance(dic[key], dict):
            if dict_any(dic[key], func):
                return True
        else:
            if func(dic[key]):
                return True
    return False


def dict_all(dic, func):
    """
    Recursively apply a function to all non-dictionary leaf values in a dictionary.
    """
    assert isinstance(dic, dict), 'input must be a dictionary'
    for key in dic.keys():
        if isinstance(dic[key], dict):
            if not dict_all(dic[key], func):
                return False
        else:
            if not func(dic[key]):
                return False
    return True


def dict_flatten(dic, sep='.'):
    """
    Flatten a nested dictionary into a dictionary with no nested dictionaries.
    """
    assert isinstance(dic, dict), 'input must be a dictionary'
    flat_dict = {}
    for key in dic.keys():
        if isinstance(dic[key], dict):
            sub_dict = dict_flatten(dic[key], sep=sep)
            for sub_key in sub_dict.keys():
                flat_dict[str(key) + sep + str(sub_key)] = sub_dict[sub_key]
        else:
            flat_dict[key] = dic[key]
    return flat_dict


# Context utils
@contextlib.contextmanager
def nested_contexts(*contexts):
    with contextlib.ExitStack() as stack:
        for ctx in contexts:
            stack.enter_context(ctx())
        yield


# Image utils
def make_grid(images, nrow=None, ncol=None, aspect_ratio=None):
    num_images = len(images)
    if nrow is None and ncol is None:
        if aspect_ratio is not None:
            nrow = int(np.round(np.sqrt(num_images / aspect_ratio)))
        else:
            nrow = int(np.sqrt(num_images))
        ncol = (num_images + nrow - 1) // nrow
    elif nrow is None and ncol is not None:
        nrow = (num_images + ncol - 1) // ncol
    elif nrow is not None and ncol is None:
        ncol = (num_images + nrow - 1) // nrow
    else:
        assert nrow * ncol >= num_images, 'nrow * ncol must be greater than or equal to the number of images'
    
    if images[0].ndim == 2:
        grid = np.zeros((nrow * images[0].shape[0], ncol * images[0].shape[1]), dtype=images[0].dtype)
    else:
        grid = np.zeros((nrow * images[0].shape[0], ncol * images[0].shape[1], images[0].shape[2]), dtype=images[0].dtype)
    for i, img in enumerate(images):
        row = i // ncol
        col = i % ncol
        grid[row * img.shape[0]:(row + 1) * img.shape[0], col * img.shape[1]:(col + 1) * img.shape[1]] = img
    return grid


def notes_on_image(img, notes=None):
    img = np.pad(img, ((0, 32), (0, 0), (0, 0)), 'constant', constant_values=0)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if notes is not None:
        img = cv2.putText(img, notes, (0, img.shape[0] - 4), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def save_image_with_notes(img, path, notes=None):
    """
    Save an image with notes.
    """
    if isinstance(img, torch.Tensor):
        img = img.cpu().numpy().transpose(1, 2, 0)
    if img.dtype == np.float32 or img.dtype == np.float64:
        img = np.clip(img * 255, 0, 255).astype(np.uint8)
    img = notes_on_image(img, notes)
    cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


# debug utils

def atol(x, y):
    """
    Absolute tolerance.
    """
    return torch.abs(x - y)


def rtol(x, y):
    """
    Relative tolerance.
    """
    return torch.abs(x - y) / torch.clamp_min(torch.maximum(torch.abs(x), torch.abs(y)), 1e-12)


# print utils
def indent(s, n=4):
    """
    Indent a string.
    """
    lines = s.split('\n')
    for i in range(1, len(lines)):
        lines[i] = ' ' * n + lines[i]
    return '\n'.join(lines)

def quantize_colors(continuous_colors: np.ndarray, n_intervals: int = 20) -> np.ndarray:
    """
    Helper function: quantize continuous color values in [0, 255] range to n_intervals discrete values.
    
    It maps each value to the *midpoint* of its corresponding interval.
    """
    if n_intervals <= 0:
        return continuous_colors.astype(np.uint8)

    print(f"  > Performing quantization, n_intervals={n_intervals}...")

    # 1. Ensure values are in [0, 255] range
    # (np.uint8 auto-truncates, but we manually clip for accuracy)
    clipped_colors = np.clip(continuous_colors, 0, 255)
    
    # 2. Define interval properties
    # Use 256.0 for float division, ensuring n_intervals evenly cover 0-255
    interval_width = 256.0 / n_intervals 
    # (e.g., 256.0 / 20 = 12.8)
    
    # Midpoint offset of each interval
    midpoint_offset = interval_width / 2.0
    # (e.g., 12.8 / 2.0 = 6.4)

    # 3. Apply quantization formula
    # Core logic:
    # 1. (clipped_colors / interval_width) -> find which interval the value belongs to (e.g., 20.0 / 12.8 = 1.56)
    # 2. np.floor(...) -> truncate to get interval index (e.g., floor(1.56) = 1.0)
    # 3. (... * interval_width) -> move to interval start (e.g., 1.0 * 12.8 = 12.8)
    # 4. (... + midpoint_offset) -> move to interval midpoint (e.g., 12.8 + 6.4 = 19.2)
    
    quantized_values = (np.floor(clipped_colors / interval_width) * interval_width) + midpoint_offset
    
    # 4. Convert back to uint8
    return quantized_values.astype(np.uint8)


def features_to_rgba(features: torch.Tensor) -> np.ndarray:
    """
    Helper function: convert [N, C] feature tensor to [N, 4] RGBA colors (0-255).
    
    (Modified: uses PCA for dimensionality reduction when C > 3)
    (Modified: adds 20-interval quantization at the end)
    """
    print(f"Computing colors for {features.shape[0]} features...")
    
    if features.is_cuda:
        features = features.cpu()
    feat_np = features.numpy()
    N, C = feat_np.shape
    
    # --- Constraint: alpha is always 255 ---
    alpha = np.full((N, 1), 255, dtype=np.uint8)
    
    # Default RGB (gray)
    rgb_colors = np.full((N, 3), 200, dtype=np.uint8) 

    try:
        if C == 1:
            # (Unchanged: 1D features use heatmap)
            print("Feature C=1, applying heatmap...")
            norm_feat = (feat_np - feat_np.min()) / (feat_np.max() - feat_np.min() + 1e-6)
            cmap = plt.get_cmap('viridis')
            rgba_colors_01 = cmap(norm_feat.squeeze())
            # (These are *continuous* float values in 0-255)
            rgb_colors = (rgba_colors_01[:, :3] * 255)
            
        elif C == 3:
            # (Unchanged: 3D features directly normalized to RGB)
            print("Feature C=3, normalizing to RGB...")
            min_vals = feat_np.min(axis=0)
            max_vals = feat_np.max(axis=0)
            range_vals = max_vals - min_vals + 1e-6
            # (These are *continuous* float values in 0-255)
            rgb_colors = ((feat_np - min_vals) / range_vals * 255)
            
        elif C > 3:
            # --- *Modified PCA logic* ---
            print(f"Feature C={C}, using PCA to reduce to 3 channels for coloring.")
            
            pca = PCA(n_components=3)
            rgb_features = pca.fit_transform(feat_np)
            
            min_vals = rgb_features.min(axis=0)
            max_vals = rgb_features.max(axis=0)
            range_vals = max_vals - min_vals + 1e-6
            # (These are *continuous* float values in 0-255)
            rgb_colors = ((rgb_features - min_vals) / range_vals * 255)
            
        else: # C == 0 or C == 2
            print(f"Warning: feature dimension {C} not supported. Using default gray.")
            
    except Exception as e:
        print(f"Error during color mapping: {e}. Using default gray.")
        # Even on error, use default rgb_colors (all 200)

    # --- 4. (New step) Quantize RGB colors ---
    # Regardless of which branch in try/except (C=1, C=3, C>3 or except),
    # quantize rgb_colors (still continuous values at this point) here.
    
    print(f"Quantizing RGB colors to {5} intervals...")
    rgb_colors = quantize_colors(rgb_colors, n_intervals=5)
    
    # 5. Combine R, G, B and Alpha
    return np.hstack((rgb_colors, alpha))


def export_indexed_coords_to_glb_cubes_colored(
    out_coords: torch.Tensor,
    output_filename: str = 'voxel_cubes_scaled.glb',
    grid_size: int = 64,
    color_rgba: list = [200, 200, 200, 255],
    slat_feat: torch.Tensor = None,
):
    """
    (V3 - Robust version)
    Convert (b,x,y,z) indexed coordinates to *colored* cube grid in [0, 1] space.
    
    Uses the reliable approach of "merge geometry first, then color all at once".
    """
    
    print(f"Starting export to {output_filename}...")

    # --- 1. Extract (x,y,z) indices ---
    if out_coords.is_cuda:
        out_coords_cpu = out_coords.cpu()
    else:
        out_coords_cpu = out_coords
        
    xyz_indices_np = out_coords_cpu[:, 1:4].numpy()
    N = xyz_indices_np.shape[0]
    
    if N == 0:
        print("Error: no coordinates to export.")
        return

    print(f"Found {N} voxels... computing scaling...")

    # --- 2. Compute scaling and positions ---
    
    voxel_size = 1.0 / float(grid_size)
    
    # Create a *base* cube (no color)
    base_cube = trimesh.creation.box(extents=[voxel_size, voxel_size, voxel_size])
    num_vertices_per_cube = len(base_cube.vertices) # Should be 8
    
    # Compute world coordinate centers
    world_centers_np = (xyz_indices_np - 0.5) * voxel_size

    # --- 3. Instantiation & Merging (geometry only) ---
    
    # 1. Instantiate *uncolored* cubes (fast)
    print("Instantiating all cubes (uncolored)...")
    all_cubes = [base_cube.copy().apply_translation(p) for p in world_centers_np]
    
    # 2. Merge them
    print("Merging all cubes...")
    final_mesh = trimesh.util.concatenate(all_cubes)
    
    # --- 4. Prepare *large* color array ---
    
    total_vertices = final_mesh.vertices.shape[0]
    # Check: total_vertices should equal N * num_vertices_per_cube
    if total_vertices != N * num_vertices_per_cube:
        print(f"Warning: vertex count mismatch! {total_vertices} != {N} * {num_vertices_per_cube}")
        # Try to continue even if mismatch
        
    if slat_feat is None:
        # *********** Goal 1: Uniform color ***********
        print(f"Building uniform color array: {color_rgba}")
        
        # Create a large list (or Numpy array)
        vertex_colors = np.array([color_rgba] * total_vertices, dtype=np.uint8)
        
    else:
        # *********** Goal 2: Color by features ***********
        print("Building feature color array...")
        if slat_feat.shape[0] != N:
            print(f"Error: coordinate count ({N}) does not match feature count ({slat_feat.shape[0]}).")
            return
            
        # 1. Compute N colors for N voxels (shape: [N, 4])
        voxel_colors_np = features_to_rgba(slat_feat) 
        
        # 2. *Key*: repeat each voxel color N_vert times (8 times)
        # np.repeat copies in order, matching the vertex order from concatenate
        # [C1, C2, C3] -> [C1,C1... (8x), C2,C2... (8x), C3,C3... (8x)]
        print(f"Expanding {N} voxel colors to {total_vertices} vertices...")
        vertex_colors = np.repeat(voxel_colors_np, num_vertices_per_cube, axis=0)

        # Check shape again
        if vertex_colors.shape[0] != total_vertices:
            print(f"Error: final color array shape {vertex_colors.shape} does not match vertex count {total_vertices}.")
            # Try to resize (this is just a safety measure)
            vertex_colors = np.resize(vertex_colors, (total_vertices, 4))


    # --- 5. Apply colors *all at once* and export ---
    print("Assigning colors to merged mesh...")
    
    # *Final assignment*
    final_mesh.visual.vertex_colors = vertex_colors
    
    print(f"Exporting to {output_filename}...")
    final_mesh.export(output_filename)
    print(f"Export of {output_filename} complete.")
