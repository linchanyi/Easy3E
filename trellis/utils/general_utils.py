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

# PIL 可选依赖：没有的话，grid 仅返回 None
try:
    from PIL import Image
    HAS_PIL = True
except Exception:
    HAS_PIL = False


ArrayLike = Union[np.ndarray, torch.Tensor]

import torch

def _smoothstep(x: torch.Tensor):
    # x ∈ [0,1] → [0,1]，C1 连续
    x = x.clamp(0, 1)
    return x * x * (3 - 2 * x)

def soft_box_weight_zyx(coords: torch.Tensor,
                        boxes_zyx,
                        band_in: float = 2.0,     # 盒内向内的平滑带宽（单位：体素）
                        band_out: float = 1.0,    # 盒外向外的平滑带宽
                        alpha_at_boundary: float = 0.5  # 边界处新特征占比（0.5=对半）
                        ) -> torch.Tensor:
    """
    coords: [N,4] 其中 1/2/3 -> x/y/z
    boxes_zyx: 形如  [ [[z0,z1],[y0,y1],[x0,x1]], ... ]（闭区间）
    返回: w_new ∈ [0,1], 形状 [N]，表示“新特征占比”
    规则：
      - 盒内深处(>=band_in): w=1
      - 盒外远处(>=band_out): w=0
      - 边界处(距离=0): w=alpha_at_boundary
      - 过渡区：平滑插值
      - 多盒：取 max（任何盒子影响都能提升 w）
    """
    if not boxes_zyx:
        return torch.zeros(coords.size(0), device=coords.device)

    x, y, z = coords[:, 1].float(), coords[:, 2].float(), coords[:, 3].float()
    w_total = torch.zeros_like(x)

    for (zr, yr, xr) in boxes_zyx:
        z0, z1 = float(zr[0]), float(zr[1])
        y0, y1 = float(yr[0]), float(yr[1])
        x0, x1 = float(xr[0]), float(xr[1])

        # --- inside 逻辑：到最近面的 L∞ 距离（体素格上更贴近“格子厚度”的感觉）
        dx_in = torch.minimum(x - x0, x1 - x)
        dy_in = torch.minimum(y - y0, y1 - y)
        dz_in = torch.minimum(z - z0, z1 - z)
        d_in = torch.minimum(torch.minimum(dx_in, dy_in), dz_in)  # <0 表示不在盒内

        inside = d_in >= 0
        w_inside = torch.zeros_like(w_total)
        if band_in > 0:
            t = (d_in / band_in).clamp(0, 1)  # 0=边界, 1=带宽内侧尽头
            w_inside = alpha_at_boundary + (1 - alpha_at_boundary) * _smoothstep(t)
        else:
            w_inside = torch.ones_like(w_total)  # 无内带宽则盒内全 1
        w_inside = torch.where(inside, w_inside, torch.zeros_like(w_inside))

        # --- outside 逻辑：到盒子的 L∞ 外部距离（0=恰在边界/投影内）
        # L∞ 外距：各轴到区间的外侧距离取 max
        dx_out = torch.maximum(x0 - x, x - x1)
        dy_out = torch.maximum(y0 - y, y - y1)
        dz_out = torch.maximum(z0 - z, z - z1)
        d_out_inf = torch.maximum(torch.maximum(dx_out, dy_out), dz_out)  # >0 在盒外，=0 与边界对齐/投影内

        outside = d_out_inf > 0
        w_outside = torch.zeros_like(w_total)
        if band_out > 0:
            t = (d_out_inf / band_out).clamp(0, 1)  # 0=边界, 1=带宽外侧尽头
            # 边界 α → 远外 0，平滑衰减
            w_outside = alpha_at_boundary * (1 - _smoothstep(t))
        else:
            w_outside = torch.zeros_like(w_total)
        w_outside = torch.where(outside, w_outside, torch.zeros_like(w_outside))

        # 合并单盒贡献：盒内或盒外都给出一个 w，取最大
        w_box = torch.maximum(w_inside, w_outside)
        w_total = torch.maximum(w_total, w_box)  # 多盒取 max

    return w_total  # [N]


import torch

import torch

def _hash4(c: torch.Tensor) -> torch.Tensor:
    c = c.long()
    return (c[:,0] << 26) | (c[:,1] << 20) | (c[:,2] << 14) | (c[:,3] << 8)

def weight_ring_distance1_outside(coords: torch.Tensor, boxes_zyx, alpha: float = 0.5) -> torch.Tensor:
    """
    仅对“盒外且到盒子 L∞ 距离==1”的体素做融合：w=alpha。
    盒内任何位置 w=1；盒外距离>=2 的位置 w=0。多盒取最大。
    coords: [N,4]（1/2/3 -> x/y/z；盒子为闭区间）
    返回: w ∈ [0,1], [N]
    """
    if not boxes_zyx:
        return torch.zeros(coords.size(0), device=coords.device)

    # 用整数网格，避免浮点 ==1 的比较问题
    x = coords[:,1].long()
    y = coords[:,2].long()
    z = coords[:,3].long()
    device = coords.device

    # w 用 float，便于后续线性融合
    w_total = torch.zeros_like(x, dtype=torch.float32, device=device)

    for (zr, yr, xr) in boxes_zyx:
        z0, z1 = int(zr[0]), int(zr[1])
        y0, y1 = int(yr[0]), int(yr[1])
        x0, x1 = int(xr[0]), int(xr[1])

        # inside（闭区间）
        inside = (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1) & (z >= z0) & (z <= z1)

        # L∞ 外距（严格整数“层数”）
        dx_out = torch.maximum(x0 - x, x - x1).clamp_min(0)
        dy_out = torch.maximum(y0 - y, y - y1).clamp_min(0)
        dz_out = torch.maximum(z0 - z, z - z1).clamp_min(0)
        d_out = torch.maximum(torch.maximum(dx_out, dy_out), dz_out)  # long

        # 盒内：w=1；盒外且 d_out==1：w=alpha；盒外且 d_out>=2：w=0
        w_box = torch.zeros_like(w_total)
        w_box = torch.where(inside, torch.ones_like(w_box), w_box)
        w_box = torch.where((~inside) & (d_out == 1), torch.full_like(w_box, float(alpha)), w_box)

        # 多盒取最大（谁更偏新听谁）
        w_total = torch.maximum(w_total, w_box)

    return w_total

def blend_feats_ring1_outside(sample, ori_sample, boxes_zyx, alpha=0.5):
    """
    只在“盒外距离=1”的体素做融合；盒内=新，盒外远处=原。
    对 ori 中不存在的新坐标：强制新（w=1）。
    """
    device = sample.feats.device
    coords      = sample.coords.to(device)
    coords_ori  = ori_sample.coords.to(device)

    # 权重
    w = weight_ring_distance1_outside(coords, boxes_zyx, alpha=alpha).to(device)  # [N], float32

    # 新增坐标（ori 无对应）强制新
    is_new = ~torch.isin(_hash4(coords), _hash4(coords_ori))
    w = torch.where(is_new, torch.ones_like(w), w)

    # —— 按你“旧风格”做坐标对齐（遍历张量行，用 c.tolist()）——
    ori_map = {tuple(c.tolist()): i for i, c in enumerate(coords_ori)}  # 关键修复：不要 .tolist() 再 .tolist()
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
    # 确保权重 dtype 与 feats 一致
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
    # 你也可以把 tau/rot90k 等作为参数传进来
    def closure(z):
        with torch.enable_grad():
            sigma = decode_voxel_fn(z)
            # 和你主流程一致（soft 版本更利于梯度）
            tau = 0.55
            sil = _silhouette_from_sigma(sigma, tau=tau)    # 或者你的 OR-soft 版本
            sil_img = project_ortho_no_center(
                sil, out_hw=edit_mask.shape[-2:],
                ortho_scale=ortho_scale, tx=tx, ty=ty, flip_y=flip_y
            )
            sil_img = apply_orient_2d(sil_img, rot90k=3)
            # 用“无 clamp”的稳定 BCE，避免 clamp 把梯度截断
            eps = 1e-6
            pred = sil_img.clamp(eps, 1-eps)  # 只为数值稳定（如果你怀疑 clamp 截梯度，可临时去掉看差异）
            L = -(edit_mask*torch.log(pred) + (1-edit_mask)*torch.log(1-pred)).mean()
            return L
    return closure


def fd_directional_check(loss_closure, x, num_dirs=3, eps_scale=1e-3):
    """
    验证 autograd 的 <grad, v> 是否≈ 有限差分 (L(x+eps v)-L(x-eps v))/(2eps)
    - loss_closure: 传入一个函数 f(z)-> L 的标量（会在 no_grad 下重新跑一次前向）
    - x: 当前 latent（叶子张量）
    """
    x = x.detach()
    x.requires_grad_(True)
    L = loss_closure(x)                # 标量
    g = torch.autograd.grad(L, x)[0]   # autograd 梯度

    gdot_vs, fd_vs, rel_errs = [], [], []
    for k in range(num_dirs):
        v = torch.randn_like(x)
        v = v / (v.norm() + 1e-8)
        # 方向导数（autograd）
        gdot = float((g * v).sum())

        # 有限差分（no_grad）
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
    打印 L 对每一层的梯度强度：dL/d(sil_img), dL/d(sil), dL/d(sigma), dL/d(x_t)
    注意：要在计算了 L 之后、图还在的时候调用。
    """
    stats = {}

    # dL/d(sil_img)
    g_silimg = torch.autograd.grad(L, sil_img, retain_graph=True, create_graph=False, allow_unused=True)[0]
    if g_silimg is not None:
        stats["dL/d(sil_img)"] = _stat(g_silimg, "g_silimg")
    else:
        stats["dL/d(sil_img)"] = "None (梯度没到 sil_img，通常是 clamp/ detach/ hard-threshold 截断)"

    # dL/d(sil)
    g_sil = torch.autograd.grad(L, sil, retain_graph=True, create_graph=False, allow_unused=True)[0]
    if g_sil is not None:
        stats["dL/d(sil)"] = _stat(g_sil, "g_sil")
    else:
        stats["dL/d(sil)"] = "None (投影/方向处理可能把梯度截断)"

    # dL/d(sigma)
    g_sigma = torch.autograd.grad(L, sigma, retain_graph=True, create_graph=False, allow_unused=True)[0]
    if g_sigma is not None:
        stats["dL/d(sigma)"] = _stat(g_sigma, "g_sigma")
    else:
        stats["dL/d(sigma)"] = "None (decoder 内部可能 detach/ 非可微)"

    # dL/d(x_t)
    g_x = torch.autograd.grad(L, x_t, retain_graph=False, create_graph=False, allow_unused=True)[0]
    if g_x is not None:
        stats["dL/d(x_t)"] = _stat(g_x, "g_x")
    else:
        stats["dL/d(x_t)"] = "None (x_t 没有参与可微路径/被替换)"

    print_stats(stats)
    return g_x, stats
def save_overlay_png(pred01, gt01, path):
    """pred01/gt01: [1,1,H,W] 或 [H,W]。红=GT(mask)，绿=Pred(投影)。"""
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
    prob_mode='logit',           # 'logit' 用 sigmoid(σ/τ); 'density' 把 σ 当非负密度再压到 [0,1]
    tau=0.6,
    hard=False,                  # True: 前向0/1 + STE；False: 纯软
    thr_voxel=0.5,
    edge_blur_ks=1.2,
    # 新增：
    agg='poisson',               # 'poisson'（推荐）| 'noisy_or'
    kappa=10.0,                   # Poisson-OR 强度（可退火 4→12）
    topk=4                    # 仅用前 K 个深度；None 表示用全部，建议 8 或 16
):
    """
    目标：视线上只要命中一次即为白。
    - Poisson-OR:  S = 1 - exp(-kappa * sum_z p_z)      （推荐，梯度稳定）
    - Noisy-OR:    S = 1 - Π_z (1 - p_z)                 （深度多时梯度易变小）
    - hard=True: 前向硬阈值 + 直通估计（反传走软 S）
    返回: [B,1,H,W]
    """
    import torch
    import torch.nn.functional as F

    # --- 1) 把“深度轴”放到 dim=2 -> [B,1,D,H,W]
    axis_to_dim = {'z': 2, 'y': 3, 'x': 4}
    da = axis_to_dim[depth_axis]
    if da != 2:
        perm = [0,1,2,3,4]
        perm[2], perm[da] = perm[da], perm[2]
        sigma = sigma.permute(*perm).contiguous()
    if invert_depth:
        sigma = sigma.flip(2)

    # --- 2) 概率化 p_z ∈ [0,1]
    if prob_mode == 'logit':
        p = torch.sigmoid(sigma / tau)       # 若 σ 是 logit，σ=0 <-> p=0.5
        thr = float(thr_voxel)               # 通常 0.5（等价 σ>0）
    elif prob_mode == 'density':
        # 若 σ 是非负密度，可用 1-exp(-c*σ) 更有物理意义；这里给个保守版：
        p = torch.clamp(sigma, min=0)
        p = p / (p.max().detach() + 1e-8)
        thr = float(thr_voxel)
    else:
        raise ValueError("prob_mode must be 'logit' or 'density'.")

    # --- 3) 仅取 Top-K 深度（可选；表面体素通常只在少数层上有响应）
    if topk is not None and topk > 0 and topk < p.shape[2]:
        p = torch.topk(p, k=topk, dim=2).values  # [B,1,K,H,W]

    # --- 4) 沿深度合成
    if agg == 'poisson':
        # S = 1 - exp(-kappa * sum_z p_z)
        S_soft = 1.0 - torch.exp(-kappa * p.sum(dim=2))
    elif agg == 'noisy_or':
        # S = 1 - Π_z (1 - p_z)
        S_soft = 1.0 - torch.prod(1.0 - p + 1e-6, dim=2)
    else:
        raise ValueError("agg must be 'poisson' or 'noisy_or'.")

    # --- 5) 可选 2D 平滑（柔化锯齿）
    if edge_blur_ks and edge_blur_ks > 1:
        k = int(edge_blur_ks); pad = k//2
        S_soft = F.avg_pool2d(F.pad(S_soft, (pad,pad,pad,pad), mode='reflect'), k, stride=1)

    if not hard:
        return S_soft  # [B,1,H,W]

    # --- 6) 硬轮廓 + STE（前向硬，反传沿软）
    P_hard = (p.max(dim=2).values > thr).float()   # [B,1,H,W]
    return P_hard + (S_soft - S_soft.detach())

def apply_orient_2d(img, flip_x=False, flip_y=False, rot90k=0, swap_xy=False):
    """img: [B,1,H,W]。按需做 2D 方向修正；可微。rot90k ∈ {0,1,2,3}"""
    # 旋转 90*k
    
    k = int(rot90k) % 4
    k = 1
    if k == 1:  # +90°
        img = img.transpose(-2, -1).flip(-2)
    elif k == 2:  # 180°
        img = img.flip(-2).flip(-1)
    elif k == 3:  # -90°
        img = img.transpose(-2, -1).flip(-1)
    # 交换 XY（如需要）
    if swap_xy:
        img = img.transpose(-2, -1)
    # 水平/竖直翻转
    if flip_x:
        img = img.flip(-1)
    if flip_y:
        img = img.flip(-2)
    return img
def project_ortho_no_center(sil_world, out_hw, ortho_scale: float,
                                tx: float = 0.0, ty: float = 0.0, flip_y: bool = False):
    """
    不做“自动居中”的正交投影采样：
        - sil_world: [B,1,Hs,Ws]，对应世界平面 (x,y)∈[-0.5,0.5]^2 的前视投影
        - out_hw: (H,W) 目标分辨率（=编辑mask分辨率）
        - ortho_scale: == Blender 的 cam.data.ortho_scale（图像宽覆盖的世界宽度）
        - tx, ty: 世界单位的平移（与 Blender 语义一致），默认0
        - flip_y: 竖直方向翻转开关（如遇上下颠倒设 True）
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
# ---------- 形状/空间对齐 & 池化 ----------
def _resolve_viz_base(feature_path, fallback="."):
    if feature_path is None:
        return fallback
    # 若存在且是目录，直接用
    if os.path.exists(feature_path) and os.path.isdir(feature_path):
        return feature_path
    # 若存在且是文件 -> 用其父目录
    if os.path.exists(feature_path) and os.path.isfile(feature_path):
        return os.path.dirname(feature_path)
    # 若不存在，但看起来像“文件路径”（有扩展名）-> 用父目录
    root, ext = os.path.splitext(feature_path)
    if ext:  # .pkl/.pt/.npy/.png 等
        return os.path.dirname(feature_path) or fallback
    # 否则当成目录用
    return feature_path
def _align_spatial(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """
    将 x 的维度对齐到 ref:  (B,C,[D,]H,W)，并在通道维做 expand。
    """
    while x.dim() < ref.dim():
        x = x.unsqueeze(1)  # 在 C 维补
    if x.size(1) != ref.size(1):
        x = x.expand(-1, ref.size(1), *([-1] * (ref.dim() - 2)))
    return x


def _avg_pool_nd(x: torch.Tensor, k: int) -> torch.Tensor:
    """
    2D/3D 自适应平均池化（保持 stride=1, padding=kernel//2）。
    """
    if x.dim() == 4:
        return F.avg_pool2d(x, k, 1, k // 2)
    elif x.dim() == 5:
        k3 = 2 * max(1, k // 2) + 1 if isinstance(k, int) else k
        return F.avg_pool3d(x, k3, 1, k3 // 2)
    return x


def _max_pool_nd(x: torch.Tensor, k: int) -> torch.Tensor:
    """
    2D/3D 自适应最大池化（保持 stride=1, padding=kernel//2）。
    """
    if x.dim() == 4:
        return F.max_pool2d(x, k, 1, k // 2)
    elif x.dim() == 5:
        k3 = 2 * max(1, k // 2) + 1 if isinstance(k, int) else k
        return F.max_pool3d(x, k3, 1, k3 // 2)
    return x


# ---------- 掩码构造/组合/度量 ----------

def apply_mask_blend(new: torch.Tensor, old: torch.Tensor, soft_mask: Optional[torch.Tensor]) -> torch.Tensor:
    """
    用软掩码在空间上做凸组合：soft*new + (1-soft)*old。
    如果 soft_mask 为 None，直接返回 new。
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
    组合手动与自动掩码：
      - "union":   A ∪ B  ≈ A + B - A*B
      - "intersect": A ∩ B = A*B
      - "blend":   (1-w)*manual + w*auto
    任一为 None 则返回另一个；两者皆 None 返回 None。
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
    从二值/软 mask 生成：
      - soft: 羽化后的软掩码（[0,1]）
      - guard_band: 保护带（膨胀 - 原 mask），[0,1]
    """
    if mask is None:
        return None, None
    m = _align_spatial(mask.float(), ref)
    # 羽化
    soft = _avg_pool_nd(m, 2 * feather + 1) if feather > 0 else m
    soft = soft.clamp(0, 1)
    # 保护带
    if guard > 0:
        dil = _max_pool_nd(m, 2 * guard + 1)
        guard_band = (dil - m).clamp(0, 1)
    else:
        guard_band = torch.zeros_like(soft)
    return soft, guard_band


def make_soft_mask(m: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    """
    兼容老逻辑的简易软掩码（轻度均值滤波）。
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
    """将软掩码二值化到 {0,1}。"""
    return (x >= thr).float()


def mask_metrics(
    auto_soft: Optional[torch.Tensor],
    manual_soft: Optional[torch.Tensor],
    thr: float = 0.5,
    eps: float = 1e-6
) -> Optional[dict]:
    """
    计算 auto vs manual 的 IoU/Precision/Recall（在 thr 处二值化）。
    输入可以是 2D/3D/Batched 张量，自动求和。
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


# ---------- Auto-mask 构建与打分 ----------

def reduce_vector_norm(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    (B,C,[D,]H,W) -> (B,1,[D,]H,W)，L2 范数。
    """
    return (x.pow(2).sum(dim=1, keepdim=True) + eps).sqrt()


def build_auto_mask_from_map(
    score_map: torch.Tensor,
    ref: Optional[torch.Tensor] = None,  # 兼容保留，不使用
    quantile: float = 0.7,
    tau: float = 0.25,
    feather: int = 2
) -> torch.Tensor:
    """
    由打分图（越大越该编辑）生成软掩码：
      - per-sample 分位阈值
      - 温度 sigmoid
      - 轻度羽化
    返回值范围 [0,1]。
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


# ---------- 可视化（np 归一化 + 2x2 grid） ----------

def _to_np_2d(x: ArrayLike) -> np.ndarray:
    """
    将 torch / numpy 数据转为 (H,W) 的 numpy.float32，并归一化到 [0,1]。
    规则：
      - 若带 batch：取第一个样本
      - 若为 (C,D,H,W)：取 ch0 + D 的中间切片
      - 若为 (C,H,W) 或 (1,H,W)：取 ch0
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
            # 无 PIL 时退化为 .npy
            np.save(path_png.replace(".png", ".npy"), arr2d)


def _save_grid_2x2(
    auto: Optional[np.ndarray],
    manual: Optional[np.ndarray],
    edited_mask: Optional[np.ndarray],
    delta: Optional[np.ndarray],
    out_png: str
) -> Optional[str]:
    """
    将四个灰度面板（归一化到 [0,1] 的 2D numpy）拼成 2x2 大图保存。
    若无 PIL，返回 None。
    面板顺序：
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
    # 形状/池化
    "_align_spatial", "_avg_pool_nd", "_max_pool_nd",
    # 掩码与融合
    "apply_mask_blend", "combine_masks", "build_soft_masks", "make_soft_mask",
    "binarize", "mask_metrics",
    # auto-mask
    "reduce_vector_norm", "build_auto_mask_from_map",
    # 可视化
    "_to_np_2d", "_save_grid_2x2",
    # 标志
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
    mask: [N] uint8 or bool tensor, 0 表示要替换
    返回：
        sample_indices: 需要替换的 sample 中的 index
        ori_indices: 对应 ori_sample 中的 index
    """
    # 先找出 sample 中需要替换的 coords
    target_indices = torch.nonzero(mask == 0, as_tuple=False).squeeze(1)
    target_coords = sample_coords[target_indices]  # [K, 3]

    # 用 hash 表加速匹配
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
    用 raw_mask（[B,1,64,64,64]）计算每个坐标的融合权重：
      - 掩码内：w=1
      - 掩码外且到掩码的 L∞ 距离==1（外侧一圈）：w=alpha
      - 其他：w=0
    坐标格式 (b,x,y,z)，与 grid[b,0,x,y,z] 对齐。
    """
    grid = raw_mask.to(coords.device)
    if grid.dtype != torch.bool:  # 0/1 浮点转 bool
        grid = grid > 0

    # 形态学膨胀实现 L∞-1 邻域：max_pool3d kernel=3, padding=1
    # 注意：必须在 float 上做池化，再阈值回 bool
    dilated = F.max_pool3d(grid.float(), kernel_size=3, stride=1, padding=1) > 0.5
    ring = (~grid) & dilated  # 外侧一圈

    b, x, y, z = coords.long().unbind(dim=1)  # (b,x,y,z)
    inside = grid[b, 0, x, y, z]
    in_ring = ring[b, 0, x, y, z]

    w = torch.zeros(coords.size(0), device=coords.device, dtype=torch.float32)
    w = torch.where(inside, torch.ones_like(w), w)
    w = torch.where((~inside) & in_ring, torch.full_like(w, float(alpha)), w)
    return w  # [N]


def blend_feats_ring1_outside_rawmask(sample, ori_sample, raw_mask: torch.Tensor, alpha=0.5):
    """
    raw_mask 版融合：
      - 掩码内：用 sample 新特征
      - 掩码外一圈（L∞=1）：新旧按 alpha 融合
      - 掩码外远处：保留 ori
      - 对“新增坐标”（ori 没有的）强制新（w=1）
    """
    device = sample.feats.device
    coords     = sample.coords.to(device)      # [Ns,4] (b,x,y,z)
    coords_ori = ori_sample.coords.to(device)  # [No,4]

    # 计算权重
    w = weight_ring1_from_rawmask(coords, raw_mask.to(device), alpha=alpha)  # [Ns]

    # 新增坐标：强制 w=1
    # 这里假设你已有 _hash4(c) -> int64 的哈希函数（与你现有代码一致）
    is_new = ~torch.isin(_hash4(coords), _hash4(coords_ori))
    w = torch.where(is_new, torch.ones_like(w), w)

    # 找出新旧坐标的重叠行，用于融合
    # 保持你原先的 dict 查找方式，简单稳妥
    ori_map = {tuple(c.tolist()): i for i, c in enumerate(coords_ori)}
    idx_s, idx_o = [], []
    for i, c in enumerate(coords):
        j = ori_map.get(tuple(c.tolist()))
        if j is not None:
            idx_s.append(i); idx_o.append(j)

    if not idx_s:
        return sample  # 无重叠，直接返回

    idx_s = torch.tensor(idx_s, dtype=torch.long, device=device)
    idx_o = torch.tensor(idx_o, dtype=torch.long, device=device)

    # 融合：new = w*new + (1-w)*old
    new_feats = sample.feats.clone()
    w_exp = w[idx_s].unsqueeze(1).to(new_feats.dtype)  # [K,1]
    new_feats[idx_s] = w_exp * new_feats[idx_s] + (1 - w_exp) * ori_sample.feats[idx_o].to(new_feats.dtype)

    return sample.replace(feats=new_feats)

def _normalize_boxes_zyx(mask_list):
    """
    将 mask_list 统一为“多个盒子”的列表，且顺序固定为 [z,y,x]。
      - 单盒子: [[z0,z1],[y0,y1],[x0,x1]] -> [ [[z0,z1],[y0,y1],[x0,x1]] ]
      - 多盒子: [[[z0,z1],[y0,y1],[x0,x1]], ...] -> 原样
    做基本 clamp 到 [0, 64]，但不改变“闭区间”的语义（仍然 <=）。
    """
    if mask_list is None:
        return []

    if isinstance(mask_list, str):
        try:
            mask_list = json.loads(mask_list)
        except Exception:
            mask_list = ast.literal_eval(mask_list)

    # 单盒子 -> 多盒子
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

        # 允许 z1==64 这种上界，因你用 <= 判断，coords 最大 63 时也不会越界
        if (z1 >= z0) and (y1 >= y0) and (x1 >= x0):
            out.append([[z0, z1], [y0, y1], [x0, x1]])
    return out

def inside_any_zyx_mask(t: torch.Tensor, boxes_zyx):
    """返回布尔向量：点是否落在任一 [z,y,x] 盒子内（闭区间）。"""
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
    """keep='inside' 或 'outside'。返回筛选后的 coords（保持 dtype/shape）。"""
    m = inside_any_zyx_mask(t, boxes_zyx)
    return t[m] if keep == 'inside' else t[~m]

import torch

def merge_coords(
    coords1: torch.Tensor,   # [N1, 4] (b,x,y,z)
    coords:  torch.Tensor,   # [N2, 4] (b,x,y,z)
    raw_mask: torch.Tensor,  # [B,1,64,64,64]，bool 或 数值>0
    device=None,             # 兼容旧签名（忽略）
    hole_fill_radius: int = 0  # 兼容旧签名（忽略）
) -> torch.Tensor:
    # 保证 dtype/设备一致（不下CPU）
    grid   = raw_mask.to(coords.device)
    if grid.dtype != torch.bool:
        grid = grid > 0

    coords1 = coords1.to(dtype=torch.long, device=coords.device)
    coords  = coords.to(dtype=torch.long, device=coords.device)

    # inside 判定：grid[b,0,x,y,z]
    b1, x1, y1, z1 = coords1.unbind(dim=1)
    b0, x0, y0, z0 = coords.unbind(dim=1)

    inside1 = grid[b1, 0, x1, y1, z1]
    inside0 = grid[b0, 0, x0, y0, z0]

    # 掩码内用新 coords1，掩码外保留旧 coords
    out = torch.cat([coords1[inside1], coords[~inside0]], dim=0)
    out = torch.unique(out, dim=0).to(torch.int32).contiguous()
    return out


def read_feature(feature_path, img_name):
    """
    从 feature_path（如 features.pkl）中读取指定 img_name 对应的 tensor。
    如果未找到该 img_name，返回 None。
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
    return None  # 没找到


def load_img_features(feature_path):
    img_feature_dict = {}

    # 以 rb 模式打开文件
    with open(feature_path, 'rb') as f:
        os.makedirs(os.path.dirname(feature_path), exist_ok=True)
        while True:
            try:
                # 加载一个字典
                feature_item = pickle.load(f)
                for key, value in feature_item.items():
                    if key.endswith('_img'):
                        img_feature_dict[key] = value
            except EOFError:
                break  # 读取完毕退出循环

    return img_feature_dict

def save_feature(feature_path, img_name, sample):
    # 确保目录存在
    os.makedirs(os.path.dirname(feature_path), exist_ok=True)

    # 确保 sample 在 CPU 上
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

    # 以追加方式写入 pickle 文件
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
    辅助函数：将 [0, 255] 范围内的连续颜色值量化为 n_intervals 个离散值。
    
    它将每个值映射到其所属区间的*中点*。
    """
    if n_intervals <= 0:
        return continuous_colors.astype(np.uint8)

    print(f"  > 正在执行量化，n_intervals={n_intervals}...")

    # 1. 确保值在 [0, 255] 范围内
    # (np.uint8 会自动截断，但为了计算准确，我们先手动 clip)
    clipped_colors = np.clip(continuous_colors, 0, 255)
    
    # 2. 定义区间的属性
    # 我们使用 256.0 来进行浮点数除法，确保 20 个区间均匀覆盖 0-255
    interval_width = 256.0 / n_intervals 
    # (例如, 256.0 / 20 = 12.8)
    
    # 区间的中点偏移量
    midpoint_offset = interval_width / 2.0
    # (例如, 12.8 / 2.0 = 6.4)

    # 3. 应用量化公式
    # 核心逻辑:
    # 1. (clipped_colors / interval_width) -> 找出值在哪个区间 (例如 20.0 / 12.8 = 1.56)
    # 2. np.floor(...) -> 取整, 得到区间索引 (例如 floor(1.56) = 1.0)
    # 3. (... * interval_width) -> 移动到该区间的起始点 (例如 1.0 * 12.8 = 12.8)
    # 4. (... + midpoint_offset) -> 移动到该区间的中点 (例如 12.8 + 6.4 = 19.2)
    
    quantized_values = (np.floor(clipped_colors / interval_width) * interval_width) + midpoint_offset
    
    # 4. 转换回 uint8
    return quantized_values.astype(np.uint8)


def features_to_rgba(features: torch.Tensor) -> np.ndarray:
    """
    辅助函数：将 [N, C] 特征张量转换为 [N, 4] RGBA 颜色 (0-255)。
    
    (已修改：当 C > 3 时使用 PCA 降维)
    (已修改：在末尾添加 20 区间量化)
    """
    print(f"开始计算 {features.shape[0]} 个特征的颜色...")
    
    if features.is_cuda:
        features = features.cpu()
    feat_np = features.numpy()
    N, C = feat_np.shape
    
    # --- 约束：透明度始终为 255 ---
    alpha = np.full((N, 1), 255, dtype=np.uint8)
    
    # 默认 RGB (灰色)
    rgb_colors = np.full((N, 3), 200, dtype=np.uint8) 

    try:
        if C == 1:
            # (逻辑不变: 1D 特征使用热力图)
            print("特征 C=1, 应用热力图...")
            norm_feat = (feat_np - feat_np.min()) / (feat_np.max() - feat_np.min() + 1e-6)
            cmap = plt.get_cmap('viridis')
            rgba_colors_01 = cmap(norm_feat.squeeze())
            # (这里得到的是 0-255 的 *连续* 浮点数)
            rgb_colors = (rgba_colors_01[:, :3] * 255)
            
        elif C == 3:
            # (逻辑不变: 3D 特征直接归一化为 RGB)
            print("特征 C=3, 归一化为 RGB...")
            min_vals = feat_np.min(axis=0)
            max_vals = feat_np.max(axis=0)
            range_vals = max_vals - min_vals + 1e-6
            # (这里得到的是 0-255 的 *连续* 浮点数)
            rgb_colors = ((feat_np - min_vals) / range_vals * 255)
            
        elif C > 3:
            # --- *修改后的 PCA 逻辑* ---
            print(f"特征 C={C}, 使用 PCA 降维至 3 通道进行着色。")
            
            pca = PCA(n_components=3)
            rgb_features = pca.fit_transform(feat_np)
            
            min_vals = rgb_features.min(axis=0)
            max_vals = rgb_features.max(axis=0)
            range_vals = max_vals - min_vals + 1e-6
            # (这里得到的是 0-255 的 *连续* 浮点数)
            rgb_colors = ((rgb_features - min_vals) / range_vals * 255)
            
        else: # C == 0 or C == 2
            print(f"警告：特征维度 {C} 不支持。使用默认灰色。")
            
    except Exception as e:
        print(f"颜色映射时发生错误: {e}。将使用默认灰色。")
        # 即使出错，也使用默认的 rgb_colors (全 200)

    # --- 4. (新步骤) 量化 RGB 颜色 ---
    # 无论 try/except 走哪个分支 (C=1, C=3, C>3 或 except)，
    # 都在这里对 rgb_colors (此时还是连续值) 进行量化。
    
    print(f"将 RGB 颜色量化为 {5} 个区间...")
    rgb_colors = quantize_colors(rgb_colors, n_intervals=5)
    
    # 5. 组合 R, G, B 和 Alpha
    return np.hstack((rgb_colors, alpha))


def export_indexed_coords_to_glb_cubes_colored(
    out_coords: torch.Tensor,
    output_filename: str = 'voxel_cubes_scaled.glb',
    grid_size: int = 64,
    color_rgba: list = [200, 200, 200, 255],
    slat_feat: torch.Tensor = None,
):
    """
    (V3 - 健壮版)
    将 (b,x,y,z) 索引坐标转换为 [0, 1] 空间中的 *带颜色* 的方块网格。
    
    统一使用 "先合并几何，后整体上色" 的可靠方法。
    """
    
    print(f"开始导出至 {output_filename}......")

    # --- 1. 提取 (x,y,z) 索引 ---
    if out_coords.is_cuda:
        out_coords_cpu = out_coords.cpu()
    else:
        out_coords_cpu = out_coords
        
    xyz_indices_np = out_coords_cpu[:, 1:4].numpy()
    N = xyz_indices_np.shape[0]
    
    if N == 0:
        print("错误：没有任何坐标可供导出。")
        return

    print(f"找到 {N} 个体素... 正在计算缩放...")

    # --- 2. 计算缩放、位置 ---
    
    voxel_size = 1.0 / float(grid_size)
    
    # 创建一个 *基础* 方块 (无色)
    base_cube = trimesh.creation.box(extents=[voxel_size, voxel_size, voxel_size])
    num_vertices_per_cube = len(base_cube.vertices) # 应该是 8
    
    # 计算世界坐标中心
    world_centers_np = (xyz_indices_np - 0.5) * voxel_size

    # --- 3. 实例化与合并 (只合并几何) ---
    
    # 1. 先实例化 *无色* 的方块 (速度快)
    print("正在实例化所有方块 (无色)...")
    all_cubes = [base_cube.copy().apply_translation(p) for p in world_centers_np]
    
    # 2. 合并它们
    print("正在合并所有方块...")
    final_mesh = trimesh.util.concatenate(all_cubes)
    
    # --- 4. 准备 *巨大* 的颜色数组 ---
    
    total_vertices = final_mesh.vertices.shape[0]
    # 检查：total_vertices 应该等于 N * num_vertices_per_cube
    if total_vertices != N * num_vertices_per_cube:
        print(f"警告：顶点数不匹配！{total_vertices} != {N} * {num_vertices_per_cube}")
        # 即使不匹配，也尝试继续
        
    if slat_feat is None:
        # *********** 目标 1: 统一颜色 ***********
        print(f"正在构建统一颜色数组: {color_rgba}")
        
        # 创建一个巨大的列表 (或 Numpy 数组)
        vertex_colors = np.array([color_rgba] * total_vertices, dtype=np.uint8)
        
    else:
        # *********** 目标 2: 根据特征分别着色 ***********
        print("正在构建特征颜色数组...")
        if slat_feat.shape[0] != N:
            print(f"错误: 坐标数 ({N}) 与特征数 ({slat_feat.shape[0]}) 不匹配。")
            return
            
        # 1. 计算 N 个体素的 N 个颜色 (shape: [N, 4])
        voxel_colors_np = features_to_rgba(slat_feat) 
        
        # 2. *关键*: 将每个体素颜色重复 N_vert 次 (8次)
        # np.repeat 会按顺序复制，正好对应 concatenate 的顶点顺序
        # [C1, C2, C3] -> [C1,C1... (8次), C2,C2... (8次), C3,C3... (8次)]
        print(f"正在将 {N} 个体素颜色扩展到 {total_vertices} 个顶点...")
        vertex_colors = np.repeat(voxel_colors_np, num_vertices_per_cube, axis=0)

        # 再次检查形状
        if vertex_colors.shape[0] != total_vertices:
            print(f"错误：最终颜色数组形状 {vertex_colors.shape} 与顶点数 {total_vertices} 不匹配。")
            # 尝试调整大小（这只是一个保险措施）
            vertex_colors = np.resize(vertex_colors, (total_vertices, 4))


    # --- 5. *一次性*应用颜色并导出 ---
    print("正在为合并后的网格指定颜色...")
    
    # *最终赋值*
    final_mesh.visual.vertex_colors = vertex_colors
    
    print(f"正在导出至 {output_filename}...")
    final_mesh.export(output_filename)
    print(f"导出 {output_filename} 完成。")
