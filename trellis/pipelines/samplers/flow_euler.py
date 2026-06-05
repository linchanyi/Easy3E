from typing import *
import torch
import numpy as np
from tqdm import tqdm
from easydict import EasyDict as edict
from .base import Sampler
from .classifier_free_guidance_mixin import ClassifierFreeGuidanceSamplerMixin
from .guidance_interval_mixin import GuidanceIntervalSamplerMixin
from torch import Tensor
from ...utils.general_utils import combine,build_coord_index_map,replace_feats,_normalize_boxes_zyx,inside_any_zyx_mask,_resolve_viz_base
import numpy as np
import torch
import torch.nn.functional as F
import os
from ...utils.general_utils import _to_mask_tensor, _silhouette_from_sigma, project_ortho_no_center,apply_orient_2d
from ...utils.general_utils import (
    _align_spatial, _avg_pool_nd, _max_pool_nd,
    apply_mask_blend, combine_masks,
    build_soft_masks, make_soft_mask,
    binarize, mask_metrics,
    reduce_vector_norm, build_auto_mask_from_map,
    _to_np_2d, _save_grid_2x2, HAS_PIL, _save_panel_png,_stat,
    blend_feats_ring1_outside_rawmask
)
from ...utils.general_utils import grad_chain_diagnostics,blend_feats_ring1_outside
from ...utils.general_utils import silhouette_at_azimuth, save_guidance_debug_viz
class FlowEulerSampler(Sampler):
    def __init__(
        self,
        sigma_min: float,
    ):
        self.sigma_min = sigma_min

    def _eps_to_xstart(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (x_t - (self.sigma_min + (1 - self.sigma_min) * t) * eps) / (1 - t)

    def _xstart_to_eps(self, x_t, t, x_0):
        assert x_t.shape == x_0.shape
        return (x_t - (1 - t) * x_0) / (self.sigma_min + (1 - self.sigma_min) * t)

    def _v_to_xstart_eps(self, x_t, t, v):
        assert x_t.shape == v.shape
        eps = (1 - t) * v + x_t
        x_0 = (1 - self.sigma_min) * x_t - (self.sigma_min + (1 - self.sigma_min) * t) * v
        return x_0, eps

    def _inference_model(self, model, x_t, t, cond=None, **kwargs):
        t = torch.tensor([1000 * t] * x_t.shape[0], device=x_t.device, dtype=torch.float32)
        if cond is not None and cond.shape[0] == 1 and x_t.shape[0] > 1:
            cond = cond.repeat(x_t.shape[0], *([1] * (len(cond.shape) - 1)))
        return model(x_t, t, cond, **kwargs)

    def _get_model_prediction(self, model, x_t, t, cond=None, **kwargs):
        
        pred_v = self._inference_model(model, x_t, t, cond, **kwargs)
        pred_x_0, pred_eps = self._v_to_xstart_eps(x_t=x_t, t=t, v=pred_v)
        return pred_x_0, pred_eps, pred_v

    @torch.no_grad()
    def sample_once(
        self,
        model,
        x_t,
        t: float,
        t_prev: float,
        cond: Optional[Any] = None,
        **kwargs
    ):
        
        pred_x_0, pred_eps, pred_v = self._get_model_prediction(model, x_t, t, cond, **kwargs)
        pred_x_prev = x_t - (t - t_prev) * pred_v
     
        return edict({"pred_x_prev": pred_x_prev, "pred_x_0": pred_x_0})

    @torch.no_grad()
    def sample(
        self,
        model,
        noise,
        cond: Optional[Any] = None,
        steps: int = 28,
        rescale_t: float = 1.0,
        verbose: bool = True,
        **kwargs
    ):
        mode = kwargs.get("mode","baseline")
        if mode == "flowedit":
            return self.sample_flowedit(model,noise,cond,steps,rescale_t,verbose,**kwargs)
        elif mode == "repaint":
            return self.sample_repaint(model,noise,cond,steps,rescale_t,verbose,**kwargs)
        else:
            return self.sample_ori(model,noise,cond,steps,rescale_t,verbose,**kwargs)
    
   
    _FE_N_MIN        = 2     
    _FE_N_AVG        = 4     
    _FE_FEATHER      = 2     
    _FE_GUARD        = 1    
    _FE_GUID_START   = 6    

    _FE_ETA          = 0.1   
    _FE_GAMMA        = 0.1   
    _FE_BETA_NORM    = 1.0   
    _FE_LAM_OUT      = 0.5   
    _FE_LAM_IN       = 0.1   

    _RP_BETA         = 0.8   
    _RP_ANCHOR_EPS   = True  
    _RP_INV_STEPS    = 4     

    @torch.no_grad()
    def sample_flowedit(
        self,
        model,
        noise,
        cond: Optional[Any] = None,
        steps: int = 25,
        rescale_t: float = 3.0,
        verbose: bool = True,
        **kwargs
    ):
        
        cond_src         = kwargs["ori_cond"]
        cond_tar         = cond
        manual_mask      = kwargs.get("mask", None)
        cfg_src_strength = float(kwargs.get("cfg_src_strength", 5.0))
        cfg_tar_strength = float(kwargs.get("cfg_tar_strength", 5.0))
        use_guidance     = bool(kwargs.get("use_guidance", True))
        azimuth_deg      = float(kwargs.get("azimuth_deg", 270.0))
        debug_guid       = bool(kwargs.get("debug_guidance_viz", False))
        debug_viz_dir    = kwargs.get("debug_viz_dir", None)
        decode_voxel_fn  = kwargs.get("decode_voxel_fn", None)

        lock_sched = lambda t: min(0.95, 0.40 + (1.0 - t) * 0.45)

        def forward_noise(x0, t_scalar, eps):
            s = 1e-5
            tv = float(t_scalar)
            return (1.0 - tv) * x0 + (s + (1 - s) * tv) * eps

        def _mul_mask(v):
            return v if combined_soft is None else v * combined_soft

        sample = noise
        x_src0 = kwargs.get("src_latent", None)
        if x_src0 is None:
            x_src0 = sample.clone()

        t_seq = np.linspace(1, 0, steps + 1)
        t_seq = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)
        t_pairs = [(t_seq[i], t_seq[i + 1]) for i in range(steps)]

        _edict = lambda d: type("edict", (object,), d)
        ret = _edict({
            "samples": None, "pred_x_t": [], "pred_x_0": [],
            "viz_grid_paths": [], "auto_masks": [], "mask_metrics": []
        })

        combined_soft = None
        if manual_mask is not None:
            combined_soft, _ = build_soft_masks(
                manual_mask, sample,
                feather=self._FE_FEATHER, guard=self._FE_GUARD
            )

        edit_mask = _to_mask_tensor(kwargs.get("edit_mask", None), sample)
        if edit_mask.shape[0] != sample.shape[0]:
            if edit_mask.shape[0] == 1:
                edit_mask = edit_mask.expand(sample.shape[0], -1, -1, -1)
            else:
                raise ValueError(
                    f"edit_mask batch mismatch with sample: {edit_mask.shape[0]} vs {sample.shape[0]}"
                )
        H_mask, W_mask = int(edit_mask.shape[-2]), int(edit_mask.shape[-1])

        
        from ...utils.general_utils import rotate_volume_z as _rot_vol_z

        def _build_3D_mask(edit_mask_2d, sigma_perm_shape):
            B, _, Y_dim, Z_dim, X_dim = sigma_perm_shape
            m = F.interpolate(edit_mask_2d, size=(X_dim, Z_dim), mode='bilinear', align_corners=True)
            m = m.flip(-2).transpose(-2, -1)
            m = m.unsqueeze(2).expand(B, 1, Y_dim, Z_dim, X_dim)
            return m.contiguous()

        
        def _guide(state, t_val, dt, step_i, prefix, v_norm_ref, xi_traj=None):
            if not use_guidance:
                return state
            try:
                with torch.enable_grad():
                    x_t   = state.detach().requires_grad_(True)
                    sigma = decode_voxel_fn(x_t)
                    tau   = 0.6 if t_val > 0.5 else 0.5

                    delta_rad = float(np.radians(azimuth_deg - 270.0))
                    rotated = sigma if abs((azimuth_deg - 270.0) % 360.0) < 1e-3 \
                              else _rot_vol_z(sigma, delta_rad)
                    sigma_perm = rotated.permute(0, 1, 3, 2, 4).contiguous()

                    M_3D = _build_3D_mask(edit_mask, sigma_perm.shape)

                    s = sigma_perm / tau
                    L_out = (F.softplus(s) * (1.0 - M_3D)).sum() \
                            / ((1.0 - M_3D).sum().clamp(min=1.0))
                    M_2D_ray = M_3D.amax(dim=2)
                    lse_y    = torch.logsumexp(s, dim=2)
                    L_in     = (F.softplus(-lse_y) * M_2D_ray).sum() \
                               / (M_2D_ray.sum().clamp(min=1.0))

                    L = self._FE_LAM_OUT * L_out + self._FE_LAM_IN * L_in
                    g = torch.autograd.grad(L, x_t)[0]
                    ret.mask_metrics.append({
                        "step": int(step_i),
                        "L_out": float(L_out.detach().cpu()),
                        "L_in":  float(L_in.detach().cpu()),
                    })

                if debug_guid and debug_viz_dir is not None:
                    with torch.no_grad():
                        sil_raw = silhouette_at_azimuth(sigma.detach(), azimuth_deg=azimuth_deg, tau=tau)
                        sil_img = project_ortho_no_center(sil_raw, out_hw=(H_mask, W_mask), ortho_scale=1.0)
                        save_guidance_debug_viz(
                            sil_img, edit_mask, step_i, azimuth_deg,
                            debug_viz_dir, prefix=prefix
                        )

                g = _mul_mask(g)
                g_norm = g.norm() + 1e-8
                scale  = self._FE_BETA_NORM * (v_norm_ref / g_norm)
                tilde_g = scale * g

                state = state - dt * (self._FE_ETA * tilde_g)
                if xi_traj is not None:
                    state = state - dt * (self._FE_GAMMA * _mul_mask(xi_traj))
            except Exception as e:
                ret.mask_metrics.append({"step": int(step_i), "err": str(e)})
            return state

        def _lock_back(x, t_val):
            if manual_mask is None:
                return x
            lk = lock_sched(t_val)
            if lk <= 0:
                return x
            keep = lk * (1 - manual_mask)
            return (1 - keep) * x + keep * x_src0

        iterator = enumerate(t_pairs)
        if verbose:
            try:
                iterator = tqdm(iterator, total=len(t_pairs), desc="Sampling (FlowEdit)")
            except Exception:
                pass

        for step_i, (t, t_prev) in iterator:
            tail_refine = (steps - step_i) <= self._FE_N_MIN
            dt = t - t_prev

            if not tail_refine:
                V = torch.zeros_like(sample)
                last_zt_src = last_zt_tar = None
                last_v_src  = last_v_tar  = None
                for _ in range(self._FE_N_AVG):
                    eps = torch.randn_like(x_src0)
                    zt_src = apply_mask_blend(forward_noise(x_src0, t, eps), x_src0, combined_soft)
                    zt_tar = sample + (zt_src - x_src0)

                    kwargs["cfg_strength"] = cfg_src_strength
                    _, _, v_src = self._get_model_prediction(model, zt_src, t, cond_src, **kwargs)
                    kwargs["cfg_strength"] = cfg_tar_strength
                    _, _, v_tar = self._get_model_prediction(model, zt_tar, t, cond_tar, **kwargs)

                    V = V + (v_tar - v_src) / float(self._FE_N_AVG)
                    last_zt_src, last_zt_tar, last_v_src, last_v_tar = zt_src, zt_tar, v_src, v_tar

                V = _mul_mask(V)
                x = sample - dt * V

                if step_i >= self._FE_GUID_START:
                    tv = float(t)
                    x0_tgt = last_zt_tar - tv * last_v_tar
                    x0_src = last_zt_src - tv * last_v_src
                    xi_traj = x0_tgt - x0_src
                    x = _guide(x, tv, dt, step_i, "mid",
                               v_norm_ref=V.detach().norm() + 1e-8,
                               xi_traj=xi_traj.detach())

                sample = _lock_back(x, float(t))
                ret.pred_x_t.append(sample)
                ret.pred_x_0.append(None)

            else:
                if (steps - step_i) == self._FE_N_MIN:
                    eps = torch.randn_like(x_src0)
                    zt_src = apply_mask_blend(forward_noise(x_src0, t, eps), x_src0, combined_soft)
                    xt_tar = sample + (zt_src - x_src0)
                else:
                    xt_tar = sample

                kwargs["cfg_strength"] = cfg_tar_strength
                pred_x0, _, v_tar = self._get_model_prediction(model, xt_tar, t, cond_tar, **kwargs)
                v_tar = _mul_mask(v_tar)
                x = xt_tar - dt * v_tar

                x = _guide(x, float(t), dt, step_i, "tail",
                           v_norm_ref=v_tar.detach().norm() + 1e-8,
                           xi_traj=None)

                sample = _lock_back(x, float(t))
                ret.pred_x_t.append(sample)
                ret.pred_x_0.append(pred_x0)

        ret.samples = sample
        return ret

    @torch.no_grad()
    def sample_ori(
        self,
        model,
        noise,
        cond: Optional[Any] = None,
        steps: int = 28,
        rescale_t: float = 1.0,
        verbose: bool = True,
        **kwargs
    ):
        sample = noise
        t_seq = np.linspace(1, 0, steps + 1)
        t_seq = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)
        t_pairs = list((t_seq[i], t_seq[i + 1]) for i in range(steps))

        ret = edict({"samples": None, "pred_x_t": [], "pred_x_0": []})

        for t, t_prev in tqdm(t_pairs, desc="Sampling", disable=not verbose):
            kwargs["t_sign"] = t
            kwargs["t_1"] = t
            out = self.sample_once(model, sample, t, t_prev, cond, **kwargs)
            sample = out.pred_x_prev
            ret.pred_x_t.append(out.pred_x_prev)
            ret.pred_x_0.append(out.pred_x_0)

        ret.samples = sample
        return ret

    @torch.no_grad()
    def sample_repaint(
        self,
        model,
        noise,
        cond: Optional[Any] = None,
        steps: int = 28,
        rescale_t: float = 1.0,
        verbose: bool = True,
        **kwargs
    ):
      
        sample = noise
        x_src = kwargs.get("x_src", None)
        mask = kwargs.get("mask", None)
        stage = kwargs.get("stage", "sparse")
        repaint_beta       = max(0.0, min(1.0, float(self._RP_BETA)))
        repaint_anchor_eps = bool(self._RP_ANCHOR_EPS)
        repaint_inv_steps  = max(0, min(int(self._RP_INV_STEPS), steps))
        cond_src           = kwargs.get("ori_cond", None)
        if repaint_inv_steps > 0 and cond_src is None:
            repaint_inv_steps = 0

        eps_anchor = None
        if repaint_anchor_eps and (x_src is not None):
            if hasattr(x_src, "feats") and hasattr(x_src, "coords"):
                eps_anchor = torch.randn_like(x_src.feats)
            else:
                eps_anchor = torch.randn_like(x_src)

        t_seq = np.linspace(1, 0, steps + 1)
        t_seq = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)
        t_pairs = list((t_seq[i], t_seq[i + 1]) for i in range(steps))

        ret = edict({"samples": None, "pred_x_t": [], "pred_x_0": []})

        for step_i, (t, t_prev) in enumerate(
            tqdm(t_pairs, desc="Sampling (Repaint)", disable=not verbose)
        ):
            kwargs["t_sign"] = t
            kwargs["t_1"] = t

            out = self.sample_once(model, sample, t, t_prev, cond, **kwargs)
            z_edit = out.pred_x_prev

            if x_src is not None and mask is not None and repaint_beta > 0.0:
                if step_i < repaint_inv_steps:
                    _drop = {
                        "x_src", "mask", "stage", "ori_cond",
                        "repaint_beta", "repaint_anchor_eps",
                        "repaint_inv_steps",
                    }
                    inner_kwargs = {k: v for k, v in kwargs.items() if k not in _drop}
                    x0_consistent = self._tweedie_self_consistent_x0(
                        model, x_src, t, cond_src,
                        eps_anchor=eps_anchor, stage=stage, **inner_kwargs,
                    )
                    z_src_t_prev = self._forward_diffuse_src(
                        x0_consistent, t_prev,
                        stage=stage, eps_override=eps_anchor,
                    )
                else:
                    z_src_t_prev = self._forward_diffuse_src(
                        x_src, t_prev,
                        stage=stage, eps_override=eps_anchor,
                    )

                z_next = self._mask_blend(
                    z_edit, z_src_t_prev, mask,
                    stage=stage, beta=repaint_beta,
                )
            else:
                z_next = z_edit

            sample = z_next
            ret.pred_x_t.append(z_next)
            ret.pred_x_0.append(out.pred_x_0)

        if x_src is not None and mask is not None:
            sample = self._mask_overwrite_outside(sample, x_src, mask, stage=stage)

        ret.samples = sample
        return ret

    def _tweedie_self_consistent_x0(
        self, model, x_src, t, cond_src, eps_anchor=None, stage="sparse", **kwargs
    ):
        
        z_src_naive = self._forward_diffuse_src(
            x_src, t, stage=stage, eps_override=eps_anchor,
        )
        _, _, v_src = self._get_model_prediction(model, z_src_naive, t, cond_src, **kwargs)
        tv = float(t)
        if hasattr(z_src_naive, "feats") and hasattr(z_src_naive, "coords"):
            x0_feats = z_src_naive.feats - tv * v_src.feats
            return z_src_naive.replace(feats=x0_feats)
        return z_src_naive - tv * v_src

    def _forward_diffuse_src(self, x_src, t_prev, stage="sparse", eps_override=None):
       
        t_val = float(t_prev)
        scale_src = 1.0 - t_val
        scale_eps = self.sigma_min + (1.0 - self.sigma_min) * t_val

        if hasattr(x_src, "feats") and hasattr(x_src, "coords"):
            eps = eps_override if eps_override is not None else torch.randn_like(x_src.feats)
            new_feats = scale_src * x_src.feats + scale_eps * eps
            return x_src.replace(feats=new_feats)

        eps = eps_override if eps_override is not None else torch.randn_like(x_src)
        return scale_src * x_src + scale_eps * eps

    def _mask_blend(self, z_edit, z_src_t, mask, stage="sparse", beta: float = 1.0):
        
        if not (hasattr(z_edit, "feats") and hasattr(z_edit, "coords")):
            m = mask
            if not torch.is_tensor(m):
                m = torch.as_tensor(m, device=z_edit.device, dtype=z_edit.dtype)
            else:
                m = m.to(device=z_edit.device, dtype=z_edit.dtype)
            return z_edit + (1.0 - m) * beta * (z_src_t - z_edit)

        if beta >= 1.0 - 1e-6:
            alpha = 0.0
            return blend_feats_ring1_outside_rawmask(z_edit, z_src_t, mask, alpha=alpha)

        device = z_edit.feats.device
        coords_e = z_edit.coords.to(device).long()
        coords_s = z_src_t.coords.to(device).long()
        m_grid = mask.to(device)
        if m_grid.dtype != torch.bool:
            m_grid = m_grid > 0
        b_e, x_e, y_e, z_e = coords_e.unbind(dim=1)
        is_inside = m_grid[b_e, 0, x_e, y_e, z_e]

        ori_map = {tuple(c.tolist()): i for i, c in enumerate(coords_s)}

        new_feats = z_edit.feats.clone()
        for i in range(coords_e.size(0)):
            if bool(is_inside[i]):
                continue
            j = ori_map.get(tuple(coords_e[i].tolist()))
            if j is None:
                continue
            new_feats[i] = beta * z_src_t.feats[j].to(new_feats.dtype) \
                         + (1.0 - beta) * new_feats[i]
        return z_edit.replace(feats=new_feats)

    def _mask_overwrite_outside(self, sample, x_src, mask, stage="sparse"):
       
        if not (hasattr(sample, "feats") and hasattr(sample, "coords")):
            m = mask
            if not torch.is_tensor(m):
                m = torch.as_tensor(m, device=sample.device, dtype=sample.dtype)
            else:
                m = m.to(device=sample.device, dtype=sample.dtype)
            return m * sample + (1.0 - m) * x_src

        
        device = sample.feats.device
        coords_s = sample.coords.to(device).long()
        coords_o = x_src.coords.to(device).long()
        m_grid = mask.to(device)
        if m_grid.dtype != torch.bool:
            m_grid = m_grid > 0
        b_s, x_s, y_s, z_s = coords_s.unbind(dim=1)
        is_inside = m_grid[b_s, 0, x_s, y_s, z_s]

        ori_map = {tuple(c.tolist()): i for i, c in enumerate(coords_o)}

        new_feats = sample.feats.clone()
        for i in range(coords_s.size(0)):
            if bool(is_inside[i]):
                continue
            j = ori_map.get(tuple(coords_s[i].tolist()))
            if j is None:
                continue
            new_feats[i] = x_src.feats[j].to(new_feats.dtype)
        return sample.replace(feats=new_feats)


class FlowEulerCfgSampler(ClassifierFreeGuidanceSamplerMixin, FlowEulerSampler):
    @torch.no_grad()
    def sample(
        self,
        model,
        noise,
        cond,
        neg_cond,
        steps: int = 50,
        rescale_t: float = 1.0,
        cfg_strength: float = 3.0,
        verbose: bool = True,
        **kwargs
    ):
        return super().sample(model, noise, cond, steps, rescale_t, verbose, neg_cond=neg_cond, cfg_strength=cfg_strength, **kwargs)


class FlowEulerGuidanceIntervalSampler(GuidanceIntervalSamplerMixin, FlowEulerSampler):
    @torch.no_grad()
    def sample(
        self,
        model,
        noise,
        cond,
        neg_cond,
        steps: int = 50,
        rescale_t: float = 1.0,
        cfg_strength: float = 3.0,
        cfg_interval: Tuple[float, float] = (0.0, 1.0),
        verbose: bool = True,
        **kwargs
    ):
        return super().sample(model, noise, cond, steps, rescale_t, verbose, neg_cond=neg_cond, cfg_strength=cfg_strength, cfg_interval=cfg_interval, **kwargs)
