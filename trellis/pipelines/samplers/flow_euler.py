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
    """
    Generate samples from a flow-matching model using Euler sampling.

    Args:
        sigma_min: The minimum scale of noise in flow.
    """
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
        """
        Sample x_{t-1} from the model using Euler method.
        
        Args:
            model: The model to sample from.
            x_t: The [N x C x ...] tensor of noisy inputs at time t.
            t: The current timestep.
            t_prev: The previous timestep.
            cond: conditional information.
            **kwargs: Additional arguments for model inference.

        Returns:
            a dict containing the following
            - 'pred_x_prev': x_{t-1}.
            - 'pred_x_0': a prediction of x_0.
        """
       
        pred_x_0, pred_eps, pred_v = self._get_model_prediction(model, x_t, t, cond, **kwargs) ####change
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
    
    def _prep_noise_table(self,sample, steps, n_avg, seed):
        import os, random, numpy as np, torch
        device, dtype = sample.device, sample.dtype
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        if device.type == "cuda": torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")

        K = max(1, n_avg)
        gen = torch.Generator(device=device).manual_seed(seed)
        eps_table = [
            torch.randn(sample.shape, generator=gen, device=device, dtype=dtype)
            for _ in range(steps * K)
        ]

        # Antithetic sampling (variance reduction, works better when n_avg is even)
        if K % 2 == 0:
            for s in range(steps):
                half = K // 2
                for k in range(half, K):
                    eps_table[s*K + k] = -eps_table[s*K + (k - half)]

        # Noise for the first step of the tail segment
        tail_eps = torch.randn(sample.shape, generator=gen, device=device, dtype=dtype)
        return eps_table, tail_eps

    @torch.no_grad()
    def sample_flowedit(
        self,
        model,
        noise,
        cond: Optional[Any] = None,   # Default fallback only; overridden by cond_src / cond_tar
        steps: int = 25,
        rescale_t: float = 3.0,
        verbose: bool = True,
        **kwargs
        ):
        """
        FlowEdit sampling (inversion-free), does not read/write latent or fuse features.
        Required kwargs (all have defaults, pass as needed):
        - cond_src, cond_tar: source/target conditions (defaults to cond)
        - n_min=0, n_max=None, n_avg=4
        - alpha_sched=lambda t: t, lock_sched=lambda t: 0.0
        - mask=None (broadcastable with sample)
        - src_latent=None (strongly recommended: clean source state; if not provided, uses a snapshot of current sample)
        - use_same_noise=True
        - forward_noise_fn=None  # default: (1-t)*x0 + t*eps
        """
        # ------ Read control parameters ------
        cond_src = kwargs["ori_cond"]
        cond_tar = cond

        n_min = int(kwargs.get("n_min", 2))
        n_max = kwargs.get("n_max", None)  # None = editable throughout all steps
        n_avg = int(kwargs.get("n_avg", 4))
        alpha_sched = kwargs.get("alpha_sched", lambda t: 1.0)     # Difference vector strength (small early, large later)
        manual_mask = kwargs.get("mask", None)
        feather = int(kwargs.get("feather", 2))
        guard   = int(kwargs.get("guard", 1))
        # Linear + lower bound, maintaining some strength in early/mid stages
        lock_sched = lambda t: min(0.95, 0.40 + (1.0 - t)*0.45)
        # t goes from 1→0: lock-back grows from ~0.40 to ~0.85 (max 0.95)


        #soft_mask = make_soft_mask(mask)
        
        forward_noise_fn = kwargs.get("forward_noise_fn", None)

        edit_mask_in = kwargs.get("edit_mask",None)
        ortho_scale = float(kwargs.get("subject_width", 1.0))
        
        tx             = float(kwargs.get("tx", 0.0))
        ty             = float(kwargs.get("ty", 0.0))
        flip_y         = bool(kwargs.get("flip_y", False))

        # Guidance switch: enabled by default; also compatible with legacy name use_ortho_guidance
        use_guidance = bool(kwargs.get("use_guidance", kwargs.get("use_ortho_guidance", True)))
        lambda0 = float(kwargs.get("lambda0", 1.0))

        # Camera azimuth for guidance projection (default 270° = front view = 012.png)
        azimuth_deg = float(kwargs.get("azimuth_deg", 270.0))
        # Debug visualization switch for guidance alignment
        debug_guidance_viz = bool(kwargs.get("debug_guidance_viz", False))
        debug_viz_dir = kwargs.get("debug_viz_dir", None)

        # CFG strength (src / tar decoupled, allows external override)
        cfg_src_strength = float(kwargs.get("cfg_src_strength", 5.0))
        cfg_tar_strength = float(kwargs.get("cfg_tar_strength", 5.0))

        decode_voxel_fn = kwargs.get("decode_voxel_fn", None)


        # ------ Initialize state ------
        sample = noise  # Note: FlowEdit does not do latent I/O or feature fusion; noise here is the original input
        T_steps = steps
        #eps_table, tail_eps = self._prep_noise_table(noise, steps=steps, n_avg=n_avg, seed=kwargs.get("seed", 20))
        # Source reference state x_src0 (best to pass a clean latent/image; otherwise uses a copy of current sample)
        x_src0 = kwargs.get("src_latent", None)
        if x_src0 is None:
            x_src0 = sample.clone()

        # Default forward noise function (Rectified Flow style)
        if forward_noise_fn is None:
            def forward_noise_fn(x0, t_scalar, eps):
                # t_scalar ∈ [0,1]; keeps dtype/device consistent with x0
                sigma_min = 1e-5 
                t_val = float(t_scalar)
                return (1.0 - t_val) * x0 + (sigma_min + (1 - sigma_min) * t_val) * eps
        
        # ------ Construct time grid (consistent with baseline) ------
        t_seq = np.linspace(1, 0, steps + 1)
        t_seq = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)
        t_pairs = [(t_seq[i], t_seq[i + 1]) for i in range(steps)]
        if n_max is None:
            n_max = T_steps

        # ------ Return container ------
        edict = lambda d: type("edict", (object,), d)
        ret = edict({
        "samples": None, "pred_x_t": [], "pred_x_0": [],
        "viz_grid_paths": [], "auto_masks": [], "mask_metrics": []
        })  # FIX: add missing fields


        iterator = enumerate(t_pairs)
        if verbose:
            try:
                from tqdm import tqdm
                iterator = tqdm(iterator, total=len(t_pairs), desc="Sampling (FlowEdit)")
            except Exception:
                pass

        manual_soft, manual_guard = (None, None)
        if manual_mask is not None:
            manual_soft, manual_guard = build_soft_masks(manual_mask, sample, feather=feather, guard=guard)
        # ------ Main loop ------
        combined_soft  = manual_soft

        # Normalize edit_mask to device/shape
        edit_mask = _to_mask_tensor(edit_mask_in, sample)  # [B?,1,H,W]
        if edit_mask.shape[0] != sample.shape[0]:
            if edit_mask.shape[0] == 1:
                edit_mask = edit_mask.expand(sample.shape[0], -1, -1, -1)
            else:
                raise ValueError(f"edit_mask batch mismatch with sample: {edit_mask.shape[0]} vs {sample.shape[0]}")
        H_mask, W_mask = int(edit_mask.shape[-2]), int(edit_mask.shape[-1])

        for step_i, (t, t_prev) in iterator:
            # Determine whether current step is in the "editable middle segment" or "tail target refinement"
            can_edit    = (T_steps - step_i) <= n_max
            tail_refine = (T_steps - step_i) <= n_min

            if not can_edit:
                continue

            if not tail_refine:
                # ========= Middle segment: difference vector field + n_avg averaging =========
                V_delta_avg = torch.zeros_like(sample)
                K = max(1, n_avg)
                
                
                for k in range(K):
                    # Same-noise differencing (recommended): src/tar share the same noise instance
                    #eps = eps_table[step_i * K + k]
                    eps = torch.randn_like(x_src0)
                    # Construct z_t^src and z_t^tar
                    zt_src = forward_noise_fn(x_src0, t, eps)
                    
                    # Keep source outside mask
                    
                    zt_src = apply_mask_blend(zt_src, x_src0, combined_soft)
                    #zt_src = soft_mask*zt_src + (1-soft_mask)*x_src0
                    zt_tar = sample + (zt_src - x_src0)

                    # Velocity field (take pred_v)
                    kwargs["cfg_strength"] = cfg_src_strength
                    _x0s, _epss, v_src = self._get_model_prediction(model, zt_src, t, cond_src, **kwargs)
                    kwargs["cfg_strength"] = cfg_tar_strength
                    _x0t, _epst, v_tar = self._get_model_prediction(model, zt_tar, t, cond_tar, **kwargs)

                    delta = (v_tar - v_src)

                    

                    V_delta_avg = V_delta_avg + delta / float(K)

                
                
                V_delta_avg = V_delta_avg * combined_soft
                # Euler step forward (same notation as baseline)
                dt = (t - t_prev)
                alpha = alpha_sched(float(t))
                pred_x_prev = sample - dt * (alpha * V_delta_avg)

                
                

                new_state = pred_x_prev
                if step_i >=8 and use_guidance:
                    try:
                        with torch.enable_grad():
                            x_t = new_state.detach().requires_grad_(True)
                            sigma = decode_voxel_fn(x_t)  # [B,1,D,H,W] (continuous)
                            tau  = 0.6 if float(t) > 0.5 else 0.5

                            # Project silhouette at the specified azimuth angle
                            sil_raw = silhouette_at_azimuth(sigma, azimuth_deg=azimuth_deg, tau=tau)
                            sil_img = project_ortho_no_center(
                                sil_raw, out_hw=(H_mask, W_mask),
                                ortho_scale=ortho_scale, tx=tx, ty=ty, flip_y=flip_y
                            )  # [B,1,H,W]

                            # Debug visualization (save every 5 steps to avoid too many files)
                            if debug_guidance_viz and debug_viz_dir and (step_i % 5 == 0):
                                save_guidance_debug_viz(
                                    sil_img, edit_mask, step_i, azimuth_deg, debug_viz_dir,
                                    prefix="mid"
                                )

                            # Loss (can stack DT/contour loss later)
                            L_ortho = F.binary_cross_entropy(sil_img.clamp(1e-6, 1-1e-6), edit_mask)
                            g_latent = torch.autograd.grad(L_ortho, x_t, retain_graph=False, create_graph=False)[0]

                            a=_stat(g_latent,"g_latent")

                            ret.mask_metrics.append({"step": int(step_i), "L_ortho": float(L_ortho.detach().cpu())})

                            if use_guidance and (g_latent is not None):
                                new_state = new_state - dt * (lambda0 * g_latent)
                    except Exception as e:
                        ret.mask_metrics.append({"step": int(step_i), "err": str(e)})

                pred_x_prev = new_state

                # Pull non-mask region back to source (optional)
                lock = lock_sched(float(t))
                if (manual_mask is not None) and (lock > 0):
                    pred_x_prev = (1 - lock * (1 - manual_mask)) * pred_x_prev + (lock * (1 - manual_mask)) * x_src0

                sample = pred_x_prev
                ret.pred_x_t.append(pred_x_prev)
                ret.pred_x_0.append(None)  # If x0 is needed, run an additional forward pass

            else:
                # ========= Tail segment: target refinement (SDEdit style, target condition only) =========
                if (T_steps - step_i) == n_min:
                    #eps = tail_eps
                    eps = torch.randn_like(x_src0)
                    xt_src = forward_noise_fn(x_src0, t, eps)
                    xt_src = apply_mask_blend(xt_src, x_src0, combined_soft)
                    xt_tar = sample + (xt_src - x_src0)
                else:
                    xt_tar = sample
                kwargs["cfg_strength"] = cfg_tar_strength
                pred_x0, pred_eps, v_tar = self._get_model_prediction(model, xt_tar, t, cond_tar, **kwargs)
                if combined_soft is not None:  # FIX: guard against None
                    v_tar = v_tar * combined_soft
                dt = (t - t_prev)
                pred_x_prev = xt_tar - dt * v_tar

                new_state = pred_x_prev
                if use_guidance :
                    try:
                        with torch.enable_grad():
                            x_t = new_state.detach().requires_grad_(True)
                            sigma = decode_voxel_fn(x_t)  # [B,1,D,H,W] (continuous)
                            tau  = 0.6 if float(t) > 0.5 else 0.5

                            # Project silhouette at the specified azimuth angle
                            sil_raw = silhouette_at_azimuth(sigma, azimuth_deg=azimuth_deg, tau=tau)
                            sil_img = project_ortho_no_center(
                                sil_raw, out_hw=(H_mask, W_mask),
                                ortho_scale=ortho_scale, tx=tx, ty=ty, flip_y=flip_y
                            )  # [B,1,H,W]

                            # Debug visualization
                            if debug_guidance_viz and debug_viz_dir and (step_i % 5 == 0):
                                save_guidance_debug_viz(
                                    sil_img, edit_mask, step_i, azimuth_deg, debug_viz_dir,
                                    prefix="tail"
                                )

                            # Loss
                            L_ortho = F.binary_cross_entropy(sil_img.clamp(1e-6, 1-1e-6), edit_mask)
                            g_latent = torch.autograd.grad(L_ortho, x_t, retain_graph=False, create_graph=False)[0]

                            ret.mask_metrics.append({"step": int(step_i), "L_ortho": float(L_ortho.detach().cpu())})

                            if use_guidance and (g_latent is not None):
                                new_state = new_state - dt * (lambda0 * g_latent)
                    except Exception as e:
                        ret.mask_metrics.append({"step": int(step_i), "err": str(e)})

                pred_x_prev = new_state

                # Pull non-mask region back to source (optional)
                lock = lock_sched(float(t))
                if (manual_mask is not None) and (lock > 0):
                    pred_x_prev = (1 - lock * (1 - manual_mask)) * pred_x_prev + (lock * (1 - manual_mask)) * x_src0

                sample = pred_x_prev
                ret.pred_x_t.append(pred_x_prev)
                ret.pred_x_0.append(pred_x0)
        
        new_state = pred_x_prev
        step_i = 25
        if use_guidance:
            try:
                with torch.enable_grad():
                    x_t = new_state.detach().requires_grad_(True)
                    sigma = decode_voxel_fn(x_t)  # [B,1,D,H,W] (continuous)
                    tau  = 0.6 if float(t) > 0.5 else 0.5

                    # Project silhouette at the specified azimuth angle
                    sil_raw = silhouette_at_azimuth(sigma, azimuth_deg=azimuth_deg, tau=tau)
                    sil_img = project_ortho_no_center(
                        sil_raw, out_hw=(H_mask, W_mask),
                        ortho_scale=ortho_scale, tx=tx, ty=ty, flip_y=flip_y
                    )  # [B,1,H,W]

                    # Debug visualization (final step)
                    if debug_guidance_viz and debug_viz_dir:
                        save_guidance_debug_viz(
                            sil_img, edit_mask, step_i, azimuth_deg, debug_viz_dir,
                            prefix="final"
                        )

                    # Loss
                    L_ortho = F.binary_cross_entropy(sil_img.clamp(1e-6, 1-1e-6), edit_mask)
                    g_latent = torch.autograd.grad(L_ortho, x_t, retain_graph=False, create_graph=False)[0]

                    ret.mask_metrics.append({"step": int(step_i), "L_ortho": float(L_ortho.detach().cpu())})

                    if use_guidance and (g_latent is not None):
                        if combined_soft is not None:
                            g_latent = g_latent * combined_soft
                        new_state = new_state - dt * (lambda0 * g_latent)
            except Exception as e:
                ret.mask_metrics.append({"step": int(step_i), "err": str(e)})

        sample = new_state
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
        """
        Baseline pure forward generation: sample from noise all the way to x_0, no repainting/inversion involved.
        """
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
        """
        SLAT Repainting sampling (inversion-free).

        Formula (for each timestep t -> t_prev):
            z_edit  = z_k + Δt * v_θ(z_k, t_k | cond_tar)
            z_src_t = (sigma_min + (1 - sigma_min) * t_prev) * eps + (1 - t_prev) * x_src
            z_{k-1} = M ⊙ z_edit + (1 - M) ⊙ z_src_t

        Required kwargs:
            - x_src: Source sample latent ("clean" x_0). SparseTensor or Tensor, shape aligned with noise.
            - mask: M, broadcastable with x_src/sample. Value 1 = editable region, 0 = keep source.
                    - Tensor (dense): shape broadcastable to sample
                    - SparseTensor: per-row (feats) 0/1 mask, aligned with sample.feats
                    - None: all editable (degenerates to normal generation)
            - stage: "sparse" | "slat", determines the specific branch for masking/noising
        """
        sample = noise
        x_src = kwargs.get("x_src", None)
        mask = kwargs.get("mask", None)
        stage = kwargs.get("stage", "sparse")

        t_seq = np.linspace(1, 0, steps + 1)
        t_seq = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)
        t_pairs = list((t_seq[i], t_seq[i + 1]) for i in range(steps))

        ret = edict({"samples": None, "pred_x_t": [], "pred_x_0": []})

        for t, t_prev in tqdm(t_pairs, desc="Sampling (Repaint)", disable=not verbose):
            kwargs["t_sign"] = t
            kwargs["t_1"] = t

            # 1) One Euler step with target condition to get z_edit
            out = self.sample_once(model, sample, t, t_prev, cond, **kwargs)
            z_edit = out.pred_x_prev

            # 2) Non-editable region: forward analytical noising of source to get z_src_t_prev
            if x_src is not None and mask is not None:
                z_src_t_prev = self._forward_diffuse_src(x_src, t_prev, stage=stage)
                z_next = self._mask_blend(z_edit, z_src_t_prev, mask, stage=stage)
            else:
                # No source/mask: degenerates to normal generation
                z_next = z_edit

            sample = z_next
            ret.pred_x_t.append(z_next)
            ret.pred_x_0.append(out.pred_x_0)

        ret.samples = sample
        return ret

    def _forward_diffuse_src(self, x_src, t_prev, stage="sparse"):
        """
        Forward analytical noising of source sample (rectified flow style):
            z_t = (1 - t) * x_src + (sigma_min + (1 - sigma_min) * t) * eps
        Returns an object of the same type as x_src (Tensor or SparseTensor).
        """
        t_val = float(t_prev)
        scale_src = 1.0 - t_val
        scale_eps = self.sigma_min + (1.0 - self.sigma_min) * t_val

        # SparseTensor: add noise to feats, keep coords unchanged
        if hasattr(x_src, "feats") and hasattr(x_src, "coords"):
            eps = torch.randn_like(x_src.feats)
            new_feats = scale_src * x_src.feats + scale_eps * eps
            return x_src.replace(feats=new_feats)

        # Dense Tensor
        eps = torch.randn_like(x_src)
        return scale_src * x_src + scale_eps * eps

    def _mask_blend(self, z_edit, z_src_t, mask, stage="sparse"):
        """
        Mask blending: z = M * z_edit + (1 - M) * z_src_t
        - stage="sparse": dense Tensor, mask broadcastable with z_edit
        - stage="slat": SparseTensor. Assumes z_edit and z_src_t have aligned coordinates (repaint requirement),
                        mask is a per-row 0/1 scalar, or 3D raw_mask pre-converted to per-row mask by caller.
        """
        # Dense Tensor branch
        if not (hasattr(z_edit, "feats") and hasattr(z_edit, "coords")):
            m = mask
            if not torch.is_tensor(m):
                m = torch.as_tensor(m, device=z_edit.device, dtype=z_edit.dtype)
            else:
                m = m.to(device=z_edit.device, dtype=z_edit.dtype)
            return m * z_edit + (1.0 - m) * z_src_t

        # SparseTensor branch: use blend_feats_ring1_outside_rawmask, compatible with project's existing mask format
        # mask here is expected to be raw_mask of shape [B,1,64,64,64]
        alpha = 0.0
        return blend_feats_ring1_outside_rawmask(z_edit, z_src_t, mask, alpha=alpha)


class FlowEulerCfgSampler(ClassifierFreeGuidanceSamplerMixin, FlowEulerSampler):
    """
    Generate samples from a flow-matching model using Euler sampling with classifier-free guidance.
    """
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
        """
        Generate samples from the model using Euler method.
        
        Args:
            model: The model to sample from.
            noise: The initial noise tensor.
            cond: conditional information.
            neg_cond: negative conditional information.
            steps: The number of steps to sample.
            rescale_t: The rescale factor for t.
            cfg_strength: The strength of classifier-free guidance.
            verbose: If True, show a progress bar.
            **kwargs: Additional arguments for model_inference.

        Returns:
            a dict containing the following
            - 'samples': the model samples.
            - 'pred_x_t': a list of prediction of x_t.
            - 'pred_x_0': a list of prediction of x_0.
        """
        return super().sample(model, noise, cond, steps, rescale_t, verbose, neg_cond=neg_cond, cfg_strength=cfg_strength, **kwargs)


class FlowEulerGuidanceIntervalSampler(GuidanceIntervalSamplerMixin, FlowEulerSampler):
    """
    Generate samples from a flow-matching model using Euler sampling with classifier-free guidance and interval.
    """
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
    ): ###change
        """
        Generate samples from the model using Euler method.
        
        Args:
            model: The model to sample from.
            noise: The initial noise tensor.
            cond: conditional information.
            neg_cond: negative conditional information.
            steps: The number of steps to sample.
            rescale_t: The rescale factor for t.
            cfg_strength: The strength of classifier-free guidance.
            cfg_interval: The interval for classifier-free guidance.
            verbose: If True, show a progress bar.
            **kwargs: Additional arguments for model_inference.

        Returns:
            a dict containing the following
            - 'samples': the model samples.
            - 'pred_x_t': a list of prediction of x_t.
            - 'pred_x_0': a list of prediction of x_0.
        """
        return super().sample(model, noise, cond, steps, rescale_t, verbose, neg_cond=neg_cond, cfg_strength=cfg_strength, cfg_interval=cfg_interval, **kwargs)
