from typing import *
from contextlib import contextmanager
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
from PIL import Image
import rembg
from .base import Pipeline
from . import samplers
from ..modules import sparse as sp
from torch import Tensor
from ..utils.general_utils import load_img_features,save_feature,read_feature,merge_coords,export_indexed_coords_to_glb_cubes_colored
import os
import torchvision.utils as vutils
import utils3d
from ..renderers import OctreeRenderer
from ..representations.octree import DfsOctree as Octree

class TrellisImageTo3DPipeline(Pipeline):
    """
    Pipeline for inferring Trellis image-to-3D models.

    Args:
        models (dict[str, nn.Module]): The models to use in the pipeline.
        sparse_structure_sampler (samplers.Sampler): The sampler for the sparse structure.
        slat_sampler (samplers.Sampler): The sampler for the structured latent.
        slat_normalization (dict): The normalization parameters for the structured latent.
        image_cond_model (str): The name of the image conditioning model.
    """
    def __init__(
        self,
        models: dict[str, nn.Module] = None,
        sparse_structure_sampler: samplers.Sampler = None,
        slat_sampler: samplers.Sampler = None,
        slat_normalization: dict = None,
        image_cond_model: str = None,
    ):
        if models is None:
            return
        super().__init__(models)
        self.sparse_structure_sampler = sparse_structure_sampler
        self.slat_sampler = slat_sampler
        self.sparse_structure_sampler_params = {}
        self.slat_sampler_params = {}
        self.slat_normalization = slat_normalization
        self.rembg_session = None
        self._init_image_cond_model(image_cond_model)

    @staticmethod
    def from_pretrained(path: str) -> "TrellisImageTo3DPipeline":
        """
        Load a pretrained model.

        Args:
            path (str): The path to the model. Can be either local path or a Hugging Face repository.
        """
        pipeline = super(TrellisImageTo3DPipeline, TrellisImageTo3DPipeline).from_pretrained(path)
        new_pipeline = TrellisImageTo3DPipeline()
        new_pipeline.__dict__ = pipeline.__dict__
        args = pipeline._pretrained_args

        new_pipeline.sparse_structure_sampler = getattr(samplers, args['sparse_structure_sampler']['name'])(**args['sparse_structure_sampler']['args'])
        new_pipeline.sparse_structure_sampler_params = args['sparse_structure_sampler']['params']

        new_pipeline.slat_sampler = getattr(samplers, args['slat_sampler']['name'])(**args['slat_sampler']['args'])
        new_pipeline.slat_sampler_params = args['slat_sampler']['params']

        new_pipeline.slat_normalization = args['slat_normalization']

        new_pipeline._init_image_cond_model(args['image_cond_model'])

        return new_pipeline
    
    def _init_image_cond_model(self, name: str):
        """
        Initialize the image conditioning model.
        """
        dinov2_model = torch.hub.load('facebookresearch/dinov2', name, pretrained=True)
        dinov2_model.eval()
        self.models['image_cond_model'] = dinov2_model
        transform = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.image_cond_model_transform = transform

    def preprocess_image(self, input: Image.Image) -> Image.Image:
        """
        Preprocess the input image.
        """
        # if has alpha channel, use it directly; otherwise, remove background
        has_alpha = False
        if input.mode == 'RGBA':
            alpha = np.array(input)[:, :, 3]
            if not np.all(alpha == 255):
                has_alpha = True
        if has_alpha:
            output = input
        else:
            input = input.convert('RGB')
            max_size = max(input.size)
            scale = min(1, 1024 / max_size)
            if scale < 1:
                input = input.resize((int(input.width * scale), int(input.height * scale)), Image.Resampling.LANCZOS)
            if getattr(self, 'rembg_session', None) is None:
                self.rembg_session = rembg.new_session('u2net')
            output = rembg.remove(input, session=self.rembg_session)
        output_np = np.array(output)
        alpha = output_np[:, :, 3]
        bbox = np.argwhere(alpha > 0.8 * 255)
        bbox = np.min(bbox[:, 1]), np.min(bbox[:, 0]), np.max(bbox[:, 1]), np.max(bbox[:, 0])
        center = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
        size = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
        size = int(size * 1.2)
        bbox = center[0] - size // 2, center[1] - size // 2, center[0] + size // 2, center[1] + size // 2
        output = output.crop(bbox)  # type: ignore
        output = output.resize((518, 518), Image.Resampling.LANCZOS)
        output = np.array(output).astype(np.float32) / 255
        output = output[:, :, :3] * output[:, :, 3:4]
        output = Image.fromarray((output * 255).astype(np.uint8))
        return output

    @torch.no_grad()
    def encode_image(self, image: Union[torch.Tensor, list[Image.Image]]) -> torch.Tensor:
        """
        Encode the image.

        Args:
            image (Union[torch.Tensor, list[Image.Image]]): The image to encode

        Returns:
            torch.Tensor: The encoded features.
        """
        if isinstance(image, torch.Tensor):
            assert image.ndim == 4, "Image tensor should be batched (B, C, H, W)"
        elif isinstance(image, list):
            assert all(isinstance(i, Image.Image) for i in image), "Image list should be list of PIL images"
            image = [i.resize((518, 518), Image.LANCZOS) for i in image]
            image = [np.array(i.convert('RGB')).astype(np.float32) / 255 for i in image]
            image = [torch.from_numpy(i).permute(2, 0, 1).float() for i in image]
            image = torch.stack(image).to(self.device)
        else:
            raise ValueError(f"Unsupported type of image: {type(image)}")
        
        image = self.image_cond_model_transform(image).to(self.device)
        features = self.models['image_cond_model'](image, is_training=True)['x_prenorm']
        patchtokens = F.layer_norm(features, features.shape[-1:])
        return patchtokens
        
    def get_cond(self, image: Union[torch.Tensor, list[Image.Image]]) -> dict:
        """
        Get the conditioning information for the model.

        Args:
            image (Union[torch.Tensor, list[Image.Image]]): The image prompts.

        Returns:
            dict: The conditioning information
        """
        cond = self.encode_image(image)
        neg_cond = torch.zeros_like(cond)
        return {
            'cond': cond,
            'neg_cond': neg_cond,
        }

    @torch.no_grad()
    def visualize_sample(self, ss: Union[torch.Tensor, dict]):
        ss = ss if isinstance(ss, torch.Tensor) else ss['ss']
        
        renderer = OctreeRenderer()
        renderer.rendering_options.resolution = 512
        renderer.rendering_options.near = 0.8
        renderer.rendering_options.far = 1.6
        renderer.rendering_options.bg_color = (0, 0, 0)
        renderer.rendering_options.ssaa = 4
        renderer.pipe.primitive = 'voxel'
        
        # Build camera
        yaws = [0, np.pi / 2, np.pi, 3 * np.pi / 2]
        yaws_offset = 0
        yaws = [y + yaws_offset for y in yaws]
        pitch = [0.0, 0.0, 0.0, 0.0]

        exts = []
        ints = []
        for yaw, pitch in zip(yaws, pitch):
            orig = torch.tensor([
                np.sin(yaw) * np.cos(pitch),
                np.cos(yaw) * np.cos(pitch),
                np.sin(pitch),
            ]).float().cuda() * 2
            fov = torch.deg2rad(torch.tensor(30)).cuda()
            extrinsics = utils3d.torch.extrinsics_look_at(orig, torch.tensor([0, 0, 0]).float().cuda(), torch.tensor([0, 0, 1]).float().cuda())
            intrinsics = utils3d.torch.intrinsics_from_fov_xy(fov, fov)
            exts.append(extrinsics)
            ints.append(intrinsics)

        images = []
        
        # Build each representation
        ss = ss.cuda()
        for i in range(ss.shape[0]):
            representation = Octree(
                depth=10,
                aabb=[-0.5, -0.5, -0.5, 1, 1, 1],
                device='cuda',
                primitive='voxel',
                sh_degree=0,
                primitive_config={'solid': True},
            )
            coords = torch.nonzero(ss[i, 0], as_tuple=False)
            representation.position = coords.float() / 64
            representation.depth = torch.full((representation.position.shape[0], 1), int(np.log2(64)), dtype=torch.uint8, device='cuda')

            image = torch.zeros(3, 1024, 1024).cuda()
            tile = [2, 2]
            for j, (ext, intr) in enumerate(zip(exts, ints)):
                res = renderer.render(representation, ext, intr, colors_overwrite=representation.position)
                image[:, 512 * (j // tile[1]):512 * (j // tile[1] + 1), 512 * (j % tile[1]):512 * (j % tile[1] + 1)] = res['color']
            images.append(image)
            
        return torch.stack(images)
       
    def sample_sparse_structure(
        self,
        cond: dict,
        num_samples: int = 1,
        sampler_params: dict = {},
        **kwargs,
    ) -> torch.Tensor:
        """
        Sample sparse structures with the given conditioning.
        
        Args:
            cond (dict): The conditioning information.
            num_samples (int): The number of samples to generate.
            sampler_params (dict): Additional parameters for the sampler.
        """
        # Sample occupancy latent
        flow_model = self.models['sparse_structure_flow_model']
        reso = flow_model.resolution
        sampler_params = {**self.sparse_structure_sampler_params, **sampler_params}
        if kwargs.get("mode","baseline") =="flowedit":
            latent=kwargs["latent"]
            z_s = self.sparse_structure_sampler.sample(
            flow_model,
            latent,
            **cond,
            **sampler_params,
            **kwargs,
            verbose=True
            ).samples
            decoder = self.models['sparse_structure_decoder']
            coords1 = torch.argwhere(decoder(z_s)>0)[:, [0, 2, 3, 4]].int()
            #####可视化结果（默认关闭，成本：每步解码 + 4视角渲染，很慢）
            enable_step_viz = bool(kwargs.get("enable_step_viz", False))
            if enable_step_viz:
                device = self.device
                features = load_img_features(kwargs["feature_path"])
                feature_dir = os.path.dirname(kwargs["feature_path"])
                parent_dir = os.path.dirname(feature_dir)
                save_dir = os.path.join(parent_dir, "visualize_latent_flowedit")
                os.makedirs(save_dir, exist_ok=True)
                for k, z_s in features.items():
                    z_s = z_s.to(self.device)
                    # print(z_s.shape)

                    decoded = decoder(z_s) > 0
                    resolution = decoded.shape[-1]
                    coords = torch.argwhere(decoded)[:, [0, 2, 3, 4]].int()  # [batch_idx, z, y, x]

                    voxel_tensor = torch.zeros((z_s.shape[0], 1, resolution, resolution, resolution), dtype=torch.float32, device=device)

                    for b in range(z_s.shape[0]):
                        batch_coords = coords[coords[:, 0] == b][:, 1:]  # 取该 batch 的 voxel 坐标
                        voxel_tensor[b, 0, batch_coords[:, 0], batch_coords[:, 1], batch_coords[:, 2]] = 1.0

                    images = self.visualize_sample(voxel_tensor)

                    for idx, img in enumerate(images):
                        name = f"{k}_{idx}.png" if z_s.shape[0] > 1 else f"{k}.png"
                        vutils.save_image(img, os.path.join(save_dir, name))
                coords = torch.argwhere(kwargs["ori_mask"])[:, [0, 2, 3, 4]].int()
                coords_union = torch.cat([coords, kwargs['latent_slat'].coords], dim=0)
                coords_union = torch.unique(coords_union, dim=0)
                coords = coords_union
                voxel_tensor = torch.zeros((z_s.shape[0], 1, resolution, resolution, resolution), dtype=torch.float32, device=device)

                for b in range(z_s.shape[0]):
                    batch_coords = coords[coords[:, 0] == b][:, 1:]  # 取该 batch 的 voxel 坐标
                    voxel_tensor[b, 0, batch_coords[:, 0], batch_coords[:, 1], batch_coords[:, 2]] = 1.0

                images = self.visualize_sample(voxel_tensor)
                for idx, img in enumerate(images):
            # 如果 batch 大于 1，保存为 k_0.png, k_1.png ...
                    name = f"mask.png" if z_s.shape[0] > 1 else f"mask.png"
                    vutils.save_image(img, os.path.join(save_dir, name))
            return coords1
        else:
            noise = torch.randn(num_samples, flow_model.in_channels, reso, reso, reso).to(self.device)
            
            z_s = self.sparse_structure_sampler.sample(
            flow_model,
            noise,
            **cond,
            **sampler_params,
            **kwargs,
            verbose=True
            ).samples
            decoder = self.models['sparse_structure_decoder']
            coords1 = torch.argwhere(decoder(z_s)>0)[:, [0, 2, 3, 4]].int()
            if kwargs["use_latent"]:
                features = load_img_features(kwargs["feature_path"])
                device = self.device
                feature_dir = os.path.dirname(kwargs["feature_path"])
                parent_dir = os.path.dirname(feature_dir)
                save_dir = os.path.join(parent_dir, "visualize_latent1")
                os.makedirs(save_dir, exist_ok=True)
                for k, z_s in features.items():
                    z_s = z_s.to(self.device)
                    # print(z_s.shape)
   
                    decoded = decoder(z_s) > 0  
                    resolution = decoded.shape[-1]
                    coords = torch.argwhere(decoded)[:, [0, 2, 3, 4]].int()  # [batch_idx, z, y, x]
                    # print("coords")
                    # print(coords.shape)
                    if k=="0.1111111111111112_img":
                        ori_coords=coords
                    voxel_tensor = torch.zeros((z_s.shape[0], 1, resolution, resolution, resolution), dtype=torch.float32, device=device)
    
                    for b in range(z_s.shape[0]):
                        batch_coords = coords[coords[:, 0] == b][:, 1:]  # 取该 batch 的 voxel 坐标
                        voxel_tensor[b, 0, batch_coords[:, 0], batch_coords[:, 1], batch_coords[:, 2]] = 1.0
    
                    images = self.visualize_sample(voxel_tensor)

                    for idx, img in enumerate(images):
        # 如果 batch 大于 1，保存为 k_0.png, k_1.png ...
                        name = f"{k}_{idx}.png" if z_s.shape[0] > 1 else f"{k}.png"
                        vutils.save_image(img, os.path.join(save_dir, name))
                coords = torch.argwhere(kwargs["ori_mask"])[:, [0, 2, 3, 4]].int()  # [batch_idx, z, y, x]
                #print("coords")
                #print(coords.shape)
                coords_union = torch.cat([coords, ori_coords], dim=0)
                coords_union = torch.unique(coords_union, dim=0)
                coords = coords_union
                voxel_tensor = torch.zeros((z_s.shape[0], 1, resolution, resolution, resolution), dtype=torch.float32, device=device)
    
                for b in range(z_s.shape[0]):
                    batch_coords = coords[coords[:, 0] == b][:, 1:]  # 取该 batch 的 voxel 坐标
                    voxel_tensor[b, 0, batch_coords[:, 0], batch_coords[:, 1], batch_coords[:, 2]] = 1.0
    
                images = self.visualize_sample(voxel_tensor)

                for idx, img in enumerate(images):
        # 如果 batch 大于 1，保存为 k_0.png, k_1.png ...
                    name = f"mask.png" if z_s.shape[0] > 1 else f"mask.png"
                    vutils.save_image(img, os.path.join(save_dir, name))
            return coords1

    def decode_slat(
        self,
        slat: sp.SparseTensor,
        formats: List[str] = ['mesh', 'gaussian', 'radiance_field'],
    ) -> dict:
        """
        Decode the structured latent.

        Args:
            slat (sp.SparseTensor): The structured latent.
            formats (List[str]): The formats to decode the structured latent to.

        Returns:
            dict: The decoded structured latent.
        """
        ret = {}
        if 'mesh' in formats:
            ret['mesh'] = self.models['slat_decoder_mesh'](slat)
        if 'gaussian' in formats:
            ret['gaussian'] = self.models['slat_decoder_gs'](slat)
        if 'radiance_field' in formats:
            ret['radiance_field'] = self.models['slat_decoder_rf'](slat)
        return ret
    
    def sample_slat(
        self,
        cond: dict,
        coords: torch.Tensor,
        sampler_params: dict = {},
        **kwargs,
    ) -> sp.SparseTensor:
        """
        Sample structured latent with the given conditioning.
        
        Args:
            cond (dict): The conditioning information.
            coords (torch.Tensor): The coordinates of the sparse structure.
            sampler_params (dict): Additional parameters for the sampler.

        kwargs:
            - mode: "baseline" | "repaint"。repaint 模式下使用 SLAT repainting 公式。
            - latent_slat (SparseTensor, optional): repaint 模式下的源 SLAT（未归一化）。
            - mask (Tensor, optional): repaint 模式下的 3D raw_mask [B,1,64,64,64]（1=可编辑区）。
        """
        # Sample structured latent
        flow_model = self.models['slat_flow_model']
        sampler_params = {**self.slat_sampler_params, **sampler_params}

        std = torch.tensor(self.slat_normalization['std'])[None].to(self.device)
        mean = torch.tensor(self.slat_normalization['mean'])[None].to(self.device)

        mode = kwargs.get("mode", "baseline")

        if mode == "repaint":
            # === SLAT Repainting：非编辑区直接用源的前向解析加噪，无需 inversion ===
            latent_slat = kwargs["latent_slat"]
            # 归一化到模型训练域
            x_src = (latent_slat - mean) / std

            # 初始噪声：与 coords 对齐的纯高斯，SparseTensor
            noise = sp.SparseTensor(
                feats=torch.randn(coords.shape[0], flow_model.in_channels).to(self.device),
                coords=coords,
            )

            # 把源 latent 对齐到目标 coords 上（只保留 coords 中的行，缺失位置填0）。
            # 为简洁起见，假设 repaint 阶段调用方已经把 coords 设成源 coords（slat 编辑典型场景下成立）。
            # 调用方需保证：x_src.coords 与 coords 对齐到相同顺序；否则 _mask_blend 会按 SparseTensor 坐标差异分支处理。
            repaint_kwargs = dict(kwargs)
            repaint_kwargs["mode"] = "repaint"
            repaint_kwargs["x_src"] = x_src
            # kwargs["mask"] 已经是 raw_mask [B,1,64,64,64]，直接透传
            slat = self.slat_sampler.sample(
                flow_model,
                noise,
                **cond,
                **sampler_params,
                **repaint_kwargs,
                verbose=True
            ).samples
            slat = slat * std + mean
        else:
            # baseline：从纯噪声生成
            noise = sp.SparseTensor(
                feats=torch.randn(coords.shape[0], flow_model.in_channels).to(self.device),
                coords=coords,
            )
            slat = self.slat_sampler.sample(
                flow_model,
                noise,
                **cond,
                **sampler_params,
                **kwargs,
                verbose=True
            ).samples
            slat = slat * std + mean
        return slat
    
    @torch.no_grad()
    def flowedit(
        self,
        ori_image: Image.Image,
        image: Image.Image,
        num_samples: int = 1,
        seed: int = 42,
        sparse_structure_sampler_params: dict = {},
        slat_sampler_params: dict = {},
        formats: List[str] = ['mesh', 'gaussian', 'radiance_field'],
        latent: Optional[Tensor] = None,
        latent_slat: Optional[Tensor] = None,
        coords: Optional[Tensor] = None,
        feature_path: Optional[str] = None,
        latent_mask: Optional[Tensor] = None,
        raw_mask: Optional[Tensor] = None,
        preprocess_image: bool = True,
        edit_mask:  Optional[Tensor] = None,
        cfg_src_strength: float = 5.0,
        cfg_tar_strength: float = 5.0,
        enable_step_viz: bool = False,
        use_guidance: bool = True,
    ):
        """
        完整编辑流程：先用 FlowEdit 编辑 sparse structure 体素，再用 Repainting 编辑 SLAT。
        整个过程共用同一个 raw_mask（3D 体素掩码，[B,1,64,64,64]，1=可编辑区）。

        新增可调参数：
            cfg_src_strength: FlowEdit 中源条件的 CFG 强度（默认 5.0）
            cfg_tar_strength: FlowEdit 中目标条件的 CFG 强度（默认 5.0）
            enable_step_viz:  是否导出 FlowEdit 每步的 4 视角可视化（默认 False，加速）
            use_guidance:     是否启用正交轮廓引导（默认 True）
        """
        if preprocess_image:
            ori_image = self.preprocess_image(ori_image)
            image = self.preprocess_image(image)
        ori_cond = cond = self.encode_image([ori_image])
        cond = self.get_cond([image])

        # ===== Stage 1: sparse structure - FlowEdit =====
        kwargs = {}
        kwargs["mode"] = "flowedit"
        kwargs["latent"] = latent
        kwargs["latent_slat"] = latent_slat
        kwargs["feature_path"] = os.path.join(feature_path, "feature_flowedit_test.pkl")
        kwargs["edit_mask"] = edit_mask
        # 新增：CFG / 引导 / 可视化开关透传到 sampler
        kwargs["cfg_src_strength"] = cfg_src_strength
        kwargs["cfg_tar_strength"] = cfg_tar_strength
        kwargs["use_guidance"] = use_guidance
        kwargs["enable_step_viz"] = enable_step_viz
        if latent_mask is not None:
            kwargs["mask"] = latent_mask
            kwargs["use_mask"] = True
            kwargs["ori_mask"] = raw_mask
        else:
            kwargs["use_mask"] = False
        kwargs["stage"] = "sparse"
        kwargs["ori_cond"] = ori_cond
        kwargs["decode_voxel_fn"] = self.models['sparse_structure_decoder']
        coords1 = self.sample_sparse_structure(cond, num_samples, sparse_structure_sampler_params, **kwargs)

        # 将新生成的 coords1 与源 coords 按 raw_mask 合并：mask 内用新 coords，mask 外保留源 coords
        coords_combined = merge_coords(
            coords1,
            latent_slat.coords,
            raw_mask,
            device=self.device,
            hole_fill_radius=0
        )

        # ===== Stage 2: slat - Repainting =====
        kwargs["decode_voxel_fn"] = None
        kwargs["stage"] = "slat"
        kwargs["mask"] = raw_mask
        kwargs["mode"] = "repaint"
        kwargs["latent_slat"] = latent_slat
        slat = self.sample_slat(cond, coords_combined, slat_sampler_params, **kwargs)

        # 导出可视化
        os.makedirs(feature_path, exist_ok=True)
        export_indexed_coords_to_glb_cubes_colored(latent_slat.coords, os.path.join(feature_path, "vox_ori.glb"), slat_feat=latent_slat.feats)
        export_indexed_coords_to_glb_cubes_colored(coords_combined, os.path.join(feature_path, "vox_edit.glb"), slat_feat=slat.feats)
        return self.decode_slat(slat, formats)

    @torch.no_grad()
    def run(
        self,
        image: Image.Image,
        num_samples: int = 1,
        seed: int = 42,
        sparse_structure_sampler_params: dict = {},
        slat_sampler_params: dict = {},
        formats: List[str] = ['mesh', 'gaussian', 'radiance_field'],
        preprocess_image: bool = True,
        latent_slat: Optional[sp.SparseTensor] = None,
        mask: Optional[Tensor] = None,
    ) -> dict:
        """
        Run the pipeline.

        两种模式：
          1) 普通生成：不传 latent_slat / mask -> 从纯噪声生成 sparse structure 和 slat
          2) SLAT Repainting：传 latent_slat（源SLAT）+ mask（3D raw_mask, [B,1,64,64,64], 1=可编辑区）
             -> sparse structure 从纯噪声生成（不做编辑），slat 阶段使用 SLAT repainting 公式：
                z_{k-1} = M ⊙ [z_k + Δt v_θ(z_k, t_k | cond)] + (1-M) ⊙ [(1-t_{k-1}) z^src + t_{k-1} ε_k]

        Args:
            image (Image.Image): 目标图像条件 I^tgt
            latent_slat (sp.SparseTensor, optional): 源 SLAT（未归一化），用于 repainting 的 z^src
            mask (Tensor, optional): 3D raw_mask，[B,1,64,64,64]，值 1 表示可编辑区
            preprocess_image (bool): Whether to preprocess the image.
        """
        if preprocess_image:
            image = self.preprocess_image(image)
        cond = self.get_cond([image])
        torch.manual_seed(seed)

        do_repaint = (latent_slat is not None) and (mask is not None)

        # ===== Stage 1: sparse structure =====
        # 不做 sparse 编辑，baseline 生成；若后续需要约束 coords，可由调用方用 latent_slat.coords 决定
        kwargs_ss = {"mode": "baseline", "stage": "sparse"}
        coords1 = self.sample_sparse_structure(
            cond, num_samples, sparse_structure_sampler_params, **kwargs_ss
        )

        # ===== Stage 2: slat =====
        if do_repaint:
            # slat repaint：coords 使用源 latent_slat 的 coords，以便与 x_src 对齐
            coords_slat = latent_slat.coords.to(self.device)
            kwargs_slat = {
                "mode": "repaint",
                "stage": "slat",
                "latent_slat": latent_slat,
                "mask": mask,
            }
            slat = self.sample_slat(cond, coords_slat, slat_sampler_params, **kwargs_slat)
        else:
            kwargs_slat = {"mode": "baseline", "stage": "slat"}
            slat = self.sample_slat(cond, coords1, slat_sampler_params, **kwargs_slat)

        return self.decode_slat(slat, formats)

    @contextmanager
    def inject_sampler_multi_image(
        self,
        sampler_name: str,
        num_images: int,
        num_steps: int,
        mode: Literal['stochastic', 'multidiffusion'] = 'stochastic',
    ):
        """
        Inject a sampler with multiple images as condition.
        
        Args:
            sampler_name (str): The name of the sampler to inject.
            num_images (int): The number of images to condition on.
            num_steps (int): The number of steps to run the sampler for.
        """
        sampler = getattr(self, sampler_name)
        setattr(sampler, f'_old_inference_model', sampler._inference_model)

        if mode == 'stochastic':
            if num_images > num_steps:
                print(f"\033[93mWarning: number of conditioning images is greater than number of steps for {sampler_name}. "
                    "This may lead to performance degradation.\033[0m")

            cond_indices = (np.arange(num_steps) % num_images).tolist()
            def _new_inference_model(self, model, x_t, t, cond, **kwargs):
                cond_idx = cond_indices.pop(0)
                cond_i = cond[cond_idx:cond_idx+1]
                return self._old_inference_model(model, x_t, t, cond=cond_i, **kwargs)
        
        elif mode =='multidiffusion':
            from .samplers import FlowEulerSampler
            def _new_inference_model(self, model, x_t, t, cond, neg_cond, cfg_strength, cfg_interval, **kwargs):
                if cfg_interval[0] <= t <= cfg_interval[1]:
                    preds = []
                    for i in range(len(cond)):
                        preds.append(FlowEulerSampler._inference_model(self, model, x_t, t, cond[i:i+1], **kwargs))
                    pred = sum(preds) / len(preds)
                    neg_pred = FlowEulerSampler._inference_model(self, model, x_t, t, neg_cond, **kwargs)
                    return (1 + cfg_strength) * pred - cfg_strength * neg_pred
                else:
                    preds = []
                    for i in range(len(cond)):
                        preds.append(FlowEulerSampler._inference_model(self, model, x_t, t, cond[i:i+1], **kwargs))
                    pred = sum(preds) / len(preds)
                    return pred
            
        else:
            raise ValueError(f"Unsupported mode: {mode}")
            
        sampler._inference_model = _new_inference_model.__get__(sampler, type(sampler))

        yield

        sampler._inference_model = sampler._old_inference_model
        delattr(sampler, f'_old_inference_model')

    @torch.no_grad()
    def run_multi_image(
        self,
        images: List[Image.Image],
        num_samples: int = 1,
        seed: int = 42,
        sparse_structure_sampler_params: dict = {},
        slat_sampler_params: dict = {},
        formats: List[str] = ['mesh', 'gaussian', 'radiance_field'],
        preprocess_image: bool = True,
        mode: Literal['stochastic', 'multidiffusion'] = 'stochastic',
    ) -> dict:
        """
        Run the pipeline with multiple images as condition

        Args:
            images (List[Image.Image]): The multi-view images of the assets
            num_samples (int): The number of samples to generate.
            sparse_structure_sampler_params (dict): Additional parameters for the sparse structure sampler.
            slat_sampler_params (dict): Additional parameters for the structured latent sampler.
            preprocess_image (bool): Whether to preprocess the image.
        """
        if preprocess_image:
            images = [self.preprocess_image(image) for image in images]
        cond = self.get_cond(images)
        cond['neg_cond'] = cond['neg_cond'][:1]
        torch.manual_seed(seed)
        ss_steps = {**self.sparse_structure_sampler_params, **sparse_structure_sampler_params}.get('steps')
        with self.inject_sampler_multi_image('sparse_structure_sampler', len(images), ss_steps, mode=mode):
            coords = self.sample_sparse_structure(cond, num_samples, sparse_structure_sampler_params)
        slat_steps = {**self.slat_sampler_params, **slat_sampler_params}.get('steps')
        with self.inject_sampler_multi_image('slat_sampler', len(images), slat_steps, mode=mode):
            slat = self.sample_slat(cond, coords, slat_sampler_params)
        return self.decode_slat(slat, formats)
