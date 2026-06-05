<div align="center">

# Easy3E: Feed-Forward 3D Asset Editing via Rectified Voxel Flow

<h3>CVPR 2026</h3>

<small>  Shimin Hu &nbsp;&nbsp; · &nbsp;&nbsp; Yuanyi Wei &nbsp;&nbsp; · &nbsp;&nbsp; Fei Zha   &nbsp;&nbsp; · &nbsp;&nbsp;   [Yudong Guo](https://yudongguo.github.io/)   &nbsp;&nbsp; · &nbsp;&nbsp;  [Juyong Zhang](http://staff.ustc.edu.cn/~juyong/)

University of Science and Technology of China

</div>

<div align="center">
  <a href="https://ustc3dv.github.io/Easy3E/"><img src="https://img.shields.io/badge/Project%20Page-333399.svg?logo=googlehome" height="22px"></a>
  <a href="https://arxiv.org/pdf/2602.21499"><img src="https://img.shields.io/badge/ArXiv-b5212f.svg?logo=arxiv" height="22px"></a>
  <a href="https://github.com/linchanyi/Easy3E/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" height="22px"></a>
</div>

<div align="center">
  <img src="assets/teaser.png" alt="Easy3E teaser" width="100%">
</div>

---

**Easy3E** is a feed-forward 3D editing framework, designed to modify 3D models from a single editing view. It introduces **Voxel FlowEdit** in sparse voxel latent space for globally consistent 3D deformation, and a **normal-guided single-to-multi-view generation** module to restore high-fidelity appearance details.

## 🗓️ Release Plan

| Stage | Component | Status |
| :--- | :--- | :---: |
| 1 | Geometry editing | ✅ Released |
| 2 | Texture refinement (normal-guided single-to-multi-view generation) | 🚧 Coming soon |

## 📋 Requirements

- **OS**: Linux (tested on Ubuntu 20.04+)
- **GPU**: NVIDIA GPU with ≥24GB VRAM (A100/A6000 recommended)
- **CUDA**: 11.8+
- **Python**: 3.10+

## 🚀 Installation

### 1. Clone the repository

```bash
git clone https://github.com/linchanyi/Easy3E.git
cd Easy3E
```

### 2. Python environment

Easy3E shares the same Python environment as [TRELLIS](https://github.com/microsoft/TRELLIS). Please follow the official TRELLIS installation guide to set up PyTorch, `spconv`, `xformers`, `flash-attn`, `vox2seq`, `kaolin`, `nvdiffrast`, etc.

```bash
# Follow https://github.com/microsoft/TRELLIS for the full setup
git clone --recurse-submodules https://github.com/microsoft/TRELLIS.git
# then run TRELLIS's setup.sh in the conda env you plan to use for Easy3E
```



### 3. Blender (placed under `tmp/`)

Easy3E renders multi-view images with Blender 4.0.0. By default the code looks for the executable at `tmp/blender-4.0.0-linux-x64/blender` (under the repo root). Just download and extract it into `tmp/`:

```bash
mkdir -p tmp
cd tmp
wget https://download.blender.org/release/Blender4.0/blender-4.0.0-linux-x64.tar.xz
tar -xf blender-4.0.0-linux-x64.tar.xz
cd ..
```

After this, you should have:

```
Easy3E/tmp/blender-4.0.0-linux-x64/blender
```

If you want to use a Blender at a different location, set the `BLENDER_PATH` environment variable to its absolute path.

### 4. Download the TRELLIS checkpoint

We use the `TRELLIS-image-large` weights. Download them into `checkpoint/`:

```bash
mkdir -p checkpoint
# Option A: huggingface-cli
huggingface-cli download microsoft/TRELLIS-image-large \
    --local-dir checkpoint/TRELLIS-image-large

# Option B: git lfs
git lfs install
git clone https://huggingface.co/microsoft/TRELLIS-image-large \
    checkpoint/TRELLIS-image-large
```


## 🔧 Usage

We organize data per case. Pick any directory as the example root (e.g. `../example/`), and put each case in a subfolder named by its ID (`girl/`, `tiger/`, ...). 

### Step 1: Preprocess

Put the source 3D model into `<case>/model/`. Supported formats: `.glb` / `.obj` / `.ply` / `.gltf` / `.fbx` / etc.

```
../example/girl/
└── model/
    └── girl.glb
```

Run preprocessing on a single case:

```bash
python inference.py --preprocess \
    --root_dir ../example/girl \
    --obj_path ../example/girl/model
```

Or batch-preprocess every case under an example root (each subfolder must contain a `model/`):

```bash
bash preprocess_batch.sh ../example
```

### Step 2: Edit

Before editing, add two folders to the same case:

- `<case>/cond/<edit_name>.png` — condition image of the edit
- `<case>/mask/<edit_name>.glb` — 3D mask mesh marking the editable region (same basename as the cond image)

```
../example/girl/
├── model/         # source model
├── cond/          # edit condition images, e.g. edit1.png
└── mask/          # 3D mask meshes,        e.g. edit1.glb
```

Edit `ID` / `BASE` in [`inference.sh`](./inference.sh) to point to your case, then run:

```bash
bash inference.sh
```

Edited results are written to `<case>/output/<edit_name>.glb`.

## 🙏 Acknowledgements

This project builds upon [TRELLIS](https://github.com/microsoft/TRELLIS) — 3D asset generation via structured latents.

## 📄 License

This project is released under the [MIT License](LICENSE).

## 📖 Citation

If you find this work useful, please cite our CVPR 2026 paper:

```bibtex
@inproceedings{hu2026easy3e,
  title={Easy3E: Feed-Forward 3D Asset Editing via Rectified Voxel Flow},
  author={Hu, Shimin and Wei, Yuanyi and Zha, Fei and Guo, Yudong and Zhang, Juyong},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2026}
}
```
