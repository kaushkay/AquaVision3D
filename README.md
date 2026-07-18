# AquaVision3D

Underwater 3D reconstruction with a **spatial hash-grid medium** on top of
[WaterSplatting](https://water-splatting.github.io/) (Li et al., 3DV 2025).

WaterSplatting models water with a direction-conditioned MLP. We keep their
3D Gaussian + volumetric rendering setup, but replace that MLP with
`HashMedium`: a multi-resolution hash grid queried at each Gaussian’s position,
producing the usual medium parameters (veiling light B∞, backscatter β_B,
attenuation β_D).

The method name in Nerfstudio is `hash-water-splatting`. Per-scene hash AABB
sizes that we locked after ablations are in `water_splatting/hash_scene_aabb.py`.

## Install

Same stack as WaterSplatting: [nerfstudio](https://docs.nerf.studio/index.html),
PyTorch, and tiny-cuda-nn.

```bash
conda create --name aquavision -y python=3.8
conda activate aquavision
python -m pip install --upgrade pip

# PyTorch (CUDA 11.8 example)
pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 \
  --extra-index-url https://download.pytorch.org/whl/cu118
conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit

pip install ninja git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
pip install nerfstudio==1.1.4
ns-install-cli

git clone https://github.com/kaushkay/AquaVision3D.git
cd AquaVision3D
git submodule update --init --recursive
pip install --no-use-pep517 -e .
```

## Data

Use COLMAP-undistorted scenes (same idea as WaterSplatting / SeaThru-NeRF):

```bash
colmap image_undistorter \
  --image_path /path/to/scene/images_wb \
  --input_path /path/to/scene/colmap/sparse/0 \
  --output_path /path/to/undistorted/scene \
  --output_type COLMAP
```

For eval we use Nerfstudio’s interval split (`eval_mode=interval`,
`eval_interval=8`) on COLMAP frame order.

## Train

**Ours (hash medium):**

```bash
export NERFSTUDIO_METHOD_CONFIGS="hash-water-splatting=water_splatting.hash_water_splatting_config:hash_water_splatting_method"

# Panama example — aabb 44.7 from hash_scene_aabb.py
ns-train hash-water-splatting \
  --max-num-iterations 15000 \
  --pipeline.model.hash-aabb-scale 44.7 \
  --pipeline.datamanager.cache-images cpu \
  --vis tensorboard \
  colmap --downscale-factor 1 --colmap-path sparse \
  --data /path/to/undistorted/Panama --images-path images
```

Defaults worth knowing: 15k steps, L1 + SSIM, medium warmup for the first 2k
steps, and soft clamping on medium logits.

**WaterSplatting baseline** (direction MLP, for comparison):

```bash
export NERFSTUDIO_METHOD_CONFIGS="water-splatting=water_splatting.water_splatting_config:water_splatting_method"

ns-train water-splatting \
  --max-num-iterations 15000 \
  --vis tensorboard \
  colmap --downscale-factor 1 --colmap-path sparse \
  --data /path/to/undistorted/Panama --images-path images
```

## Eval and render

```bash
ns-eval \
  --load-config outputs/unnamed/hash-water-splatting/<timestamp>/config.yml \
  --output-path outputs/eval/my-eval.json

ns-render dataset \
  --load-config outputs/unnamed/hash-water-splatting/<timestamp>/config.yml \
  --rendered-output-names rgb gt-rgb depth rgb_clear rgb_medium \
  --split test \
  --image-format png \
  --output-path renders/qualitative/my-run
```

Local run artifacts (`outputs/`, `renders/`, checkpoints) are gitignored.

## What’s in this repo

```
water_splatting/
  hash_medium.py              # hash-grid medium field
  hash_water_splatting.py     # 3DGS + hash medium model
  hash_water_splatting_config.py
  hash_scene_aabb.py          # locked aabb per scene
  water_splatting.py          # original WS baseline
  cuda/                       # rasterizer (from WaterSplatting)
```

## Acknowledgements

Built on [WaterSplatting](https://github.com/water-splatting/water-splatting)
(Li et al., 3DV 2025). Please cite them if you use the baseline medium or CUDA
rasterizer.

```bibtex
@article{li2025watersplatting,
  title={{W}ater{S}platting: Fast Underwater {3D} Scene Reconstruction using Gaussian Splatting},
  author={Li, Huapeng and Song, Wenxuan and Xu, Tianao and Elsig, Alexandre and Kulhanek, Jonas},
  journal={3DV},
  year={2025}
}
```
