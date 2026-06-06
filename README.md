<div align="center">
<h1>[CVPR 2026 Highlight] No Calibration, No Depth, No Problem: Cross-Sensor View Synthesis with 3D Consistency</h1>

<a href="https://arxiv.org/abs/2602.23559"><img src="https://img.shields.io/badge/arXiv-2602.23559-red" alt="arXiv"></a>
<a href="https://choyingw.github.io/3d-rgbx.github.io/"><img src="https://img.shields.io/badge/Project_Page-3D_RGBX" alt="Project Page"></a>
<a href="https://www.youtube.com/watch?v=kbOrt-hfGqU"><img src="https://img.shields.io/badge/Video-YouTube-yellow" alt="Video"></a>

<p align="center">
  <a href='https://choyingw.github.io/3d-rgbx.github.io/'>
  <img src="assets/teaser.gif" alt="teaser" style="width: 100%;">
  Check Project Page for More Visuals
  </a>
</p>
</div>

## 🧐Overview

<div class="row">
    <div class="col-md-8 col-md-offset-2">
        <section>
            <p>
              In multi-modal learning from different sensors, such as RGB + thermal cameras, Near-Infared (NIR) cameras, or Synthetic Aperture Radar (SAR), most prior works assume paired data exist and only focus on designing the network for fusing the multi-modal features. 
            </p>
            <blockquote style="font-size: 13px;">
                <strong>However, in real-world applications, especially in robotics and autonomous driving, we often encounter scenarios where perfectly aligned pairs do not exist.</strong>
            </blockquote>
            <p>
              Toward the goal, traditional pipelines require laborious calibration and depth estimation to establish cross-sensor correspondences, which can be costly and error-prone. In this work, we present the first scalable data processing framework that attempts to align the view from raw sensor sequsences.
            </p>
          </section>
    </div>
</div>

## 🔧Environment

Before running any script, update the local paths for your machine. The repository uses `/path/to/3D-RGBX` as a placeholder; replace it with the absolute path to this cloned repository, or export `RGBX_ROOT` before running the pipeline:

```bash
export RGBX_ROOT=/path/to/3D-RGBX
```

Create the basic Python environment from the repository root. Python 3.10 and a CUDA-capable GPU are recommended for the default pipeline.

```bash
conda create -n 3d-rgbx python=3.10 -y
conda activate 3d-rgbx

# Install PyTorch for your CUDA version first. Example for CUDA 12.1:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

pip install -U pip setuptools wheel
pip install -r requirements.txt

# Install 3D Gaussian Splatting CUDA extensions from the repository root.
pip install -e ./dual-diff-gaussian-ray-splatter
pip install -e ./simple-knn
```

The densification scripts import NVIDIA Apex. If Apex is not already available in your environment, install it after PyTorch:

```bash
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation \
  git+https://github.com/NVIDIA/apex.git
```

Prepare the pretrained weights under `checkpoints/` or override `CHECKPOINT_ROOT` when running the pipeline. The default pipeline expects, which can be downloaded <a href="https://drive.google.com/drive/folders/1ls0HXmf6k_kcPOYyk4LFgp9N0ignWbTQ?usp=sharing">here</a>:

```text
checkpoints/
  resnet34.pth
  densification_rgbt.pt
  minima_loftr.ckpt
  weights_xoftr_640.ckpt
  sam2.1_hiera_large.pt
  groundingdino_swint_ogc.pth
```

## 😺Data

We preprocess RGBT-Scenes with colmap. Download the data <a href="https://drive.google.com/drive/folders/1FfPg7gqwREzAKf4jkMlyEwGyvUbw6Eer?usp=sharing">here</a>. The folder contains RoadBlock, Parterre, LandScape, Dimsum, Building Scene. Each scene contains

```text
rgb: RGB images
thermal: raw thermal images
thermal_aug_1chan: augmented thermal images (1 channel for matching)
sparse: colmap results
image_generation: baseline method of image generation from StyleBooth
```

Put the dataset under the root folder ./RGBT-Scenes

## 🚀Processing

Edit `execute_pipeline_rgbt.sh` for the scene you want to process.

```bash
scene_name="Building"
SCENE_ROOT="./RGBT-Scenes/$scene_name"
RGB_DIR="$SCENE_ROOT/rgb/train"
TARGET_DIR="$SCENE_ROOT/thermal_aug_1chan/train"
```

**Important: before running, you must update the repository path.** Either edit `RGBX_ROOT` in `scripts/pipeline_common.sh` or export it in your shell:

```bash
export RGBX_ROOT=/path/to/3D-RGBX
export DENSIFICATION_ROOT=$RGBX_ROOT/densification
export CHECKPOINT_ROOT=$RGBX_ROOT/checkpoints
export RGBT_DENSIFICATION_CKPT=$CHECKPOINT_ROOT/densification_rgbt.pt
```

**Replace `/path/to/3D-RGBX` with the absolute path to this repository.** The placeholder path will not run as-is.

Run the full RGBT processing pipeline:

```bash
bash execute_pipeline_rgbt.sh
```

The script runs the following stages:

1. Generate semi-dense RGB-X matching maps with `semidense_matching.py` at three match thresholds.
2. Densify each matching result with `densification/src/first_densi.py`.
3. Average the three densified outputs with `level_mean.py`.
4. Filter the averaged maps with `filtering.py`.
5. Refine the filtered maps with `densification/src/second_densi.py` at three sample rates.
6. Average the refined outputs into the final result.

Outputs are written to:

```text
demo/pipelines/<scene_name>/
  matching/
  dens/
  mean/
  filtered/
  refined/
  refined_mean/
```

## 🎨3DGS Training and Rendering

After `refined_mean/` is generated, the pipeline trains an RGBT 3D Gaussian Splatting model with:

```text
RGB supervision:     ./RGBT-Scenes/<scene_name>/rgb/train
Thermal supervision: ./demo/pipelines/<scene_name>/refined_mean
```

`execute_pipeline_rgbt.sh` runs this 3DGS training and rendering stage automatically for the Building example:

```bash
bash execute_pipeline_rgbt.sh
```

The 3DGS outputs are saved inside the same scene pipeline folder:

```text
demo/pipelines/<scene_name>/
  gs_model/              # trained 3DGS checkpoint and config
  gs_rendered/
    train/               # rendered train views
    test/                # rendered test views
```

Training uses the `rgb/train` split only. Rendering saves train and test views separately when `rgb/test` images are available. For METU-VisTIR scenes, `render.py` automatically applies the width crop and 518 resize; RGBT-Scenes are rendered without that crop.

If your RGBT-Scenes folder is outside this repository, override the dataset root:

```bash
SCENE_ROOT=/path/to/RGBT-Scenes/Building \
  bash execute_pipeline_rgbt.sh
```

## 📚Citation

If you find our work useful, please consider cite our paper:

```bibtex
@inproceedings{choyingwu3drgbx,
  title={No Calibration, No Depth, No Problem: Cross-Sensor View Synthesis with 3D Consistency},
  author={Wu, Cho-Ying and Huang, Zixun and Huang, Xinyu and Ren, Liu},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={21836--21848},
  year={2026}
}
```
