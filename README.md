# GEO_2DBA

**GEO_2DBA** is a unified, open-source re-implementation of our cross-view
ground-to-satellite localization pipeline. Ground and satellite images are
matched in a shared 2D BEV space, and the full 6-DoF-lite pose (2D
translation + yaw) is recovered by a differentiable **cross-unnorm solver**
that closes over the bearing-only geometry of the camera rays.

One code base covers four benchmarks — **KITTI**, **RobotCar Seasons**,
**CVACT**, **VIGOR** — with a single unified solver exposed three ways:

| Solver | Gradient | Role |
|---|---|---|
| `e2e` (coarse sweep + fine soft-argmin) | differentiable | drives the training loss |
| `bisection` (closed-form C + bisection) | no | training/eval metric |
| `RANSAC` (probabilistic Procrustes) | no | training/eval metric (`--no_ransac` to skip) |

## Method overview

For every ground cell whose DINOv3 feature matches a satellite BEV cell, the
ground cell contributes a bearing ray `θ` (pinhole: `θ = atan((u−cx)/fx) − π/2`;
panorama: the equirectangular azimuth) and a weight. Under the cross-unnorm
residual

```
cross(ψ) = Δx·sin(ψ+θ) + Δy·cos(ψ+θ) = 0
```

the true yaw `ψ` zeroes *all* rays' cross products against the
ground-to-satellite offset `(Δx, Δy)`. The e2e solver sweeps `ψ` coarsely
(cheirality-weighted branch selection), then soft-argmins a fine window;
the translation `t = Δx·cos(ψ+θ) − Δy·sin(ψ+θ)` follows in closed form.

## Installation

```bash
conda create -n geo2dba python=3.10
conda activate geo2dba
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements.txt
```

The frozen backbone is a DINOv3 ViT-L/16 with
[RoMaV2](https://github.com/Parskatt/RoMaV2) weights; both are downloaded
automatically through `torch.hub` on first use (hub cache:
`~/.cache/torch/hub`). Offline machines should pre-populate the cache.

The pre-processing step additionally uses the external
[Depth-Anything-V2](https://github.com/DepthAnything/Depth-Anything-V2) and
[UniK3D](https://github.com/lpiccinelli-eth/UniK3D) repositories — see
[preprocess/README.md](preprocess/README.md) for the one-time setup.

## Data preparation

Point `config.ini` at your dataset roots, then produce the pre-computed
metric depth maps (all four datasets) and dynamic-object masks (KITTI,
RobotCar) with the scripts in [`preprocess/`](preprocess/README.md).

Expected layout per dataset (all pre-processing outputs are resumable):

```
KITTI/<root>/                        RobotCar/<root>/
├── raw_data/<day>/<drive>/          ├── robotcar_dataset.csv        (GT pairs)
│   └── image_02/data/*.png          ├── sampled_frames.csv          (sampling stage)
├── depth_anythingv2_depth/...       ├── <run>/stereo_preprocess_v2/ (levelled images + log)
├── mask2former/<day>/<drive>/...    ├── depth/<ts>.png
└── satmap/<drive>/*.png             └── mask2former/<ts>.png

CVACT/<root>/ANU_data_small/         VIGOR/<root>/<city>/
├── streetview/*.jpg                 ├── panorama/*.jpg
├── unik3d_depth/*.png               ├── unik3d_depth/*.png
└── satellite_test/...               └── satellite/*.png
```

## Released checkpoints

Validation metrics in the file names follow each legacy protocol
(KITTI/RobotCar: `loc_2_mean` on test1; CVACT/VIGOR: `translation_mean`).
All of the checkpoints below are bundled under [`checkpoints/`](checkpoints/).

| Dataset | Config | Checkpoint | Val metric |
|---|---|---|---|
| KITTI 10° | `--rotation_range 10` | `kitti10/epoch_7_step_153_0.6334873647582089_model.pt` | 0.633 m |
| KITTI 180° | `--rotation_range 180` | `kitti180/epoch_12_step_1.7016808789135593_model.pt` | 1.702 m |
| RobotCar 10° | `--rotation_range 10` | `robotcar10/epoch_95_step_0.7338818706212855_model.pt` | 0.734 m |
| RobotCar 180° | `--rotation_range 180` | `robotcar180/epoch_50_step_1.7676119810193913_model.pt` | 1.768 m |
| CVACT known-ori | `--rotation_range 0` | `cvact/known_ori/2_model_5.380451.pt` | 5.380 m |
| VIGOR same-area 0° | `--split samearea --rotation_range 0` | `vigor/samearea/known_ori/0_model_2.531872.pt` | 2.532 m |
| VIGOR same-area 180° | `--split samearea --rotation_range 180` | `vigor/samearea/unknown_ori/18_model_2.381034.pt` | 2.381 m |
| VIGOR cross-area 0° | `--split crossarea --rotation_range 0` | `vigor/crossarea/known_ori/0_model_3.635441.pt` | 3.635 m |
| VIGOR cross-area 180° | `--split crossarea --rotation_range 180` | `vigor/crossarea/unknown_ori/0_model_4.219329.pt` | 4.219 m |

Evaluate a checkpoint (all three solvers, both depth-mask modes):

```bash
python train_kitti.py --rotation_range 10 --eval_only --resume <checkpoint>
```

## Training

Each `train_<dataset>.py` is a self-contained train+eval script sharing the
same code path (datasets/, solver/, losses/, utils/). Common arguments:

| Argument | Default | Description |
|---|---|---|
| `--rotation_range` | KITTI/RobotCar 10, CVACT/VIGOR 0 | yaw sampling range {KITTI: 10/180, others: 0/180} |
| `--split` | `samearea` | VIGOR only: `samearea` / `crossarea` |
| `--epochs` / `--batch_size` / `--lr` / `--num_workers` | 100 / per dataset | optimisation |
| `--resume PATH` | — | load a checkpoint (legacy files load as-is) |
| `--eval_only` | off | evaluation only |
| `--max_depth` | KITTI/RobotCar 40, CVACT/VIGOR 35 | metric-depth clip for training |
| `--num_samples_matches` | config.ini (512) | matches sampled per step |
| `--coarse_step_deg` / `--cheir_lambda` | 10.0 / 10.0 (pinhole), 2.5 / 1.0 (panorama) | e2e solver sweep |
| `--no_ransac` | off | skip the RANSAC metrics (faster) |
| `--with_procrustes` | off | legacy weighted-Procrustes estimator (comparison only) |
| `--eval_depth_mask` | `both` | evaluate with and without the max-depth validity filter (`_nomask` suffix) |
| `--data_root` | config.ini | dataset root override |
| `--ckpt_dir` / `--log_dir` | `checkpoints/...` / `logs/` | output locations |
| `--max_steps` | 0 | cap batches per split (smoke testing) |
| `--seed` | per dataset | RNG seed |

```bash
# KITTI, cross-day test protocol (test1 + test2 evaluated every epoch)
python train_kitti.py --rotation_range 10

# VIGOR cross-area, unknown orientation
python train_vigor.py --split crossarea --rotation_range 180
```

Per epoch the scripts evaluate every split and store
`epoch_<e>_step_<loc_e2e_mean>_model.pt` (+ `_optimizer.pt`).

**Logged metrics.** Training reports all three solvers per step:
`loc_e2e`/`yaw_e2e`, `loc_bisection`/`yaw_bisection`, `loc_ransac`/`yaw_ransac`,
plus `loss_bev_cross`, `infonce`, and the monitor-only `loss_cross_gt`
(no gradient). The training objective is `loss_bev_cross + 0.1·infonce`
(KITTI/RobotCar) or `infonce + loss_bev_cross` (CVACT/VIGOR). Evaluation
reports, per depth-mask mode, `loc_*` / `lateral_*` / `long_*` (m, with R@1m /
R@5m) and `yaw_*` (deg, with R@1° / R@5°) for each solver; metrics of the
unfiltered mode carry a `_nomask` suffix.

## config.ini

Per-dataset sections hold `dataset_root` plus the geometry constants the
legacy protocols use (`grid_size_h`, `shift_range_lat/lon`, image sizes,
`random_orientation`); `[Model] sat_bev_res`, `[Matching]
num_samples_matches`, `[Loss] num_virtual_point` mirror the training
defaults. Every value can be overridden per run through CLI flags.

## Visualization

`visualization/` reproduces the four paper figures from **one model
inference** per KITTI sample (dataloader → DINOv3 features →
CrossViewMatcher → match sampling → two-stage solver). Everything is drawn
in the network's own satellite-input frame (the 656×656 input, car heading
pointing screen-right), so no external map tiles or metadata are needed —
only the dataset root, a split file and a checkpoint:

| Figure | Content |
|---|---|
| `match_gt5.png` | street \| satellite \| zoom; 5 depth-stratified GT street points matched to the model's argmax satellite cells; GT/predicted pose arrows and the error box |
| `match_model.pdf/png` | 10 model-sampled pairs on the satellite input (top) + street image (bottom), with the GT/predicted camera markers |
| `solver.png/.pdf` (+ `_a-d.png` panels) | solver 4-panel figure on the satellite-input background: (a) matched points, (b) coarse scan → soft argmin, (c) fine scan → output, (d) GT vs prediction |
| `solver_white.png/.pdf` | same figure without the satellite background |

```bash
cd GEO_2DBA
python -m visualization.run_figures --ckpt <matcher.pt> --num 3
python -m visualization.run_figures --ckpt <matcher.pt> --names <sample_name>
```

`--data-root` (default from `config.ini`), `--split` (default
`splits/kitti/test1_files.txt`), `--out-root` (default `figures/`,
resumable) and `--gpu` are overridable. `--ckpt` defaults to the bundled
best KITTI checkpoint
(`checkpoints/kitti10/epoch_7_step_153_0.6334873647582089_model.pt`,
loc 0.633 m); the paper figures were generated with the also-bundled
`epoch_32_step_153_0.6781100631905733_model.pt`, which can be passed
explicitly to reproduce them.
Every sample folder also receives
`sat_input.png`, `grd_input.png` and `data.npz` (all matching/solver
arrays, reusable by custom plots). The two stage-visualisation constants
differ from training defaults (`cheir_lambda=1.0`, 512-point fine window)
and follow the paper figures. All figures use Times New Roman.

## Acknowledgements

This project builds on [DINOv3](https://github.com/facebookresearch/dinov3)
([RoMaV2](https://github.com/Parskatt/RoMaV2) weights),
[Depth-Anything-V2](https://github.com/DepthAnything/Depth-Anything-V2),
[UniK3D](https://github.com/lpiccinelli-eth/UniK3D) and
[Mask2Former](https://huggingface.co/facebook/mask2former-swin-large-cityscapes-semantic).

## License

Released under the [AGPL-3.0 license](LICENSE).
