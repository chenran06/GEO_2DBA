# Pre-processing: metric depth + dynamic-object masks

The training/eval dataloaders consume **pre-computed** metric depth maps and
(for KITTI/RobotCar) dynamic-object masks.  This directory contains the
inference scripts that produce them.  The third-party models themselves are
**not vendored** — clone the two external repositories and download the
checkpoints as described below.

| Script | Dataset | Model | Output |
|---|---|---|---|
| `infer_depth_kitti.py` | KITTI | Depth-Anything-V2 (metric) | `<root>/depth_anythingv2_depth/...` |
| `infer_depth_robotcar.py` | RobotCar | Depth-Anything-V2 (metric) | `<root>/depth/<ts>.png` |
| `infer_depth_panorama.py` | VIGOR / CVACT | UniK3D | `<output>/<name>.png` (mm) |
| `extract_mask_kitti.py` | KITTI | Mask2Former (Cityscapes) | `<root>/mask2former/...` |
| `extract_mask_robotcar.py` | RobotCar | Mask2Former (Cityscapes) | `<root>/mask2former/<ts>.png` |

All scripts are **resumable** (existing outputs are skipped), accept
`--limit N` for a smoke run, and default their `--root` from `config.ini`.

## 1. Install external models

```bash
cd GEO_2DBA
mkdir -p external/checkpoints && cd external

# Depth-Anything-V2 (metric depth for KITTI / RobotCar pinhole images)
git clone https://github.com/DepthAnything/Depth-Anything-V2.git depthanythingv2

# UniK3D (metric depth for VIGOR / CVACT panoramas)
git clone https://github.com/lpiccinelli-eth/UniK3D.git unik3d
pip install -r unik3d/requirements.txt   # einops etc. if not already present

# Metric checkpoint for Depth-Anything-V2
huggingface-cli download depth-anything/Depth-Anything-V2-Metric-VKITTI-Large \
    depth_anything_v2_metric_vkitti_vitl.pth --local-dir checkpoints
```

Extra pip dependencies beyond the project `requirements.txt`:

```bash
pip install transformers timm einops huggingface_hub
```

Mask2Former downloads its weights automatically from HuggingFace
(`facebook/mask2former-swin-large-cityscapes-semantic`) on first use.

If you cloned the external repos elsewhere, pass
`--external-root /path/to/external` (the directory that directly contains
`depthanythingv2/` and `unik3d/`), and `--load-from` for a checkpoint outside
`external/checkpoints/`.

## 2. Depth maps

```bash
# KITTI: raw_data/<day>/<drive>/image_02/data/*.png
#    ->  depth_anythingv2_depth/<day>/<drive>/depth/*.png (1/256 m, uint16)
python preprocess/infer_depth_kitti.py

# RobotCar: frames listed in <root>/sampled_frames.csv
#    ->  <root>/depth/<ts>.png (1/256 m, uint16)
python preprocess/infer_depth_robotcar.py

# VIGOR: <VIGOR>/<City>/panorama  ->  <VIGOR>/<City>/unik3d_depth (mm, uint16)
for city in NewYork Seattle SanFrancisco Chicago; do
    python preprocess/infer_depth_panorama.py \
        --input /data/VIGOR/$city/panorama --output /data/VIGOR/$city/unik3d_depth
done

# CVACT: <root>/ANU_data_small/streetview -> <root>/ANU_data_small/unik3d_depth
python preprocess/infer_depth_panorama.py \
    --input /data/CVACT/ANU_data_small/streetview \
    --output /data/CVACT/ANU_data_small/unik3d_depth
```

Encoding conventions (match the dataloaders in `datasets/`):

- Depth-Anything-V2 outputs: `uint16 PNG = round(depth_m * 256)`, read back
  with `IMREAD_UNCHANGED` then `/ 256.0`.
- UniK3D outputs: `uint16 PNG = round(depth_m * 1000)` (clipped to 65 m),
  read back then `/ 1000.0`.

RobotCar depth additionally sets the gravity-leveling **black border** to the
max depth (invalid).  The border is reconstructed geometrically from the
per-run intrinsics and the roll/pitch log of the leveling stage
(`robotcar_leveling.py`) — no re-detection involved.

## 3. Dynamic-object masks (KITTI / RobotCar)

Mask2Former (Swin-L, Cityscapes) provides the per-pixel semantic labels used
to build the ground validity mask `mask_grd` (dynamic objects + sky are
excluded from matching).

```bash
# KITTI: mirrors raw_data layout; binary is the training convention
#    binary   (default): 255 = dynamic/sky, else 0 -> <root>/mask2former/...
#    semantic:            raw Cityscapes ids 0-18  -> <root>/mask2former_o/...
python preprocess/extract_mask_kitti.py

# RobotCar: full semantic maps + leveling border marked 255 (void)
#    ->  <root>/mask2former/<ts>.png
python preprocess/extract_mask_robotcar.py
```

Dynamic/sky classes in binary mode (Cityscapes ids): 10 sky, 11 person,
12 rider, 13 car, 14 truck, 15 bus, 16 train, 17 motorcycle, 18 bicycle
(vegetation is not excluded for KITTI; the RobotCar dataloader additionally
drops vegetation via its own class list at load time).

## 4. Prerequisites per dataset

| Dataset | Needs before this pipeline |
|---|---|
| KITTI | raw images under `<root>/raw_data/<day>/<drive>/image_02/data/` |
| VIGOR | panorama images under `<VIGOR>/<City>/panorama/` |
| CVACT | streetview images under `<root>/ANU_data_small/streetview/` |
| RobotCar | the sampled-frame stage output: `<root>/sampled_frames.csv` pointing at the levelled images `<run>/stereo_preprocess_v2/...`, plus the per-run `<run>/stereo_preprocess_v2/stereo_narrow_left.txt` and `process_log.csv` (used to reconstruct the black border) |
