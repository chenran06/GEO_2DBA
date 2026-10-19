"""UniK3D metric depth for panorama ground images (VIGOR and CVACT).

Takes a directory (or a single file) of panoramas and writes 16-bit PNG
depth maps in millimetres (``round(||points|| * 1000)`` after clipping to
65 m) to ``--output``.  The camera is the equirectangular Spherical model
from ``docs/equirectangular.json``; per-image width/height in the camera
params are updated to the actual file resolution.  This is the format
consumed by ``datasets/vigor.py`` / ``datasets/cvact.py`` (read back with
``/ 1000.0``).  Existing outputs are skipped (resumable).

Usage (VIGOR — panorama -> unik3d_depth sibling directory):
    python preprocess/infer_depth_panorama.py \
        --input /data/VIGOR/NewYork/panorama --output /data/VIGOR/NewYork/unik3d_depth
Usage (CVACT):
    python preprocess/infer_depth_panorama.py \
        --input /data/CVACT/ANU_data_small/streetview \
        --output /data/CVACT/ANU_data_small/unik3d_depth
"""

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import PROJECT_ROOT

IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
CAMERA_CLASSES = ('Fisheye624', 'Spherical', 'OPENCV', 'Pinhole', 'MEI')
MAX_DEPTH = 65.0  # metres; clipped before the mm quantisation


def save_outputs(rgb_torch, outputs, name, base_path, save_ply):
    points_torch = outputs['points']  # [B, 3, H, W]
    points_np = points_torch.detach().cpu().numpy()
    depth = np.linalg.norm(points_np[0], axis=0)

    depth_map = (np.clip(depth, 0.0, MAX_DEPTH) * 1000.0).astype(np.uint16)
    save_path = os.path.join(base_path, name + '.png')
    for attempt in range(3):
        try:
            Image.fromarray(depth_map).save(save_path)
            break
        except Exception as e:
            if attempt == 2:
                raise RuntimeError(
                    f'Failed to save depth map after 3 attempts: {save_path}') from e
            print(f'Warning: failed to save depth map on attempt {attempt + 1}/3: {e}')
            time.sleep(3)

    if save_ply:
        from unik3d.utils.visualization import save_file_ply
        valid = depth < MAX_DEPTH
        pts = points_torch[:, :, valid]
        rgb = rgb_torch[:, valid]
        save_file_ply(pts.permute(0, 2, 1).reshape(-1, 3).detach().cpu().numpy(),
                      rgb.permute(1, 0).reshape(-1, 3).detach().cpu().numpy(),
                      os.path.join(base_path, name + '.ply'))


def process_file(model, camera, rgb_path, output, save_ply):
    rgb = np.array(Image.open(rgb_path))
    rgb_torch = torch.from_numpy(rgb).permute(2, 0, 1)
    with torch.inference_mode():
        outputs = model.infer(rgb=rgb_torch, camera=camera, normalize=True, rays=None)
    name = Path(rgb_path).stem
    save_outputs(rgb_torch, outputs, name, output, save_ply)
    del rgb_torch, outputs
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return name


def main():
    parser = argparse.ArgumentParser(description='Panorama metric depth (UniK3D)')
    parser.add_argument('--input', required=True, help='input image file or directory')
    parser.add_argument('--output', required=True, help='output directory')
    parser.add_argument('--camera', default=str(PROJECT_ROOT / 'docs' / 'equirectangular.json'),
                        help='camera parameter json (Spherical equirectangular)')
    parser.add_argument('--external-root', default=str(PROJECT_ROOT / 'external'),
                        help='directory holding the cloned unik3d repo')
    parser.add_argument('--model', default='vitl',
                        help='UniK3D backbone; HF repo lpiccinelli/unik3d-<model>')
    parser.add_argument('--resolution-level', type=int, default=9,
                        choices=list(range(10)), help='resolution level in [0, 10)')
    parser.add_argument('--interpolation-mode', default='bilinear',
                        choices=['nearest', 'nearest-exact', 'bilinear'])
    parser.add_argument('--gpu', default='0', help='CUDA_VISIBLE_DEVICES')
    parser.add_argument('--save-ply', action='store_true', help='also store point clouds')
    args = parser.parse_args()

    os.environ.setdefault('CUDA_VISIBLE_DEVICES', args.gpu)

    external_root = Path(args.external_root)
    if not (external_root / 'unik3d').is_dir():
        sys.exit('unik3d repo not found under %s\n'
                 'clone: git clone https://github.com/lpiccinelli-eth/UniK3D.git %s/unik3d'
                 % (external_root, external_root))
    sys.path.insert(0, str(external_root))
    from unik3d.models import UniK3D
    from unik3d.utils.camera import MEI, OPENCV, Fisheye624, Pinhole, Spherical, BatchCamera
    cameras = {name: cls for name, cls in
               [('Fisheye624', Fisheye624), ('Spherical', Spherical), ('OPENCV', OPENCV),
                ('Pinhole', Pinhole), ('MEI', MEI), ('BatchCamera', BatchCamera)]}

    with open(args.camera) as f:
        camera_dict = json.load(f)
    camera_name = camera_dict['name']
    assert camera_name in cameras, camera_name
    params = torch.tensor(camera_dict['params'])

    print('torch version:', torch.__version__)
    model = UniK3D.from_pretrained(f'lpiccinelli/unik3d-{args.model}')
    model.resolution_level = args.resolution_level
    model.interpolation_mode = args.interpolation_mode
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device).eval()

    os.makedirs(args.output, exist_ok=True)
    input_path = Path(args.input)
    if input_path.is_file():
        # refresh the camera resolution (params indices 4/5) to the actual image
        with Image.open(input_path) as img:
            params[4], params[5] = img.size[0], img.size[1]
        name = process_file(model, cameras[camera_name](params=params),
                            str(input_path), args.output, args.save_ply)
        print(f'[1/1] Done: {name}')
    elif input_path.is_dir():
        rgb_files = sorted(p for p in input_path.iterdir()
                           if p.suffix.lower() in IMAGE_SUFFIXES)
        if not rgb_files:
            raise ValueError(f'No image files found in directory: {input_path}')
        total = len(rgb_files)
        done = 0
        for rgb_path in rgb_files:
            if os.path.exists(os.path.join(args.output, rgb_path.stem + '.png')):
                done += 1
                continue
            try:
                # camera params carry the resolution at indices 4/5; panoramas
                # of one folder can differ, so refresh per file
                with Image.open(rgb_path) as img:
                    params[4], params[5] = img.size[0], img.size[1]
                camera = cameras[camera_name](params=params)
                process_file(model, camera, str(rgb_path), args.output, args.save_ply)
            except Exception as e:
                print(f'Error occurred while processing {rgb_path}: {e}')
                continue
            done += 1
        print(f'Finished processing {done}/{total} files.')
    else:
        raise ValueError('Input path is not valid.')


if __name__ == '__main__':
    main()
