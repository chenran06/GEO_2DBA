"""Depth-Anything-V2 metric depth for the RobotCar sampled frames.

Reads the ``img_path`` column of ``<root>/sampled_frames.csv`` (produced by
the frame-sampling stage), runs metric depth per frame and writes
``<root>/depth/<timestamp>.png`` — a flat directory, 16-bit PNG
(``round(depth * 256)``, same read-back convention as ``datasets/robotcar.py``).

The gravity-leveling black border (rotation remap overflow, reconstructed
geometrically from ``robotcar_leveling``) is set to ``--max-depth``
(farthest/invalid) before clipping; genuine black pixels in the scene are
unaffected.  Existing outputs are skipped (resumable).

Usage:
    python preprocess/infer_depth_robotcar.py --limit 10     # smoke
    python preprocess/infer_depth_robotcar.py                # full run
"""

import argparse
import csv
import os
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import DAV2_CHECKPOINT, PROJECT_ROOT, dataset_root_from_config, save_depth_png, visualize_pair
from robotcar_leveling import black_border_mask, load_leveling_ctx, pixel_grid


def read_image_paths(csv_path):
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    return [r['img_path'] for r in rows]


def main():
    parser = argparse.ArgumentParser(description='RobotCar metric depth (Depth Anything V2)')
    parser.add_argument('--root', default=None, help='RobotCar root (default: config.ini [RobotCar])')
    parser.add_argument('--csv', default=None, help='default: <root>/sampled_frames.csv')
    parser.add_argument('--outdir', default=None, help='default: <root>/depth')
    parser.add_argument('--external-root', default=str(PROJECT_ROOT / 'external'),
                        help='directory holding the cloned depthanythingv2 repo')
    parser.add_argument('--encoder', default='vitl', choices=['vits', 'vitb', 'vitl', 'vitg'])
    parser.add_argument('--load-from', default=None,
                        help='checkpoint path (default: external/%s)' % DAV2_CHECKPOINT)
    parser.add_argument('--max-depth', type=float, default=80.0)
    parser.add_argument('--input-size', type=int, default=518)
    parser.add_argument('--gpu', default='0', help='CUDA_VISIBLE_DEVICES')
    parser.add_argument('--limit', type=int, default=0, help='0 = all; debug only')
    parser.add_argument('--save-numpy', action='store_true', help='also store float32 .npy')
    parser.add_argument('--vis', type=int, default=0, help='N = visualise the first N frames')
    args = parser.parse_args()

    os.environ.setdefault('CUDA_VISIBLE_DEVICES', args.gpu)

    import numpy as np
    import torch

    external_root = Path(args.external_root)
    if not (external_root / 'depthanythingv2').is_dir():
        sys.exit('depthanythingv2 repo not found under %s\n'
                 'clone: git clone https://github.com/DepthAnything/Depth-Anything-V2.git '
                 '%s/depthanythingv2  (see preprocess/README.md)'
                 % (external_root, external_root))
    sys.path.insert(0, str(external_root))

    from depth_anything import build_depth_anything

    root = Path(args.root or dataset_root_from_config('RobotCar'))
    csv_path = args.csv or (root / 'sampled_frames.csv')
    outdir = Path(args.outdir or (root / 'depth'))
    visdir = root / 'depth_vis'
    outdir.mkdir(parents=True, exist_ok=True)
    load_from = args.load_from or str(PROJECT_ROOT / 'external' / DAV2_CHECKPOINT)
    if not os.path.isfile(load_from):
        sys.exit('checkpoint not found: %s\n'
                 'download: huggingface-cli download depth-anything/'
                 'Depth-Anything-V2-Metric-VKITTI-Large '
                 'depth_anything_v2_metric_vkitti_vitl.pth --local-dir %s'
                 % (load_from, os.path.dirname(load_from)))

    paths = read_image_paths(csv_path)
    if args.limit:
        paths = paths[:max(0, args.limit)]
    print('csv rows: %d' % len(paths))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = build_depth_anything(args.encoder, args.max_depth, load_from, device)
    print('model ready on %s: %s' % (device, os.path.basename(load_from)))

    grid = pixel_grid()
    ctx = {}  # run_dir -> (K, K_inv, R_sc, {filename: (roll, pitch)})
    n_ok = n_skip = n_bad = n_black = 0
    for path in paths:
        stem = Path(path).stem
        out_path = outdir / (stem + '.png')
        if out_path.is_file():
            n_skip += 1
            continue
        raw_image = cv2.imread(path)
        if raw_image is None:
            print('Unreadable image: %s' % path)
            n_bad += 1
            continue
        run_dir = path.split('/stereo_preprocess_v2/')[0]
        if run_dir not in ctx:
            K, R_sc, rp = load_leveling_ctx(run_dir)
            ctx[run_dir] = (K, np.linalg.inv(K), R_sc, rp)
        K, K_inv, R_sc, rp = ctx[run_dir]
        with torch.inference_mode():
            depth = model.infer_image(raw_image, args.input_size)
        depth = depth.astype(np.float32)
        mask = black_border_mask(*rp[stem + '.png'], K, R_sc, K_inv, grid)
        depth[mask] = args.max_depth
        n_black += int(mask.sum())
        depth = np.clip(depth, 0.0, args.max_depth)
        save_depth_png(depth, out_path)
        if args.save_numpy:
            np.save(outdir / (stem + '.npy'), depth)
        if args.vis > 0 and n_ok < args.vis:
            visdir.mkdir(exist_ok=True)
            visualize_pair(raw_image, depth, visdir / (stem + '.jpg'))
        n_ok += 1

    print('done: %d inferred, %d skipped, %d failed, mean black-border px %d/frame -> %s'
          % (n_ok, n_skip, n_bad, n_black // max(n_ok, 1), outdir))


if __name__ == '__main__':
    main()
