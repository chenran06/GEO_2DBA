"""Depth-Anything-V2 metric depth for the KITTI ground images.

Walks ``<root>/raw_data/<day>/<drive>/image_02/data/*.png`` and writes
``<root>/depth_anythingv2_depth/<day>/<drive>/depth/*.png`` (16-bit PNG,
``round(depth * 256)`` after clipping to ``--max-depth``) — the layout
consumed by ``datasets/kitti.py``.  Existing outputs are skipped, so the
script is resumable.

Usage:
    python preprocess/infer_depth_kitti.py --limit 10        # smoke
    python preprocess/infer_depth_kitti.py                   # full run
"""

import argparse
import os
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import DAV2_CHECKPOINT, PROJECT_ROOT, dataset_root_from_config, save_depth_png, visualize_pair

MAX_DEPTH_DEFAULT = 80.0


def collect_raw_images(root):
    """All ground images under ``<root>/raw_data`` in deterministic order."""
    raw_dir = Path(root) / 'raw_data'
    images = []
    for day in sorted(p for p in raw_dir.iterdir() if p.is_dir()):
        for drive in sorted(p for p in day.iterdir() if p.is_dir()):
            img_dir = drive / 'image_02' / 'data'
            if img_dir.is_dir():
                images.extend(sorted(img_dir.glob('*.png')))
    return images


def main():
    parser = argparse.ArgumentParser(description='KITTI metric depth (Depth Anything V2)')
    parser.add_argument('--root', default=None, help='KITTI root (default: config.ini [KITTI])')
    parser.add_argument('--external-root', default=str(PROJECT_ROOT / 'external'),
                        help='directory holding the cloned depthanythingv2 repo')
    parser.add_argument('--encoder', default='vitl', choices=['vits', 'vitb', 'vitl', 'vitg'])
    parser.add_argument('--load-from', default=None,
                        help='checkpoint path (default: external/%s)' % DAV2_CHECKPOINT)
    parser.add_argument('--max-depth', type=float, default=MAX_DEPTH_DEFAULT)
    parser.add_argument('--input-size', type=int, default=518)
    parser.add_argument('--gpu', default='0', help='CUDA_VISIBLE_DEVICES')
    parser.add_argument('--limit', type=int, default=0, help='0 = all; debug only')
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

    root = Path(args.root or dataset_root_from_config('KITTI'))
    out_root = root / 'depth_anythingv2_depth'
    load_from = args.load_from or str(PROJECT_ROOT / 'external' / DAV2_CHECKPOINT)
    if not os.path.isfile(load_from):
        sys.exit('checkpoint not found: %s\n'
                 'download: huggingface-cli download depth-anything/'
                 'Depth-Anything-V2-Metric-VKITTI-Large '
                 'depth_anything_v2_metric_vkitti_vitl.pth --local-dir %s'
                 % (load_from, os.path.dirname(load_from)))

    images = collect_raw_images(root)
    if args.limit:
        images = images[:max(0, args.limit)]
    print('images: %d' % len(images))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = build_depth_anything(args.encoder, args.max_depth, load_from, device)
    print('model ready on %s: %s' % (device, os.path.basename(load_from)))

    n_ok = n_skip = n_bad = 0
    for k, img_path in enumerate(images):
        # raw_data/<day>/<drive>/image_02/data/<name>.png
        #   -> depth_anythingv2_depth/<day>/<drive>/depth/<name>.png
        # (matches the replace chain used by datasets/kitti.py)
        rel = img_path.relative_to(root / 'raw_data')
        out_path = out_root / rel.parents[2] / 'depth' / img_path.name
        if out_path.is_file():
            n_skip += 1
            continue
        raw_image = cv2.imread(str(img_path))
        if raw_image is None:
            print('Unreadable image: %s' % img_path)
            n_bad += 1
            continue
        with torch.inference_mode():
            depth = model.infer_image(raw_image, args.input_size)
        depth = np.clip(depth.astype(np.float32), 0.0, args.max_depth)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        save_depth_png(depth, out_path)
        if args.vis > 0 and n_ok < args.vis:
            vis_dir = root / 'depth_vis'
            vis_dir.mkdir(exist_ok=True)
            visualize_pair(raw_image, depth, vis_dir / (img_path.stem + '.jpg'))
        n_ok += 1
        if (k + 1) % 200 == 0:
            print('progress %d/%d (ok %d, skip %d)' % (k + 1, len(images), n_ok, n_skip))

    print('done: %d inferred, %d skipped, %d failed -> %s' % (n_ok, n_skip, n_bad, out_root))


if __name__ == '__main__':
    main()
