"""Mask2Former semantic masks for the RobotCar sampled frames.

Runs ``facebook/mask2former-swin-large-cityscapes-semantic`` on the
``img_path`` column of ``<root>/sampled_frames.csv`` and writes full
Cityscapes 19-class semantic maps (0-18) to ``<root>/mask2former/<ts>.png``
(flat directory) — consumed by ``datasets/robotcar.py`` via
``np.isin(mask, DYNAMIC_CLASSES)``.

The gravity-leveling black border (rotation remap overflow, reconstructed
geometrically from ``robotcar_leveling``) is set to 255 (void/ignore);
``datasets/robotcar.py`` treats 255 as dynamic, so border pixels are excluded
from the ground validity mask.  Existing outputs are skipped (resumable).

Usage:
    python preprocess/extract_mask_robotcar.py --limit 10    # smoke
    python preprocess/extract_mask_robotcar.py               # full run
"""

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import PROJECT_ROOT, dataset_root_from_config
from robotcar_leveling import black_border_mask, load_leveling_ctx, pixel_grid


def read_image_paths(csv_path):
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    return [r['img_path'] for r in rows]


class GrdMaskDataset(Dataset):
    """Load original images (PIL); the processor resizes internally."""

    def __init__(self, grd_paths):
        self.grd_paths = grd_paths

    def __len__(self):
        return len(self.grd_paths)

    def __getitem__(self, idx):
        grd_path = self.grd_paths[idx]
        with Image.open(grd_path) as img:
            img = img.convert('RGB')
        return img, str(grd_path)


def main():
    parser = argparse.ArgumentParser(description='RobotCar Mask2Former semantic masks')
    parser.add_argument('--root', default=None, help='RobotCar root (default: config.ini [RobotCar])')
    parser.add_argument('--csv', default=None, help='default: <root>/sampled_frames.csv')
    parser.add_argument('--outdir', default=None, help='default: <root>/mask2former')
    parser.add_argument('--model', default='facebook/mask2former-swin-large-cityscapes-semantic')
    parser.add_argument('--gpu', default='0', help='CUDA_VISIBLE_DEVICES')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='reduce if OOM (Mask2Former Swin-L is memory-heavy)')
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--limit', type=int, default=0, help='0 = all; debug only')
    args = parser.parse_args()

    os.environ.setdefault('CUDA_VISIBLE_DEVICES', args.gpu)

    root = Path(args.root or dataset_root_from_config('RobotCar'))
    csv_path = args.csv or (root / 'sampled_frames.csv')
    outdir = Path(args.outdir or (root / 'mask2former'))
    outdir.mkdir(parents=True, exist_ok=True)

    paths = read_image_paths(csv_path)
    if args.limit:
        paths = paths[:max(0, args.limit)]
    paths = [p for p in paths
             if not (outdir / (Path(p).stem + '.png')).is_file()]
    print('to process: %d (rest skipped as already present)' % len(paths))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device: %s' % device)

    from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
    processor = AutoImageProcessor.from_pretrained(args.model)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(args.model).to(device)
    model.eval()

    grid = pixel_grid()
    ctx = {}  # run_dir -> (K, K_inv, R_sc, {filename: (roll, pitch)})

    def collate_fn(batch):
        imgs, paths_ = zip(*batch)
        return list(imgs), list(paths_)

    dataloader = DataLoader(GrdMaskDataset(paths), batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers,
                            pin_memory=True, collate_fn=collate_fn)

    count = 0
    with torch.no_grad():
        for batch_imgs, batch_paths in tqdm(dataloader, desc='Mask2Former'):
            inputs = processor(images=batch_imgs, return_tensors='pt')
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs)

            # semantic segmentation restored to the original resolution (0-18)
            target_sizes = [img.size[::-1] for img in batch_imgs]
            seg_maps = processor.post_process_semantic_segmentation(
                outputs, target_sizes=target_sizes)

            for seg_map, path_str in zip(seg_maps, batch_paths):
                stem = Path(path_str).stem
                seg = seg_map.to(dtype=torch.uint8).cpu().numpy()

                run_dir = path_str.split('/stereo_preprocess_v2/')[0]
                if run_dir not in ctx:
                    K, R_sc, rp = load_leveling_ctx(run_dir)
                    ctx[run_dir] = (K, np.linalg.inv(K), R_sc, rp)
                K, K_inv, R_sc, rp = ctx[run_dir]
                mask = black_border_mask(*rp[stem + '.png'], K, R_sc, K_inv, grid)
                seg[mask] = 255

                Image.fromarray(seg).save(outdir / (stem + '.png'))
                count += 1

    print('Done. Saved %d masks to %s' % (count, outdir))


if __name__ == '__main__':
    main()
