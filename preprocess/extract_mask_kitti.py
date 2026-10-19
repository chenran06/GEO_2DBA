"""Mask2Former semantic masks for the KITTI ground images.

Runs ``facebook/mask2former-swin-large-cityscapes-semantic`` on every unique
ground image referenced by the KITTI splits (``splits/kitti/train_files.txt``
+ ``test1_files.txt`` + ``test2_files.txt``) and mirrors the ``raw_data``
layout:

``--mode binary`` (default)
    255 = dynamic/sky classes, 0 = static → ``<root>/mask2former/...``.
    This is what ``datasets/kitti.py`` consumes (pixels == 255 are treated
    as invalid and excluded from the ground validity mask).
``--mode semantic``
    raw Cityscapes 19-class ids (0-18) → ``<root>/mask2former_o/...``.

Dynamic/sky classes in binary mode (Cityscapes ids):
    10 sky, 11 person, 12 rider, 13 car, 14 truck, 15 bus, 16 train,
    17 motorcycle, 18 bicycle.  This reproduces the on-disk training masks
    bitwise; vegetation (8) is NOT excluded for KITTI.

Usage:
    python preprocess/extract_mask_kitti.py --limit 8        # smoke
    python preprocess/extract_mask_kitti.py                  # full run
"""

import argparse
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

SPLITS_DIR = PROJECT_ROOT / 'splits' / 'kitti'
# Binary mode: sky + dynamic classes — the convention of the on-disk
# training masks (verified bitwise against them; vegetation is NOT excluded).
BINARY_CLASSES = [10, 11, 12, 13, 14, 15, 16, 17, 18]
CITYSCAPES_LABELS = [
    'road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
    'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky',
    'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle',
]


def collect_grd_paths(dataset_root, limit=0):
    """Unique ground image paths across all split files.

    Split lines are ``<day>/<drive>/<stem>``; the on-disk layout inserts the
    camera directory: ``raw_data/<day>/<drive>/image_02/data/<stem>``.
    """
    grd_paths = set()
    root = Path(dataset_root)
    for split_file in sorted(SPLITS_DIR.glob('*_files.txt')):
        with open(split_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                file_name = line.split(' ')[0]
                day, drive, image_no = file_name[:10], file_name[:38], file_name[38:]
                grd_paths.add(root / 'raw_data' / drive / 'image_02' / 'data' / image_no.lower())
    paths = sorted(grd_paths)
    if limit:
        paths = paths[:limit]
    return paths


class GrdMaskDataset(Dataset):
    """Load ground images at original resolution (processor resizes internally)."""

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
    parser = argparse.ArgumentParser(description='KITTI Mask2Former masks')
    parser.add_argument('--root', default=None, help='KITTI root (default: config.ini [KITTI])')
    parser.add_argument('--mode', choices=['binary', 'semantic'], default='binary',
                        help="binary: 255=dynamic/sky (dataset convention); "
                             'semantic: raw Cityscapes ids')
    parser.add_argument('--output', default=None,
                        help='default: <root>/mask2former (binary) or <root>/mask2former_o (semantic)')
    parser.add_argument('--model', default='facebook/mask2former-swin-large-cityscapes-semantic')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='reduce if OOM (Mask2Former Swin-L is memory-heavy)')
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--limit', type=int, default=0, help='0 = all; debug only')
    args = parser.parse_args()

    device = torch.device(args.device)
    root = Path(args.root or dataset_root_from_config('KITTI'))
    output_root = Path(args.output) if args.output else root / (
        'mask2former' if args.mode == 'binary' else 'mask2former_o')
    print(f'Device: {device}  root: {root}  output: {output_root}  mode: {args.mode}')

    print('Scanning ground image paths...')
    grd_paths = collect_grd_paths(root, args.limit)
    print(f'  {len(grd_paths)} unique ground images')

    from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
    processor = AutoImageProcessor.from_pretrained(args.model)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(args.model).to(device)
    model.eval()

    def collate_fn(batch):
        imgs, paths = zip(*batch)
        return list(imgs), list(paths)

    dataloader = DataLoader(GrdMaskDataset(grd_paths), batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers,
                            pin_memory=True, collate_fn=collate_fn)

    dynamic_set = np.array(BINARY_CLASSES)
    count = 0
    with torch.no_grad():
        for batch_imgs, batch_paths in tqdm(dataloader, desc='Mask2Former'):
            inputs = processor(images=batch_imgs, return_tensors='pt')
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs)

            # per-pixel class labels at the original image resolution
            target_sizes = [img.size[::-1] for img in batch_imgs]
            seg_maps = processor.post_process_semantic_segmentation(
                outputs, target_sizes=target_sizes)

            for seg_map, path_str in zip(seg_maps, batch_paths):
                seg = seg_map.to(dtype=torch.uint8).cpu().numpy()
                if args.mode == 'binary':
                    seg = np.where(np.isin(seg, dynamic_set), 255, 0).astype(np.uint8)
                out_path = Path(path_str).relative_to(root / 'raw_data')
                out_path = output_root / out_path
                out_path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(seg).save(out_path)
                count += 1

    print(f'Done. Saved {count} masks to {output_root}')


if __name__ == '__main__':
    main()
