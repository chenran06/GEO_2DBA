"""One model inference per sample -> all four paper figures.

For every selected KITTI sample this script runs the full pipeline once
(dataloader -> DINOv3 features -> CrossViewMatcher -> match sampling ->
two-stage solver) and draws, from the same inference:

  match_gt5.png          street | satellite | zoom, model-argmax matches
  match_model.pdf/png    10 sampled pairs on the satellite input
  solver.png/.pdf + solver_{a-d}.png   solver 4-panel (satellite background)
  solver_white.png/.pdf + solver_white_{a-d}.png  white-background variant
  data.npz               all solver/matching arrays

Everything is expressed in the network's own satellite-input frame, so no
external map tiles or metadata files are needed — only the dataset root,
the split file and a matcher checkpoint.

Usage (from GEO_2DBA/):
    python -m visualization.run_figures --ckpt <matcher.pt> --num 3
    python -m visualization.run_figures --ckpt <matcher.pt> \
        --names 2011_10_03_2011_10_03_drive_0042_0000000939
"""
import argparse
import configparser
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

GEO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(GEO_ROOT))

from datasets.kitti import SatGrdDataset
from models.matcher import CrossViewMatcher
from utils.backbone import img_infer
from utils.grids import create_image_grids, create_metric_grid
from utils.matching import build_match_batch

from visualization import vis_common
from visualization.match_figure import (argmax_sat_px, draw_match_gt5,
                                        draw_match_model, select_points)
from visualization.solver_figure import (InputMapper, circ_deg, draw_solver,
                                         solve_stages)
from visualization.vis_common import GRD_H, GRD_W, SAT_GRID, kitti_root, \
    tensor2img

GRD_FEAT_H, GRD_FEAT_W = GRD_H // 16, GRD_W // 16      # 26, 88
MAX_DEPTH = 40.0
N_PAIRS10 = 10


def parse_args():
    p = argparse.ArgumentParser(description='Paper figures (one inference)')
    p.add_argument('--data-root', default=kitti_root(),
                   help='KITTI dataset root (default: config.ini)')
    p.add_argument('--ckpt',
                   default=str(GEO_ROOT / 'checkpoints' / 'kitti10'
                               / 'model.pt'),
                   help='matcher checkpoint (.pt)')
    p.add_argument('--split',
                   default=str(GEO_ROOT / 'splits' / 'kitti'
                               / 'test1_files.txt'),
                   help='split file with GT shift columns (test format)')
    p.add_argument('--names', default='', help='comma-separated sample names')
    p.add_argument('--num', type=int, default=0,
                   help='process the first N split samples (0 = all)')
    p.add_argument('--out-root', default=str(GEO_ROOT / 'figures'),
                   help='output directory (one subfolder per sample)')
    p.add_argument('--gpu', default='0', help='CUDA_VISIBLE_DEVICES')
    return p.parse_args()


def folder_name(file_name):
    """Split path -> sample directory name ('<day>_<drive>_<frame>')."""
    return file_name.replace('/', '_').replace('_sync', '') \
                    .replace('.png', '')


def main():
    args = parse_args()
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', args.gpu)

    config = configparser.ConfigParser()
    config.read(GEO_ROOT / 'config.ini')
    grid_size_h = float(config['KITTI']['grid_size_h'])
    sat_bev_res = int(config['Model']['sat_bev_res'])
    num_matches = int(config['Matching']['num_samples_matches'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = SatGrdDataset(
        root=args.data_root,
        file=args.split,
        shift_range_lat=float(config['KITTI']['shift_range_lat']),
        shift_range_lon=float(config['KITTI']['shift_range_lon']),
        rotation_range=0,
        train=False,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    model = CrossViewMatcher(device, sat_bev_res=sat_bev_res,
                             temperature=0.1, embed_dim=1024).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()

    metric_coord_sat = create_metric_grid(grid_size_h, sat_bev_res, 1, device)
    u_grid, v_grid = create_image_grids(1, device, GRD_H, GRD_W)

    want = [s for s in args.names.split(',') if s]
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    done = 0
    for data in loader:
        (sat, grd, depth, camera_k, tgt, rotation_gt, mask_grd, wh, yaw,
         paths, mpp) = data
        name = folder_name(paths[2][0])
        if want and name not in want:
            continue
        if args.num and done >= args.num:
            break

        out_dir = out_root / name
        if (out_dir / 'solver_white.png').exists():
            print(f'[{name}] exists, skip', flush=True)
            done += 1
            continue
        out_dir.mkdir(parents=True, exist_ok=True)

        # ── inference ──
        sat, grd = sat.to(device), grd.to(device)
        depth = torch.clamp(depth.to(device), 0, MAX_DEPTH)
        camera_k = camera_k.to(device)
        tgt = tgt.to(device).float()
        mask_grd = mask_grd.to(device)

        sat_size = sat.shape[-1]
        tgt_metric = (tgt / sat_size) * grid_size_h

        with torch.no_grad():
            grd_feature = img_infer(grd, H=GRD_FEAT_H, W=GRD_FEAT_W)
            sat_feature = img_infer(sat, H=SAT_GRID, W=SAT_GRID)
            (_, _, _, _, grd_indices_sampled, sat_points, _, weights,
             _, matching_score, _, _) = build_match_batch(
                model, depth, camera_k, grd_feature, sat_feature,
                metric_coord_sat, u_grid, v_grid, MAX_DEPTH, num_matches,
                mask_grd=mask_grd, training=True)

        # ── two-stage solve (figure constants) ──
        grd_uv = torch.stack([grd_indices_sampled % GRD_FEAT_W,
                              grd_indices_sampled // GRD_FEAT_W],
                             dim=-1).float()
        camera_k_scaled = camera_k.clone()
        camera_k_scaled[:, 0, :] *= (GRD_FEAT_W / GRD_W)
        camera_k_scaled[:, 1, :] *= (GRD_FEAT_H / GRD_H)

        psi_gt_deg = float(np.degrees(-float(yaw.reshape(-1)[0])))
        C_gt = tgt_metric.cpu().reshape(-1, 2)[0].numpy()
        sat_np = sat_points[0].detach().float().cpu().numpy()

        st = solve_stages(sat_points.to(device).float(),
                          grd_uv.to(device),
                          camera_k_scaled.to(device),
                          weights.to(device).float().detach())
        C_pred = st['C_pred']
        psi_pred = st['psi_pred']
        c_err = float(np.linalg.norm(C_pred - C_gt))
        y_err = circ_deg(np.degrees(psi_pred), 0.0)

        # ── shared inference artifacts ──
        sat_input = tensor2img(sat[0])
        grd_input = tensor2img(grd[0])
        sat_input.save(out_dir / 'sat_input.png')
        grd_input.save(out_dir / 'grd_input.png')

        score_t = matching_score[0].detach().float().cpu().reshape(-1)
        n_grd = GRD_FEAT_W * GRD_FEAT_H
        torch.manual_seed(1234)
        k10 = torch.multinomial(score_t, N_PAIRS10, replacement=False)
        pairs10 = []
        for kk in k10.tolist():
            s_idx, g_idx = divmod(kk, n_grd)
            u, v = g_idx % GRD_FEAT_W, g_idx // GRD_FEAT_W
            i, j = divmod(s_idx, SAT_GRID)
            pairs10.append(((u * 16 + 8, v * 16 + 8), (j * 16 + 8, i * 16 + 8)))

        np.savez(out_dir / 'data.npz',
                 sat_points=sat_np,
                 grd_uv=grd_uv.cpu().numpy(),
                 camera_k_scaled=camera_k_scaled[0].cpu().numpy(),
                 weights=weights[0].detach().float().cpu().numpy(),
                 psi_gt_deg=psi_gt_deg,
                 C_gt=C_gt,
                 C_pred=np.asarray(C_pred),
                 c_err=c_err, y_err=y_err,
                 pairs10=np.array(pairs10),
                 matching_score=matching_score[0].detach().float().cpu()
                 .numpy(),
                 score_shape=np.array([SAT_GRID ** 2, n_grd]))

        # ── figure 1: match_gt5 ──
        depth_np = depth[0, 0].cpu().numpy()
        sel = select_points(depth_np)
        sat_px = argmax_sat_px(matching_score, sel, sat_grid=SAT_GRID)
        draw_match_gt5(grd_input, sat_input, sel, sat_px, C_gt, C_pred,
                       psi_pred, c_err, y_err, str(out_dir / 'match_gt5.png'),
                       psi_gt=0.0)

        # ── figure 2: match_model (satellite input background) ──
        draw_match_model(grd_input, sat_input, pairs10, C_gt, C_pred,
                         psi_pred, str(out_dir / 'match_model'),
                         grid_m=grid_size_h)

        # ── figures 3+4: solver 4-panel (satellite input background) ──
        mapper = InputMapper(grid_size_h)
        draw_solver(mapper, st, sat_np, C_gt, c_err, y_err,
                    camera_k_scaled[0].cpu().numpy(), sat_input, str(out_dir),
                    grid_m=grid_size_h)
        draw_solver(mapper, st, sat_np, C_gt, c_err, y_err,
                    camera_k_scaled[0].cpu().numpy(), sat_input, str(out_dir),
                    grid_m=grid_size_h, bg_white=True)

        print(f'[{name}] C {c_err:.3f} m, yaw {y_err:.3f} deg', flush=True)
        done += 1

    print(f'finished {done} samples -> {out_root}')


if __name__ == '__main__':
    main()
