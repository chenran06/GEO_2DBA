"""KITTI training + evaluation (pinhole camera pipeline).

One code path serves both modes: every step back-projects the ground depth
to a BEV point cloud, matches it against the satellite BEV, and derives the
pose with three solvers whose metrics are always reported (metric keys carry
the solver name, e.g. ``loc_e2e_mean`` / ``yaw_ransac_R@1deg``):

1. procrustes - weighted similarity transform on the sampled matches
                (legacy baseline; disabled by default, ``--with_procrustes``
                to enable).
2. e2e        - differentiable cross_unnorm solve; the only solver entering
                the loss (``loss_bev_cross`` + ``infonce * 0.1``) and the
                only one running with gradients.
3. bisection  - non-differentiable coarse argmin + bisection.
4. ransac     - probabilistic solver (``--no_ransac`` to disable).

Metric BEV grids are built per sample from each frame's oxts latitude
(``grid_size_h = mpp(lat) * 656``), so heterogeneous tiles carry their own
physical scale. Evaluation additionally reports lateral/longitudinal/recall
metrics per solver and, with ``--eval_depth_mask both`` (default), one metric
set with the max-depth validity filter and one without (``_nomask`` suffix).

Examples:
    # evaluate a legacy checkpoint on 100 test1 samples
    python train_kitti.py --eval_only --rotation_range 10 \
        --resume ../checkpoints/kitti10/epoch_7_step_153_0.6334873647582089_model.pt \
        --max_eval_samples 100

    # full training (evaluates test1 + test2 every epoch)
    python train_kitti.py --rotation_range 10 --epochs 100 \
        --batch_size 128 --lr 1e-4 --num_workers 4 \
        --resume ../checkpoints/kitti10/epoch_7_step_153_0.6334873647582089_model.pt \
        --no_ransac 
"""
import argparse
import configparser
import random
import time
from pathlib import Path

import numpy as np
import torch

from datasets.kitti import GrdImg_H, GrdImg_W, build_dataloaders
from losses.bev_cross import loss_bev_cross
from losses.cross_gt import cross_unnorm_loss
from losses.infonce import compute_infonce_loss_match_all_with_scale_select_negatives
from models import CrossViewMatcher
from solver.cross_unnorm import (compute_theta_pinhole, solve_bisection,
                                 solve_e2e, solve_ransac)
from utils.backbone import img_infer
from utils.grids import (create_image_grids, create_metric_grid,
                         create_metric_grid_batch)
from utils.logger import setup_logger
from utils.matching import build_match_batch, get_point
from utils.metrics import compute_yaw_errors, diff_yaw
from utils.pose import compute_valid_pose_mask, estimate_pose

SCRIPT_DIR = Path(__file__).resolve().parent
SPLITS_DIR = SCRIPT_DIR / "splits" / "kitti"
CONFIG_PATH = SCRIPT_DIR / "config.ini"

FEATURE_H = GrdImg_H // 16  # 26 ground feature rows
FEATURE_W = GrdImg_W // 16  # 88 ground feature columns


def parse_args():
    """Command-line interface.

    Returns:
        argparse.Namespace with the training / evaluation options.
    """
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--rotation_range', type=int, default=10, choices=(10, 180),
                        help='satellite rotation amplitude in degrees')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--max_depth', type=float, default=40.0)
    parser.add_argument('--num_samples_matches', type=int, default=None,
                        help='matches sampled per step (default: config.ini)')
    parser.add_argument('--coarse_step_deg', type=float, default=10.0,
                        help='coarse psi grid step for all solvers')
    parser.add_argument('--cheir_lambda', type=float, default=10.0,
                        help='e2e cheirality penalty weight')
    parser.add_argument('--no_ransac', action='store_true',
                        help='skip the (slow) RANSAC solver metrics')
    parser.add_argument('--with_procrustes', action='store_true',
                        help='enable the legacy weighted-Procrustes pose '
                             'estimate, its valid-pose filtering and metrics '
                             '(off by default)')
    parser.add_argument('--eval_depth_mask', choices=('with', 'without', 'both'),
                        default='both',
                        help='max-depth validity filter for evaluation metrics')
    parser.add_argument('--resume', type=str, default=None,
                        help='checkpoint path (legacy files load as-is)')
    parser.add_argument('--eval_only', action='store_true')
    parser.add_argument('--max_eval_samples', type=int, default=0,
                        help='stop evaluation after this many samples (0 = all)')
    parser.add_argument('--max_steps', type=int, default=0,
                        help='cap batches per split (training or evaluation); '
                             '0 = unlimited (smoke testing)')
    parser.add_argument('--data_root', type=str, default=None,
                        help='KITTI root override (default: config.ini)')
    parser.add_argument('--ckpt_dir', type=str, default=None,
                        help='checkpoint dir (default: checkpoints/kitti<range>)')
    parser.add_argument('--log_dir', type=str, default='logs')
    parser.add_argument('--seed', type=int, default=None,
                        help='optional global seed (legacy runs are unseeded)')
    return parser.parse_args()


def resolve_settings(args):
    """Read config.ini and overlay the CLI overrides.

    Args:
        args: parsed CLI arguments.
    Returns:
        dict with sat_bev_res / num_samples_matches / num_virtual_point /
        dataset_root and the derived metric-grid extent of the default
        latitude (informational).
    """
    config = configparser.ConfigParser()
    if not config.read(CONFIG_PATH):
        raise FileNotFoundError(f'Could not read config file at {CONFIG_PATH}')
    num_samples_matches = args.num_samples_matches
    if num_samples_matches is None:
        num_samples_matches = config.getint('Matching', 'num_samples_matches')
    return {
        'sat_bev_res': config.getint('Model', 'sat_bev_res'),
        'num_samples_matches': num_samples_matches,
        'num_virtual_point': config.getint('Loss', 'num_virtual_point'),
        'dataset_root': args.data_root or config.get('KITTI', 'dataset_root'),
    }


def _loc_metrics_dict(lateral, longi, loc, yaw_err, solver, mode):
    """Build the metric dict of one solver from accumulated sample lists.

    Args:
        lateral, longi: per-sample lateral (x) / longitudinal (y) errors in
            meters.
        loc: per-sample euclidean location error (meters).
        yaw_err: per-sample yaw errors in degrees.
        solver: solver name used as metric-key tag; one of 'procrustes',
            'e2e', 'bisection', 'ransac'.
        mode: 'with' | 'nomask'; 'nomask' appends '_nomask' to every key.
    Returns:
        dict of metric name -> float, e.g. ``loc_e2e_mean``,
        ``yaw_ransac_R@1deg``, ``lateral_procrustes_R@1m_nomask``.
    """
    suffix = '_nomask' if mode == 'nomask' else ''
    lat, lon = np.asarray(lateral), np.asarray(longi)
    loc_arr, yaw_arr = np.asarray(loc), np.asarray(yaw_err)

    def _r(arr, thr):
        return float(np.mean(arr < thr)) if arr.size else float('nan')

    values = {
        f'loc_{solver}_mean': float(loc_arr.mean()),
        f'loc_{solver}_median': float(np.median(loc_arr)),
        f'lateral_{solver}_mean': float(lat.mean()),
        f'lateral_{solver}_median': float(np.median(lat)),
        f'lateral_{solver}_R@1m': _r(lat, 1.0),
        f'lateral_{solver}_R@5m': _r(lat, 5.0),
        f'long_{solver}_mean': float(lon.mean()),
        f'long_{solver}_median': float(np.median(lon)),
        f'long_{solver}_R@1m': _r(lon, 1.0),
        f'long_{solver}_R@5m': _r(lon, 5.0),
        f'yaw_{solver}_mean': float(yaw_arr.mean()),
        f'yaw_{solver}_median': float(np.median(yaw_arr)),
        f'yaw_{solver}_R@1deg': _r(yaw_arr, 1.0),
        f'yaw_{solver}_R@5deg': _r(yaw_arr, 5.0),
    }
    return {k + suffix: v for k, v in values.items()}


def run_split(name, model, loader, device, args, settings, optimizer,
              metric_coord4loss, u_grid, v_grid, training, logger):
    """Run one split in either training or evaluation mode.

    Training mode: computes the loss (``loss_bev_cross + 0.1 * infonce``),
    backpropagates and logs running per-solver metrics every step.
    Evaluation mode: collects per-sample errors and reports the full metric
    blocks (one per depth-mask mode, one per solver).

    Args:
        name: split label ('train' / 'test1' / 'test2').
        model: CrossViewMatcher.
        loader: DataLoader over the KITTI dataset.
        device: torch device.
        args: CLI arguments.
        settings: dict from :func:`resolve_settings`.
        optimizer: optimizer (ignored when ``training`` is False).
        metric_coord4loss: [1, N, 2] reference grid for ``loss_bev_cross``.
        u_grid, v_grid: ground feature-cell coordinate grids.
        training: True -> train mode, False -> eval mode.
        logger: logger instance.
    Returns:
        dict of metrics (evaluation) or the last-step running means
        (training). The checkpoint-naming metric is ``loc_e2e_mean`` of the
        'with'-mask evaluation.
    """
    model.train(training)
    sat_bev_res = settings['sat_bev_res']
    num_matches = settings['num_samples_matches']
    modes = ('with',) if training else {
        'with': ('with',), 'without': ('nomask',),
        'both': ('with', 'nomask'),
    }[args.eval_depth_mask]

    # Per-mode sample lists: one entry per processed sample.
    acc = {m: {f'{q}_{s}': []
               for q in ('loc', 'lateral', 'long', 'yaw')
               for s in ('procrustes', 'e2e', 'bisection', 'ransac')}
           for m in modes}
    for m in modes:
        acc[m].update({'loss_bev_cross': [], 'infonce': [],
                       'loss_cross_gt': [], 'loss': [], 'scales': []})
    n_done = 0

    grad_ctx = torch.enable_grad() if training else torch.no_grad()
    for step, data in enumerate(loader):
        if args.max_steps and step >= args.max_steps:
            logger.info('%s: reached max_steps=%d, stopping early',
                        name, args.max_steps)
            break
        (sat, grd, depth, camera_k, tgt, rotation_gt, mask_grd, wh, yaw,
         names, mpp) = data
        sat_size = sat.shape[-1]

        sat, grd = sat.to(device), grd.to(device)
        depth = torch.clamp(depth.to(device), 0, args.max_depth)
        camera_k, tgt = camera_k.to(device), tgt.to(device).float()
        rotation_gt, mask_grd = rotation_gt.to(device).float(), mask_grd.to(device)
        yaw, mpp = yaw.to(device).float(), mpp.to(device).float()

        # Per-sample metric grid: this frame's tile coverage in meters.
        grid_size_h = mpp * sat_size
        tgt_metric = (tgt / sat_size) * grid_size_h[:, None, None]
        metric_coord_sat = create_metric_grid_batch(grid_size_h, sat_bev_res,
                                                    device)

        grd_feature = img_infer(grd, H=FEATURE_H, W=FEATURE_W)
        sat_feature = img_infer(sat, H=sat_bev_res, W=sat_bev_res)

        with grad_ctx:
            for mode in modes:
                depth_valid = mode == 'with'
                (metric_coord_sat_batch, bev_coord_grd, mask, sat_idx,
                 grd_idx, sat_points, grd_points, weights,
                 matching_score_original, matching_score, _, _) = build_match_batch(
                    model, depth, camera_k, grd_feature, sat_feature,
                    metric_coord_sat, u_grid, v_grid, args.max_depth,
                    num_matches, mask_grd, training=True,
                    depth_valid=depth_valid)

                if args.with_procrustes:
                    rotation_pred, translation_pred, scale = estimate_pose(
                        sat_points, grd_points, weights)
                    valid_pose = compute_valid_pose_mask(rotation_pred,
                                                        translation_pred, scale)
                    if valid_pose is None or not valid_pose.any():
                        logger.warning('Skipping batch: singular transformation matrix')
                        continue
                    num_invalid = int((~valid_pose).sum().item())
                    if num_invalid:
                        logger.warning('Skipping %d samples: invalid transformation estimate',
                                       num_invalid)

                    rotation_gt_f = rotation_gt[valid_pose]
                    rotation_pred_f = rotation_pred[valid_pose]
                    translation_pred_f = translation_pred[valid_pose]
                    mpp_f = mpp[valid_pose]
                    grid_size_h_f = grid_size_h[valid_pose]
                    if scale is not None:
                        scale = scale[valid_pose]
                else:
                    rotation_gt_f = rotation_gt

                # Feature-map UVs + rescaled intrinsics for the ray angles.
                grd_uv_top, sat_points_top, camera_k_scaled = get_point(
                    sat_idx, grd_idx, camera_k, grd, metric_coord_sat,
                    FEATURE_H, FEATURE_W)
                theta = compute_theta_pinhole(grd_uv_top, camera_k_scaled,
                                              FEATURE_W)

                C_e2e, psi_e2e, _ = solve_e2e(
                    sat_points_top, theta, weights,
                    coarse_step_deg=args.coarse_step_deg,
                    cheir_lambda=args.cheir_lambda, dual_window=False)
                with torch.no_grad():
                    C_bisection, psi_bisection, _ = solve_bisection(
                        sat_points_top, theta, weights,
                        coarse_step_deg=args.coarse_step_deg)
                    if args.ransac:
                        C_ransac, psi_ransac, _, _ = solve_ransac(
                            sat_points_top, theta, weights,
                            coarse_step_deg=args.coarse_step_deg, seed=0)

                if training and mode == 'with':
                    loss_bev_cross_ = loss_bev_cross(
                        metric_coord4loss, C_e2e, psi_e2e,
                        C_gt=tgt_metric[:, 0, :], yaw_gt=yaw).mean()
                    infonce_loss = torch.mean(
                        compute_infonce_loss_match_all_with_scale_select_negatives(
                            rotation_gt_f, tgt_metric, sat_points, grd_points,
                            torch.ones(sat_points.shape[0], 1, 1,
                                       device=sat_points.device),
                            sat_idx, grd_idx,
                            matching_score_original, metric_coord_sat_batch,
                            bev_coord_grd, mask, grid_size_h, sat_bev_res))
                    loss_cross_gt_ = cross_unnorm_loss(
                        tgt_metric[:, 0, :], -yaw, sat_points_top, theta,
                        weights).mean()
                    loss = loss_bev_cross_ + infonce_loss * 0.1

                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()

                a = acc[mode]
                if training:
                    if args.with_procrustes:
                        loc_procrustes = torch.norm(
                            tgt_metric[valid_pose][:, 0, :]
                            - translation_pred_f[:, 0, :], dim=1).detach()
                        yaw_procrustes = torch.abs(
                            torch.arcsin(-rotation_pred_f[:, 0, 1]) - yaw[valid_pose]
                        ).mean().detach() * 180 / np.pi
                        a['loc_procrustes'].extend(loc_procrustes.cpu().tolist())
                        a['yaw_procrustes'].append(float(yaw_procrustes))
                    loc_e2e = torch.norm(
                        tgt_metric[:, 0, :] - C_e2e, dim=1).detach()
                    loc_bisection = torch.norm(
                        tgt_metric[:, 0, :] - C_bisection, dim=1).detach()
                    yaw_e2e = diff_yaw(psi_e2e.detach(), -yaw)
                    yaw_bisection = diff_yaw(psi_bisection.detach(), -yaw)

                    a['loc_e2e'].extend(loc_e2e.cpu().tolist())
                    a['yaw_e2e'].extend(yaw_e2e.tolist())
                    a['loc_bisection'].extend(loc_bisection.cpu().tolist())
                    a['yaw_bisection'].extend(yaw_bisection.tolist())
                    if args.ransac:
                        loc_ransac = torch.norm(
                            tgt_metric[:, 0, :] - C_ransac, dim=1).detach()
                        a['loc_ransac'].extend(loc_ransac.cpu().tolist())
                        a['yaw_ransac'].extend(
                            diff_yaw(psi_ransac.detach(), -yaw).tolist())
                    a['loss_bev_cross'].append(loss_bev_cross_.item())
                    a['infonce'].append(infonce_loss.item())
                    a['loss_cross_gt'].append(loss_cross_gt_.item())
                    a['loss'].append(loss.item())

                    parts = [f'{name} epoch {run_split.epoch} step {step}']
                    if args.with_procrustes:
                        parts.append(f'loc_procrustes: {np.mean(a["loc_procrustes"]):.4f}')
                        parts.append(f'yaw_procrustes: {np.mean(a["yaw_procrustes"]):.4f}')
                    parts.append(f'loc_e2e: {np.mean(a["loc_e2e"]):.4f}')
                    parts.append(f'yaw_e2e: {np.mean(a["yaw_e2e"]):.4f}')
                    parts.append(f'loc_bisection: {np.mean(a["loc_bisection"]):.4f}')
                    parts.append(f'yaw_bisection: {np.mean(a["yaw_bisection"]):.4f}')
                    if a['loc_ransac']:
                        parts.append(f'loc_ransac: {np.mean(a["loc_ransac"]):.4f}')
                        parts.append(f'yaw_ransac: {np.mean(a["yaw_ransac"]):.4f}')
                    parts.append(f'loss_bev_cross: {np.mean(a["loss_bev_cross"]):.4f}')
                    parts.append(f'infonce: {np.mean(a["infonce"]):.4f}')
                    parts.append(f'loss_cross_gt: {np.mean(a["loss_cross_gt"]):.4f}')
                    parts.append(f'loss: {np.mean(a["loss"]):.4f}')
                    logger.info(' '.join(parts))
                    continue

                # ---------------- evaluation metrics ----------------
                if args.with_procrustes:
                    # Procrustes: pixel errors back to meters per sample.
                    trans_px = (translation_pred_f / grid_size_h_f[:, None, None]) * sat_size
                    diff_m = (trans_px - tgt[valid_pose]) * mpp_f[:, None, None]
                    a['lateral_procrustes'].extend(diff_m[:, 0, 0].abs().cpu().tolist())
                    a['long_procrustes'].extend(diff_m[:, 0, 1].abs().cpu().tolist())
                    a['loc_procrustes'].extend(
                        diff_m.norm(dim=-1).view(-1).cpu().tolist())
                    a['yaw_procrustes'].extend(
                        compute_yaw_errors(rotation_gt_f, rotation_pred_f))
                    if scale is not None:
                        a['scales'].extend(scale.view(-1).cpu().tolist())

                # BEV solvers: errors are already in meters.
                loc_err_e2e = tgt_metric[:, 0, :] - C_e2e
                a['lateral_e2e'].extend(loc_err_e2e[:, 0].abs().cpu().tolist())
                a['long_e2e'].extend(loc_err_e2e[:, 1].abs().cpu().tolist())
                a['loc_e2e'].extend(loc_err_e2e.norm(dim=-1).cpu().tolist())
                a['yaw_e2e'].extend(diff_yaw(psi_e2e, -yaw).tolist())

                loc_err_bisection = tgt_metric[:, 0, :] - C_bisection
                a['lateral_bisection'].extend(
                    loc_err_bisection[:, 0].abs().cpu().tolist())
                a['long_bisection'].extend(
                    loc_err_bisection[:, 1].abs().cpu().tolist())
                a['loc_bisection'].extend(
                    loc_err_bisection.norm(dim=-1).cpu().tolist())
                a['yaw_bisection'].extend(diff_yaw(psi_bisection, -yaw).tolist())

                if args.ransac:
                    loc_err_ransac = tgt_metric[:, 0, :] - C_ransac
                    a['lateral_ransac'].extend(
                        loc_err_ransac[:, 0].abs().cpu().tolist())
                    a['long_ransac'].extend(
                        loc_err_ransac[:, 1].abs().cpu().tolist())
                    a['loc_ransac'].extend(
                        loc_err_ransac.norm(dim=-1).cpu().tolist())
                    a['yaw_ransac'].extend(diff_yaw(psi_ransac, -yaw).tolist())

        n_done += sat.shape[0]
        if args.max_eval_samples and not training \
                and n_done >= args.max_eval_samples:
            logger.info('%s: reached max_eval_samples=%d, stopping early',
                        name, args.max_eval_samples)
            break

    if training:
        a = acc['with']
        metrics = {'loc_e2e_mean':
                       np.mean(a['loc_e2e']) if a['loc_e2e'] else float('nan')}
        if args.with_procrustes:
            metrics['loc_procrustes_mean'] = \
                np.mean(a['loc_procrustes']) if a['loc_procrustes'] else float('nan')
        return metrics

    metrics = {}
    for mode in modes:
        a = acc[mode]
        for solver in ('procrustes', 'e2e', 'bisection', 'ransac'):
            if not a[f'loc_{solver}']:
                continue
            metrics.update(_loc_metrics_dict(
                a[f'lateral_{solver}'], a[f'long_{solver}'],
                a[f'loc_{solver}'], a[f'yaw_{solver}'], solver, mode))
        if a['scales']:
            metrics[f'scale_mean{"" if mode == "with" else "_nomask"}'] = \
                float(np.mean(a['scales']))
            metrics[f'scale_median{"" if mode == "with" else "_nomask"}'] = \
                float(np.median(a['scales']))

    for mode in modes:
        label = name if mode == 'with' else f'{name} [no depth-mask]'
        a = acc[mode]
        if not a['loc_e2e']:
            logger.warning('%s produced no valid samples.', label)
            continue
        logger.info('%s  (%d samples)', label, len(a['loc_e2e']))
        for solver in ('procrustes', 'e2e', 'bisection', 'ransac'):
            if not a[f'loc_{solver}']:
                continue
            loc = np.asarray(a[f'loc_{solver}'])
            lat = np.asarray(a[f'lateral_{solver}'])
            lon = np.asarray(a[f'long_{solver}'])
            yaw = np.asarray(a[f'yaw_{solver}'])
            logger.info('  loc_%s - Mean: %.3fm  Median: %.3fm',
                        solver, loc.mean(), np.median(loc))
            logger.info('  lateral_%s - Mean: %.3fm  Median: %.3fm  R@1m: %.3f%%  R@5m: %.3f%%',
                        solver, lat.mean(), np.median(lat),
                        100 * np.mean(lat < 1.0), 100 * np.mean(lat < 5.0))
            logger.info('  long_%s - Mean: %.3fm  Median: %.3fm  R@1m: %.3f%%  R@5m: %.3f%%',
                        solver, lon.mean(), np.median(lon),
                        100 * np.mean(lon < 1.0), 100 * np.mean(lon < 5.0))
            logger.info('  yaw_%s - Mean: %.3fdeg  Median: %.3fdeg  R@1deg: %.3f%%  R@5deg: %.3f%%',
                        solver, yaw.mean(), np.median(yaw),
                        100 * np.mean(yaw < 1.0), 100 * np.mean(yaw < 5.0))
        if a['scales']:
            logger.info('  scale - Mean: %.3f  Median: %.3f',
                        np.mean(a['scales']), np.median(a['scales']))
    return metrics


def main():
    """Entry point: build model/data, then train or evaluate."""
    args = parse_args()
    args.ransac = not args.no_ransac
    settings = resolve_settings(args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    timestamp = time.strftime('%Y%m%d%H%M%S')
    logger = setup_logger('train_kitti', args.log_dir,
                          f'train_kitti_{args.rotation_range}_{timestamp}.log')
    logger.info('args: %s', vars(args))
    logger.info('settings: %s', settings)
    logger.info('Using device: %s', device)

    model = CrossViewMatcher(device, sat_bev_res=settings['sat_bev_res'],
                             temperature=0.1, embed_dim=1024).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  betas=(0.9, 0.999))

    if args.resume:
        logger.info('Resuming from checkpoint: %s', args.resume)
        model.load_state_dict(torch.load(args.resume, map_location=device))
        optimizer_path = args.resume.replace('model', 'optimizer')
        if not args.eval_only and Path(optimizer_path).exists():
            logger.info('Resuming optimizer: %s', optimizer_path)
            optimizer.load_state_dict(
                torch.load(optimizer_path, map_location=device))

    loaders = build_dataloaders(
        settings['dataset_root'], SPLITS_DIR, args.rotation_range,
        args.batch_size, num_workers=args.num_workers, shuffle=True)
    train_loader, test1_loader, test2_loader = loaders

    metric_coord4loss = create_metric_grid(
        5.0, settings['num_virtual_point'], 1, device)
    u_grid, v_grid = create_image_grids(args.batch_size, device,
                                        GrdImg_H, GrdImg_W)

    if args.eval_only:
        test1_metrics = run_split('test1', model, test1_loader, device, args,
                                  settings, optimizer, metric_coord4loss,
                                  u_grid, v_grid, training=False, logger=logger)
        test2_metrics = run_split('test2', model, test2_loader, device, args,
                                  settings, optimizer, metric_coord4loss,
                                  u_grid, v_grid, training=False, logger=logger)
        torch.cuda.empty_cache()
        return

    ckpt_dir = Path(args.ckpt_dir or f'checkpoints/kitti{args.rotation_range}')
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    run_split.epoch = 0
    for epoch in range(args.epochs):
        run_split.epoch = epoch
        run_split('train', model, train_loader, device, args, settings,
                  optimizer, metric_coord4loss, u_grid, v_grid,
                  training=True, logger=logger)

        test1_metrics = run_split('test1', model, test1_loader, device, args,
                                  settings, optimizer, metric_coord4loss,
                                  u_grid, v_grid, training=False, logger=logger)
        test2_metrics = run_split('test2', model, test2_loader, device, args,
                                  settings, optimizer, metric_coord4loss,
                                  u_grid, v_grid, training=False, logger=logger)
        torch.cuda.empty_cache()

        loc_e2e_mean = test1_metrics['loc_e2e_mean']
        model_path = ckpt_dir / f'epoch_{epoch}_step_{loc_e2e_mean}_model.pt'
        torch.save(model.state_dict(), model_path)
        torch.save(optimizer.state_dict(),
                   ckpt_dir / f'epoch_{epoch}_step_{loc_e2e_mean}_optimizer.pt')
        logger.info('Saved checkpoint: %s', model_path)


if __name__ == '__main__':
    main()
