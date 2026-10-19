"""VIGOR training + evaluation (panorama pipeline).

One code path serves both splits and both orientation protocols: every
step back-projects the panorama depth to a BEV point cloud through the
spherical model, matches it against the satellite BEV, and derives the
pose with three solvers whose metrics are always reported (metric keys
carry the solver name, e.g. ``loc_e2e_mean`` / ``yaw_ransac_R@1deg``):

1. procrustes - weighted similarity transform on the sampled matches
                during training and a probabilistic RANSAC Procrustes over
                the full matching score at evaluation (legacy baseline;
                disabled by default, ``--with_procrustes`` to enable).
2. e2e        - differentiable cross_unnorm solve; the only solver entering
                the loss (``infonce + loss_bev_cross``) and the only one
                running with gradients.
3. bisection  - non-differentiable coarse argmin + bisection.
4. ransac     - probabilistic solver (``--no_ransac`` to disable); this is
                the solver behind the legacy ``loc_2`` evaluation block.

Ground rays use the panorama azimuth ``phi`` directly as the solver ray
angle (``theta_cross = phi``, no 90-degree shift), matching the legacy
CVACT/VIGOR convention. The e2e solver runs with the legacy panorama
parameters (coarse step 2.5 deg, cheirality weight 1.0, dual fine window);
the bisection / RANSAC solvers keep the 10 deg coarse grid of the legacy
``v2_clockwise`` family.

VIGOR geometry differs from CVACT only in scaling: satellite tiles are raw
640 px and are resized (never cropped), the BEV grid spans
``mmp * 640`` meters where ``mmp = K_MPP * cos(latitude)`` from the tile
filename, and the GT offsets are expressed in resized-satellite pixels and
rescaled to meters accordingly. ``--split`` selects samearea (random 80/20
split of the four-city balanced list; the shuffle consumes the global
numpy RNG, so runs are reproducible only with a fixed seed) or crossarea
(train on NewYork+Seattle, evaluate on SanFrancisco+Chicago).
``--rotation_range`` selects 0 = known orientation (``known_ori``
checkpoints) or 180 = unknown orientation (random satellite roll,
``unknown_ori``).

Evaluation reports lateral/longitudinal/recall metrics per solver and,
with ``--eval_depth_mask both`` (default), one metric set with the
max-depth validity filter and one without (``_nomask`` suffix).

Examples:
    # evaluate a legacy checkpoint on ~100 test samples
    python train_vigor.py --eval_only --split samearea --rotation_range 0 \
        --resume ../checkpoints/vigor/samearea/known_ori/0_model_2.531872.pt \
        --max_eval_samples 100 --with_procrustes

    # full training (evaluates the val split every epoch)
    python train_vigor.py --split crossarea --rotation_range 180 --epochs 100
"""
import argparse
import configparser
import random
import time
from pathlib import Path

import numpy as np
import torch

from datasets.vigor import GROUND_IMAGE_SIZE, build_dataloaders
from losses.bev_cross import loss_bev_cross
from losses.cross_gt import cross_unnorm_loss
from losses.infonce import compute_infonce_loss_match_all_with_scale_select_negatives
from models import CrossViewMatcher
from solver.cross_unnorm import solve_bisection, solve_e2e, solve_ransac
from utils.backbone import img_infer
from utils.grids import (create_metric_grid, create_metric_grid_batch,
                         create_spherical_grids)
from utils.logger import setup_logger
from utils.matching import build_match_batch_panorama
from utils.metrics import build_metrics_dict, diff_yaw
from utils.pose import (compute_valid_pose_mask, estimate_pose, filter_batch,
                        e2eProbabilisticProcrustesSolver)

SCRIPT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = SCRIPT_DIR / "config.ini"

FEATURE_H = GROUND_IMAGE_SIZE[0] // 16  # 32 ground feature rows
FEATURE_W = GROUND_IMAGE_SIZE[1] // 16  # 64 ground feature columns

# Legacy BEV scale: grid_size_h = mmp * 640 uses the RAW satellite pixel
# size, while the GT offsets live in resized-satellite pixels (config
# satellite_image_size). Do not derive this from the resized tensor.
RAW_SAT_SIZE = 640


def parse_args():
    """Command-line interface.

    Returns:
        argparse.Namespace with the training / evaluation options.
    """
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--split', choices=('samearea', 'crossarea'),
                        default='samearea',
                        help='VIGOR split (default: samearea)')
    parser.add_argument('--rotation_range', type=int, default=0, choices=(0, 180),
                        help='panorama roll amplitude in degrees '
                             '(0 = known orientation, 180 = unknown)')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--max_depth', type=float, default=35.0)
    parser.add_argument('--num_samples_matches', type=int, default=1024,
                        help='matches sampled per step (legacy VIGOR value)')
    parser.add_argument('--coarse_step_deg', type=float, default=10.0,
                        help='coarse psi grid step for the bisection and '
                             'RANSAC solvers (legacy 10 deg)')
    parser.add_argument('--e2e_coarse_step_deg', type=float, default=2.5,
                        help='coarse psi grid step for the e2e solver '
                             '(legacy panorama 2.5 deg)')
    parser.add_argument('--cheir_lambda', type=float, default=1.0,
                        help='e2e cheirality penalty weight (legacy 1.0)')
    parser.add_argument('--no_ransac', action='store_true',
                        help='skip the (slow) RANSAC solver metrics')
    parser.add_argument('--with_procrustes', action='store_true',
                        help='enable the legacy Procrustes pose estimate, '
                             'its valid-pose filtering and metrics (off by '
                             'default)')
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
                        help='VIGOR root override (default: config.ini)')
    parser.add_argument('--ckpt_dir', type=str, default=None,
                        help='checkpoint dir (default: checkpoints/vigor/'
                             '<split>/<known|unknown>_ori)')
    parser.add_argument('--log_dir', type=str, default='logs')
    parser.add_argument('--seed', type=int, default=None,
                        help='optional global seed (legacy runs are unseeded)')
    return parser.parse_args()


def resolve_settings(args):
    """Read config.ini and overlay the CLI overrides.

    Args:
        args: parsed CLI arguments.
    Returns:
        dict with sat_bev_res / num_virtual_point / dataset_root.
    """
    config = configparser.ConfigParser()
    if not config.read(CONFIG_PATH):
        raise FileNotFoundError(f'Could not read config file at {CONFIG_PATH}')
    dataset_root = args.data_root or config.get('VIGOR', 'dataset_root')
    return {
        'sat_bev_res': config.getint('Model', 'sat_bev_res'),
        'num_virtual_point': config.getint('Loss', 'num_virtual_point'),
        'dataset_root': dataset_root,
    }


def run_split(name, model, loader, device, args, settings, optimizer,
              metric_coord4loss, theta, phi, training, logger):
    """Run one split in either training or evaluation mode.

    Training mode: computes the loss (``infonce + loss_bev_cross``),
    backpropagates and logs running per-solver metrics every step.
    Evaluation mode: collects per-sample errors and reports the full metric
    blocks (one per depth-mask mode, one per solver).

    Args:
        name: split label ('train' / 'test').
        model: CrossViewMatcher.
        loader: DataLoader over the VIGOR dataset.
        device: torch device.
        args: CLI arguments.
        settings: dict from :func:`resolve_settings`.
        optimizer: optimizer (ignored when ``training`` is False).
        metric_coord4loss: [1, N, 2] reference grid for ``loss_bev_cross``.
        theta, phi: [B_max, 1, H/16, W/16] spherical angle grids.
        training: True -> train mode, False -> eval mode.
        logger: logger instance.
    Returns:
        dict of metrics (evaluation) or the last-step running means
        (training). The checkpoint-naming metric is ``loc_e2e_mean`` of the
        'with'-mask evaluation.
    """
    model.train(training)
    sat_bev_res = settings['sat_bev_res']
    num_matches = args.num_samples_matches
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
        if data is None:
            continue
        (grd, depth, sat, tgt, rotation_gt, _city, _resolution, _yaw,
         mmp) = data
        sat_size = sat.shape[-1]  # resized satellite pixels (656)

        grd, sat = grd.to(device), sat.to(device)
        depth = depth.to(device)
        tgt = tgt.to(device).float()
        rotation_gt = rotation_gt.to(device).float()
        mmp = mmp.to(device).float()

        grid_size_h = mmp * RAW_SAT_SIZE
        tgt_metric = (tgt / sat_size) * grid_size_h[:, None, None]
        metric_coord_sat = create_metric_grid_batch(grid_size_h, sat_bev_res,
                                                    device)

        grd_feature = img_infer(grd, H=FEATURE_H, W=FEATURE_W)
        sat_feature = img_infer(sat, H=sat_bev_res, W=sat_bev_res)

        with grad_ctx:
            for mode in modes:
                depth_valid = mode == 'with'
                batch = build_match_batch_panorama(
                    model, depth, grd_feature, sat_feature,
                    metric_coord_sat, theta, phi, args.max_depth,
                    num_matches, depth_valid=depth_valid)
                if batch is None:
                    logger.warning('Skipping batch: no valid samples')
                    continue
                (metric_coord_sat_batch, bev_coord_grd, mask, sat_idx,
                 grd_idx, sat_points, grd_points, weights,
                 matching_score_original, _sampled_idx, valid_mask,
                 matching_score) = batch

                if not valid_mask.all():
                    logger.warning('Skipping %d samples: no positive matches',
                                   int((~valid_mask).sum().item()))
                    rotation_gt = rotation_gt[valid_mask]
                    tgt_metric = tgt_metric[valid_mask]
                    grid_size_h = grid_size_h[valid_mask]

                if args.with_procrustes:
                    if training:
                        rotation_pred, translation_pred, scale = estimate_pose(
                            sat_points, grd_points, weights)
                    else:
                        pose_solver = e2eProbabilisticProcrustesSolver(
                            it_RANSAC=100, it_matches=20,
                            num_samples_matches=8192, num_corr_2d_2d=3,
                            num_refinements=4, th_inlier=2.5, th_soft_inlier=5,
                            metric_coord_sat_B=metric_coord_sat_batch,
                            metric_coord_grd_B=bev_coord_grd)
                        (rotation_pred, translation_pred, scale,
                         _, _) = pose_solver.estimate_pose(
                            matching_score, return_inliers=False)
                    if translation_pred is None:
                        logger.warning('Skipping batch: singular transformation matrix')
                        continue

                    valid_pose = compute_valid_pose_mask(
                        rotation_pred, translation_pred, scale)
                    if valid_pose is None or not valid_pose.any():
                        logger.warning('Skipping batch: singular transformation matrix')
                        continue
                    num_invalid = int((~valid_pose).sum().item())
                    if num_invalid:
                        logger.warning('Skipping %d samples: invalid transformation estimate',
                                       num_invalid)

                    (rotation_gt, tgt_metric, rotation_pred, translation_pred,
                     scale, sat_points, grd_points, weights,
                     sat_idx, grd_idx, matching_score_original,
                     bev_coord_grd, mask, grid_size_h,
                     metric_coord_sat_batch) = filter_batch(
                        valid_pose,
                        rotation_gt, tgt_metric, rotation_pred,
                        translation_pred, scale, sat_points, grd_points,
                        weights, sat_idx, grd_idx, matching_score_original,
                        bev_coord_grd, mask, grid_size_h,
                        metric_coord_sat_batch)
                    rotation_gt_f = rotation_gt
                else:
                    rotation_gt_f = rotation_gt

                # GT heading from the rotation matrix; solver convention psi = -yaw.
                yaw_gt = torch.atan2(rotation_gt_f[:, 1, 0],
                                     rotation_gt_f[:, 0, 0])
                psi_gt = -yaw_gt

                B_cur = grd_idx.shape[0]
                phi_flat = phi[:B_cur, 0, :, :].reshape(B_cur, -1)
                theta_cross = phi_flat.gather(1, grd_idx)

                C_e2e, psi_e2e, _ = solve_e2e(
                    sat_points, theta_cross, weights,
                    coarse_step_deg=args.e2e_coarse_step_deg, fine_tol_deg=0.01,
                    coarse_tau=0.05, fine_tau=0.005,
                    cheir_lambda=args.cheir_lambda, dual_window=True)
                with torch.no_grad():
                    C_bisection, psi_bisection, _ = solve_bisection(
                        sat_points, theta_cross, weights,
                        coarse_step_deg=args.coarse_step_deg, fine_tol_deg=0.01)
                    if args.ransac:
                        C_ransac, psi_ransac, _, _ = solve_ransac(
                            sat_points, theta_cross, weights,
                            it_RANSAC=100, num_corr_2d_2d=3, num_refinements=2,
                            th_inlier=5.0, th_soft_inlier=5.0,
                            minimal_step_deg=1.0,
                            coarse_step_deg=args.coarse_step_deg,
                            fine_step_deg=0.1,
                            refine_weight_mode='mask*weights', seed=0)

                if training and mode == 'with':
                    loss_bev_cross_ = loss_bev_cross(
                        metric_coord4loss, C_e2e, psi_e2e,
                        C_gt=tgt_metric[:, 0, :], yaw_gt=yaw_gt).mean()
                    infonce_loss = torch.mean(
                        compute_infonce_loss_match_all_with_scale_select_negatives(
                            rotation_gt_f, tgt_metric, sat_points, grd_points,
                            torch.ones(sat_points.shape[0], 1, 1,
                                       device=sat_points.device),
                            sat_idx, grd_idx,
                            matching_score_original, metric_coord_sat_batch,
                            bev_coord_grd, mask, grid_size_h, sat_bev_res))
                    loss_cross_gt_ = cross_unnorm_loss(
                        tgt_metric[:, 0, :], psi_gt, sat_points, theta_cross,
                        weights).mean()
                    loss = infonce_loss + loss_bev_cross_

                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()

                a = acc[mode]
                if training:
                    if args.with_procrustes:
                        loc_procrustes = torch.norm(
                            tgt_metric.squeeze(1) - translation_pred.squeeze(1),
                            dim=1).detach()
                        yaw_pred = torch.atan2(rotation_pred[:, 1, 0],
                                               rotation_pred[:, 0, 0])
                        angle_procrustes = torch.abs(yaw_gt - yaw_pred)
                        angle_procrustes = torch.minimum(
                            angle_procrustes, 2 * np.pi - angle_procrustes)
                        a['loc_procrustes'].extend(loc_procrustes.cpu().tolist())
                        a['yaw_procrustes'].append(
                            float(angle_procrustes.mean()) * 180 / np.pi)
                    loc_e2e = torch.norm(
                        tgt_metric[:, 0, :] - C_e2e, dim=1).detach()
                    loc_bisection = torch.norm(
                        tgt_metric[:, 0, :] - C_bisection, dim=1).detach()
                    yaw_e2e = diff_yaw(psi_e2e.detach(), psi_gt)
                    yaw_bisection = diff_yaw(psi_bisection.detach(), psi_gt)

                    a['loc_e2e'].extend(loc_e2e.cpu().tolist())
                    a['yaw_e2e'].extend(yaw_e2e.tolist())
                    a['loc_bisection'].extend(loc_bisection.cpu().tolist())
                    a['yaw_bisection'].extend(yaw_bisection.tolist())
                    if args.ransac:
                        loc_ransac = torch.norm(
                            tgt_metric[:, 0, :] - C_ransac, dim=1).detach()
                        a['loc_ransac'].extend(loc_ransac.cpu().tolist())
                        a['yaw_ransac'].extend(
                            diff_yaw(psi_ransac.detach(), psi_gt).tolist())
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
                    diff_m = tgt_metric.squeeze(1) - translation_pred.squeeze(1)
                    a['lateral_procrustes'].extend(diff_m[:, 0].abs().cpu().tolist())
                    a['long_procrustes'].extend(diff_m[:, 1].abs().cpu().tolist())
                    a['loc_procrustes'].extend(
                        diff_m.norm(dim=-1).cpu().tolist())
                    yaw_pred = torch.atan2(rotation_pred[:, 1, 0],
                                           rotation_pred[:, 0, 0])
                    yaw_err_procrustes = torch.abs(yaw_gt - yaw_pred)
                    yaw_err_procrustes = torch.minimum(
                        yaw_err_procrustes, 2 * np.pi - yaw_err_procrustes)
                    a['yaw_procrustes'].extend(
                        (yaw_err_procrustes * 180 / np.pi).cpu().tolist())
                    if scale is not None:
                        a['scales'].extend(scale.view(-1).cpu().tolist())

                loc_err_e2e = tgt_metric[:, 0, :] - C_e2e
                a['lateral_e2e'].extend(loc_err_e2e[:, 0].abs().cpu().tolist())
                a['long_e2e'].extend(loc_err_e2e[:, 1].abs().cpu().tolist())
                a['loc_e2e'].extend(loc_err_e2e.norm(dim=-1).cpu().tolist())
                a['yaw_e2e'].extend(diff_yaw(psi_e2e, psi_gt).tolist())

                loc_err_bisection = tgt_metric[:, 0, :] - C_bisection
                a['lateral_bisection'].extend(
                    loc_err_bisection[:, 0].abs().cpu().tolist())
                a['long_bisection'].extend(
                    loc_err_bisection[:, 1].abs().cpu().tolist())
                a['loc_bisection'].extend(
                    loc_err_bisection.norm(dim=-1).cpu().tolist())
                a['yaw_bisection'].extend(diff_yaw(psi_bisection, psi_gt).tolist())

                if args.ransac:
                    loc_err_ransac = tgt_metric[:, 0, :] - C_ransac
                    a['lateral_ransac'].extend(
                        loc_err_ransac[:, 0].abs().cpu().tolist())
                    a['long_ransac'].extend(
                        loc_err_ransac[:, 1].abs().cpu().tolist())
                    a['loc_ransac'].extend(
                        loc_err_ransac.norm(dim=-1).cpu().tolist())
                    a['yaw_ransac'].extend(diff_yaw(psi_ransac, psi_gt).tolist())

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
            metrics.update(build_metrics_dict(
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
    logger = setup_logger('train_vigor', args.log_dir,
                          f'train_vigor_{args.split}_{args.rotation_range}_{timestamp}.log')
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

    train_loader, test_loader = build_dataloaders(
        settings['dataset_root'], args.split, args.batch_size,
        args.rotation_range, num_workers=args.num_workers)

    metric_coord4loss = create_metric_grid(
        5.0, settings['num_virtual_point'], 1, device)
    # Angle grids carry one row per batch element (items are identical, the
    # per-mode code slices [:B]); legacy used a fixed headroom of 128.
    theta, phi = create_spherical_grids(GROUND_IMAGE_SIZE,
                                        max(args.batch_size, 128), device)

    if args.eval_only:
        run_split('test', model, test_loader, device, args, settings,
                  optimizer, metric_coord4loss, theta, phi,
                  training=False, logger=logger)
        torch.cuda.empty_cache()
        return

    ori_tag = 'known_ori' if args.rotation_range == 0 else 'unknown_ori'
    ckpt_dir = Path(args.ckpt_dir or f'checkpoints/vigor/{args.split}/{ori_tag}')
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    run_split.epoch = 0
    for epoch in range(args.epochs):
        run_split.epoch = epoch
        run_split('train', model, train_loader, device, args, settings,
                  optimizer, metric_coord4loss, theta, phi,
                  training=True, logger=logger)

        test_metrics = run_split('test', model, test_loader, device, args,
                                 settings, optimizer, metric_coord4loss,
                                 theta, phi, training=False, logger=logger)
        torch.cuda.empty_cache()

        loc_e2e_mean = test_metrics['loc_e2e_mean']
        model_path = ckpt_dir / f'epoch_{epoch}_step_{loc_e2e_mean}_model.pt'
        torch.save(model.state_dict(), model_path)
        torch.save(optimizer.state_dict(),
                   ckpt_dir / f'epoch_{epoch}_step_{loc_e2e_mean}_optimizer.pt')
        logger.info('Saved checkpoint: %s', model_path)


if __name__ == '__main__':
    main()
