"""CVACT split reader.

``load_cvact(root)`` returns ``(train_set, test_set)``; each entry is
``(ground_panorama_path, satellite_path, lat, lon)``.

Data layout under ``root``:
    ACT_data.mat    panoIds + utm (UTM 55S) + train/val/test row indices
    ANU_data_small/streetview/<panoId>_grdView.jpg            ground panorama
    ANU_data_small/satview_polish/<panoId>_satView_polish.jpg satellite tile

Notes:
    - The official testSet (95648) images are not shipped with
      ANU_data_small; locally only train (35532) + val (8884) exist, so by
      community convention the val split stands in as the test set.
    - UTM 55S -> WGS84 uses a pure-numpy inverse UTM transform (Snyder);
      scipy is only imported inside :func:`load_cvact`.
"""
import logging
import os

import numpy as np

logger = logging.getLogger(__name__)

GRD_DIR = 'ANU_data_small/streetview'
SAT_DIR = 'ANU_data_small/satview_polish'

# WGS84 / UTM parameters (zone 55S, central meridian 147 deg E).
_K0 = 0.9996
_A = 6378137.0
_F = 1 / 298.257223563
_E2 = _F * (2 - _F)
_EP2 = _E2 / (1 - _E2)
_LON0 = np.radians(147.0)


def utm55s_to_latlon(east, north):
    """UTM 55S (EPSG:32755) -> WGS84 (lat, lon) in degrees.

    Standard inverse UTM transform (Snyder, "Map Projections: A Working
    Manual"); sub-cm accuracy within the zone.

    Args:
        east, north: UTM easting/northing (arrays or scalars).
    Returns:
        (lat, lon) in degrees, same shape as the inputs.
    """
    x = np.asarray(east, dtype=np.float64) - 500000.0
    M = np.asarray(north, dtype=np.float64) - 10000000.0  # southern false northing
    e1 = (1 - np.sqrt(1 - _E2)) / (1 + np.sqrt(1 - _E2))
    # Northing N = k0 * meridian arc length, so invert the arc with k0 removed.
    mu = M / (_K0 * _A * (1 - _E2 / 4 - 3 * _E2 ** 2 / 64 - 5 * _E2 ** 3 / 256))
    phi1 = (mu
            + (3 * e1 / 2 - 27 * e1 ** 3 / 32) * np.sin(2 * mu)
            + (21 * e1 ** 2 / 16 - 55 * e1 ** 4 / 32) * np.sin(4 * mu)
            + (151 * e1 ** 3 / 96) * np.sin(6 * mu)
            + (1097 * e1 ** 4 / 512) * np.sin(8 * mu))
    sin1, cos1, tan1 = np.sin(phi1), np.cos(phi1), np.tan(phi1)
    C1 = _EP2 * cos1 ** 2
    T1 = tan1 ** 2
    N1 = _A / np.sqrt(1 - _E2 * sin1 ** 2)
    R1 = _A * (1 - _E2) / (1 - _E2 * sin1 ** 2) ** 1.5
    D = x / (N1 * _K0)
    lat = phi1 - (N1 * tan1 / R1) * (
        D ** 2 / 2
        - (5 + 3 * T1 + 10 * C1 - 4 * C1 ** 2 - 9 * _EP2) * D ** 4 / 24
        + (61 + 90 * T1 + 298 * C1 + 45 * T1 ** 2 - 252 * _EP2 - 3 * C1 ** 2) * D ** 6 / 720)
    lon = _LON0 + (
        D
        - (1 + 2 * T1 + C1) * D ** 3 / 6
        + (5 - 2 * C1 + 28 * T1 - 3 * C1 ** 2 + 8 * _EP2 + 24 * T1 ** 2) * D ** 5 / 120) / cos1
    return np.degrees(lat), np.degrees(lon)


def _split_indices(act, name):
    """Extract the row-index array from a 1x1 MATLAB struct field.

    ``trainSet``/``valSet``/... carry ``(xxxNum, xxxInd)``; ``xxxInd`` is a
    MATLAB 1-based index (values may reach 308900 == len(panoIds)), so it is
    decremented to a 0-based row index.

    Args:
        act:   loaded ACT_data.mat dict.
        name:  split field name, e.g. ``trainSet``.
    Returns:
        np.ndarray of 0-based row indices.
    """
    s = act[name][0, 0]
    return np.atleast_1d(s[s.dtype.names[1]]).ravel().astype(int) - 1


def _collect(root, rows, pano_ids, utm):
    """Gather ``(grd_path, sat_path, lat, lon)`` for the given rows.

    Args:
        root:     dataset root.
        rows:     row indices into pano_ids/utm.
        pano_ids: array of panorama ids.
        utm:      array of UTM coordinates.
    Returns:
        list of tuples; samples with missing images are skipped (counted in
        a warning log).
    """
    lats, lons = utm55s_to_latlon(utm[rows, 0], utm[rows, 1])
    samples, missing = [], 0
    for k, r in enumerate(rows):
        pid = pano_ids[r]
        grd = os.path.join(root, GRD_DIR, pid + '_grdView.jpg')
        sat = os.path.join(root, SAT_DIR, pid + '_satView_polish.jpg')
        if not (os.path.isfile(grd) and os.path.isfile(sat)):
            missing += 1
            continue
        samples.append((grd, sat, float(lats[k]), float(lons[k])))
    if missing:
        logger.warning('[CVACT] %d samples missing images, skipped', missing)
    return samples


def load_cvact(root, test_split='val'):
    """Load CVACT splits.

    Args:
        root:       dataset root containing ACT_data.mat.
        test_split: 'val' (default, images available locally) | 'valAll' |
                    'test' (images not shipped locally, returns empty).
    Returns:
        (train_set, test_set); each entry is
        (ground path, satellite path, lat, lon).
    """
    from scipy.io import loadmat
    act = loadmat(os.path.join(root, 'ACT_data.mat'))
    pano_ids, utm = act['panoIds'], act['utm']
    mat_name = {'val': 'valSet', 'valAll': 'valSetAll',
                'test': 'testSet'}.get(test_split, test_split)
    train = _collect(root, _split_indices(act, 'trainSet'), pano_ids, utm)
    test = _collect(root, _split_indices(act, mat_name), pano_ids, utm)
    return train, test
