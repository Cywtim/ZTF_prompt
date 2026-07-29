"""
MAD.py - Median Absolute Deviation outlier clipping for light curves.

Input:  numpy array with columns [MJD, band, flux, fluxerr]
Output: numpy array with outliers removed (same column layout)

Algorithm:
  Iteratively clips per-band outliers using MAD (Median Absolute Deviation).
  MAD = 1.4826 * median(|flux - median(flux)|)  — makes it consistent with σ for Gaussian data.
  Points beyond n_mad * MAD from the band median are removed.
  Iterates until convergence or max_iter.
"""

import numpy as np


def mad_clip(arr, n_mad=5.0, max_iter=3):
    """
    Per-band iterative MAD clipping.

    Parameters
    ----------
    arr : np.ndarray
        Shape (N, 4) with columns [MJD, band, flux, fluxerr].
    n_mad : float, default 5.0
        Threshold in MAD units. Higher = more conservative (fewer clipped).
    max_iter : int, default 3
        Maximum iterations.

    Returns
    -------
    arr_clean : np.ndarray
        Cleaned array, same column layout.
    mask : np.ndarray (bool)
        Boolean mask of kept points in the original array.
    stats : dict
        Summary: {n_total, n_kept, n_removed, per_band: {kept, removed, details}}
    """
    mjd = arr[:, 0]
    band = arr[:, 1]
    flux = arr[:, 2]
    mask = np.ones(len(flux), dtype=bool)

    removed_bands = []

    for iteration in range(max_iter):
        changed = False
        for b in np.unique(band):
            bm = (band == b) & mask
            if bm.sum() < 4:
                continue
            f = flux[bm]
            med = np.median(f)
            mad = np.median(np.abs(f - med))
            if mad == 0:
                continue
            sigma = 1.4826 * mad
            outlier = np.abs(f - med) > n_mad * sigma
            if outlier.any():
                idx = np.where(bm)[0][outlier]
                for i in idx:
                    removed_bands.append((int(b), float(flux[i]), float(mjd[i])))
                mask[idx] = False
                changed = True
        if not changed:
            break

    arr_clean = arr[mask]

    stats: dict = {
        "n_total": len(arr),
        "n_kept": int(mask.sum()),
        "n_removed": int((~mask).sum()),
    }
    for b_int in np.unique(arr[:, 1]):
        b_name = {1: "g", 2: "r", 3: "u"}.get(int(b_int), f"b{b_int}")
        b_total = int((arr[:, 1] == b_int).sum())
        b_kept = int((arr_clean[:, 1] == b_int).sum())
        stats[b_name] = {
            "total": b_total,
            "kept": b_kept,
            "removed": b_total - b_kept,
        }

    return arr_clean, mask, stats