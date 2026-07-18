"""
Fill masked pixels using the nearest non-zero source value.

Uses scipy.ndimage.distance_transform_edt, which computes the Euclidean
distance transform and, crucially, can return the indices of the nearest
non-zero source pixel for every zero (masked) location — all in one fast
C-level pass with no Python loops.
"""

import numpy as np
from scipy.ndimage import distance_transform_edt


def fill_nearest(source: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Fill masked pixels with the nearest non-zero value from source.

    For every pixel where mask is non-zero, the output copies the source
    value directly.  For every pixel where mask is zero, the output takes
    the value of the nearest (Euclidean) non-zero pixel in source.

    Parameters
    ----------
    source : 2-D array of values.  Non-zero pixels are the fill candidates.
    mask   : 2-D boolean or numeric array, same shape as source.
             Non-zero  → pixel is already known (copied from source as-is).
             Zero      → pixel needs to be filled from the nearest source.

    Returns
    -------
    np.ndarray of the same shape and dtype as source.

    Raises
    ------
    ValueError  if source and mask have different shapes, or if source
                contains no non-zero pixels (nothing to fill from).
    """
    source = np.asarray(source)
    mask   = np.asarray(mask)

    if source.shape != mask.shape:
        raise ValueError(
            f"source and mask must have the same shape; "
            f"got {source.shape} vs {mask.shape}"
        )
    if not np.any(source != 0):
        raise ValueError("source contains no non-zero pixels to fill from.")

    # distance_transform_edt treats zeros as "background" and non-zeros as
    # "foreground".  indices=True returns, for every background pixel, the
    # row/col of the nearest foreground pixel — exactly what we need.
    empty   = source == 0
    _, nearest_idx = distance_transform_edt(empty, return_indices=True)

    # Build output: start from source, then overwrite masked-off pixels
    # with the value of their nearest non-zero neighbour.
    out = source.copy()
    fill_rows, fill_cols = np.where(mask == 0)
    out[fill_rows, fill_cols] = source[
        nearest_idx[0][fill_rows, fill_cols],
        nearest_idx[1][fill_rows, fill_cols],
    ]
    return out
