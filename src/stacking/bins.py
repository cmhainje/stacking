import numpy as np
from numpy.typing import NDArray


def radial_edges() -> NDArray:
    """fine r edges, linear to 0.3 cMpc, then log out to 3.0 cMpc.
    chosen somewhat arbitrarily."""
    linear = np.arange(12) / 40
    log = np.geomspace(0.3, 3.0, 29)
    return np.concatenate((linear, log))


def wavelength_edges() -> NDArray:
    """fine rest-frame edges [um].

    we want to ~4x subdivide the observed wavelength grid everywhere when converted into
    tracers' rest frames. we'll assume tracers could span z ~ 0-2.1.

    SPHEREx has highest resolution R ~ 130 from 3.80-5.03 um
        4x subdivide that -> (Δλ/λ) / 4 ~ 0.002
        for z in 0-2.1, λ_rf can see this fine resolution down to 1.225 um

    SPHEREx has max resolution R ~ 41 from 0.73-3.87 um
        4x subdivide that -> (Δλ/λ) / 4 ~ 0.006
        for z in 0-2.1, λ_rf can be seen down to 0.235 um

    so our final bins are logarithmic,
        with Δλ/λ = 0.006 from 0.23 um to 1.225 um,
        then Δλ/λ = 0.002 up to 5.03 um.

    and we'll ensure that the coarse bins are *exactly* 3x coarser than the fine ones,
    so fine bins can be recombined to exactly match the coarse ones.
    """
    step = 0.002    # fine step; coarse is exactly 3x
    join = 1.225    # um
    n_coarse, n_fine = 276, 708  # chosen so min < 0.235, max > 5.03

    k = np.concatenate((
        np.arange(-3 * n_coarse, 0, 3),   # coarse
        np.arange(0, n_fine + 1),         # fine
    ))
    return join * np.exp(step * k)


def redshift_edges() -> NDArray:
    """Randoms live in z-slices with width 0.05: 0, 0.05, ..., 2.1."""
    return np.arange(0, 43) / 20


def slice_edges(edges: NDArray, lo: float, hi: float) -> NDArray:
    """Edges of whole bins covering [lo, hi]."""
    assert edges[0] <= lo < hi <= edges[-1]
    beg = np.searchsorted(edges, lo, side="right") - 1
    end = np.searchsorted(edges, hi, side="left") + 1
    return edges[beg:end]
