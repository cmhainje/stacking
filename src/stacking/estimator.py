from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.stats import binned_statistic_2d

from .catalog import Catalog
from .pixels import Pixels, find_near


@dataclass(frozen=True)
class Sums:
    """Rest-frame sums. All arrays have shape (n_tracer, n_r, n_wl)."""

    # fmt: off
    S: NDArray  # sum of (1+z)^4 flux
    Z: NDArray  # sum of (1+z)^4 zodi
    V: NDArray  # sum of (1+z)^8 variance
    N: NDArray  # number of measurements
    # fmt: on


def sum_around(cat: Catalog, pix: Pixels, r_edges: NDArray, wl_edges: NDArray) -> Sums:
    """Bin pixels around each object."""

    # assumes cat has reasonably small length (compared to O(100M) pixels)
    shape = (len(cat), len(r_edges) - 1, len(wl_edges) - 1)
    S, Z, V, N = (np.zeros(shape) for _ in range(4))

    for i in range(len(cat)):
        near = pix[find_near(pix, cat.vec[i], r_edges[-1] / cat.D_M[i])]
        if len(near) == 0:
            continue
        theta = np.arccos(np.clip(near.vec @ cat.vec[i], -1, +1))
        zp1 = 1 + cat.z[i]
        _s, _z, _v, _n = binned_statistic_2d(
            theta,
            near.wl,
            [
                near.flux,
                near.zodi,
                near.var,
                np.ones_like(near.flux),
            ],
            statistic='sum',
            bins=[r_edges / cat.D_M[i], wl_edges * zp1]  # type: ignore
        ).statistic
        S[i] = _s * zp1**4
        Z[i] = _z * zp1**4
        V[i] = _v * zp1**8
        N[i] = _n

    return Sums(S, Z, V, N)


def combine(sums: Sums, cat: Catalog, label: NDArray, n_labels: int) -> Sums:
    """Weighted sum within each label.

    REQUIRES SUMS AND CAT TO HAVE SAME OBJECTS IN SAME ORDER.

    sums:  arrays of shape (n_obj, n_r, n_wl)
    cat:   arrays of shape (n_obj,)
    label: (n_obj,), all ints between 0 and n_labels - 1

    output: `Sums` with shape (n_labels, n_r, n_wl)
    """
    assert label.shape == (len(cat),) and len(sums.S) == len(cat)
    assert ((0 <= label) & (label < n_labels)).all()

    shape = (n_labels, sums.S.shape[1], sums.S.shape[2])
    S, Z, V, N = (np.zeros(shape) for _ in range(4))

    for i in range(n_labels):
        m = label == i
        w = cat.weight[m, None, None]
        S[i] = np.sum(sums.S[m] * w, axis=0)
        Z[i] = np.sum(sums.Z[m] * w, axis=0)
        V[i] = np.sum(sums.V[m] * w**2, axis=0)
        N[i] = np.sum(sums.N[m] * w, axis=0)
    
    return Sums(S, Z, V, N)


def delta_J(data: Sums, rand: Sums) -> NDArray:
    """Data mean minus random mean.

    `data([S-Z]/N) - rand([S-Z]/N)`

    Expects the `Sums` objects to have performed a weighted sum over objects.
        (e.g., you should run `combine`, then pass the outputs here)
    This runs elementwise and does no summing!!"""
    J_data = (data.S - data.Z) / data.N
    J_rand = (rand.S - rand.Z) / rand.N
    return J_data - J_rand
