from dataclasses import dataclass
from pathlib import Path

import healpy as hp
import numpy as np
from astropy.io import fits
from numpy.typing import NDArray
from talltable.partition import part_to_level_index

from . import cosmology


@dataclass(frozen=True)
class Catalog:
    """Tracers or randoms."""

    # fmt: off
    id:     NDArray  # (n,)   TARGETID
    vec:    NDArray  # (n, 3) unit vectors
    z:      NDArray  # (n,)
    D_M:    NDArray  # (n,)   comoving Mpc
    weight: NDArray  # (n,)
    # fmt: on

    def __getitem__(self, idx):
        return Catalog(
            self.id[idx],
            self.vec[idx],
            self.z[idx],
            self.D_M[idx],
            self.weight[idx],
        )

    def __len__(self):
        return len(self.id)


def load_catalog(paths: list[Path]) -> Catalog:
    """Read DESI clustering catalogs."""
    pieces = []
    for path in paths:
        with fits.open(path, memmap=True) as hdul:
            x = hdul[1].data
            id = np.array(x["TARGETID"], dtype=np.int64)
            ra, dec, z, weight = (
                np.array(x[c], dtype=np.float64) for c in ["RA", "DEC", "Z", "WEIGHT"]
            )
        pieces.append((id, ra, dec, z, weight))

    id, ra, dec, z, weight = (np.concatenate(x) for x in zip(*pieces))

    vec = hp.ang2vec(ra, dec, lonlat=True)
    return Catalog(id, vec, z, cosmology.D_M(z), weight)


def in_partition(cat: Catalog, part: int) -> NDArray:
    """Whether each object lives inside `part`. Every object is in exactly one."""
    lvl, idx = part_to_level_index(int(part))
    return hp.vec2pix(2**lvl, *cat.vec.T, nest=True) == idx


def select_near(cat: Catalog, part: int, r_max: float) -> Catalog:
    """Keep objects whose disc touches `part` (and maybe some extras)."""
    lvl, idx = part_to_level_index(int(part))

    # objects that live inside the partition
    inside = in_partition(cat, part)

    # objects that live outside, but within r_max of the partition
    bounds = hp.boundaries(2**lvl, idx, step=8, nest=True).T  # (N_bound, 3)

    # widen discs by half the boundary spacing so none can slip between points
    gap = np.arccos(np.min(np.sum(bounds * np.roll(bounds, -1, axis=0), axis=1)))
    cos_theta = np.cos(r_max / cat.D_M + gap / 2)

    dots = np.einsum('ik,jk->ij', cat.vec, bounds)
    touches = np.any(dots > cos_theta[:, None], axis=1)

    return cat[inside | touches]
