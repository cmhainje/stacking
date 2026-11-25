import time
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from uuid import uuid4

import healpy as hp
import numpy as np
import pyvo
from astropy.table import Table
from numpy.typing import NDArray
from pyvo.dal import DALServiceError
from talltable.partition import part_to_level_index

from .paths import GAIA_DIR
from .pixels import Pixels, find_near

IRSA_TAP = "https://irsa.ipac.caltech.edu/TAP"
G_BRIGHT = 9.0  # stars brighter than this have large masks, so we fetch them all-sky

GAIA_EPOCH = 2016.0  # when DR3 positions are for
EPOCH = 2026.0  # when we want them: about halfway through the SPHEREx data so far
PM_MAX = 11.0  # arcsec/yr, a bit more than the fastest star's


@dataclass(frozen=True)
class Stars:
    """Gaia stars and other sources to mask."""

    # fmt: off
    vec: NDArray  # (n, 3) unit vectors
    G:   NDArray  # (n,) magnitude
    # fmt: on

    def __getitem__(self, idx):
        return Stars(self.vec[idx], self.G[idx])

    def __len__(self):
        return len(self.G)


def mask_radius(G: NDArray) -> NDArray:
    """Mask radius in radians."""
    arcsec = 8.0 + 4e3 * 10 ** (-0.25 * G)
    return np.radians(arcsec / 3600)


def query_gaia(where: str, path: Path) -> Stars:
    """Query Gaia DR3 at IRSA. The result is cached at `path`.

    `where` sees positions at GAIA_EPOCH, but the output positions are at EPOCH."""
    if not path.exists():
        query = f"""
            SELECT source_id, ra, dec, pmra, pmdec, phot_g_mean_mag
            FROM gaia_dr3_source
            WHERE {where}
        """
        table = run_tap(query)
        # write then rename, since other jobs may be fetching the same file
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{uuid4().hex}.fits")
        table.write(tmp)
        tmp.replace(path)

    t = Table.read(path).filled(0)  # some stars have no proper motion; don't move them
    ra, dec, pmra, pmdec, G = (
        np.array(t[c], dtype=np.float64)
        for c in ["ra", "dec", "pmra", "pmdec", "phot_g_mean_mag"]
    )
    return Stars(move(ra, dec, pmra, pmdec, EPOCH - GAIA_EPOCH), G)


def move(ra: NDArray, dec: NDArray, pmra: NDArray, pmdec: NDArray, dt: float) -> NDArray:
    """Unit vectors after `dt` years. Positions in deg, proper motions in mas/yr.

    Gaia's pmra already includes the cos(dec)."""
    ra, dec = np.radians(ra), np.radians(dec)
    east = np.stack([-np.sin(ra), np.cos(ra), np.zeros_like(ra)], axis=-1)
    north = np.stack(
        [-np.sin(dec) * np.cos(ra), -np.sin(dec) * np.sin(ra), np.cos(dec)], axis=-1
    )
    vec = np.stack(
        [np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)], axis=-1
    )

    mas = np.radians(1e-3 / 3600)
    vec = vec + dt * mas * (pmra[:, None] * east + pmdec[:, None] * north)
    return vec / np.linalg.norm(vec, axis=1, keepdims=True)


def run_tap(query: str, max_tries: int = 5) -> Table:
    """Run a query at IRSA, retrying if the service fails."""
    service = pyvo.dal.TAPService(IRSA_TAP)
    for i in range(max_tries):
        try:
            return service.run_async(query, maxrec=50_000_000).to_table()
        except DALServiceError:
            if i == max_tries - 1:
                raise
            time.sleep(2**i + np.random.uniform(0, 1))
    raise AssertionError("unreachable")


@cache
def fetch_bright() -> Stars:
    """Every star on the sky with G < G_BRIGHT."""
    path = GAIA_DIR / f"gaia_dr3_pm_G_lt_{G_BRIGHT:g}.fits"
    return query_gaia(f"phot_g_mean_mag < {G_BRIGHT}", path)


def fetch_stars(part: int, G_max: float) -> Stars:
    """Gaia stars that could touch `part`."""
    lvl, idx = part_to_level_index(int(part))
    center = np.array(hp.pix2vec(2**lvl, idx, nest=True))
    part_radius = hp.max_pixrad(2**lvl)  # from its center to its farthest corner

    # bright stars: cut the all-sky list to those that reach the partition
    bright = fetch_bright()
    theta = np.arccos(np.clip(bright.vec @ center, -1, +1))
    touches = theta < part_radius + mask_radius(bright.G)
    bright = bright[touches & (bright.G < G_max)]

    # faint stars: every mask is at most mask_radius(G_BRIGHT), so one cone has them all,
    # padded for those that were farther away at GAIA_EPOCH
    ra, dec = hp.vec2ang(center, lonlat=True)
    radius = np.degrees(part_radius + mask_radius(np.array(G_BRIGHT)))
    radius += PM_MAX * (EPOCH - GAIA_EPOCH) / 3600
    circle = f"CIRCLE('ICRS', {ra[0]}, {dec[0]}, {radius})"
    faint = query_gaia(
        f"1 = CONTAINS(POINT('ICRS', ra, dec), {circle})"
        f" AND phot_g_mean_mag >= {G_BRIGHT} AND phot_g_mean_mag < {G_max}",
        GAIA_DIR / f"gaia_dr3_pm_part{part}_G_{G_BRIGHT:g}_{G_max:g}_{radius:.6f}deg.fits",
    )

    return Stars(
        np.concatenate([bright.vec, faint.vec]), np.concatenate([bright.G, faint.G])
    )


def mask_stars(pix: Pixels, stars: Stars) -> Pixels:
    """Drop pixels near stars."""
    keep = np.ones(len(pix), dtype=bool)
    for vec, radius in zip(stars.vec, mask_radius(stars.G)):
        near = find_near(pix, vec, radius)
        inside = pix.vec[near] @ vec > np.cos(radius)
        keep[near[inside]] = False
    return pix[keep]
