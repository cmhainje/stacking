import time
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from uuid import uuid4

import healpy as hp
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv
import pyarrow.parquet as pq
import pyvo
from astropy.table import Table
from numpy.typing import NDArray
from pyarrow.fs import S3FileSystem
from pyvo.dal import DALQueryError, DALServiceError
from talltable.partition import part_to_level_index

from .paths import GAIA_DIR
from .pixels import Pixels, find_near

# Gaia DR3, all from the `gaia_source` table:
# - bright stars have large masks, so we fetch those on the whole sky, from IRSA
# - the rest come from STScI's copy on S3, which is split into healpix tiles
IRSA_TAP = "https://irsa.ipac.caltech.edu/TAP"
GAIA_S3 = "stpubdata/gaia/gaia_dr3/public/hats/gaia"
COLUMNS = ["ra", "dec", "pmra", "pmdec", "phot_g_mean_mag"]
G_BRIGHT = 9.0

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


@contextmanager
def replacing(path: Path) -> Generator[Path]:
    """Write to a temporary file, then rename, since other jobs may be writing `path`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{uuid4().hex}.tmp")
    yield tmp
    tmp.replace(path)


def read_stars(path: Path) -> Stars:
    """Read a table with COLUMNS, moving the stars from GAIA_EPOCH to EPOCH."""
    t = pq.read_table(path, columns=COLUMNS, partitioning=None)
    ra, dec, pmra, pmdec, G = (t[c].to_numpy().astype(np.float64) for c in COLUMNS)

    # some stars have no proper motion; don't move them
    pmra, pmdec = np.nan_to_num(pmra), np.nan_to_num(pmdec)
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
        except (DALServiceError, DALQueryError):  # IRSA's dropped connections are either
            if i == max_tries - 1:
                raise
            time.sleep(2**i + np.random.uniform(0, 1))
    raise AssertionError("unreachable")


@cache
def fetch_bright() -> Stars:
    """Every star on the sky with G < G_BRIGHT."""
    path = GAIA_DIR / f"irsa_G_lt_{G_BRIGHT:g}.parquet"
    if not path.exists():
        cols = ", ".join(COLUMNS)
        t = run_tap(f"SELECT {cols} FROM gaia_dr3_source WHERE phot_g_mean_mag < {G_BRIGHT}")
        arrays = {c: np.ma.filled(t[c].astype(np.float64), np.nan) for c in COLUMNS}
        with replacing(path) as tmp:
            pq.write_table(pa.table(arrays), tmp)
    return read_stars(path)


@cache
def gaia_s3() -> S3FileSystem:
    return S3FileSystem(anonymous=True, region="us-east-1")


@cache
def load_tiles() -> tuple[NDArray, NDArray]:
    """Order and (nested) pixel of every healpix tile on S3."""
    path = GAIA_DIR / "s3_tiles.csv"
    if not path.exists():
        with replacing(path) as tmp, gaia_s3().open_input_file(
            f"{GAIA_S3}/partition_info.csv"
        ) as f:
            tmp.write_bytes(f.read())
    t = pyarrow.csv.read_csv(path)
    return t["Norder"].to_numpy(), t["Npix"].to_numpy()


def find_tiles(part: int) -> list[tuple[int, int]]:
    """Tiles that may have faint stars that touch `part`."""
    center, radius = part_cone(part)
    radius += mask_radius(np.array(G_BRIGHT))  # the most that a faint star can reach
    radius += np.radians(PM_MAX * (EPOCH - GAIA_EPOCH) / 3600)  # tiles are at GAIA_EPOCH

    # tiles partition the sky, so each cell at the finest order is in exactly one
    order, npix = load_tiles()
    finest = order.max()
    first_cell = npix << (2 * (finest - order))
    by_cell = np.argsort(first_cell)
    cells = hp.query_disc(2**finest, center, radius, inclusive=True, nest=True)
    i = by_cell[np.searchsorted(first_cell[by_cell], cells, side="right") - 1]
    return sorted({(int(o), int(n)) for o, n in zip(order[i], npix[i])})


def fetch_tile(order: int, npix: int, G_max: float) -> Path:
    """Copy a tile's stars with G < G_max from S3, if we haven't already."""
    path = GAIA_DIR / f"s3_G_lt_{G_max:g}" / f"Norder={order}-Npix={npix}.parquet"
    if not path.exists():
        dir = npix // 10_000 * 10_000
        table = pq.read_table(
            f"{GAIA_S3}/dataset/Norder={order}/Dir={dir}/Npix={npix}.parquet",
            columns=COLUMNS,
            filters=pc.field("phot_g_mean_mag") < G_max,
            filesystem=gaia_s3(),
            partitioning=None,
        )
        with replacing(path) as tmp:
            pq.write_table(table, tmp)
    return path


def part_cone(part: int) -> tuple[NDArray, float]:
    """Center and radius [rad] of a cone that covers `part`."""
    lvl, idx = part_to_level_index(int(part))
    return np.array(hp.pix2vec(2**lvl, idx, nest=True)), hp.max_pixrad(2**lvl)


def fetch_stars(part: int, G_max: float) -> Stars:
    """Gaia stars that could touch `part`."""
    center, radius = part_cone(part)

    def touching(s: Stars) -> Stars:
        theta = np.arccos(np.clip(s.vec @ center, -1, +1))
        return s[theta < radius + mask_radius(s.G)]

    bright = fetch_bright()
    sets = [touching(bright[bright.G < G_max])]
    for order, npix in find_tiles(part):
        s = read_stars(fetch_tile(order, npix, G_max))
        sets.append(touching(s[s.G >= G_BRIGHT]))

    return Stars(
        np.concatenate([s.vec for s in sets]), np.concatenate([s.G for s in sets])
    )


def mask_stars(pix: Pixels, stars: Stars) -> Pixels:
    """Drop pixels near stars."""
    keep = np.ones(len(pix), dtype=bool)
    for vec, radius in zip(stars.vec, mask_radius(stars.G)):
        near = find_near(pix, vec, radius)
        inside = pix.vec[near] @ vec > np.cos(radius)
        keep[near[inside]] = False
    return pix[keep]
