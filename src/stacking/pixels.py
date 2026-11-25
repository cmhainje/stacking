from dataclasses import dataclass
from functools import cache

import healpy as hp
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from numpy.typing import NDArray
from talltable.constants import HP_HIGH_LEVEL
from talltable.paths import PIXEL_DB_PATH, WAVES_DB_PATH
from talltable.waveid import waveid_to_rowcoldet

BATCH_ROWS = 4 * 1024 * 1024
DISC_LEVEL = 14  # healpix level used to look up discs; cells are ~13 arcsec


@dataclass(frozen=True)
class Pixels:
    """SPHEREx measurements, sorted by hphigh."""

    # fmt: off
    hphigh: NDArray  # (n,)   nested healpix index at HP_HIGH_LEVEL
    vec:    NDArray  # (n, 3) unit vectors
    wl:     NDArray  # (n,)   observed wavelength [um]
    flux:   NDArray  # (n,)   nu I_nu [nW/m^2/sr], zodi not subtracted
    zodi:   NDArray  # (n,)   nu I_nu [nW/m^2/sr]
    var:    NDArray  # (n,)   variance of flux
    # fmt: on

    def __getitem__(self, idx):
        return Pixels(
            self.hphigh[idx],
            self.vec[idx],
            self.wl[idx],
            self.flux[idx],
            self.zodi[idx],
            self.var[idx],
        )

    def __len__(self):
        return len(self.hphigh)


def waveid_to_wave_index(wid):
    row, col, det = waveid_to_rowcoldet(wid)
    return (det - 1) * 2040 * 2040 + row * 2040 + col


@cache
def load_wavelengths() -> NDArray:
    """Wavelength [um] of every detector pixel, ordered by wave index."""
    t = pq.read_table(WAVES_DB_PATH, columns=["waveid", "wavelength"])
    waveid, wl = t["waveid"].to_numpy(), t["wavelength"].to_numpy()
    assert np.array_equal(waveid_to_wave_index(waveid), np.arange(6 * 2040 * 2040))
    return wl


def read_partition(
    part: int, wl_min: float, wl_max: float, reject_flags: int
) -> Pixels:
    """Read one talltable partition."""
    path = PIXEL_DB_PATH / f"part={part}/compacted.parquet"
    columns = ["hphigh", "flux", "variance", "zodi", "waveid"]

    # drop rows with rejected flags, non-positive variance, or non-finite values
    expr = pc.bit_wise_and(pc.field("flags"), reject_flags) == 0
    expr &= pc.field("variance") > 0
    for c in ["flux", "variance", "zodi"]:
        expr &= pc.is_finite(pc.field(c))

    # one sequential read of the whole file, then decode it in batches
    with open(path, "rb") as fh:
        f = pq.ParquetFile(pa.BufferReader(fh.read()))

    pieces = []
    for batch in f.iter_batches(batch_size=BATCH_ROWS, columns=columns + ["flags"]):
        table = pa.Table.from_batches([batch]).filter(expr)
        hphigh, flux, vari, zodi, waveid = (table.column(c).to_numpy() for c in columns)

        # cut on wavelength range before computing vec
        wl = load_wavelengths()[waveid_to_wave_index(waveid)]
        keep = (wl >= wl_min) & (wl < wl_max)
        hphigh = hphigh[keep]
        vec = np.stack(hp.pix2vec(2**HP_HIGH_LEVEL, hphigh, nest=True), axis=-1)
        pieces.append((hphigh, vec, wl[keep], flux[keep], zodi[keep], vari[keep]))

    hphigh, vec, wl, flux, zodi, vari = (np.concatenate(x) for x in zip(*pieces))

    return Pixels(
        hphigh,
        vec,
        wl,
        to_nuInu(flux, wl),
        to_nuInu(zodi, wl),
        to_nuInu(to_nuInu(vari, wl), wl),
    )


def find_near(pix: Pixels, vec: NDArray, theta: float) -> NDArray:
    """Indices of the pixels within `theta` [rad] of `vec`, plus some just outside."""
    cells = hp.query_disc(2**DISC_LEVEL, vec, theta, inclusive=True, nest=True)

    # pixels are sorted by hphigh, so those in each cell are contiguous
    shift = 2 * (HP_HIGH_LEVEL - DISC_LEVEL)
    beg = np.searchsorted(pix.hphigh, cells << shift)
    end = np.searchsorted(pix.hphigh, (cells + 1) << shift)
    return np.concatenate([np.arange(b, e) for b, e in zip(beg, end)])


def to_nuInu(I_nu: NDArray, wl: NDArray) -> NDArray:
    """Convert I_nu [MJy/sr] to nu I_nu [nW/m^2/sr]."""
    return 2997.92458 / wl * I_nu  # c / wl, with wl in um
