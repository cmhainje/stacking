from dataclasses import dataclass

from numpy.typing import NDArray

from .catalog import Catalog
from .pixels import Pixels


@dataclass(frozen=True)
class Sums:
    """Rest-frame sums, each (n, n_r, n_wl)."""

    # fmt: off
    S: NDArray  # sum of (1+z)^4 flux
    Z: NDArray  # sum of (1+z)^4 zodi
    V: NDArray  # sum of (1+z)^8 variance
    N: NDArray  # number of measurements
    # fmt: on


def sum_around(cat: Catalog, pix: Pixels, r_edges: NDArray, wl_edges: NDArray) -> Sums:
    """Bin pixels around each object."""
    raise NotImplementedError


def combine(sums: Sums, cat: Catalog, label: NDArray, n_labels: int) -> Sums:
    """Weighted sum within each label."""
    raise NotImplementedError


def delta_J(data: Sums, rand: Sums) -> NDArray:
    """Data mean minus random mean."""
    raise NotImplementedError
