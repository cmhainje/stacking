from dataclasses import dataclass
from pathlib import Path

from numpy.typing import NDArray


@dataclass(frozen=True)
class Catalog:
    """Galaxies or randoms."""

    # fmt: off
    id:     NDArray  # (n,)   TARGETID
    vec:    NDArray  # (n, 3) unit vectors
    z:      NDArray  # (n,)
    D_M:    NDArray  # (n,)   comoving Mpc
    weight: NDArray  # (n,)
    # fmt: on


def load_catalog(paths: list[Path]) -> Catalog:
    """Read DESI clustering catalogs."""
    raise NotImplementedError


def select_near(cat: Catalog, part: int, r_max: float) -> Catalog:
    """Keep objects whose disc touches `part`."""
    raise NotImplementedError
