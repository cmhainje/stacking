from dataclasses import dataclass
from pathlib import Path

from numpy.typing import NDArray

from .estimator import Sums


@dataclass(frozen=True)
class Header:
    """Metadata stored in every table."""

    # fmt: off
    r_edges:  NDArray  # comoving Mpc
    wl_edges: NDArray  # rest-frame um
    z_edges:  NDArray  # slices for the randoms
    provenance: dict[str, str]
    # fmt: on


def read_header(path: Path) -> Header:
    """Read a table's metadata."""
    raise NotImplementedError


def write_galaxies(path: Path, id: NDArray, sums: Sums, header: Header) -> None:
    """One row per galaxy."""
    raise NotImplementedError


def read_galaxies(path: Path, id: NDArray | None = None) -> tuple[NDArray, Sums]:
    """Galaxy IDs and sums."""
    raise NotImplementedError


def merge_galaxies(paths: list[Path], out: Path) -> None:
    """Add galaxies split across partitions; headers must match."""
    raise NotImplementedError


def write_randoms(path: Path, part: int, sums: Sums, header: Header) -> None:
    """One row per redshift slice."""
    raise NotImplementedError


def read_randoms(path: Path) -> tuple[int, Sums]:
    """Partition and its sums."""
    raise NotImplementedError
