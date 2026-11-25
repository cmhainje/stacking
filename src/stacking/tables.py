from dataclasses import dataclass
from pathlib import Path

import numpy as np
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


def check_headers(a: Header, b: Header) -> None:
    """Raise if the headers differ, unless it's only the git commit; then just warn."""
    for name in ["r_edges", "wl_edges", "z_edges"]:
        if not np.array_equal(getattr(a, name), getattr(b, name)):
            raise RuntimeError(f"{name} differ")

    keys = a.provenance.keys() | b.provenance.keys()
    diff = {k for k in keys if a.provenance.get(k) != b.provenance.get(k)}
    if diff - {"git_commit"}:
        raise RuntimeError(f"provenance differs: {sorted(diff)}")
    if diff:
        commits = [x.provenance.get("git_commit") for x in (a, b)]
        print(f"WARNING: headers match, but the git commits differ: {commits}")


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
