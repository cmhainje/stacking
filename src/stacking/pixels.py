from dataclasses import dataclass

from numpy.typing import NDArray


@dataclass(frozen=True)
class Pixels:
    """SPHEREx measurements."""

    vec: NDArray   # (n, 3) unit vectors
    wl: NDArray    # (n,) observed wavelength [um]
    flux: NDArray  # (n,) nu I_nu, zodi not subtracted
    zodi: NDArray  # (n,) nu I_nu
    var: NDArray   # (n,) (nu I_nu)^2


def read_partition(part: int, wl_min: float, wl_max: float, reject_flags: int) -> Pixels:
    """Read one talltable partition."""
    raise NotImplementedError


def to_nuInu(I_nu: NDArray, wl: NDArray) -> NDArray:
    """Convert I_nu to nu I_nu."""
    raise NotImplementedError
