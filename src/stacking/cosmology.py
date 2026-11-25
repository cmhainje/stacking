"""Our assumed cosmology (flat LCDM, Planck 2018), and fast versions of what we need."""

import numpy as np
from astropy.cosmology import Planck18 as COSMO
from numpy.typing import NDArray
from scipy.interpolate import CubicSpline

Z_MAX = 4.0

_z = np.linspace(0, Z_MAX, 4001)
_D_M = CubicSpline(_z, COSMO.comoving_transverse_distance(_z).to_value("Mpc"))


def D_M(z: NDArray) -> NDArray:
    """Transverse comoving distance [Mpc]."""
    assert np.all((z >= 0) & (z <= Z_MAX))
    return _D_M(z)
