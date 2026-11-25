from numpy.typing import NDArray

from .estimator import Sums


def rebin(sums: Sums, r_groups: NDArray, wl_groups: NDArray) -> Sums:
    """Merge fine bins."""
    raise NotImplementedError


def mix_slices(rand: Sums, slice_weight: NDArray) -> Sums:
    """Match randoms to the data's n(z)."""
    raise NotImplementedError


def jackknife(data: Sums, rand: Sums) -> NDArray:
    """Leave-one-region-out delta_J; axis 0 is region."""
    raise NotImplementedError


def jackknife_cov(samples: NDArray) -> NDArray:
    """Covariance from jackknife samples."""
    raise NotImplementedError


def line_excess(J: NDArray, wl_edges: NDArray) -> NDArray:
    """H-alpha bin minus continuum."""
    raise NotImplementedError
