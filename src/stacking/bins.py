from numpy.typing import NDArray


def radial_edges() -> NDArray:
    """Fine r edges [cMpc]: linear, then log."""
    raise NotImplementedError


def wavelength_edges() -> NDArray:
    """Fine rest-frame edges [um], for 0 < z < 2.1."""
    raise NotImplementedError


def redshift_edges() -> NDArray:
    """Slices for the randoms."""
    raise NotImplementedError


def slice_edges(edges: NDArray, lo: float, hi: float) -> NDArray:
    """Edges of whole bins covering [lo, hi]."""
    raise NotImplementedError
