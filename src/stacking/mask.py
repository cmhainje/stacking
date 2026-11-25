from dataclasses import dataclass

from numpy.typing import NDArray

from .pixels import Pixels


@dataclass(frozen=True)
class Stars:
    """Gaia stars and other sources to mask."""

    # fmt: off
    vec: NDArray  # (n, 3) unit vectors
    G:   NDArray  # (n,) magnitude
    # fmt: on


def fetch_stars(part: int, G_max: float) -> Stars:
    """Gaia stars that could touch `part`."""
    raise NotImplementedError


def mask_radius(G: NDArray) -> NDArray:
    """Mask radius in radians."""
    raise NotImplementedError


def mask_stars(pix: Pixels, stars: Stars) -> Pixels:
    """Drop pixels near stars."""
    raise NotImplementedError
