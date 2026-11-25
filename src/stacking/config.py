from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Config:
    """Everything that changes the numbers in the tables."""

    # fmt: off
    tracer: str                     # e.g. "LRG"
    z_range: tuple[float, float]    # whole random slices
    r_range: tuple[float, float]    # comoving Mpc
    wl_range: tuple[float, float]   # rest-frame um
    reject_flags: int               # bitmask
    G_max: float                    # faintest masked star
    rand_per_data: int
    seed: int
    # fmt: on


def load_config(path: Path) -> Config:
    """Read a TOML file; no defaults."""
    raise NotImplementedError


def provenance(cfg: Config) -> dict[str, str]:
    """Config, git commit, talltable version."""
    raise NotImplementedError
