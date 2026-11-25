from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Config:
    """Everything that changes the numbers in the tables."""

    # fmt: off
    tracer: str                     # DESI file prefix, e.g. "LRG", "ELG_LOPnotqso"
    caps: list[str]                 # e.g. ["NGC", "SGC"]
    rand_files: list[int]           # which numbered random files to use
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
    """Config, git commit (as "git_commit"), talltable version, DESI catalog version."""
    raise NotImplementedError
