import json
import subprocess
import tomllib
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from .paths import DESI_CATALOGS


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
    with open(path, "rb") as f:
        raw = tomllib.load(f)

    names = {f.name for f in fields(Config)}
    unknown, missing = sorted(raw.keys() - names), sorted(names - raw.keys())
    if unknown or missing:
        raise ValueError(f"{path}: unknown keys {unknown}, missing keys {missing}")

    for name in ["z_range", "r_range", "wl_range"]:
        raw[name] = tuple(raw[name])
    return Config(**raw)


def provenance(cfg: Config) -> dict[str, str]:
    """Config, git commit (as "git_commit"), DESI catalog version."""
    config = {f"config.{k}": json.dumps(v) for k, v in asdict(cfg).items()}

    # the commit hash, with "-dirty" appended if tracked files have been edited
    git = ["git", "describe", "--always", "--dirty", "--abbrev=40", "--exclude=*"]
    commit = subprocess.check_output(git, cwd=Path(__file__).parent, text=True).strip()

    return {**config, "git_commit": commit, "desi_catalogs": DESI_CATALOGS}
