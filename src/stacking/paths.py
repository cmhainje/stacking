import os
from pathlib import Path

DESI_CATALOGS = "dr1/iron/v1.5"  # what download_data.sh fetches; goes in provenance

DESI_DIR = Path(os.environ.get("STACKING_DESI_DIR", Path(__file__).parents[2] / "data"))


def data_paths(tracer: str, caps: list[str]) -> list[Path]:
    return [DESI_DIR / f"{tracer}_{cap}_clustering.dat.fits" for cap in caps]


def rand_paths(tracer: str, caps: list[str], rand_files: list[int]) -> list[Path]:
    return [
        DESI_DIR / f"{tracer}_{cap}_{i}_clustering.ran.fits"
        for cap in caps
        for i in rand_files
    ]
