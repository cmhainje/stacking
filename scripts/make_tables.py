from argparse import ArgumentParser, Namespace
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from stacking.bins import radial_edges, redshift_edges, slice_edges, wavelength_edges
from stacking.catalog import Catalog, in_partition, load_catalog, select_near
from stacking.config import Config, load_config, provenance
from stacking.estimator import Sums, combine, sum_around
from stacking.mask import fetch_stars, mask_stars
from stacking.paths import data_paths, rand_paths
from stacking.pixels import Pixels, read_partition
from stacking.tables import (
    Header,
    check_headers,
    read_header,
    write_galaxies,
    write_randoms,
)

RAND_BATCH = 128  # randoms to sum around at once; the full grid is ~1.3 MB per random


def parse_args() -> Namespace:
    ap = ArgumentParser()
    ap.add_argument("config", type=Path)
    ap.add_argument("-p", "--parts", type=int, nargs="+", required=True)
    ap.add_argument("-o", "--out_dir", type=Path, required=True)
    return ap.parse_args()


def try_resuming(out_dir: Path, header: Header, parts: list[int]) -> list[int]:
    """If the out_dir already exists, check whether any pq files have been written,
    if their header matches this run, and whether any partitions can be skipped."""

    if not out_dir.exists():
        return parts

    pq_files = list(out_dir.glob("part*.parquet"))
    if len(pq_files) == 0:
        return parts

    print(f"found a past run in {out_dir}; checking that it matches")
    check_headers(header, read_header(pq_files[0]))

    def still_todo(p: int) -> bool:
        return not (
            (out_dir / f"part{p}-data.parquet").exists()
            and (out_dir / f"part{p}-rand.parquet").exists()
        )

    return list(filter(still_todo, parts))


def load_catalogs(cfg: Config, z_edges: NDArray) -> tuple[Catalog, Catalog]:
    """Load the data and randoms, cut to the z range, and subsample the randoms."""

    def load(paths: list[Path]) -> Catalog:
        cat = load_catalog(paths)
        return cat[(cat.z >= z_edges[0]) & (cat.z < z_edges[-1])]

    data = load(data_paths(cfg.tracer, cfg.caps))
    rand = load(rand_paths(cfg.tracer, cfg.caps, cfg.rand_files))

    n_rand = cfg.rand_per_data * len(data)
    if n_rand > len(rand):
        raise RuntimeError(
            f"need {n_rand} randoms, but the files only have {len(rand)}"
        )
    rng = np.random.default_rng(cfg.seed)
    return data, rand[rng.choice(len(rand), size=n_rand, replace=False)]


def sum_around_slices(
    cat: Catalog, pix: Pixels, r_edges: NDArray, wl_edges: NDArray, z_edges: NDArray
) -> Sums:
    """Weighted sums around the objects in each redshift slice: (n_slice, n_r, n_wl)."""
    n_slices = len(z_edges) - 1
    shape = (n_slices, len(r_edges) - 1, len(wl_edges) - 1)
    total = Sums(*(np.zeros(shape) for _ in range(4)))

    for i in range(0, len(cat), RAND_BATCH):
        batch = cat[i : i + RAND_BATCH]
        sums = sum_around(batch, pix, r_edges, wl_edges)
        label = np.searchsorted(z_edges, batch.z, side="right") - 1
        total = total + combine(sums, batch, label, n_slices)

    return total


def main(config: Path, parts: list[int], out_dir: Path) -> None:
    """For each partition, sum around tracers and randoms, and store a table."""
    cfg = load_config(config)

    r_edges = slice_edges(radial_edges(), *cfg.r_range)
    wl_edges = slice_edges(wavelength_edges(), *cfg.wl_range)
    z_edges = slice_edges(redshift_edges(), *cfg.z_range)

    header = Header(r_edges, wl_edges, z_edges, provenance(cfg))

    parts = try_resuming(out_dir, header, parts)
    out_dir.mkdir(exist_ok=True, parents=True)

    cat_data, cat_rand = load_catalogs(cfg, z_edges)

    # observed wavelengths that land in wl_edges for some z
    wl_min = wl_edges[0] * (1 + z_edges[0])
    wl_max = wl_edges[-1] * (1 + z_edges[-1])

    for part in parts:
        pix = read_partition(part, wl_min, wl_max, cfg.reject_flags)
        pix = mask_stars(pix, fetch_stars(part, cfg.G_max))

        cat = select_near(cat_data, part, r_edges[-1])
        sums = sum_around(cat, pix, r_edges, wl_edges)
        hit = sums.N.any(axis=(1, 2))  # select_near keeps some that don't touch
        write_galaxies(
            out_dir / f"part{part}-data.parquet", cat.id[hit], sums[hit], header
        )

        cat = select_near(cat_rand, part, r_edges[-1])
        sums = sum_around_slices(cat, pix, r_edges, wl_edges, z_edges)
        home = cat[in_partition(cat, part)]
        weight = np.histogram(home.z, z_edges, weights=home.weight)[0]
        write_randoms(out_dir / f"part{part}-rand.parquet", part, weight, sums, header)


if __name__ == "__main__":
    main(**vars(parse_args()))
