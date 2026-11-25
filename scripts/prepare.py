from argparse import ArgumentParser, Namespace
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from make_tables import load_catalogs
from tqdm import tqdm

from stacking.bins import radial_edges, redshift_edges, slice_edges
from stacking.catalog import partitions_near
from stacking.config import load_config
from stacking.mask import fetch_bright, fetch_stars

IRSA_JOBS = 3  # queries to run at once; IRSA starts dropping connections by 8


def parse_args() -> Namespace:
    ap = ArgumentParser()
    ap.add_argument("config", type=Path)
    ap.add_argument("-o", "--out_dir", type=Path, required=True)
    return ap.parse_args()


def main(config: Path, out_dir: Path) -> None:
    """List the partitions that make_tables needs to run, and cache their stars."""
    cfg = load_config(config)
    r_max = slice_edges(radial_edges(), *cfg.r_range)[-1]
    z_edges = slice_edges(redshift_edges(), *cfg.z_range)

    data, rand = load_catalogs(cfg, z_edges)
    parts = sorted(partitions_near(data, r_max) | partitions_near(rand, r_max))
    out_dir.mkdir(exist_ok=True, parents=True)
    (out_dir / "parts.txt").write_text("".join(f"{p}\n" for p in parts))
    print(f"{len(data)} galaxies, {len(rand)} randoms, {len(parts)} partitions")

    fetch_bright()  # so the threads don't all fetch it
    with ThreadPoolExecutor(IRSA_JOBS) as pool:
        jobs = {pool.submit(fetch_stars, p, cfg.G_max): p for p in parts}
        done = tqdm(as_completed(jobs), total=len(jobs), desc="fetching stars")
        failed = [jobs[job] for job in done if job.exception()]
    if failed:
        raise RuntimeError(f"couldn't fetch stars for {failed}; re-run to try again")


if __name__ == "__main__":
    main(**vars(parse_args()))
