from argparse import ArgumentParser, Namespace
from pathlib import Path

from stacking.tables import merge_galaxies


def parse_args() -> Namespace:
    ap = ArgumentParser()
    ap.add_argument("table_dir", type=Path, help="make_tables's out_dir")
    ap.add_argument("-o", "--out", type=Path, required=True)
    return ap.parse_args()


def find_parts(table_dir: Path, kind: str) -> dict[int, Path]:
    """Paths to the tables, by partition number."""
    start, end = "part", f"-{kind}.parquet"
    paths = table_dir.glob(f"{start}*{end}")
    return {int(p.name.removeprefix(start).removesuffix(end)): p for p in paths}


def main(table_dir: Path, out: Path) -> None:
    """Merge the per-partition galaxy tables, so each galaxy has one row."""
    data, rand = find_parts(table_dir, "data"), find_parts(table_dir, "rand")
    if not data:
        raise RuntimeError(f"no tables in {table_dir}")

    # make_tables writes data then rand
    unfinished = sorted(data.keys() ^ rand.keys())
    if unfinished:
        raise RuntimeError(f"partitions with only one of data and rand: {unfinished}")

    # in order of partition number, neighbors are close enough together that only ~12k
    # split LRGs are ever waiting on the rest of their rows
    merge_galaxies([data[p] for p in sorted(data)], out)


if __name__ == "__main__":
    main(**vars(parse_args()))
