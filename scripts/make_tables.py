from argparse import Namespace
from pathlib import Path


def parse_args() -> Namespace:
    """config, parts, out_dir."""
    raise NotImplementedError


def main(config: Path, parts: list[int], out_dir: Path) -> None:
    """For each partition, sum around tracers and randoms, and store a table."""
    raise NotImplementedError


if __name__ == "__main__":
    main(**vars(parse_args()))
