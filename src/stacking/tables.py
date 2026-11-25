import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from numpy.typing import DTypeLike, NDArray

from .estimator import Sums

ROW_GROUP = 64  # rows; the full grid is ~0.6 MB per galaxy


@dataclass(frozen=True)
class Header:
    """Metadata stored in every table."""

    # fmt: off
    r_edges:  NDArray  # comoving Mpc
    wl_edges: NDArray  # rest-frame um
    z_edges:  NDArray  # slices for the randoms
    provenance: dict[str, str]
    # fmt: on


def check_headers(a: Header, b: Header) -> None:
    """Raise if the headers differ, unless it's only the git commit; then just warn."""
    for name in ["r_edges", "wl_edges", "z_edges"]:
        if not np.array_equal(getattr(a, name), getattr(b, name)):
            raise RuntimeError(f"{name} differ")

    keys = a.provenance.keys() | b.provenance.keys()
    diff = {k for k in keys if a.provenance.get(k) != b.provenance.get(k)}
    if diff - {"git_commit"}:
        raise RuntimeError(f"provenance differs: {sorted(diff)}")
    if diff:
        commits = [x.provenance.get("git_commit") for x in (a, b)]
        print(f"WARNING: headers match, but the git commits differ: {commits}")


def read_header(path: Path) -> Header:
    """Read a table's metadata."""
    h = json.loads(pq.read_schema(path).metadata[b"header"])
    return Header(
        np.array(h["r_edges"]),
        np.array(h["wl_edges"]),
        np.array(h["z_edges"]),
        h["provenance"],
    )


def write_table(path: Path, columns: dict, header: Header, **metadata: str) -> None:
    """Write to a temporary file, then rename, so `path` is never half-written."""
    h = {
        "r_edges": header.r_edges.tolist(),  # floats survive JSON exactly
        "wl_edges": header.wl_edges.tolist(),
        "z_edges": header.z_edges.tolist(),
        "provenance": header.provenance,
    }
    table = pa.table(columns).replace_schema_metadata(
        {"header": json.dumps(h), **metadata}
    )
    tmp = path.with_name(f".{path.name}.tmp")
    pq.write_table(
        table, tmp, compression="zstd", use_dictionary=False, row_group_size=ROW_GROUP
    )
    tmp.replace(path)


def flatten(x: NDArray, dtype: DTypeLike) -> pa.FixedSizeListArray:
    """(n, n_r, n_wl) -> n lists of length n_r * n_wl."""
    flat = pa.array(x.astype(dtype).reshape(-1))
    return pa.FixedSizeListArray.from_arrays(flat, x.shape[1] * x.shape[2])


def inflate(table: pa.Table, header: Header) -> Sums:
    """Undo `flatten` for S, Z, V, N."""
    n_r, n_wl = len(header.r_edges) - 1, len(header.wl_edges) - 1

    def get(name: str) -> NDArray:
        flat = table[name].combine_chunks().flatten()
        return flat.to_numpy().reshape(-1, n_r, n_wl)

    return Sums(get("S"), get("Z"), get("V"), get("N"))


def write_galaxies(path: Path, id: NDArray, sums: Sums, header: Header) -> None:
    """One row per galaxy."""
    assert sums.S.shape == (len(id), len(header.r_edges) - 1, len(header.wl_edges) - 1)
    columns = {
        "id": pa.array(id, pa.int64()),
        "S": flatten(sums.S, np.float32),
        "Z": flatten(sums.Z, np.float32),
        "V": flatten(sums.V, np.float32),
        "N": flatten(sums.N, np.int32),
    }
    write_table(path, columns, header)


def read_galaxies(path: Path, id: NDArray | None = None) -> tuple[NDArray, Sums]:
    """Galaxy IDs and sums. If `id` is given, only those galaxies, in that order;
    any that aren't in the table are left out."""
    header = read_header(path)
    if id is None:
        table = pq.read_table(path)
        return table["id"].to_numpy(), inflate(table, header)

    table = pq.read_table(path, filters=pc.field("id").isin(id))
    found = table["id"].to_numpy()
    id = id[np.isin(id, found)]
    order = np.argsort(found)
    rows = order[np.searchsorted(found, id, sorter=order)]
    return id, inflate(table, header)[rows]


def merge_galaxies(paths: list[Path], out: Path) -> None:
    """Add galaxies split across partitions; headers must match."""
    raise NotImplementedError


def write_randoms(
    path: Path, part: int, weight: NDArray, sums: Sums, header: Header
) -> None:
    """One row per redshift slice.

    weight: (n_slice,) summed weight of the randoms that live in the partition; unlike
        `sums`, this leaves out those in other partitions whose discs reach this one"""
    n_z, n_r, n_wl = (len(e) - 1 for e in (header.z_edges, header.r_edges, header.wl_edges))
    assert sums.S.shape == (n_z, n_r, n_wl) and weight.shape == (n_z,)
    columns = {"W": pa.array(weight, pa.float64())}
    columns |= {k: flatten(getattr(sums, k), np.float64) for k in "SZVN"}
    write_table(path, columns, header, part=str(part))


def read_randoms(path: Path) -> tuple[int, NDArray, Sums]:
    """Partition, summed weight of its randoms in each redshift slice, and its sums."""
    part = int(pq.read_schema(path).metadata[b"part"])
    table = pq.read_table(path)
    return part, table["W"].to_numpy(), inflate(table, read_header(path))
