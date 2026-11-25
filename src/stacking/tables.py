import json
from collections.abc import Generator, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace
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


def make_table(columns: dict, header: Header, **metadata: str) -> pa.Table:
    """Arrow table with the header in its metadata."""
    h = {
        "r_edges": header.r_edges.tolist(),  # floats survive JSON exactly
        "wl_edges": header.wl_edges.tolist(),
        "z_edges": header.z_edges.tolist(),
        "provenance": header.provenance,
    }
    return pa.table(columns).replace_schema_metadata(
        {"header": json.dumps(h), **metadata}
    )


@contextmanager
def open_writer(path: Path, schema: pa.Schema) -> Generator[pq.ParquetWriter]:
    """Write to a temporary file, then rename, so `path` is never half-written."""
    tmp = path.with_name(f".{path.name}.tmp")
    with pq.ParquetWriter(tmp, schema, compression="zstd", use_dictionary=False) as w:
        yield w
    tmp.replace(path)


def write_table(path: Path, table: pa.Table) -> None:
    with open_writer(path, table.schema) as w:
        w.write_table(table, row_group_size=ROW_GROUP)


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


def concat(sums: Sequence[Sums]) -> Sums:
    """Along the first axis."""
    return Sums(*(np.concatenate([getattr(s, k) for s in sums]) for k in "SZVN"))


def galaxy_table(id: NDArray, sums: Sums, header: Header) -> pa.Table:
    """One row per galaxy."""
    assert sums.S.shape == (len(id), len(header.r_edges) - 1, len(header.wl_edges) - 1)
    columns = {
        "id": pa.array(id, pa.int64()),
        "S": flatten(sums.S, np.float32),
        "Z": flatten(sums.Z, np.float32),
        "V": flatten(sums.V, np.float32),
        "N": flatten(sums.N, np.int32),
    }
    return make_table(columns, header)


def write_galaxies(path: Path, id: NDArray, sums: Sums, header: Header) -> None:
    write_table(path, galaxy_table(id, sums, header))


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


def merge_headers(headers: list[Header]) -> Header:
    """Raise if the headers differ, except in git commit; those are all kept."""
    commits = sorted({h.provenance["git_commit"] for h in headers})
    if len(commits) > 1:
        print(f"WARNING: merging tables from different git commits: {commits}")

    commit = {"git_commit": ", ".join(commits)}
    headers = [replace(h, provenance=h.provenance | commit) for h in headers]
    for h in headers:
        check_headers(headers[0], h)
    return headers[0]


def merge_galaxies(paths: list[Path], out: Path) -> None:
    """Add galaxies split across partitions; headers must match.

    Split galaxies wait in memory until all their rows have been read, so put `paths` in
    an order that keeps neighboring partitions close together."""
    header = merge_headers([read_header(p) for p in paths])

    # how many rows are there for each galaxy?
    ids = [pq.read_table(p, columns=["id"])["id"].to_numpy() for p in paths]
    for p, x in zip(paths, ids):
        if len(np.unique(x)) < len(x):
            raise RuntimeError(f"{p} has repeated ids")
    id_all, n_rows = np.unique(np.concatenate(ids), return_counts=True)

    split = n_rows > 1
    n_left = dict(zip(id_all[split].tolist(), n_rows[split].tolist()))
    partial: dict[int, Sums] = {}  # split galaxies that still have rows left

    def add_split(id: int, row: Sums) -> bool:
        """Add to the galaxy's partial sums. Was this its last row?"""
        partial[id] = partial[id] + row if id in partial else row
        n_left[id] -= 1
        return n_left[id] == 0

    def merged_rows(id: NDArray, sums: Sums) -> Iterator[tuple[NDArray, Sums]]:
        whole = np.array([i not in n_left for i in id.tolist()], dtype=bool)
        yield id[whole], sums[whole]
        for i in np.flatnonzero(~whole):
            if add_split(int(id[i]), sums[[i]]):  # [[i]] copies; a view would pin sums
                yield id[[i]], partial.pop(int(id[i]))

    shape = (0, len(header.r_edges) - 1, len(header.wl_edges) - 1)
    empty = (np.zeros(0, np.int64), Sums(*(np.zeros(shape) for _ in range(4))))

    with open_writer(out, galaxy_table(*empty, header).schema) as w:
        rows = [empty]  # merged, but not written until there's a row group's worth

        def flush() -> None:
            id, sums = zip(*rows)
            w.write_table(galaxy_table(np.concatenate(id), concat(sums), header))
            rows[:] = [empty]

        for p in paths:
            for batch in pq.ParquetFile(p).iter_batches(batch_size=ROW_GROUP):
                t = pa.Table.from_batches([batch])
                rows.extend(merged_rows(t["id"].to_numpy(), inflate(t, header)))
                if sum(len(id) for id, _ in rows) >= ROW_GROUP:
                    flush()
        flush()

    assert not partial


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
    write_table(path, make_table(columns, header, part=str(part)))


def read_randoms(path: Path) -> tuple[int, NDArray, Sums]:
    """Partition, summed weight of its randoms in each redshift slice, and its sums."""
    part = int(pq.read_schema(path).metadata[b"part"])
    table = pq.read_table(path)
    return part, table["W"].to_numpy(), inflate(table, read_header(path))
