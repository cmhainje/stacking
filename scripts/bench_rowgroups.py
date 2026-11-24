"""Benchmark talltable partition reads vs parquet row-group size, on the cluster.

Two phases, meant to be run hours apart so that neither the client page cache nor Ceph's own OSD caches still hold
the freshly written files:

    python scripts/bench_rowgroups.py write     # pick partitions, rewrite them at several row-group sizes
    python scripts/bench_rowgroups.py bench     # run the disc queries; appends a run to results.jsonl
    python scripts/bench_rowgroups.py prewarm   # whole-file warm, then queries; appends to prewarm.jsonl

`bench` can be repeated (e.g. at different times of day); each run is appended with a timestamp.

`prewarm` is independent of `bench` (it only needs `write` to have run). Per file: evict, time a whole-file
sequential read in 4 MB chunks (like spherextract_fast.warm), probe whether that stayed cached by re-reading the
first 256 MB, then time each disc query once through DuckDB and pyarrow on the warmed file. Results are appended
per file, so an interrupt keeps what finished. If `results.jsonl` from `bench` is in --out, the summary also gives
the break-even number of queries per partition: warm-file time / (mean cold query - mean query after warming).

Each query is the stacking notebook's access pattern: one 0.8' disc, row groups selected from hphigh min/max stats,
columns hphigh/waveid/flux/zodi/flags read with pyarrow pre_buffer. The same disc is also run through DuckDB with
the OR-of-BETWEEN filter talltable's PixelQuery uses (with and without DuckDB's parquet metadata cache).

Every timed query is run twice back to back: `cold` first (after posix_fadvise(DONTNEED) on the file, which drops the
kernel client's page cache on Linux; it can't touch OSD-side caches), then `warm`. cold - warm is the storage cost.

Rewritten files use the same writer settings as talltable/scripts/compact.py (zstd level 3, no dictionary), plus
row_group_size. The unmodified original partition is benchmarked too, as `orig`.
"""
import argparse
import json
import os
import socket
import time
from datetime import datetime
from pathlib import Path

import duckdb
import healpy as hp
import numpy as np
import pyarrow.parquet as pq

from talltable.partition import part_to_level_index
from talltable.query import indices_to_hphigh_ranges

DEFAULT_DB = Path('/mnt/sdceph/users/spherex/talltable')
DEFAULT_OUT = Path(f"/mnt/sdceph/users/{os.environ.get('USER', 'unknown')}/rg_bench")

RG_SIZES = {'1024k': 1 << 20, '256k': 1 << 18, '128k': 1 << 17, '64k': 1 << 16, '16k': 1 << 14}
COLS = ['hphigh', 'waveid', 'flux', 'zodi', 'flags']
RADIUS_ARCMIN = 0.8
RANGE_LEVEL = 14
HOLE, RANGE_LIMIT = 8192, 32 << 20   # arrow CacheOptions defaults used by pre_buffer


# ---------- helpers ----------

def evict(path):
    if hasattr(os, 'posix_fadvise'):
        fd = os.open(path, os.O_RDONLY)
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)


def fs_type(path):
    # filesystem type of the mount containing `path` (e.g. ceph = kernel client, fuse.ceph-fuse = FUSE client)
    try:
        best = ('', '?')
        path = os.path.realpath(path)
        for line in open('/proc/mounts'):
            _, mnt, typ = line.split()[:3]
            if (path == mnt or path.startswith(mnt.rstrip('/') + '/')) and len(mnt) > len(best[0]):
                best = (mnt, typ)
        return best[1]
    except OSError:
        return '?'


def partition_path(db, part):
    return db / 'pixels' / f'part={part}' / 'compacted.parquet'


def variant_path(out, part, name):
    return out / f'part={part}.rg{name}.parquet'


def disc_ranges(ra, dec, radius_rad):
    ipix = hp.query_disc(2**RANGE_LEVEL, hp.ang2vec(ra, dec, lonlat=True), radius_rad, nest=True, inclusive=True)
    return np.array(indices_to_hphigh_ranges(ipix, RANGE_LEVEL), dtype=np.int64).reshape(-1, 2)


def rg_bounds(md, j):
    lo = np.array([md.row_group(g).column(j).statistics.min for g in range(md.num_row_groups)])
    hi = np.array([md.row_group(g).column(j).statistics.max for g in range(md.num_row_groups)])
    return lo, hi


def select(lo, hi, ranges):
    i = np.searchsorted(ranges[:, 0], hi, side='right') - 1
    return np.flatnonzero((i >= 0) & (ranges[np.maximum(i, 0), 1] >= lo))


def io_plan(md, rgs, col_idx):
    # byte ranges pre_buffer requests: selected column chunks merged across holes <= HOLE, split at RANGE_LIMIT
    spans = []
    for g in rgs:
        rg = md.row_group(g)
        for j in col_idx:
            c = rg.column(j)
            start = c.dictionary_page_offset if c.has_dictionary_page else c.data_page_offset
            spans.append((start, start + c.total_compressed_size))
    spans.sort()
    merged = []
    for s, e in spans:
        if merged and s - merged[-1][1] <= HOLE and e - merged[-1][0] <= RANGE_LIMIT:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return len(merged), sum(e - s for s, e in merged)


# ---------- phase 1: write ----------

def pick_partitions(db, n, lo_gb, hi_gb, rng):
    with open(db / 'parts.txt') as f:
        parts = [int(line) for line in f if line.strip()]
    parts = [p for p in parts if part_to_level_index(p)[0] == 6]
    chosen = []
    for p in rng.permutation(parts):
        path = partition_path(db, p)
        if not path.exists():
            continue
        size = path.stat().st_size / 1e9
        if lo_gb < size < hi_gb:
            chosen.append(int(p))
            print(f'  part {p}: {size:.2f} GB', flush=True)
            if len(chosen) == n:
                break
    return chosen


def query_centers(path, n, rng):
    pf = pq.ParquetFile(path)
    gs = rng.choice(pf.num_row_groups, size=min(8, pf.num_row_groups), replace=False)
    h = np.concatenate([pf.read_row_group(int(g), columns=['hphigh']).column(0).to_numpy() for g in gs])
    ra, dec = hp.pix2ang(2**22, rng.choice(h, n), nest=True, lonlat=True)
    return ra.tolist(), dec.tolist()


def rewrite(src, dst, rg_rows):
    if dst.exists():
        print(f'    {dst.name} exists; skipping', flush=True)
        return
    t0 = time.perf_counter()
    pf = pq.ParquetFile(src)
    tmp = dst.with_suffix('.tmp')
    with pq.ParquetWriter(tmp, pf.schema_arrow, compression='zstd', compression_level=3,
                          use_dictionary=False) as w:
        for g in range(pf.num_row_groups):
            w.write_table(pf.read_row_group(g), row_group_size=rg_rows)
    tmp.replace(dst)
    print(f'    {dst.name}: {dst.stat().st_size / 1e9:.2f} GB in {time.perf_counter() - t0:.0f} s', flush=True)


def cmd_write(args):
    rng = np.random.default_rng(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    plan_path = args.out / 'plan.json'
    if plan_path.exists():
        plan = json.loads(plan_path.read_text())
        print(f'using existing plan {plan_path}')
    else:
        print('picking partitions:')
        parts = pick_partitions(args.db, args.n_files, args.min_gb, args.max_gb, rng)
        plan = {'db': str(args.db), 'parts': {}}
        for p in parts:
            ra, dec = query_centers(partition_path(args.db, p), args.n_queries, rng)
            plan['parts'][str(p)] = {'ra': ra, 'dec': dec}
        plan_path.write_text(json.dumps(plan))
    for p in plan['parts']:
        print(f'part {p}:', flush=True)
        for name, rg in RG_SIZES.items():
            rewrite(partition_path(Path(plan['db']), p), variant_path(args.out, p, name), rg)
    print(f"done at {datetime.now():%Y-%m-%d %H:%M}. Run `bench` in a few hours.")


# ---------- phase 2: bench ----------

def timed(fn, path):
    evict(path)
    t0 = time.perf_counter()
    fn()
    t_cold = time.perf_counter() - t0
    t0 = time.perf_counter()
    fn()
    return t_cold, time.perf_counter() - t0


def bench_file(path, ra, dec, con, con_cached):
    evict(path)
    t0 = time.perf_counter()
    pf = pq.ParquetFile(path)
    md = pf.metadata
    names = pf.schema_arrow.names
    lo, hi = rg_bounds(md, names.index('hphigh'))
    t_footer = time.perf_counter() - t0
    col_idx = [names.index(c) for c in COLS]

    rows = []
    for r, d in zip(ra, dec):
        ranges = disc_ranges(r, d, np.radians(RADIUS_ARCMIN / 60))
        rgs = select(lo, hi, ranges)
        n_req, n_bytes = io_plan(md, rgs, col_idx)

        def read_pyarrow():
            f = pq.ParquetFile(path, pre_buffer=True)   # footer re-read each query, like the notebook
            if len(rgs):
                f.read_row_groups(rgs, columns=COLS)

        ors = ' OR '.join(f'hphigh BETWEEN {a} AND {b}' for a, b in ranges)
        sql = (f"SELECT hphigh, waveid, flux, zodi, flags FROM read_parquet('{path}') "
               f"WHERE hphigh BETWEEN {ranges[0, 0]} AND {ranges[-1, 1]} AND ({ors})")

        pa_cold, pa_warm = timed(read_pyarrow, path)
        dk_cold, dk_warm = timed(lambda: con.sql(sql).fetchnumpy(), path)
        dc_cold, dc_warm = timed(lambda: con_cached.sql(sql).fetchnumpy(), path)
        rows.append(dict(n_rgs=len(rgs), n_req=n_req, mb=n_bytes / 1e6,
                         pyarrow_cold=pa_cold, pyarrow_warm=pa_warm,
                         duckdb_cold=dk_cold, duckdb_warm=dk_warm,
                         duckdb_cached_cold=dc_cold, duckdb_cached_warm=dc_warm))
    return dict(n_rg=md.num_row_groups, rows_per_rg=md.num_rows / md.num_row_groups,
                footer_kb=md.serialized_size / 1e3, file_gb=os.path.getsize(path) / 1e9,
                t_footer_cold=t_footer, queries=rows)


def cmd_bench(args):
    plan = json.loads((args.out / 'plan.json').read_text())
    db = Path(plan['db'])
    con = duckdb.connect()
    con.execute('SET enable_progress_bar = false')
    con_cached = duckdb.connect()
    con_cached.execute('SET enable_progress_bar = false')
    con_cached.execute('SET parquet_metadata_cache = true')

    run = dict(started=datetime.now().isoformat(timespec='seconds'), host=socket.gethostname(),
               fs_db=fs_type(db), fs_out=fs_type(args.out), results=[])
    print(f"host {run['host']}; filesystem: db {run['fs_db']}, out {run['fs_out']}")
    rng = np.random.default_rng()
    for p, centers in plan['parts'].items():
        ra, dec = centers['ra'][:args.n_queries], centers['dec'][:args.n_queries]
        variants = [('orig', partition_path(db, p))] + [(n, variant_path(args.out, p, n)) for n in RG_SIZES]
        print(f'part {p} ({len(ra)} queries):', flush=True)
        for k in rng.permutation(len(variants)):   # shuffled so slow drifts in cluster load don't favor one size
            name, path = variants[k]
            res = bench_file(path, ra, dec, con, con_cached)
            res.update(part=p, variant=name)
            run['results'].append(res)
            q = {key: np.array([x[key] for x in res['queries']]) for key in res['queries'][0]}
            print(f"  {name:>6s}: {res['n_rg']:5d} rgs, footer {res['footer_kb']:6.0f} kB | "
                  f"{q['n_rgs'].mean():4.1f} rgs, {q['n_req'].mean():4.1f} reqs, {q['mb'].mean():5.1f} MB / query | "
                  f"median cold/warm ms: pyarrow {1e3 * np.median(q['pyarrow_cold']):6.0f}/{1e3 * np.median(q['pyarrow_warm']):4.0f}, "
                  f"duckdb {1e3 * np.median(q['duckdb_cold']):6.0f}/{1e3 * np.median(q['duckdb_warm']):4.0f}, "
                  f"duckdb+cache {1e3 * np.median(q['duckdb_cached_cold']):6.0f}/{1e3 * np.median(q['duckdb_cached_warm']):4.0f}",
                  flush=True)

    with open(args.out / 'results.jsonl', 'a') as f:
        f.write(json.dumps(run) + '\n')
    summarize(run)


def summarize(run):
    print('\nsummary over partitions (mean of per-partition medians, cold):')
    print(f"{'variant':>8s} {'MB/query':>9s} {'reqs':>5s} {'pyarrow ms':>11s} {'duckdb ms':>10s} {'duckdb+cache ms':>16s}")
    for name in ['orig'] + list(RG_SIZES):
        rs = [r for r in run['results'] if r['variant'] == name]
        med = lambda key: np.mean([np.median([q[key] for q in r['queries']]) for r in rs])
        print(f"{name:>8s} {med('mb'):9.2f} {med('n_req'):5.1f} {1e3 * med('pyarrow_cold'):11.0f} "
              f"{1e3 * med('duckdb_cold'):10.0f} {1e3 * med('duckdb_cached_cold'):16.0f}")


# ---------- phase 3: prewarm ----------

WARM_CHUNK = 4 << 20
PROBE_BYTES = 256 << 20


def read_sequential(path, limit=None):
    n = 0
    with open(path, 'rb') as f:
        while True:
            b = f.read(WARM_CHUNK)
            if not b:
                break
            n += len(b)
            if limit is not None and n >= limit:
                break
    return n


def prewarm_file(path, ra, dec, con):
    evict(path)
    t0 = time.perf_counter()
    n = read_sequential(path)
    t_warm = time.perf_counter() - t0

    # did the warm read stay cached? (cached: several GB/s; not cached: same speed as the warm read)
    t0 = time.perf_counter()
    n_probe = read_sequential(path, PROBE_BYTES)
    t_probe = time.perf_counter() - t0

    pf = pq.ParquetFile(path)
    md = pf.metadata
    names = pf.schema_arrow.names
    lo, hi = rg_bounds(md, names.index('hphigh'))

    queries = []
    for r, d in zip(ra, dec):
        ranges = disc_ranges(r, d, np.radians(RADIUS_ARCMIN / 60))
        rgs = select(lo, hi, ranges)
        ors = ' OR '.join(f'hphigh BETWEEN {a} AND {b}' for a, b in ranges)
        sql = (f"SELECT hphigh, waveid, flux, zodi, flags FROM read_parquet('{path}') "
               f"WHERE hphigh BETWEEN {ranges[0, 0]} AND {ranges[-1, 1]} AND ({ors})")

        t0 = time.perf_counter()
        con.sql(sql).fetchnumpy()
        t_duck = time.perf_counter() - t0

        t0 = time.perf_counter()
        f = pq.ParquetFile(path, pre_buffer=True)
        if len(rgs):
            f.read_row_groups(rgs, columns=COLS)
        t_pa = time.perf_counter() - t0
        queries.append(dict(duckdb=t_duck, pyarrow=t_pa))

    return dict(file_gb=n / 1e9, t_warm=t_warm, warm_mb_s=n / 1e6 / t_warm,
                probe_mb_s=n_probe / 1e6 / t_probe, n_rg=md.num_row_groups, queries=queries)


def cold_means(out):
    # mean cold per-query time by (part, variant, method) from the last `bench` run, if available
    path = out / 'results.jsonl'
    if not path.exists():
        return {}
    run = [json.loads(line) for line in open(path)][-1]
    means = {}
    for r in run['results']:
        for method, key in [('duckdb', 'duckdb_cold'), ('pyarrow', 'pyarrow_cold')]:
            means[(r['part'], r['variant'], method)] = np.mean([q[key] for q in r['queries']])
    return means


def cmd_prewarm(args):
    plan = json.loads((args.out / 'plan.json').read_text())
    db = Path(plan['db'])
    con = duckdb.connect()
    con.execute('SET enable_progress_bar = false')
    cold = cold_means(args.out)
    started = datetime.now().isoformat(timespec='seconds')
    print(f"host {socket.gethostname()}; filesystem: db {fs_type(db)}, out {fs_type(args.out)}; "
          f"variants {args.variants}; {args.n_queries} queries per file"
          + ('' if cold else '; no results.jsonl in --out, so no break-even estimate'))

    rows = []
    rng = np.random.default_rng()
    for p, centers in plan['parts'].items():
        ra, dec = centers['ra'][:args.n_queries], centers['dec'][:args.n_queries]
        print(f'part {p}:', flush=True)
        for k in rng.permutation(len(args.variants)):
            name = args.variants[k]
            path = partition_path(db, p) if name == 'orig' else variant_path(args.out, p, name)
            res = prewarm_file(path, ra, dec, con)
            res.update(part=p, variant=name, host=socket.gethostname(), started=started)
            with open(args.out / 'prewarm.jsonl', 'a') as f:
                f.write(json.dumps(res) + '\n')
            rows.append(res)

            q = {m: np.array([x[m] for x in res['queries']]) for m in ('duckdb', 'pyarrow')}
            line = (f"  {name:>6s}: warm {res['file_gb']:.2f} GB in {res['t_warm']:5.1f} s "
                    f"({res['warm_mb_s']:5.0f} MB/s), re-read probe {res['probe_mb_s']:6.0f} MB/s | after warming, "
                    f"mean/median ms: duckdb {1e3 * q['duckdb'].mean():5.0f}/{1e3 * np.median(q['duckdb']):4.0f}, "
                    f"pyarrow {1e3 * q['pyarrow'].mean():5.0f}/{1e3 * np.median(q['pyarrow']):4.0f}")
            for m in ('duckdb', 'pyarrow'):
                c = cold.get((p, name, m))
                if c is not None and c > q[m].mean():
                    line += f" | break-even {m}: {res['t_warm'] / (c - q[m].mean()):4.0f} q"
            print(line, flush=True)

    print('\nsummary (means over partitions):')
    print(f"{'variant':>8s} {'warm s':>7s} {'MB/s':>6s} {'probe MB/s':>11s} {'duckdb ms':>10s} {'pyarrow ms':>11s}"
          + (f" {'duckdb cold':>12s} {'break-even q':>13s}" if cold else ''))
    for name in args.variants:
        rs = [r for r in rows if r['variant'] == name]
        if not rs:
            continue
        mean = lambda f: np.mean([f(r) for r in rs])
        d = mean(lambda r: np.mean([x['duckdb'] for x in r['queries']]))
        pa_ = mean(lambda r: np.mean([x['pyarrow'] for x in r['queries']]))
        tw = mean(lambda r: r['t_warm'])
        line = (f"{name:>8s} {tw:7.1f} {mean(lambda r: r['warm_mb_s']):6.0f} "
                f"{mean(lambda r: r['probe_mb_s']):11.0f} {1e3 * d:10.0f} {1e3 * pa_:11.0f}")
        cs = [cold.get((r['part'], name, 'duckdb')) for r in rs]
        if cold and all(c is not None for c in cs):
            c = np.mean(cs)
            line += f" {1e3 * c:12.0f} " + (f"{tw / (c - d):13.0f}" if c > d else f"{'never':>13s}")
        print(line)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('phase', choices=['write', 'bench', 'prewarm'])
    ap.add_argument('--db', type=Path, default=DEFAULT_DB, help='talltable DB dir (with parts.txt and pixels/)')
    ap.add_argument('--out', type=Path, default=DEFAULT_OUT, help='where rewritten files go; must be on ceph too')
    ap.add_argument('--n-files', type=int, default=3)
    ap.add_argument('--n-queries', type=int, default=None,
                    help='queries per file (default: 100 for write/bench, 50 for prewarm)')
    ap.add_argument('--variants', nargs='+', default=['1024k', '256k', '128k', '64k'],
                    choices=['orig'] + list(RG_SIZES), help='prewarm only: which files to test')
    ap.add_argument('--min-gb', type=float, default=1.0, help='partition file size range to sample from')
    ap.add_argument('--max-gb', type=float, default=1.9)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    if args.n_queries is None:
        args.n_queries = 50 if args.phase == 'prewarm' else 100
    {'write': cmd_write, 'bench': cmd_bench, 'prewarm': cmd_prewarm}[args.phase](args)


if __name__ == '__main__':
    main()
