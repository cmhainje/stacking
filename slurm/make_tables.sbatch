#!/bin/bash
#SBATCH -J make-tables
#SBATCH -C icelake
#SBATCH --nodes 4
#SBATCH --ntasks-per-node 24
#SBATCH --cpus-per-task 2
#SBATCH --mem 0
#SBATCH -t 12:00:00
#SBATCH -o /mnt/sdceph/users/%u/stacking/logs/%x-%j.out

# 24 tasks per node because each node's ceph-fuse daemon saturates around there (see
# talltable's slurm/ingest_week.sbatch).
#
# Submit from the repo root, after running scripts/prepare.py with the same CONFIG and OUT:
#
#     sbatch slurm/make_tables.sbatch CONFIG OUT
#
# To resume, submit again; finished partitions are skipped. The merge only runs if every
# task succeeds.

set -euo pipefail
: "${1:?need a config}" "${2:?need an output directory}"
export CONFIG=$1 OUT=$2

export TALLTABLE_DB_DIR=/mnt/sdceph/users/spherex/talltable
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK  # also how many threads pyarrow uses

if ! git diff --quiet HEAD; then
    echo "the code doesn't match the commit; commit and push it first:" >&2
    git status --short --untracked-files=no >&2
    exit 1
fi

# task i runs every ntasks-th partition
srun --cpus-per-task "$SLURM_CPUS_PER_TASK" --kill-on-bad-exit=0 bash -c '
    parts=$(awk -v n="$SLURM_NTASKS" -v i="$SLURM_PROCID" "NR % n == i" "$OUT/parts.txt")
    if [ -n "$parts" ]; then
        exec .venv/bin/python scripts/make_tables.py "$CONFIG" -o "$OUT" -p $parts
    fi
'

.venv/bin/python scripts/merge_tables.py "$OUT" -o "$OUT/galaxies.parquet"
