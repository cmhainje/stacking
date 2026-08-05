#!/bin/bash
set -euo pipefail

DATA_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/data"
DATA_BASE="https://data.desi.lbl.gov/public/dr1/survey/catalogs/dr1/LSS/iron/LSScats/v1.5"
RANDOMS_BASE="https://data.desi.lbl.gov/public/dr1/survey/catalogs/dr1/LSS/iron/LSScats/v1.5"

mkdir -p "$DATA_DIR"

fetch() {
    local url="$1"
    local dest="$2"
    local tmp="${dest}.part"

    if [[ -f "$dest" ]]; then
        echo "Already have $(basename "$dest"), skipping."
        return
    fi

    echo "Downloading $(basename "$dest")..."
    # Download to a .part file and only rename on success, so dest existing
    # always means a complete file. -C - resumes .part from where a prior
    # interrupted run left off instead of restarting from scratch.
    curl -fL -C - -o "$tmp" "$url"
    mv "$tmp" "$dest"
}

# ── Data catalogs ──────────────────────────────────────────────────────────
for CAP in NGC SGC; do
    fetch "${DATA_BASE}/LRG_${CAP}_clustering.dat.fits" \
        "${DATA_DIR}/LRG_${CAP}_clustering.dat.fits"
done

# ── Random catalogs (used for background estimation) ──────────────────────
for CAP in NGC SGC; do
    for N in 0 1; do
        fetch "${RANDOMS_BASE}/LRG_${CAP}_${N}_clustering.ran.fits" \
            "${DATA_DIR}/LRG_${CAP}_${N}_clustering.ran.fits"
    done
done

echo "Done."
